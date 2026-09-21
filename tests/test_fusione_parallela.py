"""Verifica senza GPU delle fasi F (fusione) e G (codifica) parallele.

Dalla radice del repository::

    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python -m unittest tests.test_fusione_parallela -v

La variabile serve solo su macOS, dove ``fork`` non e' il metodo predefinito.
Servono numpy, cv2, PIL e ffmpeg/ffprobe nel PATH; niente torch, niente pytest.
"""

import contextlib
import io
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
from fractions import Fraction
from unittest import mock

import cv2
import numpy as np

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RADICE not in sys.path:
    sys.path.insert(0, RADICE)

from musetalk.utils import fusione_parallela as fp  # noqa: E402
from tests import sintetico  # noqa: E402

# Da Python 3.12 os.fork() in un processo con altri thread emette un avviso.
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*fork.*")

MIN_SEG = 8
RIGOROSO = {"MUSETALK_FG_STRICT": "1", "MUSETALK_FG_STALLO_S": "60"}


def contesto_da(dati, **modifiche):
    campi = dict(
        facce=dati["facce"], frame_ciclo=dati["frame_ciclo"],
        face_boxes=dati["face_boxes"], maschere=dati["maschere"], parsing_mode="jaw",
    )
    campi.update(modifiche)
    return fp.ContestoFusione(**campi)


def impronte_storiche(dati, maschere=None, enhance_fn=None):
    frames = sintetico.fusione_storica(
        dati["facce"], dati["coord_ciclo"], dati["frame_ciclo"],
        dati["maschere"] if maschere is None else maschere,
        dati["extra_margin"], "jaw", enhance_fn,
    )
    return [fp.impronta_frame(f) for f in frames]


def processi_residui(cartella):
    """ffmpeg ancora vivi che scrivono nella cartella del test (None se pgrep manca)."""
    if shutil.which("pgrep") is None:
        return None
    esito = subprocess.run(["pgrep", "-f", cartella], stdout=subprocess.PIPE)
    return [int(p) for p in esito.stdout.split() if int(p) != os.getpid()]


class CasoConCartella(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cartella = self._tmp.name

    def esegui(self, dati, n_workers, nome="video", ambiente=RIGOROSO, fps=25, **opzioni):
        video = os.path.join(self.cartella, f"{nome}.mp4")
        file_impronte = os.path.join(self.cartella, f"{nome}.sha256")
        opzioni.setdefault("use_nvenc", False)
        opzioni.setdefault("min_frame_segmento", MIN_SEG)
        contesto = opzioni.pop("contesto", None) or contesto_da(dati)
        with mock.patch("builtins.print"):
            rapporto = fp.fondi_e_codifica(
                contesto, video, fps, n_workers=n_workers, dir_lavoro=self.cartella,
                percorso_impronte=file_impronte, mostra_progresso=False,
                ambiente=dict(ambiente), **opzioni)
        with open(file_impronte) as f:
            impronte = f.read().split()
        return rapporto, impronte, video


class TestFunzioniPure(unittest.TestCase):
    def test_segmenti_contigui_bilanciati(self):
        for n_frame, n_seg, minimo in [(64, 7, 8), (23599, 16, 250), (75, 4, 8), (1000, 3, 250)]:
            segmenti = fp.calcola_segmenti(n_frame, n_seg, minimo)
            self.assertEqual(segmenti[0][0], 0)
            self.assertEqual(segmenti[-1][1], n_frame)
            for (_, fine), (inizio, _) in zip(segmenti, segmenti[1:]):
                self.assertEqual(fine, inizio)
            lunghezze = [fine - inizio for inizio, fine in segmenti]
            self.assertLessEqual(max(lunghezze) - min(lunghezze), 1)
            self.assertGreaterEqual(min(lunghezze), minimo)

    def test_segmenti_rispettano_il_minimo(self):
        self.assertEqual(len(fp.calcola_segmenti(499, 16, 250)), 1)
        self.assertEqual(len(fp.calcola_segmenti(500, 16, 250)), 2)
        self.assertEqual(len(fp.calcola_segmenti(10, 16, 1)), 3)  # minimo invalicabile: 3 frame
        self.assertEqual(fp.calcola_segmenti(0, 4, 8), [])

    def test_precedenza_n_workers(self):
        self.assertEqual(fp.risolvi_n_workers(5, {"MUSETALK_BLEND_WORKERS": "3"}, 64, "linux")[0], 5)
        self.assertEqual(fp.risolvi_n_workers(None, {"MUSETALK_BLEND_WORKERS": "3"}, 64, "linux"),
                         (3, "env"))
        self.assertEqual(fp.risolvi_n_workers(None, {}, 64, "linux"), (16, "auto"))
        self.assertEqual(fp.risolvi_n_workers(None, {}, 8, "linux"), (4, "auto"))
        self.assertEqual(fp.risolvi_n_workers(None, {}, 1, "linux")[0], 1)
        self.assertEqual(fp.risolvi_n_workers(None, {}, 64, "darwin")[0], 1)
        self.assertEqual(fp.risolvi_n_workers(None, {}, 64, "win32")[0], 1)

    def test_interruttore_di_emergenza_fallisce_chiuso(self):
        """Un valore sbagliato deve spegnere il parallelo, non riaccendere l'automatico."""
        for valore in ("abc", "0", "-4", "2.5"):
            with mock.patch("builtins.print"):
                self.assertEqual(
                    fp.risolvi_n_workers(None, {"MUSETALK_BLEND_WORKERS": valore}, 64, "linux"),
                    (1, "env non valida"))
        self.assertEqual(fp.risolvi_n_workers(None, {"MUSETALK_BLEND_WORKERS": ""}, 8, "linux"),
                         (4, "auto"))

    def test_secondi_da_env(self):
        with mock.patch("builtins.print"):
            for valore in ("abc", "0", "-3", "nan", "inf"):
                self.assertEqual(fp._secondi_da_env({"X": valore}, "X", 300.0), 300.0, valore)
        self.assertEqual(fp._secondi_da_env({"X": "2.5"}, "X", 300.0), 2.5)
        self.assertEqual(fp._secondi_da_env({}, "X", 300.0), 300.0)

    def test_durate_nella_lista_concat(self):
        """Indipendente dalla versione di ffmpeg: durata esatta per ogni segmento."""
        with tempfile.TemporaryDirectory() as cartella:
            percorsi = [os.path.join(cartella, f"seg_{k:03d}.mp4") for k in range(16)]
            for fps, lunghezze in ((24, [251] * 16), (30, [19, 19, 19, 18]), (60, [251] * 16),
                                   (25, [19, 19, 19, 18])):
                with mock.patch.object(fp.subprocess, "run") as finto_run:
                    fp.concatena_segmenti(percorsi[:len(lunghezze)], "uscita.mp4", lunghezze, fps)
                self.assertIn("-c", finto_run.call_args[0][0])
                with open(os.path.join(cartella, "lista.txt")) as f:
                    righe = f.read().splitlines()
                self.assertEqual(righe[0::2], [f"file 'seg_{k:03d}.mp4'" for k in range(len(lunghezze))])
                micro = [round(float(r.split()[1]) * 1000000) for r in righe[1::2]]
                self.assertTrue(all(r.startswith("duration ") for r in righe[1::2]))
                self.assertEqual(sum(micro), round(Fraction(sum(lunghezze) * 1000000, fps)))
                # L'inizio di ogni segmento cade sul microsecondo piu' vicino al valore
                # esatto: l'errore non si accumula con il numero di segmenti.
                inizio_us, frame_prima = 0, 0
                for n, durata in zip(lunghezze, micro):
                    self.assertEqual(inizio_us, round(Fraction(frame_prima * 1000000, fps)))
                    inizio_us += durata
                    frame_prima += n

    def test_quota_cgroup(self):
        def lettore(file):
            return lambda percorso: file.get(percorso)

        self.assertEqual(fp.quota_cpu_cgroup(lettore({"/sys/fs/cgroup/cpu.max": "800000 100000\n"})), 8)
        self.assertIsNone(fp.quota_cpu_cgroup(lettore({"/sys/fs/cgroup/cpu.max": "max 100000\n"})))
        self.assertIsNone(fp.quota_cpu_cgroup(lettore({})))  # file assente = nessun limite
        self.assertEqual(fp.quota_cpu_cgroup(lettore({"/sys/fs/cgroup/cpu.max": "150000 100000"})), 1)
        v1 = {"/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "2000000\n",
              "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000\n"}
        self.assertEqual(fp.quota_cpu_cgroup(lettore(v1)), 20)
        v1["/sys/fs/cgroup/cpu/cpu.cfs_quota_us"] = "-1\n"
        self.assertIsNone(fp.quota_cpu_cgroup(lettore(v1)))
        annidato = {
            "/proc/self/cgroup": "0::/kubepods/pod1/ctr\n",
            "/sys/fs/cgroup/kubepods/cpu.max": "max 100000",
            "/sys/fs/cgroup/kubepods/pod1/cpu.max": "1600000 100000",
            "/sys/fs/cgroup/kubepods/pod1/ctr/cpu.max": "3200000 100000",
        }
        self.assertEqual(fp.quota_cpu_cgroup(lettore(annidato)), 16)  # vince l'antenato piu' stretto

    def test_cpu_effettive(self):
        nessun_file = lambda percorso: None  # noqa: E731
        cpu, dettagli = fp.risolvi_cpu_effettive(
            {"SLURM_CPUS_PER_TASK": "8"}, nessun_file, lambda: 80, lambda: 112)
        self.assertEqual(cpu, 8)
        self.assertEqual(dettagli, {"slurm": 8, "affinity": 80, "cgroup": None, "os": 112})
        self.assertEqual(fp.risolvi_cpu_effettive({}, nessun_file, lambda: None, lambda: 12)[0], 12)
        self.assertEqual(fp.risolvi_cpu_effettive({"SLURM_CPUS_PER_TASK": "x"}, nessun_file,
                                                  lambda: 6, lambda: 12)[0], 6)
        self.assertEqual(fp.risolvi_cpu_effettive({}, nessun_file, lambda: None, lambda: None)[0], 1)

    def test_thread_x264(self):
        self.assertEqual(fp.risolvi_thread_x264({}, 32, 16), 2)
        self.assertEqual(fp.risolvi_thread_x264({}, 72, 4), 8)  # tetto
        self.assertEqual(fp.risolvi_thread_x264({}, 4, 8), 1)
        self.assertEqual(fp.risolvi_thread_x264({"MUSETALK_X264_THREADS": "3"}, 32, 16), 3)

    def test_riga_di_comando_storica(self):
        """Senza parametri nuovi, ffmpeg riceve esattamente la riga di comando di prima."""
        comandi = []

        class FintoPopen:
            def __init__(self, cmd, **kwargs):
                comandi.append((list(cmd), kwargs))
                self.stdin = io.BytesIO()

            def wait(self):
                return 0

            def kill(self):
                pass

        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        with mock.patch.object(fp.subprocess, "Popen", FintoPopen):
            fp._write_video_pipe([frame], "uscita.mp4", 6, 4, 25, codec="h264_nvenc")
            fp._write_video_pipe([frame], "uscita.mp4", 6, 4, 30, codec="libx264")
        comune = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-f", "rawvideo",
                  "-pix_fmt", "bgr24", "-s", "6x4", "-r"]
        self.assertEqual(comandi[0][0], comune + [
            "25", "-i", "-", "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "18",
            "-b:v", "3M", "-maxrate", "5M", "-bufsize", "10M", "-pix_fmt", "yuv420p", "uscita.mp4"])
        self.assertEqual(comandi[1][0], comune + [
            "30", "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
            "-pix_fmt", "yuv420p", "uscita.mp4"])
        self.assertNotIn("preexec_fn", comandi[0][1])

    def test_motivi_del_percorso_sequenziale(self):
        dati = sintetico.crea_input(n_frame=40)
        ctx = contesto_da(dati)
        self.assertIsNone(fp._motivo_percorso_sequenziale(ctx, 4, MIN_SEG, "processo"))
        self.assertIn("n_workers=1", fp._motivo_percorso_sequenziale(ctx, 1, MIN_SEG, "processo"))
        self.assertIn("troppo pochi", fp._motivo_percorso_sequenziale(ctx, 4, 250, "processo"))
        dispari = sintetico.crea_input(n_frame=40, altezza=191, larghezza=255)
        self.assertIn("dispari", fp._motivo_percorso_sequenziale(
            contesto_da(dispari), 4, MIN_SEG, "processo"))
        valido = next(i for i, b in enumerate(dati["face_boxes"]) if b is not None)
        senza = list(dati["maschere"])
        senza[valido] = None
        self.assertIn("senza maschera", fp._motivo_percorso_sequenziale(
            contesto_da(dati, maschere=senza), 4, MIN_SEG, "processo"))
        storta = list(dati["maschere"])
        storta[valido] = storta[valido][:-1, :]
        self.assertIn("maschera", fp._motivo_percorso_sequenziale(
            contesto_da(dati, maschere=storta), 4, MIN_SEG, "processo"))


class TestParitaFusione(CasoConCartella):
    """Criterio 1: la sequenza dei frame della fase F e' identica al byte."""

    def test_storica_sequenziale_parallela(self):
        dati = sintetico.crea_input(n_frame=64)
        riferimento = impronte_storiche(dati)
        originali = [fp.impronta_frame(dati["frame_ciclo"][i % len(dati["frame_ciclo"])])
                     for i in range(dati["n_frame"])]
        fusi = sum(1 for a, b in zip(riferimento, originali) if a != b)
        self.assertGreater(fusi, 40, "la fusione deve davvero modificare i frame con volto")
        self.assertLess(fusi, 64, "i frame segnaposto devono restare intatti")

        rapporto, impronte, _ = self.esegui(dati, 1, "seq")
        self.assertEqual(rapporto.modo, "sequenziale")
        self.assertEqual(impronte, riferimento)
        for n in (2, 3, 4, 7):
            rapporto, impronte, _ = self.esegui(dati, n, f"par{n}")
            self.assertEqual((rapporto.modo, rapporto.n_segmenti, rapporto.fallback),
                             ("parallelo", n, False))
            self.assertEqual(impronte, riferimento, f"n_workers={n}")

    def test_esecutore_in_linea(self):
        dati = sintetico.crea_input(n_frame=48, seme=3)
        rapporto, impronte, _ = self.esegui(dati, 3, "linea", esecutore="in_linea")
        self.assertEqual((rapporto.modo, rapporto.n_segmenti), ("parallelo", 3))
        self.assertEqual(impronte, impronte_storiche(dati))

    def test_tipi_di_bbox(self):
        for tipo in ("tupla", "numpy"):
            dati = sintetico.crea_input(n_frame=40, seme=5, tipo_bbox=tipo)
            _, impronte, _ = self.esegui(dati, 3, f"bbox_{tipo}")
            self.assertEqual(impronte, impronte_storiche(dati), tipo)

    def test_maschere_lunghe_quanto_il_ciclo(self):
        """Cache hit: 2K maschere per n_frame > 2K. Lo storico andrebbe fuori indice."""
        dati = sintetico.crea_input(n_originali=6, n_frame=50, seme=9, lunghezza_maschere="ciclo")
        self.assertEqual(len(dati["maschere"]), 12)
        with self.assertRaises(IndexError):
            impronte_storiche(dati)
        estese = [dati["maschere"][i % 12] for i in range(50)]
        riferimento = impronte_storiche(dati, maschere=estese)
        _, sequenziali, _ = self.esegui(dati, 1, "ciclo_seq")
        _, parallele, _ = self.esegui(dati, 4, "ciclo_par")
        self.assertEqual(sequenziali, riferimento)
        self.assertEqual(parallele, riferimento)

    def test_miglioratore_in_linea_e_preliminare(self):
        """GFPGAN finto: in linea (sequenziale) == passaggio preliminare (parallelo)."""
        dati = sintetico.crea_input(n_frame=40, seme=11)
        inverti = lambda faccia: 255 - faccia  # noqa: E731 - applicarlo due volte si vedrebbe
        riferimento = impronte_storiche(dati, enhance_fn=inverti)
        self.assertNotEqual(riferimento, impronte_storiche(dati))
        _, sequenziali, _ = self.esegui(dati, 1, "gf_seq", migliora=inverti)
        _, parallele, _ = self.esegui(dati, 4, "gf_par", migliora=inverti)
        self.assertEqual(sequenziali, riferimento)
        self.assertEqual(parallele, riferimento)
        self.assertEqual(len(dati["facce"]), 40)  # la lista originale non viene toccata

    def test_bbox_degenere_frame_non_fuso(self):
        """cv2.resize solleva su una bbox a larghezza zero: il frame resta quello originale."""
        dati = sintetico.crea_input(n_frame=40, seme=15, con_bbox_degenere=True)
        degeneri = [i for i, b in enumerate(dati["face_boxes"]) if b is not None and b[0] == b[2]]
        self.assertTrue(degeneri)
        riferimento = impronte_storiche(dati)
        for i in degeneri:
            originale = dati["frame_ciclo"][i % len(dati["frame_ciclo"])]
            self.assertEqual(riferimento[i], fp.impronta_frame(originale))
        _, sequenziali, _ = self.esegui(dati, 1, "degenere_seq")
        _, parallele, _ = self.esegui(dati, 3, "degenere_par")
        self.assertEqual(sequenziali, riferimento)
        self.assertEqual(parallele, riferimento)

    def test_verifica_interna_di_parita(self):
        dati = sintetico.crea_input(n_frame=40, seme=13)
        ambiente = dict(RIGOROSO, MUSETALK_FG_VERIFICA="1")
        rapporto, _, _ = self.esegui(dati, 3, "autoverifica", ambiente=ambiente)
        self.assertIs(rapporto.parita, True)

    def test_verifica_interna_rileva_una_differenza(self):
        """Un figlio che altera un solo bit di un frame deve far dire KO all'autoverifica."""
        pid_padre = os.getpid()
        vero = fp.fondi_frame

        def alterato(i, ctx, migliora=None):
            frame = vero(i, ctx, migliora)
            if os.getpid() != pid_padre and i == 22:
                frame = frame.copy()
                frame[0, 0, 0] ^= 1
            return frame

        dati = sintetico.crea_input(n_frame=40, seme=17, con_segnaposto=False)
        with mock.patch.object(fp, "fondi_frame", alterato):
            rapporto, _, _ = self.esegui(dati, 3, "ko", ambiente={"MUSETALK_FG_VERIFICA": "1"})
            self.assertIs(rapporto.parita, False)
            with self.assertRaisesRegex(fp.ErroreParallelo, "parita violata al frame 22"):
                self.esegui(dati, 3, "ko_rigoroso", ambiente=dict(RIGOROSO, MUSETALK_FG_VERIFICA="1"))

    def test_configurazione_di_produzione(self):
        """Come la chiama generate(): nessuna impronta, barra di avanzamento attiva."""
        dati = sintetico.crea_input(n_frame=48, seme=19, con_segnaposto=False, maschera_piena=True)
        for n_workers, modo in ((1, "sequenziale"), (4, "parallelo")):
            video = os.path.join(self.cartella, f"prod_{n_workers}.mp4")
            with mock.patch("builtins.print"), contextlib.redirect_stderr(io.StringIO()):
                rapporto = fp.fondi_e_codifica(
                    contesto_da(dati), video, 25, use_nvenc=False, n_workers=n_workers,
                    dir_lavoro=self.cartella, min_frame_segmento=MIN_SEG, ambiente=dict(RIGOROSO))
            self.assertEqual(rapporto.modo, modo)
            frames = sintetico.decodifica_video(video, dati["larghezza"], dati["altezza"])
            letti = [sintetico.leggi_codice(f, dati["face_boxes"][i]) for i, f in enumerate(frames)]
            self.assertEqual(letti, list(range(48)))
        self.assertEqual([n for n in os.listdir(self.cartella) if not n.startswith("prod_")], [])


class TestVideo(CasoConCartella):
    """Criterio 2: stessi frame, stesso fps, stessa durata, nessun buco ai confini."""

    def controlla(self, fps):
        n_frame = 75
        dati = sintetico.crea_input(n_frame=n_frame, seme=fps, con_segnaposto=False,
                                    maschera_piena=True)
        _, _, video_seq = self.esegui(dati, 1, f"seq{fps}", fps=fps)
        rapporto, _, video_par = self.esegui(dati, 4, f"par{fps}", fps=fps, mantieni_segmenti=True)
        self.assertEqual([19, 19, 19, 18],
                         [len(sintetico.pacchetti(p)) for p in rapporto.percorsi_segmenti])

        info_seq, info_par = sintetico.info_video(video_seq), sintetico.info_video(video_par)
        for info in (info_seq, info_par):
            self.assertEqual(int(info["nb_read_packets"]), n_frame)
            self.assertEqual(info["r_frame_rate"], f"{fps}/1")
            self.assertEqual(info["durata_esatta"], Fraction(n_frame, fps))
        self.assertLessEqual(abs(info_seq["durata_contenitore"] - info_par["durata_contenitore"]), 0.01)
        self.assertEqual(info_seq["time_base"], info_par["time_base"])

        tick = Fraction(1, fps) / Fraction(info_par["time_base"])
        for percorso in [video_par] + rapporto.percorsi_segmenti:
            pts_dts = sintetico.pacchetti(percorso)
            dts = [d for _, d in pts_dts]
            pts = sorted(p for p, _ in pts_dts)
            self.assertEqual(pts[0], 0, f"{percorso}: il primo pts deve essere 0")
            self.assertEqual({b - a for a, b in zip(dts, dts[1:])}, {tick}, percorso)
            self.assertEqual({b - a for a, b in zip(pts, pts[1:])}, {tick}, percorso)

        for video in (video_seq, video_par):
            frames = sintetico.decodifica_video(video, dati["larghezza"], dati["altezza"])
            self.assertEqual(len(frames), n_frame)
            letti = [sintetico.leggi_codice(f, dati["face_boxes"][i]) for i, f in enumerate(frames)]
            self.assertEqual(letti, list(range(n_frame)), "frame duplicati, mancanti o fuori ordine")

    def test_25_fps(self):
        self.controlla(25)

    def test_24_fps(self):
        self.controlla(24)

    def test_30_fps(self):
        self.controlla(30)

    def test_concat_senza_ricodifica(self):
        dati = sintetico.crea_input(n_frame=70, seme=21, con_segnaposto=False)
        rapporto, _, video = self.esegui(dati, 3, "copia", mantieni_segmenti=True)
        self.assertEqual(len(rapporto.percorsi_segmenti), 3)
        slice_segmenti, decodificati_segmenti, avcc = [], [], set()
        for percorso in rapporto.percorsi_segmenti:
            slice_segmenti += sintetico.impronte_slice(percorso)
            decodificati_segmenti += sintetico.impronte_decodificate(percorso)
            avcc.add(fp.impronta_avcc(percorso))
        self.assertGreaterEqual(len(slice_segmenti), 70)
        self.assertEqual(sintetico.impronte_slice(video), slice_segmenti,
                         "i dati codificati devono essere copiati, non ricodificati")
        self.assertEqual(sintetico.impronte_decodificate(video), decodificati_segmenti)
        self.assertEqual(len(decodificati_segmenti), 70)
        self.assertEqual(len(avcc), 1)
        self.assertNotIn(None, avcc, "l'avcC deve essere leggibile dai box mp4")

    def test_segmenti_difformi_rifiutati(self):
        """ffmpeg concatenerebbe in silenzio: la guardia deve accorgersene."""
        percorsi = []
        for k, lato in enumerate((64, 96)):
            percorso = os.path.join(self.cartella, f"seg_{k:03d}.mp4")
            frames = [np.full((lato, lato, 3), 40 * k, dtype=np.uint8)] * 12
            fp._write_video_pipe(frames, percorso, lato, lato, 25, codec="libx264",
                                 percorso_log=os.path.join(self.cartella, "ffmpeg.log"))
            percorsi.append(percorso)
        with self.assertRaises(fp.ErroreParallelo):
            fp.verifica_segmenti(percorsi, [12, 12])
        with self.assertRaises(fp.ErroreParallelo):
            fp.verifica_segmenti(percorsi[:1], [11])
        fp.verifica_segmenti(percorsi[:1], [12])

    def _due_segmenti_uguali(self):
        percorsi = []
        for k in range(2):
            percorso = os.path.join(self.cartella, f"seg_{k:03d}.mp4")
            frames = [np.full((64, 64, 3), 30 + 9 * j, dtype=np.uint8) for j in range(12)]
            fp._write_video_pipe(frames, percorso, 64, 64, 25, codec="libx264",
                                 percorso_log=os.path.join(self.cartella, "ffmpeg.log"))
            percorsi.append(percorso)
        return percorsi

    def test_avcc_diverso_rifiutato(self):
        """Stessi campi ffprobe ma SPS/PPS diversi: decide l'impronta dell'avcC."""
        percorsi = self._due_segmenti_uguali()
        fp.verifica_segmenti(percorsi, [12, 12])
        finta = lambda percorso: "a" if percorso.endswith("seg_000.mp4") else "b"  # noqa: E731
        with mock.patch.object(fp, "impronta_avcc", finta):
            with self.assertRaisesRegex(fp.ErroreParallelo, "non uniforme"):
                fp.verifica_segmenti(percorsi, [12, 12])

    def test_verifica_video(self):
        percorsi = self._due_segmenti_uguali()
        uscita = os.path.join(self.cartella, "unito.mp4")
        fp.concatena_segmenti(percorsi, uscita, [12, 12], 25)
        fp.verifica_video(uscita, 24, 25)
        with self.assertRaisesRegex(fp.ErroreParallelo, "pacchetti invece di"):
            fp.verifica_video(uscita, 25, 25)
        with self.assertRaisesRegex(fp.ErroreParallelo, "durata"):
            fp.verifica_video(uscita, 24, 50)

    def test_guardia_prima_del_concat_nella_pipeline(self):
        """Un segmento con un frame in meno deve fermare il concat, non passare inosservato."""
        vero = fp._write_video_pipe

        def perde_un_frame(frames, output_path, *args, **kw):
            if output_path.endswith("seg_001.mp4"):
                iteratore = iter(frames)
                next(iteratore)
                frames = iteratore
            return vero(frames, output_path, *args, **kw)

        dati = sintetico.crea_input(n_frame=48, seme=23)
        with mock.patch.object(fp, "_write_video_pipe", perde_un_frame):
            with self.assertRaisesRegex(fp.ErroreParallelo, "frame invece di"):
                self.esegui(dati, 3, "frame_perso")

    def test_guardia_dopo_il_concat_nella_pipeline(self):
        vero = fp.concatena_segmenti

        def dimentica_l_ultimo(percorsi, uscita, lunghezze, fps):
            return vero(percorsi[:-1], uscita, lunghezze[:-1], fps)

        dati = sintetico.crea_input(n_frame=48, seme=25)
        with mock.patch.object(fp, "concatena_segmenti", dimentica_l_ultimo):
            with self.assertRaisesRegex(fp.ErroreParallelo, "pacchetti invece di"):
                self.esegui(dati, 3, "segmento_perso")


class TestRipieghi(CasoConCartella):
    """Criterio 3 e politica di fallimento."""

    def _salta_se_nvenc(self, dati):
        # Stessa geometria del job: sotto la risoluzione minima NVENC fallirebbe comunque.
        if fp.sonda_nvenc(dati["larghezza"], dati["altezza"], 25, 2)[0]:
            self.skipTest("questa macchina ha NVENC: il ripiego non e' osservabile")

    def test_nvenc_assente(self):
        dati = sintetico.crea_input(n_frame=40, seme=31)
        self._salta_se_nvenc(dati)
        rapporto, impronte, video = self.esegui(dati, 3, "senza_nvenc", use_nvenc=True)
        self.assertEqual((rapporto.modo, rapporto.codec, rapporto.n_segmenti),
                         ("parallelo", "libx264", 3))
        self.assertEqual(impronte, impronte_storiche(dati))
        self.assertEqual(int(sintetico.info_video(video)["nb_read_packets"]), 40)

    def test_nvenc_fallisce_su_un_segmento(self):
        """Sonda OK, poi un segmento nvenc cade: un solo nuovo tentativo, tutto libx264."""
        vero = fp._write_video_pipe

        def finto(frames, output_path, width, height, fps, codec="h264_nvenc", **kw):
            if codec != "h264_nvenc":
                return vero(frames, output_path, width, height, fps, codec=codec, **kw)
            if output_path.endswith("seg_001.mp4"):
                next(iter(frames))
                raise subprocess.CalledProcessError(234, ["ffmpeg"])
            return vero(frames, output_path, width, height, fps, codec="libx264", **kw)

        dati = sintetico.crea_input(n_frame=48, seme=33)
        with mock.patch.object(fp, "_write_video_pipe", finto), \
                mock.patch.object(fp, "sonda_nvenc", lambda *a, **k: (2, 0)):
            rapporto, impronte, video = self.esegui(dati, 4, "nvenc_ko", use_nvenc=True)
        self.assertEqual((rapporto.modo, rapporto.codec, rapporto.tentativi, rapporto.n_segmenti),
                         ("parallelo", "libx264", 2, 4))
        self.assertEqual(impronte, impronte_storiche(dati))
        self.assertEqual(int(sintetico.info_video(video)["nb_read_packets"]), 48)

    def test_nvenc_assente_nel_percorso_sequenziale(self):
        """Il ripiego storico nvenc -> libx264 con n_workers=1 (interruttore di emergenza)."""
        dati = sintetico.crea_input(n_frame=24, seme=32)
        self._salta_se_nvenc(dati)
        video = os.path.join(self.cartella, "seq_nvenc.mp4")
        with mock.patch("builtins.print") as stampa, contextlib.redirect_stderr(io.StringIO()):
            rapporto = fp.fondi_e_codifica(
                contesto_da(dati), video, 25, use_nvenc=True, n_workers=1,
                dir_lavoro=self.cartella, mostra_progresso=False, ambiente={})
        righe = [str(c.args[0]) for c in stampa.call_args_list if c.args]
        self.assertEqual((rapporto.modo, rapporto.codec), ("sequenziale", "libx264"))
        self.assertTrue(any(r.startswith("[phase G] h264_nvenc failed (rc=") and
                            r.endswith("; fallback to libx264") for r in righe), righe)
        self.assertEqual(int(sintetico.info_video(video)["nb_read_packets"]), 24)

    def test_sonda_nvenc_apre_le_sessioni_insieme(self):
        """Con un ffmpeg finto: le S sessioni devono risultare aperte nello stesso istante."""
        finto = os.path.join(self.cartella, "bin")
        os.makedirs(finto)
        registro = os.path.join(self.cartella, "sessioni")
        os.makedirs(registro)
        with open(os.path.join(finto, "ffmpeg"), "w") as f:
            f.write("#!" + sys.executable + "\n"
                    "import os, sys\n"
                    "sys.stdin.buffer.read(1)\n"
                    f"cartella = {registro!r}\n"
                    "mio = os.path.join(cartella, str(os.getpid()))\n"
                    "open(mio, 'w').close()\n"
                    f"if len(os.listdir(cartella)) > {2}: os.remove(mio); sys.exit(1)\n"
                    "open(os.path.join(cartella, '..', 'picco_%d' % len(os.listdir(cartella))), 'w').close()\n"
                    "while sys.stdin.buffer.read(1 << 20): pass\n"
                    "os.remove(mio)\n")
        os.chmod(os.path.join(finto, "ffmpeg"), 0o755)
        with mock.patch.dict(os.environ, {"PATH": finto + os.pathsep + os.environ["PATH"]}):
            self.assertEqual(fp.sonda_nvenc(64, 64, 25, 8, timeout_s=20), (2, 0))
        self.assertIn("picco_2", os.listdir(self.cartella), "due sessioni aperte insieme")

    def test_sessioni_nvenc_da_env(self):
        vero = fp._write_video_pipe
        richieste = []

        def finto(frames, output_path, width, height, fps, codec="h264_nvenc", **kw):
            return vero(frames, output_path, width, height, fps, codec="libx264", **kw)

        def sonda(larghezza, altezza, fps, sessioni, **kw):
            richieste.append(sessioni)
            return sessioni, 0

        dati = sintetico.crea_input(n_frame=48, seme=34)
        ambiente = dict(RIGOROSO, MUSETALK_NVENC_SESSIONS="3")
        with mock.patch.object(fp, "_write_video_pipe", finto), \
                mock.patch.object(fp, "sonda_nvenc", sonda):
            rapporto, _, _ = self.esegui(dati, 4, "env_sessioni", use_nvenc=True, ambiente=ambiente)
        self.assertEqual(richieste, [3])
        self.assertEqual((rapporto.codec, rapporto.n_segmenti), ("h264_nvenc", 3))

    def test_una_sola_sessione_nvenc(self):
        dati = sintetico.crea_input(n_frame=24, seme=36)
        with mock.patch.object(fp, "sonda_nvenc", lambda *a, **k: (1, 0)):
            with self.assertRaisesRegex(fp.ErroreParallelo, "una sola sessione nvenc"):
                self.esegui(dati, 4, "una_sessione", use_nvenc=True)

    def test_tetto_sessioni_nvenc(self):
        vero = fp._write_video_pipe

        def finto(frames, output_path, width, height, fps, codec="h264_nvenc", **kw):
            return vero(frames, output_path, width, height, fps, codec="libx264", **kw)

        dati = sintetico.crea_input(n_frame=48, seme=35)
        with mock.patch.object(fp, "_write_video_pipe", finto), \
                mock.patch.object(fp, "sonda_nvenc", lambda *a, **k: (2, 0)):
            rapporto, _, _ = self.esegui(dati, 4, "tetto", use_nvenc=True)
        self.assertEqual((rapporto.codec, rapporto.n_segmenti, rapporto.tentativi),
                         ("h264_nvenc", 2, 1))

    def _fragile(self, indice_fatale):
        pid_padre = os.getpid()
        vero = fp.fondi_frame

        def fragile(i, ctx, migliora=None):
            if os.getpid() != pid_padre and i == indice_fatale:
                os._exit(7)
            return vero(i, ctx, migliora)

        return fragile

    def test_worker_morto_ripiego_sequenziale(self):
        dati = sintetico.crea_input(n_frame=48, seme=37)
        inverti = lambda faccia: 255 - faccia  # noqa: E731
        with mock.patch.object(fp, "fondi_frame", self._fragile(29)):
            rapporto, impronte, video = self.esegui(dati, 4, "crollo", ambiente={}, migliora=inverti)
        self.assertEqual((rapporto.modo, rapporto.fallback), ("sequenziale", True))
        self.assertEqual(impronte, impronte_storiche(dati, enhance_fn=inverti),
                         "il ripiego non deve applicare il miglioramento due volte")
        self.assertEqual(int(sintetico.info_video(video)["nb_read_packets"]), 48)
        residui = [n for n in os.listdir(self.cartella) if n.startswith("fg_")]
        self.assertEqual(residui, [], "le cartelle dei tentativi vanno rimosse")
        self.assertEqual(multiprocessing.active_children(), [])
        self.assertIn(processi_residui(self.cartella), (None, []), "ffmpeg orfani dopo il ripiego")

    def test_stallo_rilevato_e_superstiti_uccisi(self):
        pid_padre = os.getpid()
        vero = fp.fondi_frame

        def si_pianta(i, ctx, migliora=None):
            if os.getpid() != pid_padre and i == 30:
                time.sleep(3600)
            return vero(i, ctx, migliora)

        dati = sintetico.crea_input(n_frame=48, seme=45)
        ambiente = dict(RIGOROSO, MUSETALK_FG_STALLO_S="2")
        inizio = time.monotonic()
        with mock.patch.object(fp, "fondi_frame", si_pianta):
            with self.assertRaisesRegex(fp.ErroreParallelo, "stallo"):
                self.esegui(dati, 4, "stallo", ambiente=ambiente)
        self.assertLess(time.monotonic() - inizio, 30)
        self.assertEqual(multiprocessing.active_children(), [])
        self.assertIn(processi_residui(self.cartella), (None, []), "worker o ffmpeg sopravvissuti")

    def test_automatico_troppo_corto_non_solleva(self):
        """Rigoroso + dimensionamento automatico: un job corto va in sequenziale senza errori."""
        dati = sintetico.crea_input(n_frame=24, seme=47)
        with mock.patch.object(fp, "risolvi_n_workers", lambda *a, **k: (4, "auto")):
            rapporto, impronte, _ = self.esegui(dati, None, "auto_corto", min_frame_segmento=250)
        self.assertEqual(rapporto.modo, "sequenziale")
        self.assertEqual(impronte, impronte_storiche(dati))

    def test_worker_morto_in_modo_rigoroso(self):
        dati = sintetico.crea_input(n_frame=48, seme=39)
        with mock.patch.object(fp, "fondi_frame", self._fragile(5)):
            with self.assertRaises(fp.ErroreParallelo):
                self.esegui(dati, 4, "rigoroso")

    def test_eccezione_nel_worker(self):
        pid_padre = os.getpid()
        vero = fp.fondi_frame

        def esplode(i, ctx, migliora=None):
            if os.getpid() != pid_padre and i == 17:
                raise ValueError("dato corrotto")
            return vero(i, ctx, migliora)

        dati = sintetico.crea_input(n_frame=48, seme=41)
        with mock.patch.object(fp, "fondi_frame", esplode):
            with self.assertRaisesRegex(fp.ErroreParallelo, "dato corrotto"):
                self.esegui(dati, 4, "eccezione")

    def test_parallelo_impraticabile_in_modo_rigoroso(self):
        dati = sintetico.crea_input(n_frame=40, seme=43)
        with self.assertRaises(fp.ErroreParallelo):
            self.esegui(dati, 4, "pochi", min_frame_segmento=250)
        rapporto, impronte, _ = self.esegui(dati, 4, "pochi_ok", ambiente={}, min_frame_segmento=250)
        self.assertEqual(rapporto.modo, "sequenziale")
        self.assertEqual(impronte, impronte_storiche(dati))


class TestCv2DopoFork(CasoConCartella):
    def test_resize_nel_figlio_dopo_pool_nel_padre(self):
        """Il padre usa prima il parallelismo di OpenCV; i figli non devono bloccarsi."""
        grande = np.random.RandomState(0).randint(0, 255, (1600, 1600, 3)).astype(np.uint8)
        cv2.resize(grande, (800, 800))
        cv2.GaussianBlur(grande, (31, 31), 0)
        # Volti grandi come in produzione: sotto ~98k pixel cv2.resize gira in linea e
        # il figlio non toccherebbe mai il pool, rendendo il test inutile.
        dati = sintetico.crea_input(n_frame=40, seme=51, altezza=768, larghezza=768,
                                    volto=(350, 440), rumore=2)
        aree = [(b[2] - b[0]) * (b[3] - b[1]) for b in dati["face_boxes"] if b is not None]
        self.assertGreater(min(aree), 98304)

        eventi = []
        vero_set, vero_fork = cv2.setNumThreads, os.fork

        def spia_set(n):
            eventi.append(("set", n))
            return vero_set(n)

        def spia_fork():
            pid = vero_fork()
            if pid:
                eventi.append(("fork",))
            return pid

        ambiente = dict(RIGOROSO, MUSETALK_FG_STALLO_S="30")
        with mock.patch.object(cv2, "setNumThreads", spia_set), \
                mock.patch.object(os, "fork", spia_fork):
            rapporto, impronte, _ = self.esegui(dati, 4, "dopo_pool", ambiente=ambiente)
        self.assertEqual(rapporto.modo, "parallelo")
        self.assertEqual(impronte, impronte_storiche(dati))
        # 1 thread come ultima azione prima dei fork, valore predefinito subito dopo.
        self.assertEqual(eventi, [("set", 1)] + [("fork",)] * 4 + [("set", -1)])
        # Il padre deve ritrovare il parallelismo predefinito di OpenCV.
        cv2.resize(grande, (800, 800))


if __name__ == "__main__":
    unittest.main(verbosity=2)
