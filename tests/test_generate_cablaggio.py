"""Cablaggio di ``MuseTalkInference.generate()`` con le fasi F e G, senza GPU.

Si carica il vero ``api/inference_service.py`` sostituendo con controfigure solo
cio' che richiede torch o i pesi (whisper, UNet, VAE, BiSeNet, GFPGAN, face
detection). Tutto il resto e' codice di produzione: estrazione dei frame,
preparazione delle bbox, costruzione delle maschere, fusione, codifica, concat,
mux dell'audio. Il file viene caricato per percorso, perche' ``api/__init__``
importerebbe FastAPI e torch.

    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python -m unittest tests.test_generate_cablaggio -v
"""

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
import warnings
from unittest import mock

import cv2
import numpy as np
from PIL import Image, ImageDraw

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RADICE not in sys.path:
    sys.path.insert(0, RADICE)

from musetalk.utils import fusione_parallela as fp  # noqa: E402
from musetalk.utils.blending import _build_blend_mask_from_parsing, get_crop_box  # noqa: E402
from tests import sintetico  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*fork.*")

FPS = 25
ALTEZZA, LARGHEZZA = 192, 256


class _Lotto:
    """Controfigura di un tensore di latenti: ricorda solo gli indici dei frame."""

    def __init__(self, indici):
        self.indici = list(indici)

    def to(self, *args, **kwargs):
        return self


def _parsing_finto(immagini):
    """La forma dipende dai pixel del ritaglio: un ritaglio sbagliato cambia la maschera."""
    risultati = []
    for immagine in immagini:
        pixel = np.asarray(immagine.convert("L"), dtype=np.int64)
        spinta = int(pixel.sum() % 97)
        parsing = Image.new("L", (512, 512), 0)
        ImageDraw.Draw(parsing).ellipse([60 + spinta, 100 + spinta // 2, 440, 500], fill=255)
        risultati.append(parsing)
    return risultati


class _FaceParsingFinto:
    def __init__(self, left_cheek_width=90, right_cheek_width=90):
        pass

    def batch_call(self, images, mode="jaw"):
        return _parsing_finto(images)


def _carica_motore(stato):
    """Importa api/inference_service.py con i moduli pesanti sostituiti."""
    torch = types.ModuleType("torch")
    torch.no_grad = lambda: (lambda funzione: funzione)
    torch.device = lambda nome: nome
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, synchronize=lambda: None)
    torch.Tensor = object
    torch.float32, torch.float16 = "float32", "float16"
    torch.tensor = lambda *a, **k: None

    transformers = types.ModuleType("transformers")
    transformers.WhisperModel = object

    modulo_tqdm = types.ModuleType("tqdm")
    modulo_tqdm.tqdm = lambda iterabile=None, **k: iterabile

    face_parsing = types.ModuleType("musetalk.utils.face_parsing")
    face_parsing.FaceParsing = _FaceParsingFinto
    audio_processor = types.ModuleType("musetalk.utils.audio_processor")
    audio_processor.AudioProcessor = object

    utilita = types.ModuleType("musetalk.utils.utils")
    utilita.get_file_type = lambda percorso: (
        "image" if percorso.lower().endswith((".png", ".jpg")) else "video")
    utilita.load_all_model = lambda **k: (None, None, None)

    def datagen(whisper_chunks, vae_encode_latents, batch_size=8, delay_frame=0, device="cpu"):
        indici = list(range(len(whisper_chunks)))
        for inizio in range(0, len(indici), batch_size):
            lotto = indici[inizio:inizio + batch_size]
            yield lotto, _Lotto(lotto)

    utilita.datagen = datagen

    preprocessing = types.ModuleType("musetalk.utils.preprocessing")
    preprocessing.coord_placeholder = sintetico.SEGNAPOSTO

    def get_landmark_and_bbox(img_list, upperbondrange=0, precomputed_bboxes=None, frames=None):
        if precomputed_bboxes is not None:
            return list(precomputed_bboxes)[:len(frames)], frames
        return list(stato["coordinate_rilevate"])[:len(frames)], frames

    preprocessing.get_landmark_and_bbox = get_landmark_and_bbox

    sostituti = {
        "torch": torch, "transformers": transformers,
        "musetalk.utils.face_parsing": face_parsing,
        "musetalk.utils.audio_processor": audio_processor,
        "musetalk.utils.utils": utilita,
        "musetalk.utils.preprocessing": preprocessing,
    }
    try:
        import tqdm  # noqa: F401
    except ImportError:
        sostituti["tqdm"] = modulo_tqdm
    with mock.patch.dict(sys.modules, sostituti):
        spec = importlib.util.spec_from_file_location(
            "motore_sotto_test", os.path.join(RADICE, "api", "inference_service.py"))
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
    return modulo


class TestCablaggioGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stato = {"coordinate_rilevate": []}
        cls.modulo = _carica_motore(cls.stato)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cartella = self._tmp.name
        self.rng = np.random.RandomState(7)
        self.facce = {}

    # ---- costruzione degli ingressi -------------------------------------
    def crea_video(self, n_originali):
        dati = sintetico.crea_input(n_originali=n_originali, n_frame=1, altezza=ALTEZZA,
                                    larghezza=LARGHEZZA, seme=3)
        percorso = os.path.join(self.cartella, "ingresso.mp4")
        fp._write_video_pipe(dati["frame_ciclo"][:n_originali], percorso, LARGHEZZA, ALTEZZA,
                             FPS, codec="libx264",
                             percorso_log=os.path.join(self.cartella, "ingresso.log"))
        cattura = cv2.VideoCapture(percorso)
        frames = []
        while True:
            ok, frame = cattura.read()
            if not ok:
                break
            frames.append(frame)
        cattura.release()
        self.assertEqual(len(frames), n_originali)
        return percorso, frames

    def crea_audio(self, n_frame):
        percorso = os.path.join(self.cartella, "audio.wav")
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
            f"sine=frequency=440:duration={n_frame / FPS + 0.5:.3f}", "-ar", "16000", percorso,
        ], check=True, stdin=subprocess.DEVNULL)
        return percorso

    def crea_motore(self, n_frame):
        motore = self.modulo.MuseTalkInference(use_float16=False)
        motore.models_loaded = True
        motore.device = "cpu"
        motore.weight_dtype = None
        motore.timesteps = None
        motore.pe = lambda lotto: lotto
        motore.whisper = None
        motore.audio_processor = types.SimpleNamespace(
            get_audio_feature=lambda percorso: (None, n_frame),
            get_whisper_chunk=lambda *a, **k: list(range(n_frame)),
        )
        motore.unet = types.SimpleNamespace(
            model=lambda latenti, passi, encoder_hidden_states=None:
            types.SimpleNamespace(sample=latenti))

        def faccia(i):
            if i not in self.facce:
                self.facce[i] = np.ascontiguousarray(sintetico.faccia_con_codice(i, self.rng))
            return self.facce[i]

        def decode_latents(latenti):
            # Come vae.decode_latents: lotto RGB, poi vista BGR a stride negativo.
            return np.stack([faccia(i)[..., ::-1] for i in latenti.indici])[..., ::-1]

        def get_latents_for_unet_batch(ritagli):
            return np.zeros((len(ritagli), 8, 32, 32), dtype=np.float32)

        motore.vae = types.SimpleNamespace(
            decode_latents=decode_latents, get_latents_for_unet_batch=get_latents_for_unet_batch)
        motore._load_gfpgan = lambda: None
        motore._enhance_face_aligned = lambda faccia_in, peso=0.5: 255 - faccia_in
        return motore

    def coordinate(self, n, con_segnaposto=True):
        rng = np.random.RandomState(11)
        risultato = []
        for j in range(n):
            if con_segnaposto and j % 6 == 4:
                risultato.append(sintetico.SEGNAPOSTO)
                continue
            largo, alto = int(rng.randint(64, 100)), int(rng.randint(70, 110))
            x1 = int(rng.randint(0, LARGHEZZA - largo))
            y1 = int(rng.randint(0, ALTEZZA - alto))
            risultato.append((x1, y1, x1 + largo, y1 + alto))
        return risultato

    def genera(self, motore, n_workers, nome, **parametri):
        """Esegue generate() intercettando le impronte dei frame della fase F."""
        file_impronte = os.path.join(self.cartella, f"{nome}.sha256")
        vero = self.modulo.fondi_e_codifica
        rapporti = []
        self.sincronizzazioni = 0
        self.fp_visti_dai_figli = []
        vero_fork = fp._esegui_con_fork

        def conta_sincronizzazioni():
            self.sincronizzazioni += 1

        def spia_fork(parametri_segmenti, *args, **kwargs):
            self.fp_visti_dai_figli += [parametri[0].fp for parametri in parametri_segmenti]
            return vero_fork(parametri_segmenti, *args, **kwargs)

        motore._sincronizza_gpu = conta_sincronizzazioni

        def spia(*args, **kwargs):
            rapporto = vero(*args, percorso_impronte=file_impronte, mostra_progresso=False, **kwargs)
            rapporti.append(rapporto)
            return rapporto

        ambiente = {"MUSETALK_FG_STRICT": "1", "MUSETALK_FG_VERIFICA": "1",
                    "MUSETALK_FG_STALLO_S": "60"}
        with mock.patch.object(self.modulo, "fondi_e_codifica", spia), \
                mock.patch.object(fp, "_esegui_con_fork", spia_fork), \
                mock.patch.object(fp, "MIN_FRAME_SEGMENTO", 8), \
                mock.patch.dict(os.environ, ambiente), \
                mock.patch("builtins.print"):
            uscita = motore.generate(
                result_dir=os.path.join(self.cartella, f"risultati_{nome}"),
                output_name=nome, use_nvenc=False, n_workers=n_workers, batch_size=16,
                **parametri)
        with open(file_impronte) as f:
            return uscita, f.read().split(), rapporti[0]

    def riferimento(self, frames, coordinate, n_frame, maschere=None, enhance_fn=None,
                    extra_margin=10):
        """Impronte del ciclo storico, maschere comprese (blocco storico copiato alla lettera)."""
        frame_ciclo = frames + frames[::-1]
        coord_ciclo = list(coordinate) + list(coordinate)[::-1]
        if maschere is None:
            maschere = [None] * n_frame
            for i in range(n_frame):
                bbox = coord_ciclo[i % len(coord_ciclo)]
                if bbox == sintetico.SEGNAPOSTO:
                    continue
                ori_frame = frame_ciclo[i % len(frame_ciclo)]
                x1, y1, x2, y2 = bbox
                face_box = (x1, y1, x2, min(y2 + extra_margin, ori_frame.shape[0]))
                crop_box, _ = get_crop_box(face_box, 1.5)
                face_large_pil = Image.fromarray(ori_frame[:, :, ::-1]).crop(crop_box)
                ori_shape = face_large_pil.size
                parsing_pil = _parsing_finto([face_large_pil])[0].resize(ori_shape)
                maschere[i] = _build_blend_mask_from_parsing(parsing_pil, face_box, crop_box, ori_shape)
        facce = [self.facce[i][..., ::-1][..., ::-1] for i in range(n_frame)]
        fusi = sintetico.fusione_storica(facce, coord_ciclo, frame_ciclo, maschere,
                                         extra_margin, "jaw", enhance_fn)
        return [fp.impronta_frame(f) for f in fusi]

    # ---- casi -------------------------------------------------------------
    def test_cache_hit_video(self):
        """precomputed_bboxes/latents/masks: sequenziale == parallelo == storico, video integro."""
        n_originali, n_frame = 30, 50
        video, frames = self.crea_video(n_originali)
        audio = self.crea_audio(n_frame)
        coordinate = [list(c) for c in self.coordinate(n_originali, con_segnaposto=False)]
        maschere = []
        for bbox, frame in zip(coordinate, frames):
            x1, y1, x2, y2 = bbox
            riquadro, _ = get_crop_box((x1, y1, x2, min(y2 + 10, frame.shape[0])), 1.5)
            maschere.append(np.full((riquadro[3] - riquadro[1], riquadro[2] - riquadro[0]), 255,
                                    dtype=np.uint8))
        comuni = dict(audio_path=audio, video_path=video, precomputed_bboxes=coordinate,
                      precomputed_latents=[np.zeros((1, 8, 32, 32))] * n_originali,
                      precomputed_masks=maschere)
        motore = self.crea_motore(n_frame)

        def vietato(*args, **kwargs):
            raise AssertionError("su cache hit generate() non deve costruire ritagli ne' parsing")

        # Solo il nome Image visto da generate(): blending.py usa il proprio import.
        senza_pil = types.SimpleNamespace(fromarray=vietato, Image=Image.Image)
        with mock.patch.object(self.modulo, "FaceParsing", vietato), \
                mock.patch.object(self.modulo, "Image", senza_pil):
            _, sequenziali, rapporto_seq = self.genera(motore, 1, "hit_seq", **comuni)
            self.assertEqual(self.sincronizzazioni, 0, "nessun fork nel percorso sequenziale")
            uscita, parallele, rapporto_par = self.genera(motore, 4, "hit_par", **comuni)
        self.assertEqual(self.sincronizzazioni, 1, "torch.cuda.synchronize() prima dei fork")
        self.assertEqual(rapporto_seq.modo, "sequenziale")
        self.assertEqual((rapporto_par.modo, rapporto_par.n_segmenti, rapporto_par.parita),
                         ("parallelo", 4, True))
        ciclo_maschere = maschere + maschere[::-1]
        atteso = self.riferimento(frames, coordinate, n_frame,
                                  maschere=[ciclo_maschere[i % 60] for i in range(n_frame)])
        self.assertEqual(sequenziali, atteso)
        self.assertEqual(parallele, atteso)

        # File finale dopo il mux: audio presente, tutti i frame, nell'ordine giusto.
        self.assertTrue(uscita.endswith("hit_par.mp4"))
        self.assertEqual(int(sintetico.info_video(uscita)["nb_read_packets"]), n_frame)
        flussi = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", uscita],
            check=True, stdout=subprocess.PIPE).stdout.decode().split()
        self.assertEqual(sorted(flussi), ["audio", "video"])
        ciclo = coordinate + coordinate[::-1]
        decodificati = sintetico.decodifica_video(uscita, LARGHEZZA, ALTEZZA)
        letti = []
        for i, frame in enumerate(decodificati):
            x1, y1, x2, y2 = ciclo[i % 60]
            letti.append(sintetico.leggi_codice(frame, (x1, y1, x2, min(y2 + 10, ALTEZZA))))
        self.assertEqual(letti, list(range(n_frame)))
        self.assertFalse(os.path.exists(os.path.join(self.cartella, "risultati_hit_par", "temp")))

    def test_senza_cache_con_parsing(self):
        """Percorso senza cache: BiSeNet (finto) nel padre, ritagli a lotti, segnaposto."""
        n_originali, n_frame = 24, 44
        video, frames = self.crea_video(n_originali)
        audio = self.crea_audio(n_frame)
        self.stato["coordinate_rilevate"] = self.coordinate(n_originali)
        motore = self.crea_motore(n_frame)
        comuni = dict(audio_path=audio, video_path=video, parsing_batch_size=5)
        _, sequenziali, _ = self.genera(motore, 1, "miss_seq", **comuni)
        _, parallele, rapporto = self.genera(motore, 3, "miss_par", **comuni)
        self.assertEqual((rapporto.modo, rapporto.n_segmenti), ("parallelo", 3))
        self.assertEqual(self.fp_visti_dai_figli, [None, None, None],
                         "il FaceParsing (GPU) non deve arrivare ai processi figli")
        atteso = self.riferimento(frames, self.stato["coordinate_rilevate"], n_frame)
        self.assertEqual(sequenziali, atteso)
        self.assertEqual(parallele, atteso)

    def test_enhance(self):
        """enhance=True: in linea nel sequenziale, passaggio preliminare nel parallelo."""
        n_originali, n_frame = 24, 40
        video, frames = self.crea_video(n_originali)
        audio = self.crea_audio(n_frame)
        self.stato["coordinate_rilevate"] = self.coordinate(n_originali)
        motore = self.crea_motore(n_frame)
        pesi = []

        def migliora_finto(faccia_in, peso=0.5):
            pesi.append(peso)
            return 255 - faccia_in

        motore._enhance_face_aligned = migliora_finto
        comuni = dict(audio_path=audio, video_path=video, enhance=True, gfpgan_weight=0.7)
        _, sequenziali, _ = self.genera(motore, 1, "gf_seq", **comuni)
        ciclo = self.stato["coordinate_rilevate"] + self.stato["coordinate_rilevate"][::-1]
        con_volto = sum(1 for i in range(n_frame) if ciclo[i % len(ciclo)] != sintetico.SEGNAPOSTO)
        self.assertEqual(pesi, [0.7] * con_volto)
        del pesi[:]
        _, parallele, rapporto = self.genera(motore, 4, "gf_par", **comuni)
        # Passaggio preliminare nel padre: una chiamata per faccia, mai nei figli. La
        # verifica interna di parita' non deve migliorare di nuovo.
        self.assertEqual(pesi, [0.7] * con_volto)
        self.assertEqual(rapporto.modo, "parallelo")
        atteso = self.riferimento(frames, self.stato["coordinate_rilevate"], n_frame,
                                  enhance_fn=lambda faccia: 255 - faccia)
        self.assertEqual(sequenziali, atteso)
        self.assertEqual(parallele, atteso)

    def test_immagine_singola(self):
        n_frame = 36
        dati = sintetico.crea_input(n_originali=1, n_frame=1, altezza=ALTEZZA, larghezza=LARGHEZZA)
        immagine = os.path.join(self.cartella, "ritratto.png")
        cv2.imwrite(immagine, dati["frame_ciclo"][0])
        audio = self.crea_audio(n_frame)
        self.stato["coordinate_rilevate"] = [(60, 40, 150, 150)]
        motore = self.crea_motore(n_frame)
        comuni = dict(audio_path=audio, video_path=immagine)
        _, sequenziali, _ = self.genera(motore, 1, "img_seq", **comuni)
        uscita, parallele, rapporto = self.genera(motore, 3, "img_par", **comuni)
        self.assertEqual((rapporto.modo, rapporto.n_segmenti), ("parallelo", 3))
        atteso = self.riferimento([cv2.imread(immagine)], self.stato["coordinate_rilevate"], n_frame)
        self.assertEqual(sequenziali, atteso)
        self.assertEqual(parallele, atteso)
        self.assertEqual(int(sintetico.info_video(uscita)["nb_read_packets"]), n_frame)

    def test_generate_senza_spie(self):
        """generate() esattamente come la chiama l'handler: nessun parametro nuovo, env per i worker."""
        n_originali, n_frame = 30, 40
        video, _ = self.crea_video(n_originali)
        audio = self.crea_audio(n_frame)
        coordinate = [list(c) for c in self.coordinate(n_originali, con_segnaposto=False)]
        maschere = []
        for x1, y1, x2, y2 in coordinate:
            riquadro, _ = get_crop_box((x1, y1, x2, min(y2 + 10, ALTEZZA)), 1.5)
            maschere.append(np.full((riquadro[3] - riquadro[1], riquadro[2] - riquadro[0]), 255,
                                    dtype=np.uint8))
        motore = self.crea_motore(n_frame)
        righe = []
        ambiente = {"MUSETALK_BLEND_WORKERS": "3", "MUSETALK_FG_STRICT": "1"}
        with mock.patch.object(fp, "MIN_FRAME_SEGMENTO", 8), \
                mock.patch.dict(os.environ, ambiente), \
                mock.patch("builtins.print", lambda *a, **k: righe.append(" ".join(map(str, a)))), \
                contextlib.redirect_stderr(io.StringIO()):
            uscita = motore.generate(
                audio_path=audio, video_path=video, use_nvenc=False, batch_size=16,
                result_dir=os.path.join(self.cartella, "risultati"), output_name="produzione",
                precomputed_bboxes=coordinate, precomputed_masks=maschere,
                precomputed_latents=[np.zeros((1, 8, 32, 32))] * n_originali)
        self.assertRegex("\n".join(righe), r"\[phase F\] blending: 40 frames in \d+\.\d\ds")
        self.assertRegex("\n".join(righe), r"\[phase G\] video encode: \d+\.\d\ds")
        self.assertRegex("\n".join(righe), r"\[phase F\+G\] modo=parallelo n_workers=3 segmenti=3 ")
        ciclo = coordinate + coordinate[::-1]
        letti = []
        for i, frame in enumerate(sintetico.decodifica_video(uscita, LARGHEZZA, ALTEZZA)):
            x1, y1, x2, y2 = ciclo[i % 60]
            letti.append(sintetico.leggi_codice(frame, (x1, y1, x2, min(y2 + 10, ALTEZZA))))
        self.assertEqual(letti, list(range(n_frame)))

    def test_firma_retrocompatibile(self):
        import inspect
        parametri = list(inspect.signature(self.modulo.MuseTalkInference.generate).parameters)
        self.assertEqual(parametri[-1], "n_workers")
        self.assertEqual(parametri[:3], ["self", "audio_path", "video_path"])
        firma = inspect.signature(self.modulo.MuseTalkInference.generate)
        self.assertIsNone(firma.parameters["n_workers"].default)
        self.assertTrue(callable(self.modulo._write_video_pipe))


if __name__ == "__main__":
    unittest.main(verbosity=2)
