"""Verifica e misura delle fasi F e G su una macchina reale, senza GPU.

Pensato per girare dentro le immagini Docker/Singularity e sui nodi HPC, dove
contano la versione vera di ffmpeg, il backend parallelo di OpenCV e le CPU
realmente concesse. Genera un input sintetico, esegue il percorso sequenziale e
quello parallelo, confronta le impronte sha256 dei frame e i dati di ffprobe e
stampa i tempi. Esce con codice 0 solo se tutto coincide.

    python tests/verifica_fusione_parallela.py --lato 768 --frame 1500
    python tests/verifica_fusione_parallela.py --solo-diagnostica
"""

import argparse
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
import PIL

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RADICE not in sys.path:
    sys.path.insert(0, RADICE)

from musetalk.utils import fusione_parallela as fp  # noqa: E402
from tests import sintetico  # noqa: E402


def _misura(funzione, ripetizioni=20):
    funzione()
    t = time.perf_counter()
    for _ in range(ripetizioni):
        funzione()
    return (time.perf_counter() - t) / ripetizioni * 1000.0


def diagnostica(lato: int) -> None:
    """Il "passo 0": i numeri che decidono quanto rende il percorso parallelo."""
    print("== diagnostica ==")
    print(f"python {platform.python_version()} | numpy {np.__version__} | "
          f"cv2 {cv2.__version__} | PIL {PIL.__version__} | {platform.platform()}")
    riga_ffmpeg = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE,
                                 stdin=subprocess.DEVNULL).stdout.decode().splitlines()[0]
    print(riga_ffmpeg)
    for riga in cv2.getBuildInformation().splitlines():
        if "Parallel framework" in riga:
            print("OpenCV", riga.strip())
    cpu, dettagli = fp.risolvi_cpu_effettive()
    print(f"cpu effettive: {cpu}  dettagli: {dettagli}")
    print(f"n_workers automatico: {fp.risolvi_n_workers(None, os.environ, cpu)}")
    for percorso in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max",
                     "/proc/sys/vm/overcommit_memory"):
        testo = fp._leggi_testo(percorso)
        if testo is not None:
            print(f"{percorso}: {testo.strip()}")
    import multiprocessing
    print(f"metodi di avvio: {multiprocessing.get_all_start_methods()} | "
          f"vfork in subprocess: {getattr(subprocess, '_USE_VFORK', 'n/d')} | "
          f"thread attivi: {threading.active_count()}")

    frame = np.random.RandomState(0).randint(0, 255, (lato, lato, 3)).astype(np.uint8)
    vista = frame[:, :, ::-1]
    t_contiguo = _misura(lambda: frame.tobytes())
    t_strided = _misura(lambda: vista.tobytes())
    print(f"tobytes {lato}x{lato}: contiguo {t_contiguo:.2f} ms, stride negativo {t_strided:.2f} ms "
          f"(x{t_strided / max(t_contiguo, 1e-6):.0f}): e' il costo che tiene il GIL nel feeder")

    # Quanto corre x264 se nessuno lo fa aspettare: dice se la fase G di oggi e'
    # limitata dall'encoder o dal processo Python che lo alimenta.
    n = 150
    dati = sintetico.crea_input(n_originali=10, n_frame=1, altezza=lato, larghezza=lato)
    blocco = b"".join(f.tobytes() for f in dati["frame_ciclo"][:10])
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{lato}x{lato}", "-r", "25", "-i", "-",
           *fp.argomenti_encoder("libx264"), "-f", "null", "-"]
    t = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for _ in range(n // 10):
            proc.stdin.write(blocco)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()
    print(f"x264 -preset slow su -f null, thread automatici: {n / (time.perf_counter() - t):.0f} fps")


def esegui(dati, n_workers, cartella, nome, fps, min_seg, use_nvenc, con_impronte):
    """Una corsa di F+G. I tempi si misurano SENZA impronte: calcolare gli sha256 costa
    quanto una parte rilevante della fase F e nel sequenziale non e' parallelizzato."""
    contesto = fp.ContestoFusione(
        facce=dati["facce"], frame_ciclo=dati["frame_ciclo"], face_boxes=dati["face_boxes"],
        maschere=dati["maschere"], parsing_mode="jaw")
    video = os.path.join(cartella, f"{nome}.mp4")
    impronte = os.path.join(cartella, f"{nome}.sha256") if con_impronte else None
    t = time.perf_counter()
    rapporto = fp.fondi_e_codifica(
        contesto, video, fps, use_nvenc=use_nvenc, n_workers=n_workers, dir_lavoro=cartella,
        min_frame_segmento=min_seg, percorso_impronte=impronte, mostra_progresso=False,
        ambiente=dict(os.environ, MUSETALK_FG_STRICT="1"))
    durata = time.perf_counter() - t
    lette = []
    if impronte:
        with open(impronte) as f:
            lette = f.read().split()
    return rapporto, lette, video, durata


def main() -> int:
    analizzatore = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analizzatore.add_argument("--lato", type=int, default=768, help="lato del frame quadrato")
    analizzatore.add_argument("--frame", type=int, default=1500, help="frame da produrre")
    analizzatore.add_argument("--originali", type=int, default=150, help="frame del video sorgente")
    analizzatore.add_argument("--fps", type=int, default=25)
    analizzatore.add_argument("--workers", type=int, default=None,
                              help="default: MUSETALK_BLEND_WORKERS o meta' delle CPU effettive")
    analizzatore.add_argument("--min-seg", type=int, default=fp.MIN_FRAME_SEGMENTO)
    analizzatore.add_argument("--nvenc", action="store_true", help="prova h264_nvenc")
    analizzatore.add_argument("--rumore", type=int, default=2,
                              help="rumore per pixel dello sfondo: 0 = molto comprimibile, "
                                   "10 = caso peggiore per x264")
    analizzatore.add_argument("--solo-diagnostica", action="store_true")
    argomenti = analizzatore.parse_args()

    diagnostica(argomenti.lato)
    if argomenti.solo_diagnostica:
        return 0

    n_workers = argomenti.workers
    if n_workers is None:
        cpu, _ = fp.risolvi_cpu_effettive()
        n_workers = fp.risolvi_n_workers(None, os.environ, cpu, "linux")[0]
    print(f"\n== confronto: {argomenti.frame} frame {argomenti.lato}x{argomenti.lato}, "
          f"n_workers={n_workers} ==")
    volto = (argomenti.lato // 3, argomenti.lato // 2)
    dati = sintetico.crea_input(
        n_originali=argomenti.originali, n_frame=argomenti.frame, altezza=argomenti.lato,
        larghezza=argomenti.lato, lunghezza_maschere="ciclo", volto=volto,
        rumore=argomenti.rumore)

    with tempfile.TemporaryDirectory() as cartella:
        comuni = (argomenti.fps, argomenti.min_seg, argomenti.nvenc)
        # Prima i tempi (senza impronte), poi la parita' (con impronte, non cronometrata).
        rap_seq, _, video_seq, t_seq = esegui(dati, 1, cartella, "sequenziale", *comuni, False)
        rap_par, _, video_par, t_par = esegui(dati, n_workers, cartella, "parallelo", *comuni, False)
        _, imp_seq, _, _ = esegui(dati, 1, cartella, "parita_seq", *comuni, True)
        rap_par_imp, imp_par, _, _ = esegui(dati, n_workers, cartella, "parita_par", *comuni, True)
        info_seq, info_par = sintetico.info_video(video_seq), sintetico.info_video(video_par)
        dimensioni = os.path.getsize(video_seq), os.path.getsize(video_par)

    controlli = {
        "percorso parallelo davvero eseguito": rap_par.modo == rap_par_imp.modo == "parallelo"
        and not rap_par.fallback,
        "impronte sha256 dei frame identiche": imp_seq == imp_par and len(imp_par) == argomenti.frame,
        "stesso numero di pacchetti": int(info_seq["nb_read_packets"])
        == int(info_par["nb_read_packets"]) == argomenti.frame,
        "stesso fps": info_seq["r_frame_rate"] == info_par["r_frame_rate"],
        "durata esatta e identica": info_seq["durata_esatta"] == info_par["durata_esatta"],
    }
    print(f"\nsequenziale: {t_seq:7.2f} s  (F {rap_seq.t_fusione:.2f} + G {rap_seq.t_codifica:.2f})"
          f"  {dimensioni[0] / 1e6:.1f} MB")
    print(f"parallelo:   {t_par:7.2f} s  ({rap_par.n_segmenti} segmenti, codec {rap_par.codec}, "
          f"x264_threads {rap_par.thread_x264})  {dimensioni[1] / 1e6:.1f} MB")
    print(f"accelerazione F+G: x{t_seq / t_par:.2f}  "
          f"(dimensione {100.0 * (dimensioni[1] / dimensioni[0] - 1):+.1f}%: GOP e rate control "
          f"ripartono a ogni segmento)")
    for nome, esito in controlli.items():
        print(f"  [{'OK' if esito else 'KO'}] {nome}")
    return 0 if all(controlli.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
