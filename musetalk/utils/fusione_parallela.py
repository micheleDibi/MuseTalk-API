"""Fusione (blending) e codifica video per le fasi F e G di ``generate()``.

Il modulo e' volutamente privo di torch: importa solo stdlib, numpy, cv2, PIL e
``musetalk.utils.blending``. In questo modo le due fasi sono testabili senza GPU
(vedi ``tests/test_fusione_parallela.py``) e i processi figli non hanno alcun
motivo di toccare CUDA.

Due percorsi, stesso kernel per frame (:func:`fondi_frame`):

- **sequenziale** (``n_workers=1``): e' il percorso storico, trapiantato qui
  istruzione per istruzione. Resta il riferimento di correttezza.
- **parallelo**: i frame vengono divisi in N segmenti contigui; ogni segmento e'
  affidato a un processo creato con ``fork`` che fonde i propri frame e li invia
  in streaming al proprio ffmpeg (``seg_NNN.mp4``). Il padre verifica che i
  segmenti siano uniformi, li concatena con il concat demuxer in ``-c copy`` e
  restituisce il video muto; il mux dell'audio resta in ``generate()``.

Perche' ``fork`` e non ``spawn`` + ``shared_memory``: con ``spawn`` il figlio
ri-esegue ``__main__`` (in ``musetalk_runpod/handler.py`` i modelli vengono
caricati a livello di modulo) e i ~42 GB di frame andrebbero copiati in
``/dev/shm``. Con ``fork`` gli argomenti del processo sono ereditati, mai
serializzati, e i buffer numpy vengono letti in copy-on-write.

Regole per il codice eseguito nei figli: niente torch, niente print, niente
import, niente ``cv2.setNumThreads`` (nel figlio farebbe ``pthread_join`` su
thread che non esistono), uscita sempre con ``os._exit``.
"""

import dataclasses
import gc
import hashlib
import json
import logging
import mmap
import multiprocessing
import multiprocessing.connection
import os
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import traceback
import uuid
from fractions import Fraction
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from musetalk.utils.blending import get_crop_box, get_image

try:  # tqdm e' presente in produzione; nei test minimali puo' mancare
    from tqdm import tqdm
except Exception:  # pragma: no cover - dipende dall'ambiente
    tqdm = None


# Tetto del dimensionamento automatico: oltre, il costo seriale dei fork e le
# page table per figlio superano il guadagno (x264 domina il lavoro di CPU).
TETTO_WORKERS = 16
# Lunghezza minima di un segmento. Il valore di produzione evita segmenti
# minuscoli; il minimo invalicabile e' 3 frame: sotto, libx264 non usa B-frame,
# il ritardo DTS cambia e il confine fra segmenti si disallinea.
MIN_FRAME_SEGMENTO = 250
MIN_FRAME_INVALICABILE = 3
TETTO_THREAD_X264 = 8
SESSIONI_NVENC_PREDEFINITE = 2
STALLO_PREDEFINITO_S = 300.0

# Campi che devono coincidere fra tutti i segmenti perche' il concat in
# ``-c copy`` sia legittimo: ffmpeg concatena in silenzio anche segmenti difformi.
CAMPI_UNIFORMITA = (
    "codec_name", "profile", "level", "width", "height", "pix_fmt",
    "r_frame_rate", "time_base", "has_b_frames",
)

# Serializza la finestra "cv2 a 1 thread -> ultimo fork" fra chiamate concorrenti.
_LOCK_FORK = threading.Lock()


class ErroreParallelo(RuntimeError):
    """Il percorso parallelo non e' riuscito a produrre un video valido."""


class ErroreEncoder(ErroreParallelo):
    """Un segmento e' fallito dentro ffmpeg (tipicamente h264_nvenc)."""

    def __init__(self, messaggio: str, returncode: Optional[int] = None):
        super().__init__(messaggio)
        self.returncode = returncode


@dataclasses.dataclass
class ContestoFusione:
    """Dati in sola lettura necessari a fondere il frame ``i``.

    ``face_boxes[i]`` vale ``None`` per i frame senza volto (placeholder): la
    classificazione resta in ``generate()``, cosi' questo modulo non importa
    ``preprocessing`` (che carica torch).
    """

    facce: Sequence[np.ndarray]
    frame_ciclo: Sequence[np.ndarray]
    face_boxes: Sequence[Optional[Tuple[int, int, int, int]]]
    maschere: Sequence[Optional[np.ndarray]]
    parsing_mode: str = "jaw"
    fp: Any = None


@dataclasses.dataclass
class RapportoFusione:
    """Esito di :func:`fondi_e_codifica`, usato dai log e dai test."""

    modo: str
    n_frame: int
    n_workers: int
    n_segmenti: int
    codec: str
    motivo: str = ""
    tentativi: int = 0
    fallback: bool = False
    fork_ms: List[float] = dataclasses.field(default_factory=list)
    t_fusione: float = 0.0
    t_codifica: float = 0.0
    thread_x264: Optional[int] = None
    percorsi_segmenti: List[str] = dataclasses.field(default_factory=list)
    parita: Optional[bool] = None


# --------------------------------------------------------------------------
# Codifica (fase G)
# --------------------------------------------------------------------------

def argomenti_encoder(codec: str) -> List[str]:
    """Argomenti dell'encoder: unica sorgente per sequenziale, segmenti e sonda."""
    if codec == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-rc", "vbr",
            "-cq", "18",
            "-b:v", "3M",
            "-maxrate", "5M",
            "-bufsize", "10M",
            "-pix_fmt", "yuv420p",
        ]
    return [
        "-c:v", codec,
        "-preset", "slow",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
    ]


def _chiudi_senza_errori(flusso: Any) -> None:
    try:
        flusso.close()
    except OSError:
        pass


def _write_video_pipe(
    frames: Iterable[np.ndarray],
    output_path: str,
    width: int,
    height: int,
    fps: int,
    codec: str = "h264_nvenc",
    argomenti_extra: Optional[Sequence[str]] = None,
    percorso_log: Optional[str] = None,
    osservatore: Optional[Callable[[bytes], None]] = None,
) -> None:
    """Pipe uint8 BGR frames through ffmpeg's stdin into ``output_path``.

    Avoids writing PNG intermediates to disk. Raises ``CalledProcessError`` on
    non-zero ffmpeg exit so the caller can fall back to libx264.

    Parametri aggiunti (tutti opzionali: con i default la riga di comando e'
    identica a quella storica). ``frames`` puo' essere un generatore.
    ``argomenti_extra`` finisce subito prima del file di uscita; ``percorso_log``
    riceve lo stderr di ffmpeg (mai una PIPE: senza un thread che la svuota si
    blocca); ``osservatore`` riceve gli stessi byte scritti sulla pipe.
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        *argomenti_encoder(codec),
        *(argomenti_extra or []),
        output_path,
    ]
    flusso_log = open(percorso_log, "ab") if percorso_log else None
    try:
        # Mai preexec_fn: manterrebbe un fork completo al posto di vfork, e da un
        # processo con decine di GB di RSS ogni ffmpeg costerebbe secondi.
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=flusso_log)
        try:
            try:
                for frame in frames:
                    dati = frame.tobytes()
                    proc.stdin.write(dati)
                    if osservatore is not None:
                        osservatore(dati)
                proc.stdin.close()
            except BrokenPipeError:
                _chiudi_senza_errori(proc.stdin)
            ret = proc.wait()
        except BaseException:
            # Il generatore ha sollevato (o siamo stati interrotti): ffmpeg non
            # deve restare orfano ne' lasciare un file parziale.
            proc.kill()
            _chiudi_senza_errori(proc.stdin)
            proc.wait()
            try:
                os.remove(output_path)
            except OSError:
                pass
            raise
    finally:
        if flusso_log is not None:
            flusso_log.close()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def codifica_con_fallback(
    frames: Sequence[np.ndarray],
    percorso_video: str,
    larghezza: int,
    altezza: int,
    fps: int,
    use_nvenc: bool,
) -> str:
    """Blocco storico nvenc -> libx264 su un solo ffmpeg. Ritorna il codec usato."""
    encoded_ok = False
    if use_nvenc:
        try:
            _write_video_pipe(frames, percorso_video, larghezza, altezza, fps, codec="h264_nvenc")
            encoded_ok = True
        except subprocess.CalledProcessError as e:
            print(f"[phase G] h264_nvenc failed (rc={e.returncode}); fallback to libx264")
        except FileNotFoundError:
            print("[phase G] ffmpeg not found in PATH; cannot encode")
            raise
    if not encoded_ok:
        _write_video_pipe(frames, percorso_video, larghezza, altezza, fps, codec="libx264")
    return "h264_nvenc" if encoded_ok else "libx264"


# --------------------------------------------------------------------------
# Fusione (fase F)
# --------------------------------------------------------------------------

def fondi_frame(
    i: int,
    ctx: ContestoFusione,
    migliora: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Kernel unico della fase F: stesse istruzioni, stesso ordine del loop storico."""
    frame_originale = ctx.frame_ciclo[i % len(ctx.frame_ciclo)]
    face_box = ctx.face_boxes[i]
    if face_box is None:
        return frame_originale
    x1, y1, x2, y2c = face_box
    ori_frame = frame_originale.copy()

    face_crop = ctx.facce[i].astype(np.uint8)
    if migliora is not None:
        face_crop = migliora(face_crop)
    try:
        face_resized = cv2.resize(face_crop, (x2 - x1, y2c - y1))
    except Exception:
        return ori_frame

    # Il modulo allinea le maschere a frame e bbox (che gia' ciclano): e'
    # un'identita' in tutti i casi che funzionavano, e non va piu' fuori indice
    # quando l'audio supera il doppio della durata del video su cache hit.
    maschera = ctx.maschere[i % len(ctx.maschere)]
    return get_image(
        ori_frame,
        face_resized,
        [x1, y1, x2, y2c],
        mode=ctx.parsing_mode,
        fp=ctx.fp,
        precomputed_mask=maschera,
    )


def fondi_sequenziale(
    ctx: ContestoFusione,
    migliora: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    mostra_progresso: bool = True,
) -> List[np.ndarray]:
    """Percorso storico della fase F: un frame alla volta, tutto in memoria."""
    indici: Iterable[int] = range(len(ctx.facce))
    if tqdm is not None and mostra_progresso:
        indici = tqdm(indici, desc="Blending", total=len(ctx.facce))
    combine_frames: List[np.ndarray] = []
    for i in indici:
        combine_frames.append(fondi_frame(i, ctx, migliora))
    return combine_frames


def impronta_frame(frame: np.ndarray) -> str:
    """sha256 dei byte esatti che finiscono sulla pipe di ffmpeg."""
    return hashlib.sha256(frame.tobytes()).hexdigest()


# --------------------------------------------------------------------------
# Dimensionamento
# --------------------------------------------------------------------------

def _leggi_testo(percorso: str) -> Optional[str]:
    try:
        with open(percorso, "r") as f:
            return f.read()
    except OSError:
        return None


def _affinita_cpu() -> Optional[int]:
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except OSError:
            return None
    return None


def _directory_cgroup(leggi: Callable[[str], Optional[str]]) -> Tuple[List[str], List[str]]:
    """Directory candidate per la quota CPU: (cgroup v2, cgroup v1).

    Si prova sempre il percorso piatto (namespace cgroup del container) e, se
    ``/proc/self/cgroup`` indica un percorso annidato, tutti i suoi antenati.
    """
    v2 = ["/sys/fs/cgroup"]
    v1 = ["/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"]
    contenuto = leggi("/proc/self/cgroup")
    for riga in (contenuto or "").splitlines():
        parti = riga.strip().split(":", 2)
        if len(parti) != 3:
            continue
        controller, relativo = parti[1], parti[2]
        antenati = []
        while relativo and relativo != "/":
            antenati.append(relativo)
            relativo = os.path.dirname(relativo)
        if controller == "":
            v2.extend("/sys/fs/cgroup" + a for a in antenati)
        elif "cpu" in controller.split(","):
            for base in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):
                v1.extend(base + a for a in antenati)
    return v2, v1


def quota_cpu_cgroup(leggi: Callable[[str], Optional[str]] = _leggi_testo) -> Optional[int]:
    """CPU concesse dalla quota CFS, minimo lungo gli antenati. ``None`` = nessun limite.

    Un file assente significa "nessun limite" (es. Apptainer su MareNostrum).
    """
    v2, v1 = _directory_cgroup(leggi)
    limiti: List[int] = []
    for cartella in v2:
        testo = leggi(cartella + "/cpu.max")
        if not testo:
            continue
        campi = testo.split()
        if len(campi) != 2 or campi[0] == "max":
            continue
        try:
            quota, periodo = int(campi[0]), int(campi[1])
        except ValueError:
            continue
        if quota > 0 and periodo > 0:
            limiti.append(max(1, quota // periodo))
    for cartella in v1:
        testo_quota = leggi(cartella + "/cpu.cfs_quota_us")
        testo_periodo = leggi(cartella + "/cpu.cfs_period_us")
        if not testo_quota or not testo_periodo:
            continue
        try:
            quota, periodo = int(testo_quota.strip()), int(testo_periodo.strip())
        except ValueError:
            continue
        if quota > 0 and periodo > 0:
            limiti.append(max(1, quota // periodo))
    return min(limiti) if limiti else None


def risolvi_cpu_effettive(
    ambiente: Optional[Dict[str, str]] = None,
    leggi: Callable[[str], Optional[str]] = _leggi_testo,
    affinita: Callable[[], Optional[int]] = _affinita_cpu,
    conta_cpu: Callable[[], Optional[int]] = os.cpu_count,
) -> Tuple[int, Dict[str, Optional[int]]]:
    """CPU realmente utilizzabili: ``os.cpu_count()`` da solo ignora SLURM e cgroup."""
    ambiente = os.environ if ambiente is None else ambiente
    slurm: Optional[int] = None
    testo_slurm = ambiente.get("SLURM_CPUS_PER_TASK")
    if testo_slurm:
        try:
            slurm = int(testo_slurm) if int(testo_slurm) > 0 else None
        except ValueError:
            slurm = None
    dettagli = {
        "slurm": slurm,
        "affinity": affinita(),
        "cgroup": quota_cpu_cgroup(leggi),
        "os": conta_cpu(),
    }
    valori = [v for v in dettagli.values() if v]
    return (max(1, min(valori)) if valori else 1), dettagli


def _intero_da_env(ambiente: Dict[str, str], nome: str) -> Optional[int]:
    testo = ambiente.get(nome)
    if testo is None or str(testo).strip() == "":
        return None
    try:
        valore = int(str(testo).strip())
    except ValueError:
        valore = 0
    if valore < 1:
        print(f"[phase F+G] {nome}={testo!r} non valido: ignorato")
        return None
    return valore


def _secondi_da_env(ambiente: Dict[str, str], nome: str, predefinito: float) -> float:
    """Secondi positivi e finiti, altrimenti il valore predefinito (mai un'eccezione)."""
    testo = ambiente.get(nome)
    if testo is None or str(testo).strip() == "":
        return predefinito
    try:
        valore = float(str(testo).strip())
    except ValueError:
        valore = 0.0
    if not (0.0 < valore < float("inf")):  # scarta anche nan
        print(f"[phase F+G] {nome}={testo!r} non valido: uso {predefinito:.0f}s")
        return predefinito
    return valore


def risolvi_n_workers(
    n_workers: Optional[int],
    ambiente: Optional[Dict[str, str]] = None,
    cpu_effettive: int = 1,
    piattaforma: Optional[str] = None,
) -> Tuple[int, str]:
    """Precedenza: parametro esplicito > env MUSETALK_BLEND_WORKERS > automatico."""
    ambiente = os.environ if ambiente is None else ambiente
    piattaforma = sys.platform if piattaforma is None else piattaforma
    if n_workers is not None:
        return max(1, int(n_workers)), "parametro"
    testo_env = ambiente.get("MUSETALK_BLEND_WORKERS")
    if testo_env is not None and str(testo_env).strip() != "":
        try:
            da_env = int(str(testo_env).strip())
        except ValueError:
            da_env = 0
        if da_env >= 1:
            return da_env, "env"
        # E' l'interruttore di emergenza: un valore sbagliato (0, refuso) deve
        # spegnere il parallelo, non riaccendere il dimensionamento automatico.
        print(f"[phase F+G] MUSETALK_BLEND_WORKERS={testo_env!r} non valido: percorso sequenziale")
        return 1, "env non valida"
    if not piattaforma.startswith("linux"):
        return 1, "auto: piattaforma non Linux"
    # Meta' delle CPU: ogni worker pilota anche un ffmpeg con i suoi thread.
    return max(1, min(TETTO_WORKERS, cpu_effettive // 2)), "auto"


def calcola_segmenti(
    n_frame: int,
    n_seg_max: int,
    min_frame: int = MIN_FRAME_SEGMENTO,
) -> List[Tuple[int, int]]:
    """Partizione contigua e bilanciata di ``[0, n_frame)`` in intervalli ``[inizio, fine)``.

    Ogni segmento, l'ultimo compreso, ha almeno ``min_frame`` frame e le
    lunghezze differiscono al piu' di uno.
    """
    if n_frame <= 0:
        return []
    min_frame = max(MIN_FRAME_INVALICABILE, int(min_frame))
    n_seg = max(1, min(int(n_seg_max), n_frame // min_frame))
    base, resto = divmod(n_frame, n_seg)
    segmenti = []
    inizio = 0
    for k in range(n_seg):
        lunghezza = base + (1 if k < resto else 0)
        segmenti.append((inizio, inizio + lunghezza))
        inizio += lunghezza
    return segmenti


def risolvi_thread_x264(ambiente: Dict[str, str], cpu_effettive: int, n_seg: int) -> int:
    da_env = _intero_da_env(ambiente, "MUSETALK_X264_THREADS")
    if da_env is not None:
        return da_env
    return max(1, min(TETTO_THREAD_X264, cpu_effettive // max(1, n_seg)))


# --------------------------------------------------------------------------
# NVENC
# --------------------------------------------------------------------------

def sonda_nvenc(
    larghezza: int,
    altezza: int,
    fps: int,
    sessioni: int,
    timeout_s: float = 30.0,
) -> Tuple[int, int]:
    """Quante sessioni h264_nvenc concorrenti si aprono davvero. ``(0, rc)`` = nessuna.

    Lancia ``sessioni`` encode minuscoli con la stessa geometria e gli stessi
    argomenti del job; se anche uno solo fallisce dimezza e riprova. ffmpeg apre
    l'encoder solo all'arrivo del primo frame: per avere le sessioni aperte
    davvero insieme, un thread per processo scrive i frame e poi aspetta gli
    altri a una barriera prima di chiudere lo stdin. I thread sono tutti
    terminati al ritorno (la sonda gira prima dei fork). Sul return code non si
    ramifica oltre ``!= 0``: cambia fra le versioni di ffmpeg.
    """
    frame = bytes(larghezza * altezza * 3)
    # Oltre la probesize di ffmpeg (5 MB): a scrittura finita l'encoder e' aperto.
    n_frame = max(5, 5_000_000 // len(frame) + 3)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{larghezza}x{altezza}", "-r", str(fps), "-i", "-",
        *argomenti_encoder("h264_nvenc"),
        "-f", "null", "-",
    ]
    ultimo_rc = 1
    tentativo = max(1, int(sessioni))
    while tentativo >= 1:
        processi = [
            subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            for _ in range(tentativo)
        ]
        codici = [-1] * tentativo
        barriera = threading.Barrier(tentativo)

        def alimenta(j: int, proc: Any) -> None:
            try:
                for _ in range(n_frame):
                    proc.stdin.write(frame)
                proc.stdin.flush()
            except OSError:
                pass  # ffmpeg e' gia' uscito (encoder assente o sessione rifiutata)
            try:
                barriera.wait(timeout=timeout_s)
            except threading.BrokenBarrierError:
                pass
            _chiudi_senza_errori(proc.stdin)
            try:
                codici[j] = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                pass

        fili = [threading.Thread(target=alimenta, args=(j, proc), daemon=True)
                for j, proc in enumerate(processi)]
        for filo in fili:
            filo.start()
        scadenza = time.monotonic() + 2 * timeout_s + 5
        for filo in fili:
            filo.join(timeout=max(0.0, scadenza - time.monotonic()))
        for proc in processi:
            if proc.poll() is None:  # appeso: ucciderlo sblocca anche il suo thread
                proc.kill()
                proc.wait()
        for filo in fili:
            filo.join()
        if all(c == 0 for c in codici):
            return tentativo, 0
        ultimo_rc = next(c for c in codici if c != 0)
        tentativo //= 2
    return 0, ultimo_rc


# --------------------------------------------------------------------------
# Verifiche sui segmenti e sul video concatenato
# --------------------------------------------------------------------------

def _sonda_flusso(percorso: str, conta_pacchetti: bool = False) -> Dict[str, Any]:
    voci = ",".join(CAMPI_UNIFORMITA + ("nb_frames", "duration_ts", "nb_read_packets"))
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if conta_pacchetti:
        cmd.append("-count_packets")
    cmd += ["-show_entries", "stream=" + voci, "-of", "json", percorso]
    uscita = subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE).stdout
    flussi = json.loads(uscita.decode("utf-8")).get("streams") or []
    if not flussi:
        raise ErroreParallelo(f"nessun flusso video in {percorso}")
    return flussi[0]


def _corpo_moov(percorso: str) -> Optional[bytes]:
    """Legge solo il box ``moov``, saltando ``mdat`` con seek (i segmenti pesano decine di MB)."""
    with open(percorso, "rb") as f:
        dimensione_file = os.fstat(f.fileno()).st_size
        posizione = 0
        while posizione + 8 <= dimensione_file:
            f.seek(posizione)
            dimensione, tipo = struct.unpack(">I4s", f.read(8))
            intestazione = 8
            if dimensione == 1:
                dimensione = struct.unpack(">Q", f.read(8))[0]
                intestazione = 16
            elif dimensione == 0:
                dimensione = dimensione_file - posizione
            if dimensione < intestazione:
                return None
            if tipo == b"moov":
                return f.read(dimensione - intestazione)
            posizione += dimensione
    return None


def _cerca_avcc(dati: bytes, inizio: int, fine: int) -> Optional[bytes]:
    posizione = inizio
    while posizione + 8 <= fine:
        dimensione, tipo = struct.unpack(">I4s", dati[posizione:posizione + 8])
        intestazione = 8
        if dimensione == 1:
            dimensione = struct.unpack(">Q", dati[posizione + 8:posizione + 16])[0]
            intestazione = 16
        elif dimensione == 0:
            dimensione = fine - posizione
        if dimensione < intestazione or posizione + dimensione > fine:
            return None
        corpo, termine = posizione + intestazione, posizione + dimensione
        trovato = None
        if tipo == b"avcC":
            return dati[corpo:termine]
        if tipo in (b"trak", b"mdia", b"minf", b"stbl"):
            trovato = _cerca_avcc(dati, corpo, termine)
        elif tipo == b"stsd":
            trovato = _cerca_avcc(dati, corpo + 8, termine)  # version/flags + entry_count
        elif tipo in (b"avc1", b"avc3"):
            trovato = _cerca_avcc(dati, corpo + 78, termine)  # campi fissi del VisualSampleEntry
        if trovato is not None:
            return trovato
        posizione = termine
    return None


def impronta_avcc(percorso: str) -> Optional[str]:
    """sha256 dell'avcC (SPS/PPS) letto dai box mp4; ``None`` se non trovato.

    Non ci si affida a ``extradata_hash`` di ffprobe, assente nelle versioni vecchie.
    """
    moov = _corpo_moov(percorso)
    if moov is None:
        return None
    avcc = _cerca_avcc(moov, 0, len(moov))
    return hashlib.sha256(avcc).hexdigest() if avcc is not None else None


def verifica_segmenti(percorsi: Sequence[str], lunghezze: Sequence[int]) -> None:
    """Uniformita' dei parametri e numero di frame di ogni segmento, prima del concat."""
    riferimento: Optional[Tuple] = None
    for percorso, attesi in zip(percorsi, lunghezze):
        flusso = _sonda_flusso(percorso)
        n_frame = int(flusso.get("nb_frames") or -1)
        if n_frame != attesi:
            raise ErroreParallelo(
                f"{os.path.basename(percorso)}: {n_frame} frame invece di {attesi}"
            )
        firma = tuple(flusso.get(c) for c in CAMPI_UNIFORMITA) + (impronta_avcc(percorso),)
        if riferimento is None:
            riferimento = firma
        elif firma != riferimento:
            raise ErroreParallelo(
                f"{os.path.basename(percorso)} non uniforme: {firma} != {riferimento}"
            )


def concatena_segmenti(
    percorsi: Sequence[str],
    percorso_uscita: str,
    lunghezze: Sequence[int],
    fps: int,
) -> None:
    """Concat demuxer in ``-c copy``.

    Ogni riga ``file`` (nome relativo alla lista) e' seguita dalla ``duration``
    esatta al microsecondo. Senza, ffmpeg <= 5.1 (in produzione c'e' la 4.4) usa
    la durata del contenitore mp4, arrotondata per eccesso al millisecondo: a
    24/30/60 fps ogni confine slitterebbe di una frazione di millisecondo. La
    durata e' la differenza fra cumulate arrotondate, cosi' l'errore resta sotto
    il microsecondo qualunque sia il numero di segmenti. Con ffmpeg >= 6 il file
    prodotto e' identico al byte a quello senza ``duration``.

    Restano vietati ``inpoint``/``outpoint`` (tagliano pacchetti) e ogni flag sui
    timestamp: il DTS iniziale negativo dei segmenti con B-frame deve restare
    simmetrico.
    """
    cartella = os.path.dirname(percorsi[0])
    lista = os.path.join(cartella, "lista.txt")
    frame_cumulati = 0
    with open(lista, "w") as f:
        for percorso, n_frame in zip(percorsi, lunghezze):
            inizio_us = round(Fraction(frame_cumulati * 1000000, fps))
            frame_cumulati += n_frame
            durata_us = round(Fraction(frame_cumulati * 1000000, fps)) - inizio_us
            f.write(f"file '{os.path.basename(percorso)}'\n")
            f.write(f"duration {durata_us // 1000000}.{durata_us % 1000000:06d}\n")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "concat", "-safe", "0", "-i", lista,
        "-c", "copy", percorso_uscita,
    ]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)


def verifica_video(percorso: str, n_frame: int, fps: int) -> None:
    """Il video concatenato ha tutti i frame e la durata attesa (tolleranza: mezzo frame)."""
    flusso = _sonda_flusso(percorso, conta_pacchetti=True)
    pacchetti = int(flusso.get("nb_read_packets") or -1)
    if pacchetti != n_frame:
        raise ErroreParallelo(f"video concatenato: {pacchetti} pacchetti invece di {n_frame}")
    try:
        durata = Fraction(int(flusso["duration_ts"])) * Fraction(flusso["time_base"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        raise ErroreParallelo("video concatenato: durata non leggibile")
    if abs(durata - Fraction(n_frame, fps)) >= Fraction(1, 2 * fps):
        raise ErroreParallelo(
            f"video concatenato: durata {float(durata):.4f}s invece di {n_frame / fps:.4f}s"
        )


# --------------------------------------------------------------------------
# Esecuzione di un segmento (nel figlio, oppure in linea nei test)
# --------------------------------------------------------------------------

class _Contatori:
    """Frame completati per segmento in una mmap anonima condivisa.

    Ereditata col fork, non passa da ``/dev/shm``. Alimenta la barra di
    avanzamento e il controllo di stallo nel padre.
    """

    def __init__(self, n: int):
        self._mm = mmap.mmap(-1, 8 * max(1, n))

    def scrivi(self, k: int, valore: int) -> None:
        struct.pack_into("<q", self._mm, 8 * k, valore)

    def totale(self, n: int) -> int:
        return sum(struct.unpack_from("<q", self._mm, 8 * k)[0] for k in range(n))

    def chiudi(self) -> None:
        self._mm.close()


def _nome_segmento(cartella: str, k: int, estensione: str) -> str:
    return os.path.join(cartella, f"seg_{k:03d}{estensione}")


def _esegui_segmento(
    ctx: ContestoFusione,
    k: int,
    inizio: int,
    fine: int,
    codec: str,
    cartella: str,
    larghezza: int,
    altezza: int,
    fps: int,
    argomenti_extra: Optional[Sequence[str]],
    con_impronte: bool,
    contatori: Optional[_Contatori] = None,
) -> Dict[str, Any]:
    """Fonde i frame ``[inizio, fine)`` e li invia in streaming al proprio ffmpeg.

    Nessuna lista di frame fusi: ogni frame vive solo il tempo della scrittura.
    """
    impronte: Optional[List[str]] = [] if con_impronte else None
    tempi = {"fusione": 0.0}
    t_avvio = time.perf_counter()

    def genera():
        for fatti, i in enumerate(range(inizio, fine), start=1):
            t = time.perf_counter()
            frame = fondi_frame(i, ctx)
            tempi["fusione"] += time.perf_counter() - t
            yield frame
            if contatori is not None:
                contatori.scrivi(k, fatti)

    osservatore = None
    if impronte is not None:
        osservatore = lambda dati: impronte.append(hashlib.sha256(dati).hexdigest())  # noqa: E731

    _write_video_pipe(
        genera(), _nome_segmento(cartella, k, ".mp4"), larghezza, altezza, fps,
        codec=codec, argomenti_extra=argomenti_extra,
        percorso_log=_nome_segmento(cartella, k, ".ffmpeg.log"),
        osservatore=osservatore,
    )
    if impronte is not None:
        with open(_nome_segmento(cartella, k, ".sha256"), "w") as f:
            f.write("\n".join(impronte) + "\n")
    return {
        "k": k, "inizio": inizio, "fine": fine, "codec": codec,
        "t_fusione": tempi["fusione"], "t_totale": time.perf_counter() - t_avvio,
    }


def _scrivi_esito(cartella: str, k: int, esito: Dict[str, Any]) -> None:
    """Scrittura atomica: il padre non deve mai leggere un JSON a meta'."""
    definitivo = _nome_segmento(cartella, k, ".json")
    provvisorio = definitivo + ".tmp"
    with open(provvisorio, "w") as f:
        json.dump(esito, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(provvisorio, definitivo)


def _lavoratore_segmento(pid_padre: int, prctl: Any, parametri: Tuple) -> None:
    """Corpo del processo figlio. Non ritorna mai: termina con ``os._exit``.

    ``os._exit`` salta l'epilogo di multiprocessing (flush degli stream, che puo'
    bloccarsi su un lock ereditato da un thread che nel figlio non esiste) e ogni
    distruttore di torch/CUDA ereditato dal padre.
    """
    codice = 1
    k = parametri[1]
    cartella = parametri[5]
    try:
        gc.disable()
        try:
            signal.set_wakeup_fd(-1)
        except (ValueError, OSError):
            pass
        # Gli handler ereditati (RunPod, uvicorn) ignorerebbero SIGTERM.
        for segnale in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                signal.signal(segnale, signal.SIG_DFL)
            except (ValueError, OSError):
                pass
        try:
            os.setpgid(0, 0)  # gruppo proprio: killpg colpisce anche il suo ffmpeg
        except OSError:
            pass
        if prctl is not None:
            try:
                prctl(1, int(signal.SIGKILL), 0, 0, 0)  # PR_SET_PDEATHSIG
            except Exception:
                pass
            if os.getppid() != pid_padre:
                os._exit(1)
        try:
            with open("/proc/self/oom_score_adj", "w") as f:
                f.write("1000")  # in caso di OOM muoia un figlio, non il padre con la GPU
        except OSError:
            pass

        esito = _esegui_segmento(*parametri)
        esito["stato"] = "ok"
        _scrivi_esito(cartella, k, esito)
        codice = 0
    except BaseException as exc:  # noqa: BLE001 - nel figlio nulla deve risalire
        try:
            _scrivi_esito(cartella, k, {
                "stato": "errore",
                "k": k,
                "tipo": type(exc).__name__,
                "messaggio": str(exc),
                "encoder": isinstance(exc, subprocess.CalledProcessError),
                "returncode": getattr(exc, "returncode", None),
                "traceback": traceback.format_exc(),
            })
        except BaseException:  # noqa: BLE001
            pass
    finally:
        os._exit(codice)


def _risolvi_prctl() -> Any:
    """Risolto nel padre, cosi' il figlio non importa nulla dopo il fork."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes
        return ctypes.CDLL(None, use_errno=True).prctl
    except Exception:
        return None


def _termina_processi(processi: Sequence[Any]) -> None:
    """SIGKILL al gruppo di ogni worker non uscito pulito, join, raccolta dei nipoti.

    Si colpisce il gruppo anche se il worker e' gia' morto (OOM, crash): il suo
    ffmpeg gli sopravvive, riceve EOF e continuerebbe a codificare durante il
    ripiego. I worker usciti con 0 hanno gia' atteso il proprio ffmpeg e non si
    toccano, cosi' non si manda mai un segnale a un pid riciclato.
    """
    for p in processi:
        if p.pid is None or (not p.is_alive() and p.exitcode == 0):
            continue
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            try:
                p.kill()
            except Exception:
                pass
    for p in processi:
        if p.pid is not None:
            p.join(timeout=10)
    # Se il padre e' il PID 1 del container (RunPod, uvicorn) adotta gli ffmpeg
    # rimasti orfani: senza waitpid resterebbero zombie fino alla fine del container.
    scadenza = time.monotonic() + 5.0
    for p in processi:
        if p.pid is None or p.exitcode in (None, 0):
            continue  # capogruppo non ancora raccolto da multiprocessing, o uscita pulita
        while True:
            try:
                pid, _ = os.waitpid(-p.pid, os.WNOHANG)
            except OSError:  # ECHILD: nessun nostro figlio in quel gruppo (caso normale)
                break
            if pid == 0:
                if time.monotonic() > scadenza:
                    break
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
                time.sleep(0.02)


def _leggi_esito(cartella: str, k: int) -> Optional[Dict[str, Any]]:
    testo = _leggi_testo(_nome_segmento(cartella, k, ".json"))
    if not testo:
        return None
    try:
        return json.loads(testo)
    except ValueError:
        return None


def _errore_da_esito(k: int, codice: Optional[int], esito: Optional[Dict[str, Any]]) -> ErroreParallelo:
    if esito and esito.get("stato") == "errore":
        messaggio = f"segmento {k}: {esito.get('tipo')}: {esito.get('messaggio')}"
        if esito.get("encoder"):
            return ErroreEncoder(messaggio, esito.get("returncode"))
        return ErroreParallelo(messaggio)
    return ErroreParallelo(f"segmento {k}: worker terminato con codice {codice}")


def _esegui_con_fork(
    parametri_segmenti: List[Tuple],
    cartella: str,
    n_frame: int,
    prima_del_fork: Optional[Callable[[], None]],
    stallo_s: float,
    mostra_progresso: bool,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    contesto_mp = multiprocessing.get_context("fork")
    n_seg = len(parametri_segmenti)
    contatori = _Contatori(n_seg)
    prctl = _risolvi_prctl()
    pid_padre = os.getpid()
    processi: List[Any] = []
    fork_ms: List[float] = []
    barra = None
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        for gestore in logging.root.handlers:
            try:
                gestore.flush()
            except Exception:
                pass
        if prima_del_fork is not None:
            prima_del_fork()

        with _LOCK_FORK:
            gc_attivo = gc.isenabled()
            # Il figlio nasce gia' col GC spento: nessuna raccolta ciclica puo'
            # liberare tensori CUDA ereditati prima che il suo codice parta.
            gc.disable()
            # Ultima azione prima dei fork. Con numThreads == 1 OpenCV esegue i
            # parallel_for in linea su ogni backend (pthreads, GCD): il figlio
            # non aspetta thread del pool che per lui non esistono.
            cv2.setNumThreads(1)
            try:
                for parametri in parametri_segmenti:
                    p = contesto_mp.Process(
                        target=_lavoratore_segmento,
                        args=(pid_padre, prctl, parametri + (contatori,)),
                        daemon=True,
                    )
                    t = time.perf_counter()
                    try:
                        p.start()
                    except (OSError, AssertionError) as exc:
                        raise ErroreParallelo(f"fork del segmento {parametri[1]} fallito: {exc}")
                    fork_ms.append((time.perf_counter() - t) * 1000.0)
                    processi.append(p)
                    try:
                        os.setpgid(p.pid, p.pid)
                    except OSError:
                        pass
            finally:
                cv2.setNumThreads(-1)  # valore predefinito della libreria
                if gc_attivo:
                    gc.enable()

        # La barra (e il thread monitor di tqdm) nascono solo dopo l'ultimo fork.
        if tqdm is not None and mostra_progresso:
            barra = tqdm(total=n_frame, desc="Blending")
        in_attesa = {p.sentinel: (parametri_segmenti[j][1], p) for j, p in enumerate(processi)}
        visti = 0
        ultimo_progresso = time.monotonic()
        while in_attesa:
            pronti = multiprocessing.connection.wait(list(in_attesa), timeout=0.5)
            for sentinella in pronti:
                k, p = in_attesa.pop(sentinella)
                # Uscito ma non ancora raccolto: il pid non e' riciclabile. Se e'
                # morto male il suo ffmpeg gli sopravvive nel gruppo e va fermato
                # subito; se e' uscito bene il gruppo contiene solo lui.
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
                p.join()
                if p.exitcode != 0:
                    raise _errore_da_esito(k, p.exitcode, _leggi_esito(cartella, k))
            totale = contatori.totale(n_seg)
            if totale > visti:
                if barra is not None:
                    barra.update(totale - visti)
                visti = totale
                ultimo_progresso = time.monotonic()
            elif in_attesa and time.monotonic() - ultimo_progresso > stallo_s:
                raise ErroreParallelo(f"nessun avanzamento da {stallo_s:.0f}s (stallo)")

        esiti = []
        for parametri in parametri_segmenti:
            k = parametri[1]
            esito = _leggi_esito(cartella, k)
            if not esito or esito.get("stato") != "ok":
                raise _errore_da_esito(k, 0, esito)
            esiti.append(esito)
        return esiti, fork_ms
    finally:
        if barra is not None:
            barra.close()
        _termina_processi(processi)
        # I Process tengono un riferimento forte agli argomenti (tutti i frame).
        del processi[:]
        contatori.chiudi()


def _esegui_in_linea(parametri_segmenti: List[Tuple]) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Stesso codice dei figli ma nel processo corrente: serve ai test senza fork."""
    esiti = []
    for parametri in parametri_segmenti:
        try:
            esiti.append(_esegui_segmento(*parametri))
        except subprocess.CalledProcessError as exc:
            raise ErroreEncoder(f"segmento {parametri[1]}: {exc}", exc.returncode)
    return esiti, []


# --------------------------------------------------------------------------
# Orchestrazione
# --------------------------------------------------------------------------

def _motivo_percorso_sequenziale(
    ctx: ContestoFusione,
    n_workers: int,
    min_frame: int,
    esecutore: str,
) -> Optional[str]:
    """Controlli prima di qualunque fork. Una stringa = si resta sul percorso storico.

    Gli input anomali vanno al percorso storico di proposito: l'errore che ne
    esce e' identico a quello di oggi, senza sprecare un passaggio parallelo.
    """
    n_frame = len(ctx.facce)
    if n_workers <= 1:
        return "n_workers=1"
    if esecutore == "processo" and "fork" not in multiprocessing.get_all_start_methods():
        return "fork non disponibile su questa piattaforma"
    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        return "ffmpeg/ffprobe assenti dal PATH"
    if len(calcola_segmenti(n_frame, n_workers, min_frame)) < 2:
        return f"{n_frame} frame: troppo pochi per due segmenti"
    if not ctx.frame_ciclo or not ctx.maschere:
        return "contesto senza frame o senza maschere"
    altezza, larghezza = ctx.frame_ciclo[0].shape[:2]
    if altezza % 2 or larghezza % 2:
        return f"dimensioni dispari {larghezza}x{altezza}"
    for i in range(n_frame):
        face_box = ctx.face_boxes[i]
        if face_box is None:
            continue
        maschera = ctx.maschere[i % len(ctx.maschere)]
        if maschera is None:
            return f"frame {i} senza maschera precalcolata"
        riquadro, _ = get_crop_box(face_box, 1.5)
        atteso = (riquadro[3] - riquadro[1], riquadro[2] - riquadro[0])
        if tuple(maschera.shape[:2]) != atteso:
            return f"frame {i}: maschera {tuple(maschera.shape[:2])} invece di {atteso}"
    return None


def _riversa_log_ffmpeg(cartella: str, max_byte: int = 2000, max_segmenti: int = 4) -> None:
    """Solo nel padre: porta su stderr la coda dei log ffmpeg non vuoti dei segmenti.

    Nel percorso storico ffmpeg scrive direttamente sullo stderr del job; qui i
    figli non stampano, quindi senza questo passaggio avvisi ed errori andrebbero
    persi con la cartella del tentativo.
    """
    try:
        nomi = sorted(n for n in os.listdir(cartella) if n.endswith(".ffmpeg.log"))
    except OSError:
        return
    mostrati = 0
    for nome in nomi:
        try:
            with open(os.path.join(cartella, nome), "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - max_byte))
                coda = f.read().decode("utf-8", "replace").strip()
        except OSError:
            continue
        if coda:
            print(f"[phase G] {nome}:\n{coda}", file=sys.stderr)
            mostrati += 1
            if mostrati >= max_segmenti:
                break


def _rss_gb() -> Optional[float]:
    try:
        with open("/proc/self/statm", "r") as f:
            pagine = int(f.read().split()[1])
        return pagine * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (OSError, ValueError, IndexError):
        return None


def _riga_di_log(rapporto: RapportoFusione, dettagli_cpu: Dict[str, Optional[int]],
                 cpu: int, extra: str = "") -> str:
    riga = (
        f"[phase F+G] modo={rapporto.modo} n_workers={rapporto.n_workers} "
        f"segmenti={rapporto.n_segmenti} codec={rapporto.codec}"
    )
    if rapporto.thread_x264 is not None:
        riga += f" x264_threads={rapporto.thread_x264}"
    if rapporto.fork_ms:
        ordinati = sorted(rapporto.fork_ms)
        riga += (f" fork_ms={ordinati[0]:.0f}/{ordinati[len(ordinati) // 2]:.0f}"
                 f"/{ordinati[-1]:.0f}")
    riga += (
        f" tentativi={rapporto.tentativi} cpu(slurm={dettagli_cpu.get('slurm')},"
        f"affinity={dettagli_cpu.get('affinity')},cgroup={dettagli_cpu.get('cgroup')},"
        f"os={dettagli_cpu.get('os')})->{cpu}"
    )
    if rapporto.motivo:
        riga += f" motivo={rapporto.motivo!r}"
    return riga + extra


def fondi_e_codifica(
    ctx: ContestoFusione,
    percorso_video: str,
    fps: int,
    use_nvenc: bool = True,
    n_workers: Optional[int] = None,
    dir_lavoro: Optional[str] = None,
    migliora: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    min_frame_segmento: Optional[int] = None,
    esecutore: str = "processo",
    prima_del_fork: Optional[Callable[[], None]] = None,
    t_inizio_fase_f: Optional[float] = None,
    percorso_impronte: Optional[str] = None,
    mantieni_segmenti: bool = False,
    mostra_progresso: bool = True,
    ambiente: Optional[Dict[str, str]] = None,
) -> RapportoFusione:
    """Fasi F e G: fonde i frame e produce il video muto ``percorso_video``.

    Macchina a stati: parallelo col codec sondato -> (se un segmento nvenc
    fallisce) un solo nuovo tentativo parallelo in libx264 -> percorso
    sequenziale storico. Il codec e' sempre uniforme dentro un video.

    ``migliora`` (GFPGAN) gira sulla GPU del padre. Nel percorso parallelo
    diventa un passaggio preliminare che produce una lista nuova di facce; da li'
    in poi nessuno lo richiama, quindi nemmeno il ripiego puo' applicarlo due volte.
    """
    ambiente = os.environ if ambiente is None else ambiente
    t_f = time.perf_counter() if t_inizio_fase_f is None else t_inizio_fase_f
    if min_frame_segmento is None:
        min_frame_segmento = MIN_FRAME_SEGMENTO  # letto a ogni chiamata, non alla definizione
    n_frame = len(ctx.facce)
    rigoroso = ambiente.get("MUSETALK_FG_STRICT") == "1"
    verifica_parita = ambiente.get("MUSETALK_FG_VERIFICA") == "1"
    cpu, dettagli_cpu = risolvi_cpu_effettive(ambiente)
    n_w, origine = risolvi_n_workers(n_workers, ambiente, cpu)

    def sequenziale(contesto: ContestoFusione, funzione_migliora, motivo: str,
                    fallback: bool, tentativi: int) -> RapportoFusione:
        frames = fondi_sequenziale(contesto, funzione_migliora, mostra_progresso)
        t_fusione = time.perf_counter() - t_f
        print(f"[phase F] blending: {len(frames)} frames in {t_fusione:.2f}s")
        if percorso_impronte:  # solo test e diagnostica: fuori dai tempi di fase
            with open(percorso_impronte, "w") as f:
                f.write("\n".join(impronta_frame(fr) for fr in frames) + "\n")
        t_g = time.perf_counter()
        altezza, larghezza = frames[0].shape[:2]
        codec_usato = codifica_con_fallback(frames, percorso_video, larghezza, altezza, fps, use_nvenc)
        t_codifica = time.perf_counter() - t_g
        print(f"[phase G] video encode: {t_codifica:.2f}s")
        rapporto = RapportoFusione(
            modo="sequenziale", n_frame=len(frames), n_workers=1, n_segmenti=1,
            codec=codec_usato, motivo=motivo, tentativi=tentativi, fallback=fallback,
            t_fusione=t_fusione, t_codifica=t_codifica,
        )
        print(_riga_di_log(rapporto, dettagli_cpu, cpu))
        return rapporto

    # In modo rigoroso una richiesta esplicita di parallelo non deve poter
    # finire in silenzio sul sequenziale; il dimensionamento automatico si'.
    pretende_parallelo = rigoroso and n_w > 1 and origine in ("parametro", "env")
    motivo = _motivo_percorso_sequenziale(ctx, n_w, min_frame_segmento, esecutore)
    if motivo is not None:
        if pretende_parallelo:
            raise ErroreParallelo(f"percorso parallelo non praticabile: {motivo}")
        return sequenziale(ctx, migliora, f"{motivo} ({origine})", False, 0)

    altezza, larghezza = ctx.frame_ciclo[0].shape[:2]
    codec = "libx264"
    limite_nvenc = 0
    if use_nvenc:
        sessioni = _intero_da_env(ambiente, "MUSETALK_NVENC_SESSIONS") or SESSIONI_NVENC_PREDEFINITE
        try:
            limite_nvenc, rc_sonda = sonda_nvenc(larghezza, altezza, fps, sessioni)
        except FileNotFoundError:
            print("[phase G] ffmpeg not found in PATH; cannot encode")
            raise
        if limite_nvenc > 0:
            codec = "h264_nvenc"
        else:
            print(f"[phase G] h264_nvenc failed (rc={rc_sonda}); fallback to libx264")
    if codec == "h264_nvenc" and len(calcola_segmenti(n_frame, min(n_w, limite_nvenc),
                                                       min_frame_segmento)) < 2:
        # Una sola sessione nvenc utilizzabile: il percorso storico fa gia' questo.
        if pretende_parallelo:
            raise ErroreParallelo("percorso parallelo non praticabile: una sola sessione nvenc "
                                  "(use_nvenc=False per il parallelo in libx264)")
        return sequenziale(ctx, migliora, "una sola sessione nvenc", False, 0)

    # Passaggio preliminare GFPGAN, nel padre e prima di qualunque fork.
    t_prepass = 0.0
    ctx_pronto = dataclasses.replace(ctx, fp=None)
    if migliora is not None:
        t = time.perf_counter()
        facce_pronte = [
            migliora(faccia.astype(np.uint8)) if ctx.face_boxes[i] is not None else faccia
            for i, faccia in enumerate(ctx.facce)
        ]
        ctx_pronto = dataclasses.replace(ctx_pronto, facce=facce_pronte)
        t_prepass = time.perf_counter() - t

    # Riscaldamento nel padre: le inizializzazioni pigre di PIL, cv2 e subprocess
    # avvengono qui e non nei figli, dove un lock ereditato a meta' non si
    # libererebbe mai.
    primo_valido = next((i for i in range(n_frame) if ctx_pronto.face_boxes[i] is not None), None)
    if primo_valido is not None:
        fondi_frame(primo_valido, ctx_pronto)
    subprocess.run(["ffmpeg", "-version"], stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    stallo_s = _secondi_da_env(ambiente, "MUSETALK_FG_STALLO_S", STALLO_PREDEFINITO_S)
    con_impronte = bool(percorso_impronte) or verifica_parita
    dir_base = os.path.join(dir_lavoro or os.path.dirname(percorso_video) or ".",
                            f"fg_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    codec_tentativi = [codec] + (["libx264"] if codec == "h264_nvenc" else [])
    errore = ""
    rapporto: Optional[RapportoFusione] = None
    tentativi = 0
    try:
        for codec_tentativo in codec_tentativi:
            tentativi += 1
            n_seg_max = min(n_w, limite_nvenc) if codec_tentativo == "h264_nvenc" else n_w
            segmenti = calcola_segmenti(n_frame, n_seg_max, min_frame_segmento)
            thread_x264 = None
            argomenti_extra: Optional[List[str]] = None
            if codec_tentativo != "h264_nvenc":
                thread_x264 = risolvi_thread_x264(ambiente, cpu, len(segmenti))
                argomenti_extra = ["-threads", str(thread_x264)]
                if len(segmenti) * thread_x264 > cpu:
                    print(f"[phase F+G] attenzione: {len(segmenti)} segmenti x {thread_x264} "
                          f"thread x264 > {cpu} CPU")
            cartella = os.path.join(dir_base, f"tentativo_{tentativi}")
            os.makedirs(cartella, exist_ok=True)
            parametri_segmenti = [
                (ctx_pronto, k, inizio, fine, codec_tentativo, cartella, larghezza, altezza,
                 fps, argomenti_extra, con_impronte)
                for k, (inizio, fine) in enumerate(segmenti)
            ]
            rss = _rss_gb()
            try:
                if esecutore == "processo":
                    esiti, fork_ms = _esegui_con_fork(
                        parametri_segmenti, cartella, n_frame, prima_del_fork, stallo_s,
                        mostra_progresso)
                else:
                    esiti, fork_ms = _esegui_in_linea(parametri_segmenti)
                t_fusione = time.perf_counter() - t_f
                t_g = time.perf_counter()
                percorsi = [_nome_segmento(cartella, k, ".mp4") for k in range(len(segmenti))]
                lunghezze = [fine - inizio for inizio, fine in segmenti]
                verifica_segmenti(percorsi, lunghezze)
                concatena_segmenti(percorsi, percorso_video, lunghezze, fps)
                verifica_video(percorso_video, n_frame, fps)
            except ErroreEncoder as exc:
                _riversa_log_ffmpeg(cartella)
                if codec_tentativo == "h264_nvenc":
                    print(f"[phase G] h264_nvenc failed (rc={exc.returncode}); fallback to libx264")
                    continue
                # Solo il testo: tenere l'eccezione terrebbe vivo, via traceback, il
                # frame di chiamata con tutti i frame del job fino al GC ciclico.
                errore = f"{type(exc).__name__}: {exc}"
                break
            except (ErroreParallelo, OSError, subprocess.SubprocessError, ValueError) as exc:
                _riversa_log_ffmpeg(cartella)
                errore = f"{type(exc).__name__}: {exc}"
                break
            _riversa_log_ffmpeg(cartella)  # eventuali avvisi di ffmpeg, come nel percorso storico

            print(f"[phase F] blending: {n_frame} frames in {t_fusione:.2f}s")
            t_codifica = time.perf_counter() - t_g
            print(f"[phase G] video encode: {t_codifica:.2f}s")
            rapporto = RapportoFusione(
                modo="parallelo", n_frame=n_frame, n_workers=n_w, n_segmenti=len(segmenti),
                codec=codec_tentativo, motivo=origine, tentativi=tentativi, fork_ms=fork_ms,
                t_fusione=t_fusione, t_codifica=t_codifica, thread_x264=thread_x264,
                percorsi_segmenti=percorsi if mantieni_segmenti else [],
            )
            extra = f" prepass_gfpgan={t_prepass:.2f}s uniformita=OK"
            if rss is not None:
                extra += f" rss_fork={rss:.1f}GB"
            if esiti:
                extra += (f" fusione_cpu={sum(e['t_fusione'] for e in esiti):.1f}s"
                          f" seg_max={max(e['t_totale'] for e in esiti):.1f}s")
            print(_riga_di_log(rapporto, dettagli_cpu, cpu, extra))

            if con_impronte:
                impronte: List[str] = []
                for k in range(len(segmenti)):
                    testo = _leggi_testo(_nome_segmento(cartella, k, ".sha256")) or ""
                    impronte.extend(testo.split())
                if percorso_impronte:
                    with open(percorso_impronte, "w") as f:
                        f.write("\n".join(impronte) + "\n")
                if verifica_parita:
                    # Riferimento ricalcolato qui, sugli stessi dati in memoria: la
                    # fase E non e' riproducibile al bit fra due job diversi.
                    diverso = next(
                        (i for i in range(n_frame)
                         if i >= len(impronte)
                         or impronta_frame(fondi_frame(i, ctx_pronto)) != impronte[i]),
                        None,
                    )
                    rapporto.parita = diverso is None
                    if diverso is None:
                        print(f"[phase F] verifica parita: OK ({n_frame} frame)")
                    else:
                        print(f"[phase F] verifica parita: KO al frame {diverso}")
                        if rigoroso:
                            raise ErroreParallelo(f"parita violata al frame {diverso}")
            return rapporto

        # Tutti i tentativi paralleli sono falliti.
        if rigoroso:
            print(f"[phase F] PARALLELO FALLITO ({errore})")
            raise ErroreParallelo(errore)
        print(f"[phase F] PARALLELO FALLITO ({errore}); ripiego sul percorso sequenziale")
        try:
            os.remove(percorso_video)
        except OSError:
            pass
        # ctx_pronto contiene gia' le facce migliorate: migliora resta None.
        return sequenziale(ctx_pronto, None, f"ripiego: {errore}", True, tentativi)
    finally:
        if not mantieni_segmenti:
            shutil.rmtree(dir_base, ignore_errors=True)


__all__ = [
    "ContestoFusione",
    "ErroreEncoder",
    "ErroreParallelo",
    "RapportoFusione",
    "argomenti_encoder",
    "calcola_segmenti",
    "codifica_con_fallback",
    "concatena_segmenti",
    "fondi_e_codifica",
    "fondi_frame",
    "fondi_sequenziale",
    "impronta_avcc",
    "impronta_frame",
    "quota_cpu_cgroup",
    "risolvi_cpu_effettive",
    "risolvi_n_workers",
    "sonda_nvenc",
    "verifica_segmenti",
    "verifica_video",
]
