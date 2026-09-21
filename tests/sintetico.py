"""Input sintetici e strumenti ffmpeg/ffprobe per i test delle fasi F e G.

Nessuna GPU e nessun torch: si usano solo numpy, cv2, PIL e ffmpeg dal PATH.
Ogni faccia generata porta un codice a blocchi con il proprio indice, cosi' dal
video finale si puo' verificare che nessun frame sia duplicato, mancante o fuori
ordine ai confini fra segmenti.
"""

import hashlib
import json
import os
import subprocess
import sys
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RADICE not in sys.path:
    sys.path.insert(0, RADICE)

from musetalk.utils.blending import (  # noqa: E402
    _build_blend_mask_from_parsing,
    get_crop_box,
    get_image,
)

# Stesso valore di ``coord_placeholder`` in musetalk/utils/preprocessing.py, che
# qui non si puo' importare perche' carica torch.
SEGNAPOSTO = (0.0, 0.0, 0.0, 0.0)
LATO_CODICE = 4  # griglia 4x4 = 16 bit


def faccia_con_codice(indice: int, rng: np.random.RandomState) -> np.ndarray:
    """Faccia 256x256 con l'indice scritto in una griglia di blocchi chiari/scuri.

    Ritorna una vista a stride negativo sull'ultimo asse, come i frame prodotti
    da ``vae.decode_latents`` (``image[..., ::-1]``).
    """
    cella = 256 // LATO_CODICE
    immagine = np.empty((256, 256, 3), dtype=np.uint8)
    for bit in range(LATO_CODICE * LATO_CODICE):
        riga, colonna = divmod(bit, LATO_CODICE)
        valore = 230 if (indice >> bit) & 1 else 25
        immagine[riga * cella:(riga + 1) * cella, colonna * cella:(colonna + 1) * cella] = valore
    rumore = rng.randint(-6, 7, size=immagine.shape)
    immagine = np.clip(immagine.astype(np.int16) + rumore, 0, 255).astype(np.uint8)
    return immagine[..., ::-1]


def leggi_codice(frame: np.ndarray, face_box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = face_box
    regione = frame[y1:y2, x1:x2].astype(np.float32).mean(axis=2)
    alto, largo = regione.shape
    indice = 0
    for bit in range(LATO_CODICE * LATO_CODICE):
        riga, colonna = divmod(bit, LATO_CODICE)
        y0 = int((riga + 0.3) * alto / LATO_CODICE)
        y9 = int((riga + 0.7) * alto / LATO_CODICE)
        x0 = int((colonna + 0.3) * largo / LATO_CODICE)
        x9 = int((colonna + 0.7) * largo / LATO_CODICE)
        if regione[y0:max(y9, y0 + 1), x0:max(x9, x0 + 1)].mean() > 128:
            indice |= 1 << bit
    return indice


def calcola_face_boxes(coord_ciclo, frame_ciclo, n_frame: int, extra_margin: int):
    """Specchio del ciclo di preparazione di ``generate()``: ``None`` = segnaposto."""
    face_boxes: List[Optional[Tuple[int, int, int, int]]] = [None] * n_frame
    for i in range(n_frame):
        bbox = coord_ciclo[i % len(coord_ciclo)]
        if bbox == SEGNAPOSTO:
            continue
        ori_frame = frame_ciclo[i % len(frame_ciclo)]
        x1, y1, x2, y2 = bbox
        y2c = min(y2 + extra_margin, ori_frame.shape[0])
        face_boxes[i] = (x1, y1, x2, y2c)
    return face_boxes


def crea_input(
    n_originali: int = 10,
    n_frame: int = 64,
    altezza: int = 192,
    larghezza: int = 256,
    seme: int = 0,
    con_segnaposto: bool = True,
    tipo_bbox: str = "lista",
    lunghezza_maschere: str = "n_frame",
    maschera_piena: bool = False,
    extra_margin: int = 10,
    volto: Tuple[int, int] = (64, 110),
    rumore: int = 10,
    con_bbox_degenere: bool = False,
) -> Dict[str, Any]:
    """Frame originali, bbox, facce generate e maschere, tutto deterministico."""
    rng = np.random.RandomState(seme)
    yy, xx = np.mgrid[0:altezza, 0:larghezza]
    frames = []
    for j in range(n_originali):
        base = np.stack([
            (xx * 255 // larghezza + 7 * j) % 256,
            (yy * 255 // altezza + 13 * j) % 256,
            ((xx + yy) // 2 + 29 * j) % 256,
        ], axis=2).astype(np.int16)
        if rumore > 0:
            base += rng.randint(-rumore, rumore + 1, size=base.shape)
        frames.append(np.clip(base, 0, 255).astype(np.uint8))

    coordinate = []
    for j in range(n_originali):
        if con_segnaposto and j % 5 == 3:
            coordinate.append(SEGNAPOSTO)
            continue
        largo = int(rng.randint(volto[0], volto[1]))
        alto = int(rng.randint(volto[0] + 6, volto[1] + 10))
        if j % 4 == 0:
            # Vicino al bordo: il riquadro espanso 1.5x esce dal frame.
            x1, y1 = 1, altezza - alto - 2
        else:
            x1 = int(rng.randint(0, larghezza - largo))
            y1 = int(rng.randint(0, altezza - alto))
        bbox = [x1, y1, x1 + largo, y1 + alto]
        if con_bbox_degenere and j == 1:
            # Larghezza zero: cv2.resize solleva e lo storico lascia il frame non fuso.
            bbox = [x1, y1, x1, y1 + alto]
        if tipo_bbox == "tupla":
            bbox = tuple(bbox)
        elif tipo_bbox == "numpy":
            bbox = tuple(np.int64(v) for v in bbox)
        coordinate.append(bbox)

    frame_ciclo = frames + frames[::-1]
    coord_ciclo = coordinate + coordinate[::-1]
    face_boxes = calcola_face_boxes(coord_ciclo, frame_ciclo, n_frame, extra_margin)
    facce = [faccia_con_codice(i, rng) for i in range(n_frame)]

    def maschera_per(face_box):
        riquadro, _ = get_crop_box(face_box, 1.5)
        forma = (int(riquadro[2] - riquadro[0]), int(riquadro[3] - riquadro[1]))
        if maschera_piena or face_box[2] == face_box[0]:
            return np.full((forma[1], forma[0]), 255, dtype=np.uint8)
        parsing = Image.new("L", forma, 0)
        ImageDraw.Draw(parsing).ellipse(
            [forma[0] // 5, forma[1] // 5, forma[0] * 4 // 5, forma[1] * 4 // 5], fill=255)
        return _build_blend_mask_from_parsing(
            parsing, tuple(int(v) for v in face_box), riquadro, forma)

    n_maschere = n_frame if lunghezza_maschere == "n_frame" else len(frame_ciclo)
    boxes_per_maschere = calcola_face_boxes(coord_ciclo, frame_ciclo, n_maschere, extra_margin)
    maschere = [None if b is None else maschera_per(b) for b in boxes_per_maschere]

    return {
        "frame_ciclo": frame_ciclo,
        "coord_ciclo": coord_ciclo,
        "face_boxes": face_boxes,
        "facce": facce,
        "maschere": maschere,
        "extra_margin": extra_margin,
        "n_frame": n_frame,
        "altezza": altezza,
        "larghezza": larghezza,
    }


def fusione_storica(res_frame_list, coord_list_cycle, frame_list_cycle, blend_masks,
                    extra_margin, parsing_mode="jaw", enhance_fn=None):
    """Copia testuale del ciclo di fusione di ``generate()`` prima della modifica.

    E' il riferimento del criterio di parita': non va "migliorata". Nota: usa
    ``blend_masks[i]`` senza modulo, come l'originale.
    """
    coord_placeholder = SEGNAPOSTO
    n_blend = len(res_frame_list)
    face_boxes = [None] * n_blend
    ori_frames_for_blend = [None] * n_blend
    for i in range(n_blend):
        bbox = coord_list_cycle[i % len(coord_list_cycle)]
        if bbox == coord_placeholder:
            continue
        ori_frame = frame_list_cycle[i % len(frame_list_cycle)]
        x1, y1, x2, y2 = bbox
        y2c = min(y2 + extra_margin, ori_frame.shape[0])
        face_boxes[i] = (x1, y1, x2, y2c)
        ori_frames_for_blend[i] = ori_frame

    combine_frames = []
    for i, res_frame in enumerate(res_frame_list):
        bbox = coord_list_cycle[i % len(coord_list_cycle)]
        if bbox == coord_placeholder:
            combine_frames.append(frame_list_cycle[i % len(frame_list_cycle)])
            continue
        x1, y1, x2, y2c = face_boxes[i]
        ori_frame = ori_frames_for_blend[i].copy()

        face_crop = res_frame.astype(np.uint8)
        if enhance_fn is not None:
            face_crop = enhance_fn(face_crop)
        try:
            face_resized = cv2.resize(face_crop, (x2 - x1, y2c - y1))
        except Exception:
            combine_frames.append(ori_frame)
            continue

        combine_frame = get_image(
            ori_frame,
            face_resized,
            [x1, y1, x2, y2c],
            mode=parsing_mode,
            fp=None,
            precomputed_mask=blend_masks[i],
        )
        combine_frames.append(combine_frame)
    return combine_frames


# --------------------------------------------------------------------------
# ffmpeg / ffprobe
# --------------------------------------------------------------------------

def _esegui(cmd: List[str]) -> bytes:
    return subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def info_video(percorso: str) -> Dict[str, Any]:
    uscita = _esegui([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries",
        "stream=codec_name,r_frame_rate,time_base,duration_ts,nb_read_packets,nb_frames,"
        "width,height,pix_fmt:format=duration",
        "-of", "json", percorso,
    ])
    dati = json.loads(uscita.decode("utf-8"))
    flusso = dati["streams"][0]
    flusso["durata_contenitore"] = float(dati["format"]["duration"])
    flusso["durata_esatta"] = Fraction(int(flusso["duration_ts"])) * Fraction(flusso["time_base"])
    return flusso


def pacchetti(percorso: str) -> List[Tuple[int, int]]:
    """Lista di (pts, dts) nell'ordine del file."""
    uscita = _esegui([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts,dts", "-of", "csv=p=0", percorso,
    ]).decode("utf-8")
    risultato = []
    for riga in uscita.splitlines():
        campi = [c for c in riga.strip().split(",") if c != ""]
        if len(campi) >= 2:
            risultato.append((int(campi[0]), int(campi[1])))
    return risultato


def impronte_slice(percorso: str) -> List[str]:
    """sha256 di ogni NAL di slice (tipi 1 e 5) del flusso H.264, nell'ordine del file.

    E' la prova che il concat non ricodifica: i dati codificati delle slice devono
    essere identici al byte. Non si confrontano i pacchetti interi perche' il concat
    demuxer inserisce da solo ``h264_mp4toannexb`` (SPS/PPS in banda sui keyframe) e
    la lunghezza degli start code cambia fra le versioni di ffmpeg. Dentro una NAL
    la sequenza ``00 00 01`` non puo' comparire (emulation prevention), quindi
    basta dividere sugli start code.
    """
    grezzo = _esegui([
        "ffmpeg", "-v", "error", "-i", percorso, "-map", "0:v:0", "-c", "copy",
        "-bsf:v", "h264_mp4toannexb", "-f", "h264", "-",
    ])
    impronte = []
    for nal in grezzo.split(b"\x00\x00\x01")[1:]:
        nal = nal.rstrip(b"\x00")  # lo zero iniziale di uno start code a 4 byte
        if nal and (nal[0] & 0x1F) in (1, 5):
            impronte.append(hashlib.sha256(nal).hexdigest())
    return impronte


def impronte_decodificate(percorso: str) -> List[str]:
    uscita = _esegui([
        "ffmpeg", "-v", "error", "-i", percorso, "-map", "0:v:0",
        "-f", "framehash", "-hash", "sha256", "-",
    ]).decode("utf-8")
    return [riga.split(",")[-1].strip() for riga in uscita.splitlines()
            if riga.strip() and not riga.startswith("#")]


def decodifica_video(percorso: str, larghezza: int, altezza: int) -> List[np.ndarray]:
    grezzo = _esegui([
        "ffmpeg", "-v", "error", "-i", percorso, "-map", "0:v:0",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ])
    passo = larghezza * altezza * 3
    return [
        np.frombuffer(grezzo[k:k + passo], dtype=np.uint8).reshape(altezza, larghezza, 3)
        for k in range(0, len(grezzo) - passo + 1, passo)
    ]
