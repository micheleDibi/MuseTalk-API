# MuseTalk-API

Pipeline production-ready per generare video **lipsync di lunga durata** (15+ min) di un avatar parlante, partendo da una directory di clip brevi + una traccia audio. Fork di [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) ottimizzato per esecuzione su **RunPod Serverless** (H100 SXM 80GB) con storage **Cloudflare R2**, target di **rapporto ≤ 1:1** tra durata audio e tempo di processing.

> **Documentazione tecnica completa**: vedi [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) per il modello, l'architettura della pipeline, la storia delle versioni v0→v8 e le motivazioni di ogni cambiamento.

---

## Cosa fa

Il client `synth_random_lipsync.py`:

1. Prende N clip brevi (es. 5 clip da ~6s) dello stesso soggetto.
2. Le campiona random con sostituzione e le concatena per coprire la durata della traccia audio.
3. Invia il video assemblato + l'audio a un endpoint RunPod Serverless che esegue MuseTalk + face parsing + blending + encoding.
4. Salva il video lipsynced + un JSON con metadati (tempi, parametri, run_id).

Cache per-frame su R2 → seconda esecuzione e successive con le stesse clip sono ~14-15 min per 15.7 min di audio (rapporto **0.95:1**), con **zero drift cumulativo** audio/video.

---

## Architettura in sintesi

```
┌──────── CLIENT (Windows/Linux PC) ─────────┐
│  synth_random_lipsync.py                   │
│  ├─ build_random_video → mp4 a 25fps       │
│  ├─ get_or_compute_full_manifest           │
│  │  └─ R2 cache hit / preprocess_full job  │
│  └─ submit_job lipsync                     │
└────────────────────┬───────────────────────┘
                     │ HTTPS / R2
                     ▼
┌─────── RUNPOD SERVERLESS (H100 SXM) ───────┐
│  musetalk_runpod/handler.py                │
│  └─ MuseTalkInference.generate()           │
│     Phase A: cv2 frame extract             │
│     Phase B: Whisper features              │
│     Phase C: bbox (skip cache)             │
│     Phase D: VAE encode (skip cache)       │
│     Phase E: UNet batch=16 inference       │
│     Phase F: blending + cached masks       │
│     Phase G: ffmpeg pipe encode            │
└────────────────────────────────────────────┘
```

Dettagli in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Quick start

### Prerequisiti

- **Account RunPod** con endpoint Serverless configurato (vedi sotto).
- **Bucket Cloudflare R2** + access key.
- Python 3.10+ in venv locale (solo per il client).
- ffmpeg + ffprobe installati e nel PATH.

### Setup credenziali

Crea `runpod.env` nella root del repo (è in `.gitignore`):

```env
RUNPOD_API_KEY=...
RUNPOD_ENDPOINT_ID=...
R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com
R2_BUCKET=musetalk-io
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
```

Le variabili sono auto-caricate da `scripts/client/runpod_client.py` ad ogni invocazione del CLI.

### Setup endpoint RunPod

Crea un endpoint Serverless con:

| Parametro | Valore |
|---|---|
| Container Image | `micheledibi/musetalk-runpod:v7` |
| GPU Types | H100 SXM, H100 PCIe, L40S (escludi A100 e Blackwell sm_120) |
| Workers Min (Active) | 1 |
| Workers Max | 3 |
| Idle Timeout | 60 s |
| Execution Timeout | 3 600 000 ms (60 min) |
| Env vars | `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` |

### Installazione locale del client

```powershell
# Windows + venv (PowerShell)
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install boto3 requests
```

(Il client usa solo `boto3` e `requests` — niente PyTorch/CUDA lato locale.)

### Run

```powershell
python -m scripts.client.synth_random_lipsync `
  --clips-dir .\avatar4university\videos\michele_monaco\ `
  --audio .\avatar4university\audios\audio.wav `
  --output .\data\generated\final.mp4 `
  --seed 42 --use-nvenc
```

Output:
- `.\data\generated\final.mp4` — video lipsynced
- `.\data\generated\final.json` — metadati (run_id, parametri, tempi, R2 keys)

---

## CLI reference

`scripts/client/synth_random_lipsync.py`:

| Flag | Default | Descrizione |
|---|---|---|
| `--clips-dir` | required | Directory con le clip sorgente (.mp4/.mov/.mkv) |
| `--audio` | required | File audio (wav/mp3) — la traccia da lipsynchare |
| `--output` | required | Path del video .mp4 finale |
| `--minutes` | `audio_duration` | Override durata target in minuti (default: durata audio) |
| `--seed` | random | Seed per il sampling delle clip (riproducibilità) |
| `--fps` | 25 | FPS target. **Usa 25 — altri valori non multipli di 50 crashano upstream** |
| `--batch-size` | 16 | Batch size UNet. H100 80GB regge 16-32 |
| `--bbox-shift` | 0 | Shift verticale della bbox (-7..+7) |
| `--extra-margin` | 10 | Margine sotto la mascella (5-20) |
| `--parsing-mode` | `jaw` | `jaw` (consigliato), `raw`, `neck` |
| `--left-cheek-width` | 90 | Larghezza maschera guancia sinistra |
| `--right-cheek-width` | 90 | Larghezza maschera guancia destra |
| `--enhance` | off | Attiva GFPGAN face restoration (+5-7 min su 15 min audio) |
| `--quality` | off | Preset: `--enhance` automatico + `gfpgan-weight 0.6` |
| `--gfpgan-weight` | 0.6 | Blend GFPGAN (0=originale, 1=fully enhanced) |
| `--use-nvenc` | off | Concat ffmpeg con NVENC (più veloce su RTX locale) |
| `--skip-manifest` | off | Salta il cache R2 e fa face detection runtime su tutti i frame (lento) |
| `--skip-full-preprocess` | off | Manifest bbox-only senza latents/parsing cache |
| `--manifest-cache-dir` | `data/manifests` | Cache locale dei manifest |
| `--keep-r2-objects` | off | Non eliminare gli oggetti R2 dopo il run |
| `--keep-intermediate` | off | Non eliminare `video_completo.mp4` locale |
| `--api-timeout-seconds` | nessuno | Timeout polling RunPod (default: nessuno) |
| `--preprocess-timeout-seconds` | 1800 | Timeout job preprocess (default: 30 min) |

---

## Performance

Sul setup di riferimento (H100 SXM 80GB, audio 15.73 min, 5 clip uniche da 5.9s @ 24fps):

| Versione | Cache | Tempo | Rapporto | Drift |
|---|---|---|---|---|
| v0 (local RTX 4070) | nessuno | 137 min | 8.7:1 | — |
| v5 (RunPod, prime ottimizzazioni) | latents+masks | 13.7 min | 0.87:1 | spaziale (bbox statica) |
| v7 (RunPod, encoding bump + per-frame bbox) | v3 per-frame | 14-15 min | 0.95:1 | 520ms cumulativo |
| **v8** (RunPod, fix drift) — **attuale** | v3 per-frame | **14-15 min** | **0.95:1** | **0 ms** ✓ |
| v8 + `--quality` (GFPGAN) | v3 per-frame | 20-21 min | 1.3:1 | 0 ms |

Tabella completa con phase-by-phase breakdown in [`docs/ARCHITECTURE.md` §6](docs/ARCHITECTURE.md#6-numeri-di-performance).

---

## Struttura del progetto

```
MuseTalk-API/
├─ scripts/client/                  # Client lato utente (Python puro)
│  ├─ synth_random_lipsync.py       # Entry-point CLI
│  ├─ video_assembler.py            # probe/sample/concat/trim ffmpeg
│  ├─ clip_manifest.py              # Cache logic + preprocess_full request
│  └─ runpod_client.py              # R2 + RunPod API (submit/poll)
├─ musetalk_runpod/                 # Container RunPod Serverless
│  ├─ handler.py                    # action dispatch (lipsync, preprocess_full, ...)
│  ├─ storage.py                    # R2 client server-side
│  └─ Dockerfile                    # CUDA 11.8 + Python 3.10 + MuseTalk stack
├─ api/                             # MuseTalkInference engine
│  ├─ inference_service.py          # generate() — pipeline completa
│  └─ main.py                       # FastAPI server (uso opzionale locale)
├─ musetalk/                        # Modello upstream + patches
│  ├─ models/{vae,unet}.py          # con batch helpers per VAE
│  ├─ utils/{audio_processor,blending,preprocessing,face_parsing}.py
│  └─ ...
├─ models/                          # Pesi (~8.6 GB, scaricati con download_models.py)
├─ docs/
│  └─ ARCHITECTURE.md               # Documento di riferimento tecnico
└─ data/
   ├─ manifests/                    # Cache locale (auto-gestita)
   └─ generated/                    # Output video
```

---

## Modalità alternative

### API HTTP locale (legacy / dev)

Lo storico endpoint FastAPI è ancora funzionante per single-shot lipsync (senza cloud, senza random sampling):

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoint disponibili:
- `POST /generate` (multipart) — upload source+audio
- `POST /generate/json` — path-based per file già sul filesystem
- `GET /download/{filename}` — scarica il risultato
- `GET /health` — health check con GPU info

Non si scala per video lunghi (no caching) e non supporta il random sampling. Per workload reali usa il CLI client.

### Solo bbox cache (no latents)

Se non vuoi pagare il preprocess_full (~60-90s su 5 clip) e accetti un lipsync senza cache di latents/parsing:

```powershell
python -m scripts.client.synth_random_lipsync ... --skip-full-preprocess
```

Tempo atteso: ~30-40 min per 15 min audio (manca la cache di Phase D + F).

---

## Limiti noti

Non risolvibili senza cambiare modello (vedi [`docs/ARCHITECTURE.md` §7](docs/ARCHITECTURE.md#7-limiti-intrinseci)):

- **Forma labbra fonema-specifica**: MuseTalk usa training set generalista, certe consonanti occlusive (p/b/m) non sempre chiudono completamente.
- **Espressività emotiva**: MuseTalk genera solo lower-face, occhi/sopracciglia restano del frame originale.
- **Risoluzione ≥256×256**: il modello è locked a 256×256. Faccia molto piccola (<150px nel sorgente) viene upscalata e perde dettaglio. Usa `--quality` per attivare GFPGAN.
- **Jitter frame-to-frame**: inferenza per-frame indipendente, no temporal smoothing.

---

## Sviluppo: cosa è cambiato rispetto all'upstream

| Caratteristica | Upstream | Questo fork |
|---|---|---|
| Use case primario | CLI batch, Gradio demo | CLI client → RunPod Serverless |
| Random sampling clip | manuale | automatico via `--clips-dir` |
| Cache face detection | nessuna | per-frame su R2 (`v3` schema) |
| Cache VAE latents | nessuna | per-frame su R2 |
| Cache face parsing | nessuna | per-frame 512×512 su R2 |
| Encode finale | `libx264 fast crf 18` | `libx264 slow crf 16` o `h264_nvenc cq 18` con bitrate floor 3 Mbps |
| GFPGAN integration | full pipeline (con redetection) | `has_aligned=True` (~1.8× più veloce) |
| Frame extraction | imageio PNG su disco | `cv2.VideoCapture` in-memory |
| Batched VAE | no | sì (`get_latents_for_unet_batch`) |
| Batched FaceParsing | no | sì (`FaceParsing.batch_call`) |
| Drift audio/video | n/a (CLI single-clip) | 0 ms su 15+ min audio (v8) |

---

## Licenza

- **Codice**: MIT
- **Pesi MuseTalk**: vedere licenza upstream TMElyralab
- **GFPGAN, Whisper, ecc.**: licenze rispettive

---

## Citation

```bibtex
@article{musetalk,
  title={MuseTalk: Real-Time High-Fidelity Video Dubbing via Spatio-Temporal Sampling},
  author={Zhang, Yue and Zhong, Zhizhou and Liu, Minhao and Chen, Zhaokang and Wu, Bin and Zeng, Yubin and Zhan, Chao and He, Yingjie and Huang, Junxin and Zhou, Wenjiang},
  journal={arxiv},
  year={2025}
}
```
