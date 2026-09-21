# MuseTalk Pipeline — Architettura, Modello e Storia delle Versioni

> Documento di riferimento tecnico per il progetto `MuseTalk-API`. Spiega
> il modello sottostante (MuseTalk), l'obiettivo del progetto, l'architettura
> della pipeline (locale e su RunPod Serverless), l'evoluzione attraverso le
> versioni `v0`→`v7` e le motivazioni specifiche di ogni cambiamento.

---

## Indice

1. [Modello: MuseTalk](#1-modello-musetalk)
2. [Obiettivo e scope del progetto](#2-obiettivo-e-scope-del-progetto)
3. [Architettura della pipeline](#3-architettura-della-pipeline)
4. [Storia delle versioni](#4-storia-delle-versioni)
5. [Evoluzione del cache](#5-evoluzione-del-cache)
6. [Numeri di performance](#6-numeri-di-performance)
7. [Limiti intrinseci e cosa NON è risolvibile in questa pipeline](#7-limiti-intrinseci)
8. [Riferimenti](#8-riferimenti)

---

## 1. Modello: MuseTalk

### 1.1 Cos'è MuseTalk

**MuseTalk** è un modello di lipsync (sincronizzazione labiale audio-driven)
rilasciato da Tencent Music Entertainment — Lyra Lab nel 2024. Versione
attualmente integrata nel progetto: **MuseTalk V1.5** (pesi in
`models/musetalkV15/unet.pth`).

Genera **solo la zona bocca/mascella** del volto: gli occhi, le sopracciglia,
il naso, i capelli, lo sfondo restano del frame originale. La novità rispetto
a Wav2Lip e modelli precedenti è che lavora **nello spazio latente** di un
VAE pre-addestrato (Stable Diffusion `sd-vae-ft-mse`) invece che in pixel
space, ottenendo:

- Output a **256×256** con qualità nettamente superiore a Wav2Lip 96×96.
- Inferenza **real-time** (~30 fps su V100, ~100+ fps su H100 con batching).
- Nessun ri-addestramento per speaker specifico (zero-shot).

### 1.2 Architettura ad alto livello

```
                                      ┌──────────────────┐
                                      │   audio.wav      │
                                      └────────┬─────────┘
                                               │
                                               ▼
                                   ┌─────────────────────────┐
                                   │   Whisper-tiny encoder  │
                                   │   (./models/whisper)    │
                                   │   16 kHz → features     │
                                   │     [T, 10, 5, 384]     │
                                   └────────┬────────────────┘
                                            │ audio_padding_length_left/right=2
                                            ▼
                                ┌──────────────────────────────┐
                                │  PositionalEncoding (d=384)  │
┌─────────────────┐             └────────┬─────────────────────┘
│  video frame    │                      │ audio_feature_batch [B, 50, 384]
└────────┬────────┘                      │
         ▼                               │
┌─────────────────┐                      │
│ face detection  │                      │
│  s3fd + DWPose  │                      │
│  (bbox 240px²)  │                      │
└────────┬────────┘                      │
         │ crop + resize 256×256         │
         ▼                               │
┌─────────────────┐                      │
│  VAE encoder    │                      │
│ (sd-vae-ft-mse) │                      │
│ 256×256→32×32×4 │                      │
└────────┬────────┘                      │
         │                               │
         │  masked_latent  ref_latent    │
         │  (half-masked)  (full face)   │
         └─────────────┬─────────────────┘
                       │
                       ▼  [B, 8, 32, 32]
        ┌──────────────────────────────────────┐
        │   UNet (musetalkV15/unet.pth)        │
        │   cross-attention with audio_feature │
        │   timestep = 0 (deterministic)       │
        └────────┬─────────────────────────────┘
                 │ [B, 4, 32, 32]
                 ▼
        ┌──────────────────────────┐
        │   VAE decoder            │
        │   32×32×4 → 256×256×3    │
        └────────┬─────────────────┘
                 │
                 ▼  generated face 256×256
        ┌────────────────────────────────────────┐
        │  Blending:                             │
        │  1. resize generated to crop size      │
        │  2. paste over face_large region       │
        │  3. blend with FaceParsing mask        │
        │     (BiSeNet — jaw/raw/neck modes)     │
        │  4. (opzionale) GFPGAN enhancement     │
        └────────┬───────────────────────────────┘
                 ▼
         output video frame
```

### 1.3 Modelli concreti caricati

Tutti baked nell'immagine Docker, totale ~8.6 GB:

| Path | Modello | Ruolo |
|---|---|---|
| `./models/musetalkV15/unet.pth` | UNet 2D condizionato | Cuore del lipsync |
| `./models/sd-vae/` | VAE Stable Diffusion ft-mse | Encoder/decoder face latent |
| `./models/whisper/` | Whisper-tiny (HuggingFace) | Audio→feature 50fps |
| `./models/dwpose/dw-ll_ucoco_384.pth` | RTMPose-l Ubody 384 | Face landmarks (68pt) |
| `./models/face-parse-bisent/79999_iter.pth` | BiSeNet | Face parsing (mask blending) |
| `./models/face-parse-bisent/resnet18-5c106cde.pth` | ResNet-18 backbone | — |
| `s3fd-619a316812.pth` (`face_alignment` package) | S3FD | Face detection |
| `GFPGANv1.4.pth` (lazy, scaricato al primo `--enhance`) | GFPGAN v1.4 | Face restoration opzionale |

### 1.4 Caratteristiche numeriche del modello

- **Risoluzione di lavoro**: 256×256 (faccia croppata) → 32×32 nel latent space.
- **Frame rate di addestramento**: 25 fps. Whisper features sono a 50 fps;
  `whisper_idx_multiplier = audio_fps / fps = 2.0` quando si lavora a 25.
- **Contesto temporale audio per frame**: 2 frame passati + 2 frame futuri +
  frame corrente = **5 frame** × 50fps interna = 100 ms × 10 livelli Whisper
  hidden states × 384 dim = `[10, 5, 384]` per frame.
- **Indipendenza temporale del video**: ogni frame video è generato in modo
  **indipendente** dagli altri. Il UNet non ha contesto temporale sui frame
  passati: l'unica continuità viene dal `ref_latent` (frame corrente non
  mascherato come riferimento) e dal contesto audio (5 frame).
- **Timestep**: fissato a 0 (diffusione "one-step"). Non c'è un loop di
  denoising — è una singola forward pass.

### 1.5 Punti di forza e limiti intrinseci di MuseTalk

**Forze**:
- Qualità della zona bocca superiore a Wav2Lip.
- Velocità di inferenza alta (one-step, batch friendly).
- Zero-shot speaker generalization decente.
- Pipeline modulare (VAE, UNet, Whisper sono componenti separati).

**Limiti**:
- **Risoluzione fissa 256×256**: faccia molto grande nel sorgente (>500px)
  perde dettaglio nel resize a 256 e poi ri-upscale. Faccia molto piccola
  (<150px) viene upscalata interpolata.
- **Niente temporal consistency**: jitter frame-to-frame visibile, soprattutto
  in pause/silenzi.
- **Forma della bocca limitata al training set**: certi fonemi (consonanti
  occlusive labiali, "p/b/m") non sempre chiudono completamente le labbra.
  Non risolvibile senza ri-addestramento o swap modello.
- **Solo lower-face**: la mascella si muove, ma occhi/cigli/fronte restano
  bloccati al frame originale → manca espressività emotiva.
- **Dipende fortemente dalla qualità del face parsing**: bordi di blending
  visibili se la mask BiSeNet sbaglia.

Per fonemi specifici o espressività emotiva, l'unico path è cambiare modello
(SadTalker, DiffTalk, Wav2Lip-384, GeneFace++, etc.) o fine-tuning su un
dataset specifico — entrambi fuori scope di questo progetto.

---

## 2. Obiettivo e scope del progetto

### 2.1 Obiettivo

Generare un **video lipsync di lunga durata** (15+ minuti) per un avatar
parlante a partire da:

- Una **directory di clip brevi** del soggetto (5-30 clip da 5-10s)
- Una **traccia audio** lunga (la "lezione" che l'avatar deve "dire")

Il client:
1. **Campiona random con sostituzione** dalle clip per coprire l'intera durata
   dell'audio (es. 159 segmenti per coprire 15.7 min).
2. **Concatena** i segmenti in un singolo `video_completo.mp4`.
3. **Invia** video + audio al motore MuseTalk.
4. **Riceve** il video lipsync-ato e lo salva localmente.

### 2.2 Vincoli

| Vincolo | Valore |
|---|---|
| **Rapporto target** processing / audio_duration | **≤ 1:1** (15 min audio → ≤ 15 min processing) |
| **Qualità output target** | Professionale: niente artefatti di blending, niente jump temporali |
| **Sicurezza/privacy** | Le clip non devono restare a tempo indeterminato in cloud |
| **Costo cloud** | Tollerato H100 SXM Active worker permanente |
| **Hardware utente** | Windows 11 + RTX 4070 Laptop 8GB VRAM (sviluppo locale) |

### 2.3 Scope

**Incluso**:
- Setup locale Windows con Python 3.10 + venv.
- Migrazione del motore inferenza a RunPod Serverless (H100 80GB).
- Storage degli artefatti via Cloudflare R2 (S3-compatibile, no egress).
- Cache intelligente per evitare ricalcolo di bbox/latents/parsing sulle clip
  unica → 1:1 raggiunto da run #2 in poi.
- Ottimizzazioni per portare la pipeline da 3-4 ore a ~14 min.

**Escluso** (e perché):

| Out of scope | Motivo |
|---|---|
| Avatar pre-renderizzato singolo (no random sampling) | Già supportato upstream da `realtime_inference.py`; il valore del progetto è il sampling. |
| Multi-speaker | Caso d'uso single-avatar (lezioni universitarie). |
| Streaming live audio→video | Servirebbe latenza <500ms end-to-end; non realistico via RunPod. |
| Modello migliore di MuseTalk | Cambio di modello = ri-engineering completo. |
| Fine-tuning su speaker specifico | Richiederebbe dataset annotato del soggetto + GPU training (~24h H100). |
| UI grafica | Tooling CLI sufficiente per questa fase. |

---

## 3. Architettura della pipeline

### 3.1 Diagramma end-to-end (versione attuale: `v7`)

```
┌─────────────────── CLIENT (PC Windows) ──────────────────────────────────┐
│                                                                          │
│  scripts/client/synth_random_lipsync.py                                  │
│     1. probe_duration(audio) → target_minutes                            │
│     2. build_random_video(clips_dir, target_minutes, target_fps=25)      │
│        ├─ probe_duration + probe_fps per ogni clip                       │
│        ├─ sample_clips_until_duration (random con sostituzione)          │
│        ├─ concat_clips_reencode (ffmpeg filter_complex + scale+pad)      │
│        ├─ trim_to_exact_duration                                         │
│        └─ _build_frame_to_clip_map (per-frame mapping a 25 fps,          │
│           applicando source_fps/target_fps ratio per non eccedere        │
│           la dimensione del cache nativo)                                │
│                                                                          │
│     3. get_or_compute_full_manifest (clip_manifest.py)                   │
│        ├─ compute_full_set_hash (sha256 di path+size+mtime+params)       │
│        ├─ cache_load locale → JSON con R2 keys                           │
│        │  └─ se cache HIT + R2 ha gli oggetti → re-presign URLs          │
│        │  └─ se cache MISS → request_full_preprocess_runpod              │
│        │     ├─ upload clip → R2 cache/<hash>/clips/<name>               │
│        │     ├─ submit_job action=preprocess_full → bbox+latents+parsing │
│        │     ├─ scarica result, salva su cache locale + R2               │
│        │     └─ cleanup R2 clips                                         │
│        └─ return (bboxes_urls, latents_urls, parsing_urls, meta)         │
│                                                                          │
│     4. Upload video_completo.mp4 + audio.wav su R2                       │
│                                                                          │
│     5. submit_job action=lipsync con tutti gli URL + frame_to_clip[_idx] │
│                                                                          │
│     6. poll_job → ~12-14 min su H100 SXM 80GB (con cache hit)            │
│                                                                          │
│     7. r2_download(output_url) → salva su disco                          │
│                                                                          │
│     8. Cleanup R2 (input + output) salvo --keep-r2-objects               │
└──────────────────────────────────────────────────────────────────────────┘
                                    │ HTTPS / S3 API
                                    ▼
┌──────────────────── RUNPOD SERVERLESS H100 SXM 80GB ────────────────────┐
│                                                                          │
│  musetalk_runpod/handler.py                                              │
│     ENGINE = MuseTalkInference(use_float16=True, gpu_id=0)               │
│     ENGINE.load_models()  # eager, una volta per worker                  │
│                                                                          │
│     def handler(job):                                                    │
│        action = job["input"]["action"] (default: "lipsync")              │
│        ┌──────────────────────────────────────────────────────────┐      │
│        │ action == "preprocess_clips"                             │      │
│        │   Single PNG probe per clip → bbox (face_detection)      │      │
│        │   [usata solo nel path --skip-full-preprocess]           │      │
│        └──────────────────────────────────────────────────────────┘      │
│        ┌──────────────────────────────────────────────────────────┐      │
│        │ action == "preprocess_full"  [il path veloce in v7]      │      │
│        │   Per ogni clip unica:                                   │      │
│        │     - download mp4 da R2                                 │      │
│        │     - decode tutti i frame (cv2.VideoCapture)            │      │
│        │     - face detection PER OGNI FRAME (no static bbox)     │      │
│        │     - crop + resize 256x256 + VAE encode (batched 16)    │      │
│        │     - FaceParsing BiSeNet (batched 16) → parsing 512×512 │      │
│        │     - upload R2:                                         │      │
│        │       * <clip>_bboxes.json (per-frame + frame_shape)     │      │
│        │       * <clip>_latents.pt  (per-frame [N,8,32,32])       │      │
│        │       * <clip>_parsing.npz (per-frame [N,512,512])       │      │
│        └──────────────────────────────────────────────────────────┘      │
│        ┌──────────────────────────────────────────────────────────┐      │
│        │ action == "lipsync"                                      │      │
│        │   Se cache (bboxes_urls + latents_urls + parsing_urls):  │      │
│        │     - scarica i blob per ogni clip unica                 │      │
│        │     - per ogni frame assembled: lookup (clip, idx)       │      │
│        │       e ricostruisci blend mask al volo dal parsing      │      │
│        │       usando la bbox PER QUEL frame                      │      │
│        │     - chiama ENGINE.generate(precomputed_*)              │      │
│        │   La engine.generate(...):                               │      │
│        │     Phase A: cv2.VideoCapture in-memory frames           │      │
│        │     Phase B: whisper features                            │      │
│        │     Phase C: bbox (SKIP se precomputed)                  │      │
│        │     Phase D: VAE encode (SKIP se precomputed)            │      │
│        │     Phase E: UNet inference (batch=16)                   │      │
│        │     Phase F: maschere (precomputed_mask se cache,        │      │
│        │              altrimenti BiSeNet nel padre)               │      │
│        │     Phase F+G: fusione + encode, n_workers processi      │      │
│        │       (musetalk/utils/fusione_parallela.py)              │      │
│        │       n_workers=1 -> ciclo storico + un solo ffmpeg      │      │
│        │       n_workers>1 -> N segmenti contigui, un fork per    │      │
│        │         segmento: blend + ffmpeg pipe stdin in           │      │
│        │         streaming su seg_NNN.mp4, poi concat demuxer     │      │
│        │         -c copy (nessuna ricodifica)                     │      │
│        │       codec uniforme: h264_nvenc cq=18 +3M floor se      │      │
│        │         la sonda passa (max MUSETALK_NVENC_SESSIONS      │      │
│        │         segmenti), altrimenti libx264 crf=16             │      │
│        │       parallelo fallito -> ripiego sul sequenziale       │      │
│        │     mux audio AAC                                        │      │
│        └──────────────────────────────────────────────────────────┘      │
│                                                                          │
│   Container image: micheledibi/musetalk-runpod:v7                        │
│   GPU types allowed: H100 SXM, H100 PCIe, L40S (no Blackwell sm_120,     │
│     no A100 — A100 manca NVENC hardware ma è OK come fallback compute    │
│     se H100 esauriti; il fallback libx264 copre l'assenza NVENC).        │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 File principali e responsabilità

| File | Cosa fa |
|---|---|
| `scripts/client/synth_random_lipsync.py` | Entry-point CLI. Orchestrazione end-to-end. |
| `scripts/client/video_assembler.py` | ffmpeg probe/sample/concat/trim + `_build_frame_to_clip_map`. |
| `scripts/client/clip_manifest.py` | Cache hash + `get_or_compute_full_manifest`. |
| `scripts/client/runpod_client.py` | R2 upload/download + RunPod submit/poll. |
| `musetalk_runpod/handler.py` | Handler RunPod: action dispatch + per-frame cache expansion. |
| `musetalk_runpod/storage.py` | R2 client-side dentro al container. |
| `musetalk_runpod/Dockerfile` | Base CUDA 11.8 + Python 3.10 + tutta la stack MuseTalk. |
| `api/inference_service.py` | `MuseTalkInference.generate()` — engine vero e proprio. |
| `musetalk/utils/preprocessing.py` | `get_landmark_and_bbox` (s3fd + DWPose). |
| `musetalk/utils/audio_processor.py` | Whisper feature extraction + chunking per frame. |
| `musetalk/utils/blending.py` | `get_image` (blend con FaceParsing mask). |
| `musetalk/utils/fusione_parallela.py` | Fasi F e G: percorso sequenziale storico e percorso a segmenti (fork + concat `-c copy`). Senza torch. |
| `musetalk/utils/face_parsing/__init__.py` | BiSeNet wrapper con `batch_call`. |
| `musetalk/models/vae.py` | VAE wrapper con `preprocess_img_batch` + `get_latents_for_unet_batch`. |
| `musetalk/models/unet.py` | UNet wrapper. |

### 3.3 Endpoint RunPod — configurazione

| Parametro | Valore | Motivazione |
|---|---|---|
| **Image** | `micheledibi/musetalk-runpod:v7` | Versione attuale (vedi §4) |
| **GPU types** | H100 SXM, H100 PCIe, L40S | sm_80–sm_90 compatibili PyTorch 2.0.1+cu118 |
| **GPU exclude** | RTX PRO 6000 Blackwell (sm_120) | non supportato dalla nostra build PyTorch |
| **Active workers** | 1 | warm start, no cold start latency |
| **Max workers** | 3 | tetto per picchi concorrenti |
| **Idle timeout** | 60s | bilanciamento costo/latency |
| **Execution timeout** | 3 600 000 ms = 60 min | margine 4× sul tempo medio (15 min) |
| **Scaler** | REQUEST_COUNT, value=1 | 1 job per worker |
| **Flashboot** | true | rapido restart in-place |
| **Env vars** | R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY | tutto via env, niente secrets in image |

---

## 4. Storia delle versioni

> **Notazione**: `v0` indica il setup locale pre-Docker. Da `v1` in poi
> sono tag di immagine Docker pushati su Docker Hub
> (`micheledibi/musetalk-runpod:vN`).

### 4.1 v0 — Setup locale (Piano #1)

**Trigger**: clone iniziale del repo `MuseTalk-API`.

**Cosa è stato fatto**:
- Installato Python 3.10 (winget) e ffmpeg.
- Creato venv `.venv`.
- PyTorch 2.0.1+cu118, mmpose 1.1.0 (`--no-build-isolation` per chumpy),
  diffusers 0.30.2, transformers 4.39.2, ecc.
- `python download_models.py` (~8.6 GB).
- Avviato `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
- Implementato `scripts/client/`: `video_assembler.py`,
  `synth_random_lipsync.py`, `musetalk_client.py` (REMOVED in v2).

**Performance**: 8.7× real-time su RTX 4070 Laptop 8GB.
- Per 15.73 min di audio → **137 min** di processing.

**Motivazione del passaggio successivo**: rapporto 8.7:1 inservibile per
qualsiasi workload reale; serve hardware più grosso (H100) ed è ovvio che
girare in cloud è più conveniente che comprare l'hardware.

---

### 4.2 v1–v4 — Migrazione RunPod Serverless (Piano #2, iterazioni di
debugging)

**Trigger**: dopo il benchmark locale 137 min, decisione di migrare a RunPod
Serverless per H100 80GB.

**Cosa è stato fatto in totale tra v1 e v4** (iterazioni di tipo "build →
push → test → bug → fix"):
- Creato `musetalk_runpod/` con:
  - `handler.py`: module-scope `ENGINE = MuseTalkInference(use_float16=True)`,
    eager `load_models()`.
  - `storage.py`: helper boto3 per R2.
  - `Dockerfile`: base `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` con
    Python 3.10 + tutta la stack pip, modelli COPY in `/app/models/`.
- Rimosso `scripts/client/musetalk_client.py` (path localhost non più usato).
- Aggiunto `scripts/client/runpod_client.py`: R2 upload/download +
  `submit_job` + `poll_job` con backoff.
- `scripts/client/synth_random_lipsync.py` riscritto per usare
  RunPod invece di localhost.
- Variabili d'ambiente caricate automaticamente da `runpod.env`/`.env`.

**Bug risolti tra v1 e v4 (tutti durante test reali)**:
- Path Dockerfile sbagliato (`docker build -t ...` senza `-f
  musetalk_runpod/Dockerfile` aveva preso il Dockerfile root con uvicorn
  entrypoint).
- `R2_BUCKET` impostato al valore del token invece del nome bucket.
- Git Bash che mangiava i backslash dei path Windows.
- ffmpeg subprocess interactive mode → aggiunto `stdin=subprocess.DEVNULL`.
- `executionTimeoutMs` di default 30 min → portato a 60 min.

**Performance dopo v4**: il job FAILED al timeout di 30 min senza completare.
La pipeline naive (face detection su tutti i 23 599 frame, single-frame VAE
encode, single-frame FaceParsing, encode PNG su disco) impiegava oltre 3 ore
su H100.

**Motivazione del passaggio successivo**: serve un'ottimizzazione massiccia
per arrivare a 1:1. Il piano #3 nasce qui.

---

### 4.3 v5 — Piano #3: 7 ottimizzazioni in un solo rebuild

**Trigger**: pipeline inutilizzabile a 3+ ore su 15 min audio.

**Analisi del bottleneck** (profiling sui log dei job v4):

| Fase | Tempo stimato | Causa |
|---|---|---|
| Face detection | 30-90 min | s3fd + DWPose single-frame su 23 599 frame |
| Frame extract | 4-6 min | `imageio` legge + scrive PNG |
| Whisper | 1.5-2 min | OK, no problema |
| VAE encoding | 70-100 min | `vae.encode` 2× per frame, single-batch |
| UNet inference | 14-21 min | batch=8 lascia VRAM H100 sotto-utilizzata |
| Blending | 70-100 min | BiSeNet single-frame + cv2.imwrite PNG |
| Encode finale | 8-17 min | libx264 software |

**Le 7 ottimizzazioni implementate tutte insieme in `v5`**:

1. **Batched VAE encoding**: nuovo `vae.get_latents_for_unet_batch(img_list)`
   in `musetalk/models/vae.py` che fa 2 forward del VAE (masked + ref) su
   batch di 16 invece di 2 forward per ogni singolo frame. **Speedup ~10×**
   sulla fase VAE.

2. **Cache per-clip dei latents**: nuovo cache R2 `cache/<hash>/<clip>_latents.pt`.
   Sul secondo run (stesse clip) la fase VAE viene saltata interamente
   (idx-lookup di tensor pre-computati).

3. **Batch UNet inference**: `--batch-size 16` di default (era 8). H100 80GB
   ha VRAM più che sufficiente. **Speedup ~2×** sulla fase UNet.

4. **Batched FaceParsing**: nuovo `FaceParsing.batch_call(images, mode)` in
   `musetalk/utils/face_parsing/__init__.py` che fa una singola forward
   BiSeNet su batch invece di 23 599 forward single-image. **Speedup ~6-8×**.

5. **Cache per-clip delle blend masks**: `cache/<hash>/<clip>_masks.npz`.
   Stessa idea dei latents.

6. **Skip PNG su disco**: nuovo `_write_video_pipe()` che fa
   `subprocess.Popen` di ffmpeg con stdin pipe BGR raw. Niente più 23 599
   PNG temporanei scritti + riletti.

7. **h264_nvenc encode finale** con fallback automatico a libx264 se NVENC
   non disponibile sulla GPU (H100 non ha NVENC hardware: scoperto runtime,
   fallback funziona).

**Refactor cross-cutting `api/inference_service.py:generate()`**:
- Aggiunti parametri `precomputed_bboxes`, `precomputed_latents`,
  `precomputed_masks`, `use_nvenc`, `vae_batch_size`, `parsing_batch_size`.
- Sostituito `imageio.get_reader` → `cv2.VideoCapture` in-memory.
- Sostituiti i 4 loop frame-by-frame con varianti batched.

**Refactor `musetalk_runpod/handler.py`**:
- Nuova action `preprocess_full` che fa tutto il preprocessing pesante
  (decode, face detection, VAE batched, parsing batched) e carica i blob
  su R2.
- Action `lipsync` esistente accetta i nuovi parametri `latents_urls`,
  `masks_urls`, `frame_to_clip`, `frame_to_clip_idx`.

**Refactor `scripts/client/clip_manifest.py`**:
- Nuova funzione `compute_full_set_hash` che include `(clips, bbox_shift,
  extra_margin, parsing_mode, left/right_cheek_width)` per invalidare il
  cache quando uno di questi parametri cambia.
- Nuova `get_or_compute_full_manifest` che gestisce la cache lookup +
  invocazione `preprocess_full` se cache miss + re-presign URLs se cache hit.

**Cache key `v2`**: marcatore nel hash per invalidare i cache delle
versioni precedenti che usavano solo le bbox precomputate.

**Performance dopo v5**:

| Fase | Tempo |
|---|---|
| preprocess_clips (5 clip) | 11 s |
| preprocess_full (5 clip × 142 frame) | 72 s |
| download R2 + cache expand | 9 s |
| Phase A frame extract | 25 s |
| Phase B whisper | 12 s |
| Phase C bbox | 0 s (cache hit) |
| Phase D VAE encode | SKIPPED (cache hit) |
| Phase E UNet inference | 183 s |
| Phase F blending | 395 s |
| Phase G video encode (libx264 fallback) | 201 s |
| Phase G audio mux | 3 s |
| **Totale engine.generate** | **820 s = 13.7 min** |

**Rapporto raggiunto**: **0.87 : 1** (sub-1:1) su 15.73 min di audio.
**Obiettivo 1:1 raggiunto.** ✅

**Motivazione del passaggio successivo**: l'utente, ispezionando il video,
ha notato che "il jaw si muove indipendente dalla testa". Diagnosi: la
bbox era statica per clip (solo frame 0). Quando la testa si muove nel
corso della clip (cosa che succede in clip reali), il volto generato
appare in posizione "vecchia". Da qui il bisogno della v6.

---

### 4.4 v6 — Per-frame bbox tracking (fix qualità head motion)

**Trigger**: feedback utente "il volto generato non segue il movimento
della testa".

**Diagnosi precisa**:
- In v5, `preprocess_full` faceva face detection **solo sul primo frame**
  di ogni clip.
- Lo stesso bbox veniva poi usato per croppare TUTTI i frame della clip e
  per costruire la blend mask di TUTTI i frame della clip.
- Risultato: se la testa scendeva durante la clip, la bocca generata
  appariva nella posizione del frame 0 (in alto), mentre la testa reale
  era già scesa. Il "jaw che parte prima della testa".

**Cosa è cambiato in `v6`**:

1. **`_handle_preprocess_full` rifattorizzato**: chiama
   `get_landmark_and_bbox(frames=frames)` per fare detection
   **frame-by-frame** (s3fd + DWPose). Costo aggiuntivo una tantum:
   ~60-80s sui 5 clip da 142 frame.

2. **Schema cache aggiornato v2 → v3**:
   - Vecchio (v2): `<clip>_latents.pt` (per-frame) + `<clip>_masks.npz`
     (per-frame, ma costruite tutte con la stessa bbox).
   - Nuovo (v3): `<clip>_bboxes.json` (per-frame, formato
     `{frame_shape: [H,W], bboxes: [[x1,y1,x2,y2]...]}`) +
     `<clip>_latents.pt` (per-frame, ora effettivamente per-frame perché
     ogni latent usa la sua bbox individuale) + `<clip>_parsing.npz`
     (per-frame, 512×512 raw BiSeNet output — la blend mask vera viene
     ricostruita a lipsync time).

3. **`_expand_cached_per_clip` rifattorizzato nel handler**:
   - Scarica i 3 blob (bboxes, latents, parsing) per clip.
   - Per ogni frame del video assembled, lookup (clip_name, idx).
   - Ricostruisce la blend mask al volo da `parsing[idx]` usando la
     `bbox[idx]` (cambiata frame-by-frame) → produce `precomputed_masks`
     allineate al movimento della testa.

4. **`compute_full_set_hash` cache key bumpata a `v3`**: invalida
   automaticamente i cache v5 (formato v2).

**Performance dopo v6**:
- preprocess_full: ~140s (+70s rispetto a v5 per la face detection
  per-frame).
- Phase F: leggermente più alta (rebuild blend mask al volo) ma compensata
  dal fatto che il blob parsing.npz è più piccolo e veloce da scaricare.
- Totale stimato: ~14-15 min (vs ~14 min di v5), perfettamente nel target.

**Cosa NON è stato fixato in v6**:
- Il sintomo iniziale ("jaw che parte prima della testa") è risolto, ma
  l'utente nota un **altro** problema: il video sembra ancora poco
  professionale. Sintomi sospettati: micro-glitch periodici, bassa
  definizione della bocca. → trigger per v7.

---

### 4.5 v7 — Fix bug fps mismatch + encoding quality + GFPGAN preset

**Trigger**: `test_3.mp4` (output di v6) descritto come "inguardabile e poco
professionale" malgrado v6 risolva il problema della bbox statica.

**Diagnosi attraverso ispezione frame-by-frame + analisi codice**:

#### Bug critico — fps mismatch source(24) vs concat(25)

Le clip sorgente del soggetto `michele_monaco/` sono a **24 fps**:
```
clip_1.mp4: fps=24.000 duration=5.917s n_native=142 frame
```

Ma la pipeline assembla a **25 fps** (default di `--fps`), perché:
- MuseTalk è addestrato a 25 fps.
- L'audio_processor crasha su fps che non sono multipli di 25 o 50.

Conseguenza: ogni segmento di clip nella concat occupa
`int(round(5.917 × 25)) = 148 frame`, ma il cache contiene solo 142 latents
(perché preprocess_full decodifica le clip a 24 fps native).

In `scripts/client/video_assembler.py:_build_frame_to_clip_map`:
```python
idx_per_frame[f] = f - start_f   # bug: idx va da 0 a 147
```

In `musetalk_runpod/handler.py:_expand_cached_per_clip`:
```python
lat_idx = int(idx) % n_lat       # bug: 147 % 142 = 5, wrap-around
```

**Risultato visivo**: ogni ~5.9 secondi (a ogni boundary di segmento clip,
quindi ~159 volte in un video da 15.7 min), gli ultimi 6-7 frame del
segmento riusano latents/parsing/bboxes dei PRIMI 6-7 frame della stessa
clip → il volto generato fa un "rewind" di 0.25s.

**Fix in `v7`**:

1. **Nuova `probe_fps()` in `video_assembler.py`** che usa ffprobe per
   leggere `r_frame_rate` di ogni clip sorgente.

2. **`_build_frame_to_clip_map` rifattorizzato**: usa
   `source_fps / target_fps` ratio per produrre idx che restano
   nell'intervallo [0, n_native−1]:
   ```python
   scale = source_fps_map[clip] / target_fps        # 24/25 = 0.96
   n_native = int(round(durations[clip] * source_fps_map[clip]))
   for f in range(start_f, end_f):
       local_target = f - start_f                   # 0..147
       idx_per_frame[f] = min(int(local_target * scale), n_native - 1)
                                                    # 0..141 ✓
   ```
   Questa logica replica esattamente ciò che fa il filtro ffmpeg `fps=25`
   internamente (nearest-neighbor frame duplication) quando l'input è a
   24 fps.

3. **Clamp difensivo nel handler**: sostituito `% n_lat` con
   `min(idx, n_lat - 1)`. Se mai un client più vecchio (pre-v7) inviasse
   un idx out-of-range, invece di wrappare al frame 0 ora duplichiamo
   l'ultimo frame valido — degradazione graceful invece di jump.

4. **Nessuna modifica al formato del cache**: i blob R2 esistenti restano
   validi. La cache `v3` continua a funzionare; cambia solo l'indicizzazione
   client-side.

#### Bug encoding bitrate

In `test_3.mp4`: bitrate medio **455 kbps** su 768×768 @ 25fps. Molto basso,
specie sulla zona bocca (la più importante visivamente).

**Fix in `v7`** (`api/inference_service.py:_write_video_pipe`):

```python
# h264_nvenc:
"-preset", "p5",       # era p4
"-cq", "18",           # era 20 (più alto = più compressione)
"-b:v", "3M",          # NUOVO: bitrate target
"-maxrate", "5M",      # NUOVO: tetto
"-bufsize", "10M",     # NUOVO: VBR rate control buffer

# libx264 fallback:
"-preset", "slow",     # era fast
"-crf", "16",          # era 18
```

Aspettativa: bitrate output ~2.5-3 Mbps (5-7× test_3). Costo: +30-60s su
fase G (encoding software libx264 più lento con preset slow).

#### Quality preset CLI

Aggiunto flag `--quality` in `scripts/client/synth_random_lipsync.py` che
attiva automaticamente `--enhance` (GFPGAN). Default `--gfpgan-weight`
bumpato da 0.5 a 0.6 (più dettagli pelle/labbra, ancora naturale).

Costo `--quality`: ~+5-7 min su 15 min audio. Totale stimato ~20-21 min
(rapporto ~1.3:1 — fuori dall'obiettivo 1:1 ma giustificato dal salto
qualitativo se serve).

**Cosa NON è stato fatto in v7** (intenzionalmente):
- **Audio padding window** (`audio_padding_length_left/right`): l'utente ha
  scelto di non sperimentare ora. Resta hardcoded a `(2,2)` come da default
  upstream.
- **Smoothing temporale latents**: richiederebbe modifica dell'inference
  loop. Escluso per non aggiungere variabili.
- **Modello migliore di MuseTalk**: cambio scope, escluso.

**Performance attesa dopo v7**:

| Modalità | Tempo | Rapporto |
|---|---|---|
| Standard (no `--quality`) | ~14-15 min | 0.95:1 |
| `--quality` (GFPGAN on) | ~20-21 min | 1.3:1 |

**Cosa NON era ancora fixato in v7**: il drift residuo nella mappa
`assembled_frame → (clip, idx)` non era stato preso, perché v7 corregge solo
l'idx **dentro** alla clip (`source_fps/target_fps` scaling), ma il calcolo
dei **boundary tra clip** continuava a usare `cumulative_s × target_fps`
floating-point. Su 160 clip questo errore di rounding si accumulava → trigger
per v8.

---

### 4.6 v8 — Fix drift cumulativo (client-only, immagine resta v7)

**Trigger**: l'utente nota che, anche dopo v7, "la mascella si muove prima,
e più passano i minuti più si vede la differenza" — drift **lineare crescente**
nel tempo, non a salti come prima.

**Diagnosi quantitativa sul file v7 `test_4.mp4`**:
- Video: 768×768 @ 25 fps, 23 598 frame, 943.920 s — coerente
- Audio sorgente: 24 kHz, 943.957 s, librosa resample a 16 kHz **corretto**
- num_frames audio_processor = floor(943.957 × 25) = 23 598 → matcha video
- ffmpeg con `fps=25` filter sulla clip 24fps produce **esattamente 148
  frame** (verificato empiricamente con `ffmpeg -filter:v fps=25
  -c:v rawvideo`)

Il bug è **nella formula di `start_f` per ogni clip occurrence**, in
`scripts/client/video_assembler.py:_build_frame_to_clip_map`:

```python
# v7 (buggy):
cumulative_s = 0.0
for clip in sampled:
    start_f = int(round(cumulative_s * target_fps))   # ← drift accumulativo!
    cumulative_s += durations[clip]                   # 5.916667 per clip
    end_f = int(round(cumulative_s * target_fps))
```

Per clip N, `start_f = round(N × 5.917 × 25) = round(N × 147.9167)`. Il
residuo di rounding (~0.083 frame per clip) si accumula:

| N (clip occurrence) | v7 calcola `start_f` | ffmpeg actual (N × 148) | Drift cumulativo |
|---|---|---|---|
| 10 | round(1479.17) = 1479 | 1480 | 1 frame (40 ms) |
| 50 | round(7395.83) = 7396 | 7400 | 4 frame (160 ms) |
| 100 | round(14791.67) = 14792 | 14800 | 8 frame (320 ms) |
| 159 (ultimo) | round(23518.75) = 23519 | 23532 | **13 frame (520 ms)** |

A frame 23 519, il client v7 dice "inizia clip 159 a idx 0". Il contenuto
reale del video lì è ancora **clip 158** a posizione locale 135. Bbox/latents/
parsing scaricati = **WRONG CLIP** → la mascella generata viene incollata
con timing e posizione di un altro clip → "mascella in anticipo" che cresce
linearmente.

**Fix in `v8`** (puramente lato client):

1. **Tracking in frame interi invece di secondi**. `_build_frame_to_clip_map`
   passa da `cumulative_s` (float arithmetic con rounding error per ogni
   clip) a `cumulative_frames` (intero esatto):

   ```python
   start_f = 0
   for clip in sampled:
       # Deve matchare `fps={target_fps}:round=up` in concat_clips_reencode
       clip_frames = math.ceil(durations[clip] * target_fps)
       end_f = min(start_f + clip_frames, total_frames)
       source_fps = source_fps_map[clip]
       n_native = max(1, int(round(durations[clip] * source_fps)))
       scale = source_fps / target_fps
       for f in range(start_f, end_f):
           clip_per_frame[f] = clip
           local = f - start_f
           idx_per_frame[f] = min(int(local * scale), n_native - 1)
       start_f = end_f
   ```

2. **`fps={target}:round=up` esplicito nel filtro ffmpeg**
   (`concat_clips_reencode`). Garantisce che ffmpeg produca esattamente
   `ceil(duration × target_fps)` frame per clip, in ogni run, su ogni
   versione di ffmpeg. Il calcolo `start_f` lato client adesso combacia
   per costruzione.

**Verifica logica del fix** (smoke test su seed=42, stesso scenario di
test_4):

| Metrica | v7 (buggy) | v8 (fix) |
|---|---|---|
| Drift a fine video | 520 ms | **0 ms** ✓ |
| Boundary clip 159 calcolato | 23519 (sbagliato) | 23532 (= 159×148) ✓ |
| Max idx per ogni clip | 141 (= n_native-1) | 141 (= n_native-1) ✓ |

**Cosa NON cambia in v8**:
- **Docker image RunPod**: resta v7. Il clamp difensivo `min(idx, n_lat-1)`
  inserito nel handler v7 è già sufficiente per gestire i nuovi idx.
- **Cache R2 v3**: resta valida. Il cache contiene per-frame nativo della
  clip (142 frame @ 24fps native), indipendente dal target_fps.
- **Runtime**: identico a v7. La fix sono microsecondi di CPU client-side
  (pure Python integer arithmetic).
- **Tutto il resto della pipeline**: invariato.

**File modificato**:
- `scripts/client/video_assembler.py`:
  - `concat_clips_reencode` → aggiunto `:round=up` (1 riga)
  - `_build_frame_to_clip_map` → riscritta con cumulative_frames (~50 righe)

**Pubblicazione**: nessun docker build, nessun push, nessun redeploy endpoint.
Lato utente, basta `git pull` e rilanciare il run.

**Performance attesa dopo v8**:

| Modalità | Tempo | Rapporto | Drift a fine video |
|---|---|---|---|
| Standard (no `--quality`) | ~14-15 min | 0.95:1 | **0 ms** ✓ |
| `--quality` (GFPGAN on) | ~20-21 min | 1.3:1 | **0 ms** ✓ |

---

## 5. Evoluzione del cache

### 5.1 Cache key version `v2` (in `v5`)

Hash key:
```
v2|bbox_shift=N|extra_margin=N|parsing_mode=X|left_cw=N|right_cw=N
+ per ogni clip: <path>\t<size>\t<mtime>
```

R2 layout:
```
cache/<full_set_hash>/
  ├─ clips/<clip_name>            (upload temporaneo, eliminato dopo
  │                                  preprocess)
  ├─ <clip>_latents.pt            (per-frame latents [N,8,32,32])
  └─ <clip>_masks.npz             (per-frame blend masks, MA costruite con
                                   bbox STATICA del primo frame)
```

Locale: `data/manifests/<full_set_hash>.json` con `clip_bboxes` (singolo bbox
per clip), `full_payload` (R2 keys), parametri preprocess.

### 5.2 Cache key version `v3` (da `v6` in poi, attuale)

Hash key:
```
v3|bbox_shift=N|extra_margin=N|parsing_mode=X|left_cw=N|right_cw=N
+ per ogni clip: <path>\t<size>\t<mtime>
```

R2 layout:
```
cache/<full_set_hash>/
  ├─ clips/<clip_name>            (upload temporaneo)
  ├─ <clip>_bboxes.json           ({"frame_shape": [H,W],
                                    "bboxes": [[x1,y1,x2,y2]...]})
  ├─ <clip>_latents.pt            (per-frame latents, ora coerenti con
                                   bbox per-frame)
  └─ <clip>_parsing.npz           (per-frame BiSeNet raw 512×512;
                                   la blend mask viene costruita al volo a
                                   lipsync time usando la bbox per-frame)
```

Locale: `data/manifests/<full_set_hash>.json` con `full_payload` (R2 keys
+ n_frames + frame_shape per clip).

### 5.3 Invalidazione automatica

Tutti i parametri che influenzano il preprocessing fanno parte del hash:

| Parametro | Effetto se cambia |
|---|---|
| Path/size/mtime delle clip | Hash diverso → fresh preprocess |
| `bbox_shift` | Bbox cambiate → fresh preprocess |
| `extra_margin` | Crop region diversa → fresh preprocess |
| `parsing_mode` (`jaw`/`raw`/`neck`) | BiSeNet output diverso → fresh preprocess |
| `left_cheek_width` / `right_cheek_width` | Solo dentro a BiSeNet → fresh preprocess |

Parametri che **NON** invalidano il cache:
- `batch_size` (solo inferenza UNet)
- `enhance` / `gfpgan_weight` (post-blending, locale a inference)
- `fps` target (l'indicizzazione del cache si adatta via
  `_build_frame_to_clip_map`)
- Audio file (cambia solo lo step Whisper, lipsync time)

---

## 6. Numeri di performance

### 6.1 Tabella riassuntiva per versione (15.73 min audio = target 15.73 min)

| Versione | Hardware | Cache | Totale | Rapporto | Drift fine video | Note |
|---|---|---|---|---|---|---|
| v0 | RTX 4070 8GB local | nessuno | **137 min** | **8.7:1** | — | Inservibile |
| v1-v4 | H100 SXM RunPod | nessuno | **>180 min (timeout)** | **>11.5:1** | — | FAILED |
| v5 (cache hit) | H100 SXM RunPod | latents + masks | **13.7 min** | **0.87:1** | n/a (bbox statica → drift spaziale) | Primo 1:1 |
| v5 (cache miss) | H100 SXM RunPod | freshly built | ~15-16 min | ~1:1 | n/a | One-time cost |
| v6 (cache hit) | H100 SXM RunPod | bboxes + latents + parsing | ~14-15 min | ~0.95:1 | ~250 ms jump ogni 5.9s | Fix bbox spaziale, bug fps mascherato |
| v6 (cache miss) | H100 SXM RunPod | freshly built | ~16-17 min | ~1.05:1 | ~250 ms jump ogni 5.9s | +face detection per-frame |
| v7 (cache hit) | H100 SXM RunPod | v3 cache (idem v6) | ~14-15 min | ~0.95:1 | **520 ms cumulativo** | Niente jump periodici ma drift residuo |
| v7 + `--quality` | H100 SXM RunPod | v3 cache + GFPGAN | ~20-21 min | ~1.3:1 | **520 ms cumulativo** | Drift cumulativo ancora visibile |
| **v8 (cache hit)** | H100 SXM RunPod (v7) | v3 cache | ~14-15 min | **~0.95:1** | **0 ms** ✓ | **Target finale** |
| **v8 + `--quality`** | H100 SXM RunPod (v7) | v3 cache + GFPGAN | ~20-21 min | ~1.3:1 | **0 ms** ✓ | Qualità max + zero drift |

L'obiettivo del progetto è **v8 standard**: 1:1 rispettato (~14-15 min su
15.73 min audio) **+ zero drift cumulativo**. Il preset `--quality` resta
disponibile per chi privilegia nitidezza del volto sul tempo.

### 6.2 Phase-by-phase su cache hit (v6+, valori medi)

Profilo **misurato** del percorso sequenziale (`n_workers=1`), che resta
disponibile come riferimento:

| Fase | Tempo | Note |
|---|---|---|
| Download cache blobs (5 clip) | ~5-10 s | dipende dalla connessione R2→worker |
| Expand cache per-frame | ~1-3 s | torch tensors → list |
| **Phase A** frame extract | 25 s | cv2.VideoCapture in-memory di 23 599 frame 768×768 |
| **Phase B** whisper | 12 s | encoder Whisper-tiny su 16 kHz × 15 min |
| **Phase C** bbox | 0 s | skip totale grazie a `precomputed_bboxes` |
| **Phase D** VAE encode | 0-1 s | skip totale grazie a `precomputed_latents` |
| **Phase E** UNet inference | 180-200 s | batch=16, 23 599 forward su H100 fp16 |
| **Phase F** blending | 390-400 s | mask resize + paste + cv2 ops per 23 599 frame |
| **Phase G** video encode | 200 s | libx264 (NVENC non disponibile su H100) |
| **Phase G** audio mux | 3 s | AAC encode |
| **Totale engine.generate** | ~830-850 s | ≈ 14 min |

#### 6.2.1 Fasi F e G parallele

Il 70% del tempo qui sopra è in F+G, che usavano un solo core. Dal modulo
`musetalk/utils/fusione_parallela.py` (senza torch, testabile senza GPU):

- I frame vengono divisi in **N segmenti contigui** (`n_seg = min(n_workers,
  n_frame // 250)`, lunghezze bilanciate, mai sotto 3 frame). Ogni segmento è
  affidato a un processo creato con `fork` che fonde i propri frame e li invia
  **in streaming** al proprio ffmpeg (`seg_NNN.mp4`): il buffer
  `combine_frames` (~42 GB) nel percorso parallelo non esiste.
- Il padre verifica che i segmenti siano **uniformi** (codec, profilo, livello,
  geometria, `time_base`, `has_b_frames`, hash dell'avcC, numero di frame),
  li concatena con il concat demuxer in `-c copy` e controlla pacchetti e durata
  del risultato. Il mux dell'audio è invariato. Nella lista ogni riga `file` è
  seguita dalla `duration` esatta al microsecondo: senza, ffmpeg ≤ 5.1 (nelle
  immagini c'è la 4.4) userebbe la durata del contenitore mp4, arrotondata al
  millisecondo, e a 24/30/60 fps ogni confine slitterebbe di una frazione di
  millisecondo. Con ffmpeg ≥ 6 il file prodotto è identico al byte.
- **Perché fork e non spawn + shared_memory**: con spawn il figlio ri-esegue
  `__main__` (in `handler.py` i modelli si caricano a livello di modulo) e i
  frame andrebbero copiati in `/dev/shm`, non dimensionabile su RunPod
  Serverless. Con fork gli argomenti sono ereditati, mai serializzati, e i
  buffer numpy si leggono in copy-on-write. I figli non toccano torch/CUDA, non
  stampano, ed escono con `os._exit`.
- **GFPGAN** (`enhance=True`) resta sulla GPU del padre: nel percorso parallelo
  è un passaggio preliminare sequenziale, prima dei fork, che produce una lista
  nuova di facce; da lì in poi nessuno lo richiama (nemmeno il ripiego).
- **Senza cache** BiSeNet resta nel padre come prima; i ritagli PIL si
  costruiscono a lotti e su cache hit non si costruiscono più (erano lavoro
  inutile: ~14 GB e una parte seriale rilevante della fase F).
- **NVENC**: una sonda nel padre apre `MUSETALK_NVENC_SESSIONS` encode minuscoli
  in contemporanea. Se passa, i segmenti sono al più quante le sessioni (il
  blending è quindi parallelo solo S volte: con NVENC disponibile
  `use_nvenc=False` può risultare più veloce). Se un segmento nvenc fallisce si
  rifà **tutto** in libx264: il codec è sempre uniforme dentro un video, perché
  ffmpeg concatena in silenzio anche segmenti difformi.
- **Fallimento** del percorso parallelo (fork, worker morto, stallo, segmenti
  difformi, conteggio frame): riga `[phase F] PARALLELO FALLITO (...)` e ripiego
  sul percorso sequenziale, così il job termina comunque.
- Il bitstream mp4 **non** è identico al bit fra `n_workers=1` e `n_workers=N`
  (GOP e rate control ripartono a ogni segmento). La parità è definita sui frame
  in uscita dalla fase F (sha256 identici) e su numero di frame, fps e durata.

Log: le righe `[phase F] blending: N frames in X.XXs` e `[phase G] video
encode: X.XXs` restano nel formato di prima. In parallelo la prima misura
fusione+codifica sovrapposte (passaggio GFPGAN e sonda NVENC compresi), la
seconda verifica dei segmenti e concat. Si aggiunge sempre una riga:

```
[phase F+G] modo=parallelo n_workers=16 segmenti=16 codec=libx264 x264_threads=2
  fork_ms=min/mediana/max tentativi=1 cpu(slurm=..,affinity=..,cgroup=..,os=..)->N
  motivo='auto' prepass_gfpgan=0.00s uniformita=OK rss_fork=52.3GB
  fusione_cpu=410.2s seg_max=38.1s
```

(su una riga sola; nel percorso sequenziale si ferma a `motivo=`, che dice
perché non si è andati in parallelo). `cpu(...)->N` sono i valori grezzi e le
CPU effettive che ne derivano; `fusione_cpu` è la somma del tempo di fusione di
tutti i worker (≈ la vecchia fase F senza preparazione), `seg_max` il segmento
più lento: `seg_max` molto maggiore di `fusione_cpu / segmenti` significa che i
worker aspettano x264 e conviene alzare i thread o ridurre i worker. Gli avvisi
e gli errori di ffmpeg dei segmenti vengono riportati sullo stderr del job
(`[phase G] seg_NNN.ffmpeg.log:`).

| Variabile | Default | Effetto |
|---|---|---|
| `MUSETALK_BLEND_WORKERS` | automatico | Numero di processi. `1` = percorso sequenziale (interruttore di emergenza, senza rebuild); anche un valore non valido (`0`, refuso) spegne il parallelo. Il parametro `n_workers` di `generate()` ha la precedenza. |
| `MUSETALK_NVENC_SESSIONS` | 2 | Sessioni h264_nvenc concorrenti (= segmenti con NVENC). |
| `MUSETALK_X264_THREADS` | `cpu // n_seg`, max 8 | Thread di libx264 per segmento. |
| `MUSETALK_FG_STRICT` | off | `1` = nessun ripiego: il fallimento del parallelo solleva, e solleva anche se il parallelo era stato chiesto in modo esplicito (parametro o env) ma non è praticabile. Per test e validazione. |
| `MUSETALK_FG_VERIFICA` | off | `1` = dopo il parallelo rifonde tutti i frame nel padre e confronta gli sha256 (`[phase F] verifica parita: OK/KO`). Costa quanto una fase F sequenziale. |
| `MUSETALK_FG_STALLO_S` | 300 | Secondi senza avanzamento prima di dichiarare lo stallo. |

Il default automatico è `min(16, cpu_effettive // 2)`, solo su Linux, dove
`cpu_effettive = min(SLURM_CPUS_PER_TASK, affinity, quota cgroup, os.cpu_count())`:
`os.cpu_count()` da solo ignora SLURM e i limiti dei container. Si divide per
due perché ogni worker pilota anche un ffmpeg. Con il template SLURM attuale
(`--cpus-per-task=8`) i worker sono 4; su MN5 ACC conviene chiedere 20 core per
GPU (i template non sono stati modificati).

**Tempi attesi di F+G — STIME NON MISURATE** sull'hardware di destinazione.
Ipotesi: 23 599 frame 768×768, cache hit, x264 `-preset slow` ≈ 90 ms per
vCPU-frame (intervallo 50-130), fork 0,2-0,85 s per figlio. Vanno sostituite
con i numeri reali della riga `[phase F+G]` al primo job.

| Scenario | F+G | Nota |
|---|---|---|
| Sequenziale, H100 SXM (**misurato**) | 590-600 s | riferimento, tabella sopra |
| 8 vCPU (template SLURM attuale) | ~220-460 s *(stima)* | 4 worker × 2 thread |
| 16-20 vCPU | ~95-195 s *(stima)* | |
| 32 vCPU | ~60-125 s *(stima)* | 16 worker × 2 thread; la fase E diventa il collo di bottiglia |
| 72 vCPU | ~32-65 s *(stima)* | 16 worker × 4 thread |
| `enhance=True`, 32 vCPU | ~360 s + ~90 s *(stima)* | il passaggio GFPGAN resta seriale: guadagno ~2× |
| NVENC con 2 sessioni | ~200 s *(stima)* | blending solo 2 volte parallelo |

Uniche misure disponibili, **non rappresentative** dell'hardware di produzione
(MacBook 8 core, numpy 1.24, ffmpeg 7.1, 1 500 frame 768×768 sintetici, 4
worker × 2 thread x264, tempi senza il calcolo delle impronte):

| Contenuto sintetico | Sequenziale (F + G) | Parallelo | Guadagno |
|---|---|---|---|
| comprimibile, ~1,8 Mbps | 24,1 s (8,8 + 15,3) | 20,2 s | ×1,19 |
| rumore leggero | 33,9 s (8,7 + 25,3) | 30,7 s | ×1,11 |
| sfondo rumoroso, ~25 Mbps | 130,9 s (9,0 + 121,9) | 125,6 s | ×1,04 |

Su quella macchina la fase G domina e x264 satura già tutti gli 8 core nel
percorso sequenziale: dividere in segmenti non crea capacità che non c'è. Il
guadagno atteso in produzione viene da due condizioni da **verificare sul
posto**, non dimostrate qui: che la fase F pesi molto più della G (395 s contro
200 s, l'opposto del Mac) e che restino core liberi, cioè che la G di oggi sia
limitata dal processo Python che alimenta ffmpeg (`tobytes` su array a stride
negativo: ~5 ms a frame) o dal fatto che x264 a 768×768 non scala oltre una
quindicina di thread. Lo misura `python tests/verifica_fusione_parallela.py`,
che stampa anche gli fps di x264 lasciato libero e il costo di `tobytes`.

**RAM** (stima): picco di oggi ~107-117 GB (frame 42 + `combine_frames` 42 +
ritagli PIL 14 + facce e maschere ~10); in parallelo ~55-60 GB, più ~0,13 GB di
page table per figlio.

**Verifica**: `python -m unittest tests.test_fusione_parallela
tests.test_generate_cablaggio` gira senza GPU e senza torch (su macOS con
`OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`), anche dentro le immagini montando
`tests/`. Non è verificabile senza GPU: il fork da un padre con contesto CUDA e
i thread dell'SDK RunPod, NVENC reale, GFPGAN e BiSeNet veri, ffmpeg 4.4 e il
backend pthreads di OpenCV su Linux. Per questo il primo job reale va lanciato
con `MUSETALK_FG_VERIFICA=1 MUSETALK_FG_STRICT=1`: il confronto avviene dentro
lo stesso job perché la fase E non è riproducibile al bit fra job diversi.

### 6.3 Costo cloud per run (15 min audio)

H100 SXM: $0.00093/s × 850s = **$0.79 per job** (cache hit).
Active worker 24/7 idle: ~$2.4k/mese. Da spegnere via API fuori dagli orari
di lavoro per ridurre il costo.

R2 storage: ~150 MB per clip set cachato (5 clip × ~30 MB blob).
Bitrate output: ~340 MB per video 15 min @ 3 Mbps. Lifecycle rule R2 a 7
giorni consigliato.

---

## 7. Limiti intrinseci

### 7.1 Cosa NON è risolvibile senza cambiare modello

| Limite | Causa | Workaround possibile |
|---|---|---|
| Forma esatta delle labbra per ogni fonema | MuseTalk training set generalista | Cambiare modello (Wav2Lip 384, SadTalker, DiffTalk, GeneFace++) |
| Espressività emotiva | MuseTalk genera solo lower-face | Solo MuseTalkV2+ o modelli emotive (es. EmoTalk) |
| Jitter frame-to-frame in pause | Inference indipendente per frame | Implementare temporal EMA sui pred_latents (richiede modifica engine) |
| Definizione bocca <250px sorgente | Resolution lock 256×256 | GFPGAN (`--quality`) o cambiare modello a 512px |
| Sincronizzazione fonema-precisa | Whisper-tiny limitato + audio_padding=2 | Provare Whisper-base + padding=3 (untested) |

### 7.2 Cosa è risolvibile ma fuori scope per ora

| Limite | Soluzione | Costo stima |
|---|---|---|
| Bordi blending visibili in alcune frame | Poisson blending (cv2.seamlessClone) invece di paste | 2-3h di lavoro, +20-30s su fase F |
| Risoluzione output | Pre-process source a 1280×720 + GFPGAN | Già usabile col preset `--quality` |
| Lipsync drift su audio molto lungo (>30 min) | Riavviare engine ogni N minuti | Non testato |

---

## 8. Riferimenti

### Repository upstream
- **MuseTalk official**: https://github.com/TMElyralab/MuseTalk
- **Paper**: "MuseTalk: Real-Time High Quality Lip Synchronization with
  Latent Space Inpainting" (Tencent Music Entertainment Lyra Lab, 2024)

### Componenti
- **Stable Diffusion VAE ft-mse**: https://huggingface.co/stabilityai/sd-vae-ft-mse
- **Whisper**: OpenAI Whisper (HuggingFace `transformers.WhisperModel`)
- **DWPose / RTMPose**: https://github.com/IDEA-Research/DWPose
- **face-parse-bisent**: https://github.com/zllrunning/face-parsing.PyTorch
- **S3FD**: via `face_alignment` package
- **GFPGAN v1.4**: https://github.com/TencentARC/GFPGAN

### Infrastruttura
- **RunPod Serverless docs**: https://docs.runpod.io/serverless/overview
- **Cloudflare R2 docs**: https://developers.cloudflare.com/r2/

### File chiave nel progetto (snapshot v8)

**Client** (modificato da v8 per fix drift cumulativo):
- `scripts/client/video_assembler.py:79-110` — `probe_fps()`
- `scripts/client/video_assembler.py:90-135` — `concat_clips_reencode` (con `fps:round=up`)
- `scripts/client/video_assembler.py:265-320` — `_build_frame_to_clip_map` (cumulative_frames)
- `scripts/client/clip_manifest.py:60-90` — `compute_full_set_hash`
- `scripts/client/clip_manifest.py:280-410` — `get_or_compute_full_manifest`
- `scripts/client/synth_random_lipsync.py` — entry-point CLI

**Engine** (l'immagine RunPod `v7` contiene ancora le fasi F e G sequenziali:
la parallelizzazione di §6.2.1 richiede il rebuild delle immagini Docker e
Singularity):
- `api/inference_service.py` — `MuseTalkInference.generate`
- `musetalk/utils/fusione_parallela.py` — fasi F e G (`fondi_e_codifica`)
- `musetalk_runpod/handler.py` — `_handle_preprocess_full`,
  `_expand_cached_per_clip`, `_handle_lipsync`

**Versioning duale**:
- **Script client**: versionato in questo documento (v0...v8). Cambia ad
  ogni release.
- **Immagine Docker `micheledibi/musetalk-runpod`**: versionata indipendentemente
  con tag `:v1, :v2, ..., :v7`. Resta a `v7` finché un fix non richiede
  cambio dell'engine o del handler. v8 è puramente client-side.

---

*Documento mantenuto manualmente. Aggiornare ad ogni release client (`vN →
v(N+1)`), ad ogni bump della docker image, o ad ogni cambio del formato
cache (`v3 → v4`).*
