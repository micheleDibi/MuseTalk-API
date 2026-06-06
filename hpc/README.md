# MuseTalk-API su HPC (MareNostrum 5 ACC + altri cluster EuroHPC)

Deploy della pipeline MuseTalk-API su HPC accademici, primariamente
**MareNostrum 5 ACC (BSC)** ma compatibile con qualsiasi cluster che offra:

- GPU NVIDIA Ampere/Hopper (A100, H100, L40S) — sm_80…sm_90
- Driver NVIDIA ≥ 11.8 (CUDA 12.x driver compatibile con runtime 11.8
  per [forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/))
- SLURM scheduler
- Singularity ≥ 3.7 o Apptainer ≥ 1.0
- ≥ 35 GB di scratch per la build dell'immagine

**Vincoli HPC tipici gestiti dal setup**:
- ❌ Niente outbound internet sui compute node → tutto baked nell'immagine
- ❌ Niente Docker → solo Singularity/Apptainer
- ❌ Niente cloud storage → cache su filesystem `$PROJECT`

---

## Quick start (MareNostrum 5 ACC)

### 1. Login + clone repo

```bash
# SSH al login node
ssh <utente>@mn5-acc.bsc.es

# Clone repo nel tuo $HOME o $PROJECT
cd $PROJECT
git clone https://github.com/micheleDibi/MuseTalk-API.git
cd MuseTalk-API
```

### 2. Scarica pesi modello

I pesi (~8.6 GB) servono dentro l'immagine. Il download richiede internet,
fattibile solo sul login node:

```bash
module load python/3.10
python download_models.py    # ~8.6 GB su ./models
```

### 3. Configura env

```bash
cp hpc/env.example hpc/env.local
$EDITOR hpc/env.local        # imposta MUSETALK_ACCOUNT, MUSETALK_QOS, paths
source hpc/env.local
```

Per scoprire i tuoi account/qos disponibili:
```bash
sacctmgr show user $USER format=user,account,qos,partition
```

### 4. Build dell'immagine Singularity

```bash
module load singularity      # o "module load apptainer"
bash hpc/scripts/build_image.sh
```

Cosa fa:
- Costruisce il `.sif` (~13 GB) sotto `$PROJECT/musetalk/musetalk-v8-hpc.sif`
- Usa `$SCRATCH/sing-tmp` e `$SCRATCH/sing-cache` per non saturare `$HOME`
- Pre-fetcha GFPGAN + S3FD nell'immagine (lazy downloads bloccati offline)
- Verifica con un smoke test che gli import funzionino offline

Durata: **~30-45 min** sul login MN5 (dipende dalla velocità del pip mirror).

### 5. Smoke test offline (consigliato prima del primo job vero)

```bash
bash hpc/scripts/verify_offline.sh
```

Riavvia il network namespace dentro al container per assicurarsi che
nulla scarichi pesi al volo. Se passa, l'immagine è pronta.

### 6. Submit del primo job

Carica le tue clip e il tuo audio su `$PROJECT/inputs/`:

```bash
mkdir -p $PROJECT/inputs/clips
scp -r /path/to/local/clips/* mn5-acc.bsc.es:$PROJECT/inputs/clips/
scp /path/to/local/audio.wav  mn5-acc.bsc.es:$PROJECT/inputs/audio.wav
```

Poi lancia il job:

```bash
source hpc/env.local
sbatch \
    --account=$MUSETALK_ACCOUNT --qos=$MUSETALK_QOS \
    --export=ALL,CLIPS_DIR=$PROJECT/inputs/clips,AUDIO=$PROJECT/inputs/audio.wav,OUTPUT_NAME=lesson1.mp4 \
    hpc/slurm/lipsync_single.sbatch
```

Controlla lo stato:
```bash
squeue -u $USER
tail -f logs/musetalk_<JOBID>.out
```

L'output finale sarà in `$PROJECT/musetalk_outputs/lesson1.mp4` + un
`lesson1.json` con tempi per fase, hash del cache, ecc.

---

## Tempistiche attese

Su H100 PCIe MN5 ACC, audio da 15.73 min, 5 clip uniche da ~6s @ 24fps:

| Run | Cache | Tempo | Rapporto | Note |
|---|---|---|---|---|
| Primo | miss (preprocess full) | ~30 min | ~1.9:1 | One-time per clip set |
| Successivi (stesse clip) | hit | **~14-15 min** | **~0.95:1** | Target 1:1 raggiunto |
| Con `--quality` (GFPGAN) | hit | ~20-21 min | ~1.3:1 | Qualità professionale |

Il primo run paga ~80-90 s per clip unica di face detection + VAE batch +
parsing batch (totale ~7-10 min su 5 clip). I run successivi caricano i
blob da `$PROJECT/musetalk_cache/blobs/<hash>/` (vedi `cache` flag in
`*.json` di output).

---

## Template SLURM disponibili

| File | Quando usarlo |
|---|---|
| `slurm/lipsync_single.sbatch` | Singolo audio → singolo mp4. Use case base. |
| `slurm/lipsync_quality.sbatch` | Idem ma con `--quality` (GFPGAN attivo). Più lento, più nitido. |
| `slurm/lipsync_array.sbatch` | SLURM array per batch processing di N audio contro lo stesso set di clip. |

Tutti accettano gli stessi env vars (vedi commenti in cima a ogni file).

---

## Architettura del deploy

```
┌──── LOGIN NODE (internet OK) ───┐    ┌── COMPUTE NODE (no internet) ──┐
│                                 │    │                                │
│  download_models.py             │    │  singularity exec --nv         │
│       ↓ ~8.6 GB pesi            │    │     --bind paths               │
│  bash build_image.sh            │    │     musetalk-v8-hpc.sif        │
│       ↓                         │    │     -m scripts.client          │
│  Singularity.def %post:         │    │     .synth_random_lipsync      │
│   - pip install torch+stack     │    │     --backend local            │
│   - pre-fetch GFPGAN, S3FD      │    │                                │
│   - smoke test offline          │ →  │  Engine: MuseTalkInference     │
│       ↓ ~13 GB                  │    │  Cache: /cache (= $PROJECT/    │
│  musetalk-v8-hpc.sif            │    │           musetalk_cache)      │
│                                 │    │  Output: /outputs              │
└─────────────────────────────────┘    └────────────────────────────────┘
                       │                              ↑
                       │     $PROJECT (GPFS)          │
                       └──────────────────────────────┘
```

---

## Backend `--backend local` vs cloud

Il client `scripts/client/synth_random_lipsync.py` espone `--backend`:

- `--backend runpod` (default) — uploads to R2 + RunPod Serverless. Cloud
  path, invariato.
- `--backend local` — carica `MuseTalkInference` in-process, no R2, no
  network. **Path HPC**.

Le modifiche al codice per supportare HPC sono minime:
- nuovo flag `--backend` in `synth_random_lipsync.py`
- nuovo modulo `scripts/client/local_backend.py` (preprocess_full +
  cache + lipsync in-process)
- nuovo env var `GFPGAN_MODEL_PATH` rispettato da `api/inference_service.py:_load_gfpgan`

Tutto il path cloud RunPod resta inalterato.

---

## Cache su HPC

Stessa logica del cloud (cache key v3, hash deterministico su
clip+params), ma su disco invece che su R2:

```
$PROJECT/musetalk_cache/
├── <hash>.json                          # manifest JSON (paths ai blob)
└── blobs/<hash>/
    ├── clip_1.mp4_bboxes.json           # per-frame [x1,y1,x2,y2]
    ├── clip_1.mp4_latents.pt            # per-frame [N,8,32,32]
    ├── clip_1.mp4_parsing.npz           # per-frame parsing 512×512
    └── ...
```

Il cache è **persistente** tra job, condivisibile tra utenti del progetto
(se i permessi GPFS lo consentono). Pulisci manualmente con:

```bash
rm -rf $PROJECT/musetalk_cache/<hash>*
# oppure tutto:
rm -rf $PROJECT/musetalk_cache
```

---

## Troubleshooting

### Build fallisce con "out of disk space"

`$SCRATCH/sing-tmp` o `$SCRATCH/sing-cache` saturati. Pulisci e rilancia:
```bash
rm -rf $SCRATCH/sing-tmp $SCRATCH/sing-cache
bash hpc/scripts/build_image.sh
```

### Job fallisce con `OSError: [Errno 101] Network is unreachable`

Qualche modulo prova ancora a scaricare a runtime. Lancia
`hpc/scripts/verify_offline.sh` per identificare quale weight manca.
Comune: `~/.cache/torch/hub/checkpoints/` mancante → vedi
`hpc/scripts/prefetch_lazy_models.py`.

### Job termina con "module 'mmcv._ext' has no attribute ..."

Il wheel di mmcv non matcha la versione di PyTorch nel container.
Solitamente significa che la build ha scaricato un mmcv per CUDA 12.x
invece di 11.8. Forza il rebuild con:
```bash
rm $PROJECT/musetalk/musetalk-v8-hpc.sif
SINGULARITY_CACHEDIR=$SCRATCH/sing-cache bash hpc/scripts/build_image.sh
```

### `singularity: command not found`

```bash
module avail | grep -i sing
module load singularity      # o "apptainer"
```

Se nessun modulo è disponibile sul tuo cluster, contatta il supporto BSC.

### Job timeout

Se il primo run sfora 1 h (preprocess_full ha tantissime clip), aumenta
in `slurm/lipsync_single.sbatch`:
```
#SBATCH --time=02:00:00
```

### NVENC non disponibile

H100 PCIe non ha NVENC hardware (decisione NVIDIA). Il flag `--use-nvenc`
viene comunque passato a ffmpeg; l'engine fa automaticamente fallback a
`libx264` con `-preset slow -crf 16` (più lento, stessa qualità).

---

## File chiave

| Path | Contenuto |
|---|---|
| `Singularity.def` | Definition file per la build dell'immagine |
| `scripts/build_image.sh` | Wrapper SCRATCH-aware per `singularity build` |
| `scripts/prefetch_lazy_models.py` | Bake di GFPGAN + S3FD nell'immagine |
| `scripts/verify_offline.sh` | Smoke test offline sull'immagine costruita |
| `slurm/lipsync_single.sbatch` | Template SLURM single run |
| `slurm/lipsync_quality.sbatch` | Template SLURM con GFPGAN |
| `slurm/lipsync_array.sbatch` | Template SLURM array (batch) |
| `env.example` | Variabili d'ambiente da copiare in `env.local` |

---

## Riferimenti

- **Architettura del progetto** (model, pipeline, version history): vedi
  `../docs/ARCHITECTURE.md`
- **README principale**: vedi `../README.md`
- **MareNostrum 5 docs**: https://www.bsc.es/supportkc/docs/MareNostrum5/intro
- **NVIDIA CUDA forward compatibility**: https://docs.nvidia.com/deploy/cuda-compatibility/
- **Apptainer docs**: https://apptainer.org/docs/
- **SLURM docs**: https://slurm.schedmd.com/documentation.html
