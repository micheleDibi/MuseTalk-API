#!/bin/bash
# ============================================================================
# Single-run MuseTalk lipsync job on MareNostrum 5 ACC.
#
# Submit example (from the repo root on a login node):
#
#   export MUSETALK_ACCOUNT=bsc99
#   export MUSETALK_QOS=acc_bsccs
#   sbatch \
#       --export=ALL,CLIPS_DIR=$PROJECT/inputs/michele_monaco,\
#               AUDIO=$PROJECT/inputs/audio.wav,OUTPUT_NAME=lesson1.mp4 \
#       hpc/slurm/lipsync_single.sbatch
#
# All paths below assume MN5 ACC conventions ($PROJECT, $SCRATCH). On other
# EuroHPC machines (Leonardo, MeluXina, ...) the env var names may differ —
# adjust IMAGE / CACHE_DIR / OUTPUT_DIR accordingly via --export.
# ============================================================================

#SBATCH --job-name=musetalk-lipsync
#SBATCH --output=logs/musetalk_%j.out
#SBATCH --error=logs/musetalk_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
mkdir -p logs

# --------- load Singularity / Apptainer ---------
# MN5 module name may be "singularity" or "apptainer"; try both.
if module load apptainer 2>/dev/null; then
    :
else
    module load singularity
fi

# --------- defaults (overridable via --export) ---------
: "${IMAGE:=${PROJECT}/musetalk/musetalk-v8-hpc.sif}"
: "${CACHE_DIR:=${PROJECT}/musetalk_cache}"
: "${OUTPUT_DIR:=${PROJECT}/musetalk_outputs}"
: "${CLIPS_DIR:?CLIPS_DIR is required (e.g. \$PROJECT/inputs/clips)}"
: "${AUDIO:?AUDIO is required (e.g. \$PROJECT/inputs/audio.wav)}"
: "${OUTPUT_NAME:=lipsync_${SLURM_JOB_ID}.mp4}"
: "${SEED:=42}"
: "${EXTRA_MARGIN:=10}"
: "${LEFT_CHEEK_WIDTH:=90}"
: "${RIGHT_CHEEK_WIDTH:=90}"
: "${PARSING_MODE:=jaw}"
: "${BATCH_SIZE:=16}"
: "${USE_NVENC:=1}"           # H100 has no NVENC; engine falls back to libx264
: "${QUALITY:=0}"             # 1 → enable GFPGAN (~+5-7 min)

# --------- sanity checks ---------
if [ ! -f "$IMAGE" ]; then
    echo "ERROR: image $IMAGE not found. Build it on the login node first" >&2
    echo "       bash hpc/scripts/build_image.sh" >&2
    exit 2
fi
if [ ! -d "$CLIPS_DIR" ]; then
    echo "ERROR: CLIPS_DIR=$CLIPS_DIR does not exist or is not a directory" >&2
    exit 2
fi
if [ ! -f "$AUDIO" ]; then
    echo "ERROR: AUDIO=$AUDIO is not a file" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
WORK_DIR="${SCRATCH:-/tmp}/musetalk_${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR/generated"

NVENC_FLAG=""
[ "$USE_NVENC" = "1" ] && NVENC_FLAG="--use-nvenc"

QUALITY_FLAG=""
[ "$QUALITY" = "1" ] && QUALITY_FLAG="--quality"

# --------- run ---------
echo "[slurm] job_id=$SLURM_JOB_ID node=$SLURMD_NODENAME"
echo "[slurm] image=$IMAGE"
echo "[slurm] clips=$CLIPS_DIR  audio=$AUDIO  output=$OUTPUT_DIR/$OUTPUT_NAME"
echo "[slurm] cache=$CACHE_DIR  work=$WORK_DIR"
echo "[slurm] seed=$SEED quality=$QUALITY use_nvenc=$USE_NVENC"

# Pick the right binary — both apptainer and singularity accept the same flags.
BUILDER=$(command -v apptainer || command -v singularity)

$BUILDER exec --nv \
    --bind "$CLIPS_DIR:/inputs/clips:ro" \
    --bind "$AUDIO:/inputs/audio.wav:ro" \
    --bind "$WORK_DIR:/work" \
    --bind "$CACHE_DIR:/cache" \
    --bind "$OUTPUT_DIR:/outputs" \
    "$IMAGE" \
    -m scripts.client.synth_random_lipsync \
        --backend local \
        --clips-dir /inputs/clips \
        --audio /inputs/audio.wav \
        --output "/outputs/$OUTPUT_NAME" \
        --intermediate-dir /work/generated \
        --manifest-cache-dir /cache/manifests \
        --seed "$SEED" \
        --extra-margin "$EXTRA_MARGIN" \
        --left-cheek-width "$LEFT_CHEEK_WIDTH" \
        --right-cheek-width "$RIGHT_CHEEK_WIDTH" \
        --parsing-mode "$PARSING_MODE" \
        --batch-size "$BATCH_SIZE" \
        $NVENC_FLAG \
        $QUALITY_FLAG

# Cleanup the per-job scratch (the persistent cache under $PROJECT stays).
rm -rf "$WORK_DIR"

echo "[slurm] done. output: $OUTPUT_DIR/$OUTPUT_NAME"
