#!/bin/bash
# ============================================================================
# SLURM array job — batch-process N audio files against the same clip set.
#
# Layout expected:
#   $PROJECT/inputs/<job_set_name>/
#       clips/                      # the clip set (shared across all tasks)
#       audios/audio_001.wav        # one per array task index
#       audios/audio_002.wav
#       ...
#
# Submit:
#   sbatch --array=1-10 \
#       --export=ALL,AUDIO_BASE=$PROJECT/inputs/lessons/audios,\
#               CLIPS_DIR=$PROJECT/inputs/lessons/clips,\
#               OUTPUT_DIR=$PROJECT/lessons_out \
#       hpc/slurm/lipsync_array.sbatch
#
# Cache is persistent + shared across array tasks (same clip set), so only
# the FIRST task pays the preprocess_full cost (~2 min). All subsequent
# tasks hit the cache and run at ~14-15 min each.
# ============================================================================

#SBATCH --job-name=musetalk-array
#SBATCH --output=logs/musetalk_array_%A_%a.out
#SBATCH --error=logs/musetalk_array_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
mkdir -p logs

if module load apptainer 2>/dev/null; then :; else module load singularity; fi

: "${IMAGE:=${PROJECT}/musetalk/musetalk-v8-hpc.sif}"
: "${CACHE_DIR:=${PROJECT}/musetalk_cache}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${CLIPS_DIR:?CLIPS_DIR is required}"
: "${AUDIO_BASE:?AUDIO_BASE is required (directory containing audio_NNN.wav)}"
: "${SEED:=42}"
: "${QUALITY:=0}"

# Resolve THIS task's audio file from the array index.
TASK_ID=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
AUDIO="$AUDIO_BASE/audio_${TASK_ID}.wav"
OUTPUT_NAME="lipsync_${SLURM_ARRAY_JOB_ID}_task${TASK_ID}.mp4"

if [ ! -f "$AUDIO" ]; then
    echo "ERROR: $AUDIO not found for task $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
WORK_DIR="${SCRATCH:-/tmp}/musetalk_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORK_DIR/generated"

QUALITY_FLAG=""
[ "$QUALITY" = "1" ] && QUALITY_FLAG="--quality"

echo "[slurm-array] task=$SLURM_ARRAY_TASK_ID  audio=$AUDIO"
echo "[slurm-array] output=$OUTPUT_DIR/$OUTPUT_NAME"

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
        --seed "$SEED" --use-nvenc \
        $QUALITY_FLAG

rm -rf "$WORK_DIR"
echo "[slurm-array] task=$SLURM_ARRAY_TASK_ID done."
