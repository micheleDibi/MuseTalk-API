#!/usr/bin/env bash
# Build the MuseTalk-API Singularity/Apptainer image on a login node with
# internet. Run from the repository root.
#
# The build needs ~35 GB of disk for: base image cache, pip downloads,
# squashfs construction, and the final .sif (~13 GB). $HOME quota on
# MareNostrum 5 is typically too small — point TMPDIR + CACHEDIR at
# $SCRATCH so the build doesn't fill up $HOME.

set -euo pipefail

# ----- paths (override via environment) -----
: "${IMAGE_OUT:=${PROJECT:-$PWD}/musetalk/musetalk-v8-hpc.sif}"
: "${DEF_FILE:=hpc/Singularity.def}"
: "${SCRATCH:=$PWD/.singularity-build}"

# ----- ensure required things exist -----
if [ ! -f "$DEF_FILE" ]; then
    echo "ERROR: $DEF_FILE not found. Run from the repository root." >&2
    exit 2
fi
if [ ! -d "models" ]; then
    echo "ERROR: ./models is missing — run 'python download_models.py' first" >&2
    echo "       (you need internet for this step, it pulls ~8.6 GB)" >&2
    exit 2
fi

# ----- build under SCRATCH so $HOME quota is not consumed -----
export SINGULARITY_TMPDIR="$SCRATCH/sing-tmp"
export SINGULARITY_CACHEDIR="$SCRATCH/sing-cache"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR" "$(dirname "$IMAGE_OUT")"

# ----- pick singularity or apptainer (MN5 may have either) -----
if command -v apptainer >/dev/null 2>&1; then
    BUILDER=apptainer
elif command -v singularity >/dev/null 2>&1; then
    BUILDER=singularity
else
    echo "ERROR: neither 'apptainer' nor 'singularity' found on PATH" >&2
    echo "HINT:  on MareNostrum 5 try: module load singularity   (or apptainer)" >&2
    exit 3
fi

echo "[build] builder: $BUILDER"
echo "[build] def file: $DEF_FILE"
echo "[build] output:   $IMAGE_OUT"
echo "[build] tmp:      $SINGULARITY_TMPDIR"
echo "[build] cache:    $SINGULARITY_CACHEDIR"

t0=$(date +%s)
$BUILDER build --force "$IMAGE_OUT" "$DEF_FILE"
t1=$(date +%s)

echo "[build] done in $((t1 - t0)) s"
echo "[build] image:    $IMAGE_OUT  ($(du -h "$IMAGE_OUT" | cut -f1))"
echo
echo "Next: submit a SLURM job, e.g."
echo "  sbatch --export=CLIPS_DIR=\$PROJECT/inputs/clips,AUDIO=\$PROJECT/inputs/audio.wav \\"
echo "      hpc/slurm/lipsync_single.sbatch"
