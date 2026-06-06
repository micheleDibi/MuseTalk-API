#!/bin/bash
# ============================================================================
# Quality preset: same as lipsync_single.sbatch but with GFPGAN enabled and
# a wider time limit. Use for production-quality output where polish matters
# more than the 1:1 ratio target.
#
# Expected runtime: ~20-21 min for 15 min of audio (vs ~14-15 min standard).
# ============================================================================

#SBATCH --job-name=musetalk-quality
#SBATCH --output=logs/musetalk_quality_%j.out
#SBATCH --error=logs/musetalk_quality_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Force quality preset and delegate everything else to the single template.
export QUALITY=1
export OUTPUT_NAME="${OUTPUT_NAME:-quality_${SLURM_JOB_ID}.mp4}"
exec bash "$(dirname "$0")/lipsync_single.sbatch"
