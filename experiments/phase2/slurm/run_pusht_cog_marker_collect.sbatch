#!/bin/bash
# CV_project hidden-CoG PushT v5 (marker), Nov 9: collect 128/32/32 families on CPU.
# MARKER_RADIUS=4 with OUTPUT=.../pusht-cog-v5b-marker-r4-20261120 gives v5b (same seeds, larger dot).
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96GB
#SBATCH --time=1:00:00
#SBATCH --job-name=pusht_cog_marker_collect
#SBATCH --output=/projects/work/yang-lab/users/yz12129/logs/pusht_cog_marker_collect_%j.out
set -euo pipefail
REPO=/projects/work/yang-lab/users/yz12129/CV_project   # venv and build/ live here
CODE=${CODE:-$REPO}                                      # set to a worktree to run its code
export SDL_VIDEODRIVER=dummy PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$REPO/.venv/bin/python" -I "$CODE/experiments/phase2/prepare_pusht_cog_marker_data.py" \
  --output "${OUTPUT:-$REPO/build/pusht-cog-v5-marker-20261109}" --processes "$SLURM_CPUS_PER_TASK" \
  --marker-radius "${MARKER_RADIUS:-2}" --train-families "${TRAIN_FAMILIES:-128}" \
  --distractor-dots "${DISTRACTOR_DOTS:-0}"
