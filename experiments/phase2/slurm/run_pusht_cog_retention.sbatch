#!/bin/bash
# CV_project hidden-CoG PushT: deletion-cost retention on marker data with a frozen predictor (Nov 21).
# Needs predictor checkpoints (train_pusht_cog_predictor.py --save-dir). CPU only.
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=16:00:00
#SBATCH --job-name=pusht_cog_retention
#SBATCH --output=/projects/work/yang-lab/users/yz12129/logs/pusht_cog_retention_%j.out
set -euo pipefail
REPO=/projects/work/yang-lab/users/yz12129/CV_project   # venv and build/ live here
CODE=${CODE:-$REPO}                                      # set to a worktree to run its code
DATA=${DATA:-$REPO/build/pusht-cog-v5b-marker-r4-t960-20261120}
FEAT=${FEAT:-$DATA-ijepa}
PRED=${PRED:-$FEAT/predictors}
OUT=${OUT:-$CODE/results/pusht_cog/$(basename "$DATA")_retention.json}
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
"$REPO/.venv/bin/python" -I "$CODE/experiments/phase2/retention_pusht_cog.py" \
  --data "$DATA" --features "$FEAT" --predictors "$PRED" --output "$OUT" --threads "$SLURM_CPUS_PER_TASK" ${EXTRA_ARGS:-}
