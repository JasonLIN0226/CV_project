#!/bin/bash
# CV_project hidden-CoG PushT: train the latent predictor on marker data and evaluate retention
# conditions (Nov 20). CPU only. DATA defaults to v5b; FEAT to "$DATA-ijepa"; CODE to the repo.
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --job-name=pusht_cog_predictor
#SBATCH --output=/projects/work/yang-lab/users/yz12129/logs/pusht_cog_predictor_%j.out
set -euo pipefail
REPO=/projects/work/yang-lab/users/yz12129/CV_project   # venv and build/ live here
CODE=${CODE:-$REPO}                                      # set to a worktree to run its code
DATA=${DATA:-$REPO/build/pusht-cog-v5b-marker-r4-20261120}
FEAT=${FEAT:-$DATA-ijepa}
OUT=${OUT:-$CODE/results/pusht_cog/$(basename "$DATA")_predictor.json}
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
"$REPO/.venv/bin/python" -I "$CODE/experiments/phase2/train_pusht_cog_predictor.py" \
  --data "$DATA" --features "$FEAT" --output "$OUT" --threads "$SLURM_CPUS_PER_TASK" ${EXTRA_ARGS:-}
