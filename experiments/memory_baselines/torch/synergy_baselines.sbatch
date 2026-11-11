#!/bin/bash
#SBATCH --job-name=cvmem_synergy
#SBATCH --account=torch_pr_147_courant
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ys4907/cv_project_runs/logs/%x_%j.log
# Trains the literature baselines on the Synergy TRAIN split and scores them, with the team's frozen
# students and reference rungs, on the locked 48-family confirmation set. CPU only.
# Needs synergy_assets.sbatch first.
set -euo pipefail
ROOT=/scratch/ys4907/CV_project
SYN=/scratch/ys4907/cv_project_runs/synergy
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
source /scratch/ys4907/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/ys4907/miniconda3/envs/cvmem
cd "$ROOT"
echo "== host $(hostname) $(date -Is)"
python -m experiments.memory_baselines.synergy_run \
  --terminal "$SYN/terminal/run" \
  --agg "$SYN/conf_full/build/reader-aggregation-20261025/hpc/extracted/run" \
  --train-xfer "$SYN/trainvaldev/xfer" --conf-xfer "$SYN/conf_xfer" --released "$SYN/conf_full" \
  --out "${OUT:-$ROOT/results/memory_baselines/synergy}" ${EXTRA:-}
echo "SYNERGY_BASELINES_DONE $(date -Is)"
