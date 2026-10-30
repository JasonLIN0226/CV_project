#!/bin/bash
#SBATCH --job-name=cv-qualified-reader-conf
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/xy3257/cv-phase2-20261020/qualified-reader-conf-%j.log
set -euo pipefail
umask 077
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
BASE=/scratch/xy3257/cv-phase2-20261020
"$BASE/venv/bin/python" "$BASE/qualified-reader-conf-job/conf_encode_project.py" --base "$BASE"
