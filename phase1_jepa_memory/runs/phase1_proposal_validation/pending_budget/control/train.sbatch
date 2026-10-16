#!/usr/bin/env bash
#SBATCH --job-name=cv-phase1-anchor
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=04:56:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/scratch/xl4358/phase1_proposal_validation/pending_budget/logs/train-%j.out
#SBATCH --error=/scratch/xl4358/phase1_proposal_validation/pending_budget/logs/train-%j.err
set -euo pipefail
ROOT=/scratch/xl4358/phase1_proposal_validation/pending_budget
CONTROL="$ROOT/control"
export PYTHONPATH="$CONTROL/src"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export HF_HOME=/scratch/xl4358/phase1_hf_cache
export PYTHONUNBUFFERED=1
hostname
hostname -I
cat /etc/ssh/ssh_host_ed25519_key.pub
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
/scratch/xl4358/phase1_env/bin/python "$CONTROL/run_trial.py" train \
    --control "$CONTROL" --output "$ROOT/training/job_$SLURM_JOB_ID"
