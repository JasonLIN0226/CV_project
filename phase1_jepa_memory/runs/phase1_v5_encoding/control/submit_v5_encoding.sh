#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-encode
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/scratch/xl4358/phase1_jepa_memory_v5_encoding/logs/encode-%j.out
#SBATCH --error=/scratch/xl4358/phase1_jepa_memory_v5_encoding/logs/encode-%j.err
set -euo pipefail
export HF_HOME=/scratch/xl4358/phase1_hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_CONTROL=/scratch/xl4358/phase1_jepa_memory_v5_encoding
cd "$TASK_CONTROL"
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python scripts/v5_encode_real.py --control "$TASK_CONTROL" --source-lock-sha 384ef961b067c6c0d7189e02ca370a13d680c5d8e11e13df67d1fdf8b3537ca2 --rgb-root /scratch/xl4358/phase1_v5_data/rgb --output /scratch/xl4358/phase1_v5_real_encoding/job_${SLURM_JOB_ID}
