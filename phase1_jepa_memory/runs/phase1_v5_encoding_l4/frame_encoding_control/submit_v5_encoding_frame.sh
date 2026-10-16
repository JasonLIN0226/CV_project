#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-encode
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/scratch/xl4358/phase1_jepa_memory_v5_encoding_frame/logs/encode-%j.out
#SBATCH --error=/scratch/xl4358/phase1_jepa_memory_v5_encoding_frame/logs/encode-%j.err
set -euo pipefail
export HF_HOME=/scratch/xl4358/phase1_hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_CONTROL=/scratch/xl4358/phase1_jepa_memory_v5_encoding_frame
cd "$TASK_CONTROL"
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python scripts/v5_encode_real.py --control "$TASK_CONTROL" --source-lock-sha cd667fc4f029660c198353570686a8ed8f069a0e2dab1434e912776f5521985e --rgb-root /scratch/xl4358/phase1_v5_data/rgb --output /scratch/xl4358/phase1_v5_real_encoding_frame/job_${SLURM_JOB_ID}
