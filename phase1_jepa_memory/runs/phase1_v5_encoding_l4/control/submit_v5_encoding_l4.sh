#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-encode-l4
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/scratch/xl4358/phase1_jepa_memory_v5_encoding_l4/logs/encode-%j.out
#SBATCH --error=/scratch/xl4358/phase1_jepa_memory_v5_encoding_l4/logs/encode-%j.err
set -euo pipefail
export HF_HOME=/scratch/xl4358/phase1_hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_CONTROL=/scratch/xl4358/phase1_jepa_memory_v5_encoding_l4
cd "$TASK_CONTROL"
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python scripts/v5_encode_real_l4.py --control "$TASK_CONTROL" --source-lock-sha 39fe05f2e7ddb65bf9f144b13e0cd327cf2cdf289c3c204d25ee52334280d88c --rgb-root /scratch/xl4358/phase1_v5_data/rgb --output /scratch/xl4358/phase1_v5_real_encoding_l4/job_${SLURM_JOB_ID}
