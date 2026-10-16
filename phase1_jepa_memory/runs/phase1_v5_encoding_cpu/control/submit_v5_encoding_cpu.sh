#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-encode-cpu
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/scratch/xl4358/phase1_jepa_memory_v5_encoding_cpu/logs/encode-%j.out
#SBATCH --error=/scratch/xl4358/phase1_jepa_memory_v5_encoding_cpu/logs/encode-%j.err
set -euo pipefail
export HF_HOME=/scratch/xl4358/phase1_hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_CONTROL=/scratch/xl4358/phase1_jepa_memory_v5_encoding_cpu
cd "$TASK_CONTROL"
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python scripts/v5_encode_real_cpu.py --control "$TASK_CONTROL" --source-lock-sha 01f0adc454082c050331fc7d30911814f46c8cbf4740b56697b12796dc9797df --rgb-root /scratch/xl4358/phase1_v5_data/rgb --output /scratch/xl4358/phase1_v5_real_encoding_cpu/job_${SLURM_JOB_ID}
