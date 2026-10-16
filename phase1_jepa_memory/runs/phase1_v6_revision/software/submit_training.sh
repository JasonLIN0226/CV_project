#!/usr/bin/env bash
#SBATCH --job-name=cv-v6-real-wd020
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/scratch/xl4358/phase1_jepa_memory_v6_real_wd020/logs/train-%j.out
#SBATCH --error=/scratch/xl4358/phase1_jepa_memory_v6_real_wd020/logs/train-%j.err
set -euo pipefail
export HF_HOME=/scratch/xl4358/phase1_hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /scratch/xl4358/phase1_jepa_memory_v6_real_wd020
hostname
hostname -I
cat /etc/ssh/ssh_host_ed25519_key.pub
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python scripts/v5_train_real.py --control /scratch/xl4358/phase1_jepa_memory_v6_real_wd020 --source-lock-sha 24116c56a15ead3f47d5109be0a389eab179f01fbc63ba96fb69e9813bf3b5e5 --encoding-root /scratch/xl4358/phase1_v5_real_encoding_frame/job_7187 --encoding-control /scratch/xl4358/phase1_jepa_memory_v5_encoding_frame --encoding-job 7187 --output /scratch/xl4358/phase1_v6_real_wd020/job_${SLURM_JOB_ID}
