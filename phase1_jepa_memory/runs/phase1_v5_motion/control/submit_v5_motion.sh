#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-motion
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/scratch/xl4358/phase1_jepa_memory_v5_motion/logs/train-%j.out
#SBATCH --error=/scratch/xl4358/phase1_jepa_memory_v5_motion/logs/train-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME=/scratch/xl4358/phase1_hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TASK_CONTROL=/scratch/xl4358/phase1_jepa_memory_v5_motion
cd "$TASK_CONTROL"
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python scripts/v5_train_motion.py --control "$TASK_CONTROL" --source-lock-sha f230af5ce3b0f2bf85d88eeecca886db218970547975e34629b61eabc843896c --output /scratch/xl4358/phase1_v5_motion_development
