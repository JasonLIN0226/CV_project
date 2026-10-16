#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-gate
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/scratch/xl4358/phase1_jepa_memory_v5_gate/logs/train-%j.out
#SBATCH --error=/scratch/xl4358/phase1_jepa_memory_v5_gate/logs/train-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME=/scratch/xl4358/phase1_hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TASK_CONTROL=/scratch/xl4358/phase1_jepa_memory_v5_gate
cd "$TASK_CONTROL"
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python scripts/v5_train_synthetic.py --control "$TASK_CONTROL" --source-lock-sha 9e2fec4cc7a2b526a275b53f6a96c6180cfe470004f805310e6075ffdb5f9830 --output /scratch/xl4358/phase1_v5_gate_development
