#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-gate-eval
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xl4358/phase1_jepa_memory_v5_gate_evaluation/logs/eval-%j.out
#SBATCH --error=/scratch/xl4358/phase1_jepa_memory_v5_gate_evaluation/logs/eval-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME=/scratch/xl4358/phase1_hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TASK_CONTROL=/scratch/xl4358/phase1_jepa_memory_v5_gate_evaluation
cd "$TASK_CONTROL"
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python scripts/v5_evaluate_gate.py --control "$TASK_CONTROL" --source-lock-sha 2055b466634d59adcd951088eab03d6ac9d87d811f49c48d51d4d34958bd14a5
