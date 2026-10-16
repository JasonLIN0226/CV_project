#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-diag-bind
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/xl4358/phase1_v5_real_diagnostic_handoff_7209/logs/bind-%j.out
#SBATCH --error=/scratch/xl4358/phase1_v5_real_diagnostic_handoff_7209/logs/bind-%j.err
set -euo pipefail
export HF_HOME=/scratch/xl4358/phase1_hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
TASK_CONTROL=/scratch/xl4358/phase1_v5_real_diagnostic_handoff_7209
cd "$TASK_CONTROL"
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python bind_completed_real_diagnostics.py --control "$TASK_CONTROL" --source-lock-sha 870f1e43047e733158dce14cfe14544c7a64d7b362dc8ff80847f70c9b37a3c9
