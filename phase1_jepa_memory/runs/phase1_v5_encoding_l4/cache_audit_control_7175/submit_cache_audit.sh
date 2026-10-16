#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-cache-audit
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xl4358/phase1_v5_l4_cache_audit_7175/logs/audit-%j.out
#SBATCH --error=/scratch/xl4358/phase1_v5_l4_cache_audit_7175/logs/audit-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME=/scratch/xl4358/phase1_hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TASK_CONTROL=/scratch/xl4358/phase1_v5_l4_cache_audit_7175
cd "$TASK_CONTROL"
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python scripts/v5_audit_l4_cache.py --control "$TASK_CONTROL" --source-lock-sha 962dedf7a55416c3f91ebaa1685de4689cbacb3f52ae3dc7627b467d1fe57536
