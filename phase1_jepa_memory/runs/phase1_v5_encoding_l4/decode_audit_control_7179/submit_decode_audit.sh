#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-decode-audit
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xl4358/phase1_v5_decode_audit_7179/logs/audit-%j.out
#SBATCH --error=/scratch/xl4358/phase1_v5_decode_audit_7179/logs/audit-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
TASK_CONTROL=/scratch/xl4358/phase1_v5_decode_audit_7179
cd "$TASK_CONTROL"
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python -u decode_performance_audit.py --control "$TASK_CONTROL" --source-lock-sha 9abe3450f43ae9e4d7a9f010178c88af0c333b4e6841f9a1204caed9037b8cdd
