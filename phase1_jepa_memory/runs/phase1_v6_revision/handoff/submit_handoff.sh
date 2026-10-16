#!/usr/bin/env bash
#SBATCH --job-name=cv-v6-proof-handoff
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/xl4358/phase1_v6_diagnostic_handoff_7251/logs/handoff-%j.out
#SBATCH --error=/scratch/xl4358/phase1_v6_diagnostic_handoff_7251/logs/handoff-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /scratch/xl4358/phase1_v6_diagnostic_handoff_7251
hostname
hostname -I
cat /etc/ssh/ssh_host_ed25519_key.pub
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python bind_completed.py --control /scratch/xl4358/phase1_v6_diagnostic_handoff_7251 --source-lock-sha 63ace4bb0ab4adada2cd55d0618d4a6de0fde1840966b80887b8dadddd49a662
