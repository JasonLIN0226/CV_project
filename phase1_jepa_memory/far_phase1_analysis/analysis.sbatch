#!/bin/bash
#SBATCH --job-name=cv-far-phase1-audit
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=1
#SBATCH --mem=1500M
#SBATCH --time=00:15:00
#SBATCH --chdir=/scratch/xl4358/phase1_far_phase1_analysis
#SBATCH --output=/scratch/xl4358/phase1_far_phase1_analysis/logs/audit-%j.out
#SBATCH --error=/scratch/xl4358/phase1_far_phase1_analysis/logs/audit-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
hostname
hostname -I
cat /etc/ssh/ssh_host_ed25519_key.pub
/opt/slurm/bin/scontrol show job "$SLURM_JOB_ID"
exec /scratch/xl4358/phase1_env/bin/python /scratch/xl4358/phase1_far_phase1_analysis/aggregate.py
