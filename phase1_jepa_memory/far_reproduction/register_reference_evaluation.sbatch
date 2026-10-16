#!/bin/bash
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --job-name=cv-far-reference-bind
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/xl4358/phase1_far_reference/logs/bind-%j.out
#SBATCH --error=/scratch/xl4358/phase1_far_reference/logs/bind-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
hostname
hostname -I
ssh-keygen -l -E sha256 -f /etc/ssh/ssh_host_ed25519_key.pub
cat /etc/ssh/ssh_host_ed25519_key.pub
/opt/slurm/bin/scontrol show job "$SLURM_JOB_ID"
cd /scratch/xl4358/phase1_far_reference
/scratch/xl4358/phase1_env/bin/python register_reference_evaluation.py
