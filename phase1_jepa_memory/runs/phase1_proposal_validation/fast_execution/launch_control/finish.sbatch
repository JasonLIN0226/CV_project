#!/usr/bin/env bash
#SBATCH --job-name=cv-phase1-fast-preflight
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=1
#SBATCH --mem=1500M
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/xl4358/phase1_proposal_validation/fast_execution/logs/preflight-%j.out
#SBATCH --error=/scratch/xl4358/phase1_proposal_validation/fast_execution/logs/preflight-%j.err
set -euo pipefail
export PYTHONPATH=/scratch/xl4358/phase1_proposal_validation/fast_execution/control/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
hostname
hostname -I
cat /etc/ssh/ssh_host_ed25519_key.pub
/scratch/xl4358/phase1_env/bin/python /scratch/xl4358/phase1_proposal_validation/fast_execution/launch_control/finish.py
