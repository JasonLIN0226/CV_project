#!/usr/bin/env bash
#SBATCH --job-name=cv-phase1-repair-independent
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=1
#SBATCH --mem=1500M
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/xl4358/phase1_predictor_repair/repair_execution/logs/independent-%j.out
#SBATCH --error=/scratch/xl4358/phase1_predictor_repair/repair_execution/logs/independent-%j.err
set -euo pipefail
ROOT=/scratch/xl4358/phase1_predictor_repair/repair_execution
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
hostname
hostname -I
cat /etc/ssh/ssh_host_ed25519_key.pub
/scratch/xl4358/phase1_env/bin/python "$ROOT/independent_audit/run_audit.py"
