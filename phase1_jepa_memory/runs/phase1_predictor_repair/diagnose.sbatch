#!/bin/bash
#SBATCH --job-name=cv-phase1-diagnose
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xl4358/phase1_predictor_repair/logs/diagnose-%j.out
#SBATCH --error=/scratch/xl4358/phase1_predictor_repair/logs/diagnose-%j.err
set -euo pipefail
hostname
hostname -I
cat /etc/ssh/ssh_host_ed25519_key.pub
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export HF_HOME=/scratch/xl4358/phase1_hf_cache
exec /scratch/xl4358/phase1_env/bin/python /scratch/xl4358/phase1_predictor_repair/diagnose_native.py
