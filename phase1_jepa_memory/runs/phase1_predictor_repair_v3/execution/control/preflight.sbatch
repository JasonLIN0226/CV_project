#!/usr/bin/env bash
#SBATCH --job-name=cv-phase1-repair-preflight
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xl4358/phase1_predictor_repair_v3/execution/logs/preflight-%j.out
#SBATCH --error=/scratch/xl4358/phase1_predictor_repair_v3/execution/logs/preflight-%j.err
set -euo pipefail
ROOT=/scratch/xl4358/phase1_predictor_repair_v3/execution
CONTROL="$ROOT/control"
export PYTHONPATH="$CONTROL/src"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export HF_HOME=/scratch/xl4358/phase1_hf_cache PYTHONUNBUFFERED=1
hostname
hostname -I
cat /etc/ssh/ssh_host_ed25519_key.pub
/scratch/xl4358/phase1_env/bin/python "$CONTROL/native_preflight.py"
