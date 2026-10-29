#!/bin/bash
#SBATCH --job-name=gate-v2-validate
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --export=ALL
#SBATCH --output=/scratch/xl4358/clean_future_gate_20261022/gate_supervision_v2_20261029/run_20261029/validate-%j.out
#SBATCH --error=/scratch/xl4358/clean_future_gate_20261022/gate_supervision_v2_20261029/run_20261029/validate-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4
cd /scratch/xl4358/clean_future_gate_20261022/gate_supervision_v2_20261029/run_20261029/source
sha256sum -c SHA256SUMS
exec /scratch/xl4358/clean_future_retention_audit_20261022/venv/bin/python experiments/gate_supervision_comparison_design_v2/run_trainval.py validate
