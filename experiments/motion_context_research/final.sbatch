#!/bin/bash
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=1
#SBATCH --mem=1500M
#SBATCH --time=00:15:00
#SBATCH --job-name=motion-context-final
#SBATCH --output=/scratch/xl4358/phase1_motion_context_locked_test/final-%j.out
#SBATCH --error=/scratch/xl4358/phase1_motion_context_locked_test/final-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
exec /scratch/xl4358/phase1_env/bin/python -u /scratch/xl4358/phase1_motion_context_locked_test/final_verify.py
