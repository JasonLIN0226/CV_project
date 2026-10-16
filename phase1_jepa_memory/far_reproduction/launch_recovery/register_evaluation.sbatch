#!/bin/bash
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=interactive
#SBATCH --job-name=cv-far-launch-bind
#SBATCH --cpus-per-task=1
#SBATCH --mem=1500M
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/xl4358/phase1_far_launch_recovery/logs/register-%j.out
#SBATCH --error=/scratch/xl4358/phase1_far_launch_recovery/logs/register-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /scratch/xl4358/phase1_far_launch_recovery
/scratch/xl4358/phase1_env/bin/python register_evaluation.py "$1"
