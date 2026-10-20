#!/bin/bash
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:05:00
#SBATCH --job-name=motion-context-krr
#SBATCH --output=/scratch/xl4358/phase1_motion_context_research/kernel_round1/fit-%j.out
#SBATCH --error=/scratch/xl4358/phase1_motion_context_research/kernel_round1/fit-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME=/scratch/xl4358/phase1_hf_cache
cd /scratch/xl4358/phase1_motion_context_research/kernel_round1
exec /scratch/xl4358/phase1_env/bin/python gpu_kernel.py "$SLURM_JOB_ID"
