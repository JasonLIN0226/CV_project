#!/bin/bash
#SBATCH --job-name=cv-far-phase1
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=03:19:00
#SBATCH --chdir=/scratch/xl4358/phase1_far_phase1
#SBATCH --output=/scratch/xl4358/phase1_far_phase1/logs/phase1-%j.out
#SBATCH --error=/scratch/xl4358/phase1_far_phase1/logs/phase1-%j.err
set -euo pipefail
export HF_HOME=/scratch/xl4358/phase1_hf_cache
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
hostname
hostname -I
cat /etc/ssh/ssh_host_ed25519_key.pub
/opt/slurm/bin/scontrol show job "$SLURM_JOB_ID"
nvidia-smi
exec /scratch/xl4358/phase1_env/bin/python /scratch/xl4358/phase1_far_phase1/run_phase1.py
