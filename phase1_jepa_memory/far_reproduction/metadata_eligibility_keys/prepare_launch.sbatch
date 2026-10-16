#!/bin/bash
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=interactive
#SBATCH --job-name=cv-far-meta-prepare
#SBATCH --cpus-per-task=1
#SBATCH --mem=1500M
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xl4358/phase1_far_metadata_keys/logs/prepare-%j.out
#SBATCH --error=/scratch/xl4358/phase1_far_metadata_keys/logs/prepare-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export HF_HOME=/scratch/xl4358/phase1_hf_cache HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1
export FAR_ROOT=/scratch/xl4358/phase1_far_reference/upstream
export FAR_MODELS_ROOT=$FAR_ROOT/models FAR_RESULT_ROOT=$FAR_ROOT/results
export PYTHONPATH=/scratch/xl4358/phase1_far_reference/vendor:$FAR_ROOT/src
hostname
hostname -I
ssh-keygen -l -E sha256 -f /etc/ssh/ssh_host_ed25519_key.pub
cat /etc/ssh/ssh_host_ed25519_key.pub
/opt/slurm/bin/scontrol show job "$SLURM_JOB_ID"
cd /scratch/xl4358/phase1_far_metadata_keys
/scratch/xl4358/phase1_env/bin/python prepare_launch.py
