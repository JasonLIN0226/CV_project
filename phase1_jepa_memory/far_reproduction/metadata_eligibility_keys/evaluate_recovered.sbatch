#!/bin/bash
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=cv-far-recovered-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=03:33:40
#SBATCH --output=/scratch/xl4358/phase1_far_metadata_keys/logs/evaluate-%j.out
#SBATCH --error=/scratch/xl4358/phase1_far_metadata_keys/logs/evaluate-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export HF_HOME=/scratch/xl4358/phase1_hf_cache HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1
export FAR_ROOT=/scratch/xl4358/phase1_far_reference/upstream
export FAR_MODELS_ROOT=$FAR_ROOT/models FAR_RESULT_ROOT=$FAR_ROOT/results
export PYTHONPATH=/scratch/xl4358/phase1_far_reference/vendor:$FAR_ROOT/src
hostname
hostname -I
ssh-keygen -l -E sha256 -f /etc/ssh/ssh_host_ed25519_key.pub
cat /etc/ssh/ssh_host_ed25519_key.pub
/opt/slurm/bin/scontrol show job "$SLURM_JOB_ID"
nvidia-smi
cd /scratch/xl4358/phase1_far_metadata_keys
/scratch/xl4358/phase1_env/bin/python evaluate_recovered.py
