#!/bin/bash
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --job-name=cv-far-dependencies
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xl4358/phase1_far_dependency_recovery/logs/prepare-%j.out
#SBATCH --error=/scratch/xl4358/phase1_far_dependency_recovery/logs/prepare-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
export HF_HOME=/scratch/xl4358/phase1_hf_cache HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONUNBUFFERED=1
hostname
hostname -I
ssh-keygen -l -E sha256 -f /etc/ssh/ssh_host_ed25519_key.pub
cat /etc/ssh/ssh_host_ed25519_key.pub
/opt/slurm/bin/scontrol show job "$SLURM_JOB_ID"
cd /scratch/xl4358/phase1_far_dependency_recovery
/scratch/xl4358/phase1_env/bin/python recover_dependencies.py
