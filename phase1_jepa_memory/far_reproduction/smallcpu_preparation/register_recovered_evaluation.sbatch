#!/bin/bash
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=interactive
#SBATCH --job-name=cv-far-recovered-bind
#SBATCH --cpus-per-task=1
#SBATCH --mem=1500M
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/xl4358/phase1_far_dependency_smallcpu/logs/bind-%j.out
#SBATCH --error=/scratch/xl4358/phase1_far_dependency_smallcpu/logs/bind-%j.err
set -euo pipefail
export TMPDIR=/scratch/xl4358/phase1_far_dependency_smallcpu/tmp
export PIP_DISABLE_PIP_VERSION_CHECK=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
hostname
hostname -I
ssh-keygen -l -E sha256 -f /etc/ssh/ssh_host_ed25519_key.pub
cat /etc/ssh/ssh_host_ed25519_key.pub
/opt/slurm/bin/scontrol show job "$SLURM_JOB_ID"
cd /scratch/xl4358/phase1_far_dependency_smallcpu
/scratch/xl4358/phase1_env/bin/python register_recovered_evaluation.py
