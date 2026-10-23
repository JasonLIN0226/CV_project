#!/bin/bash
#SBATCH --job-name=cv-age-control-a1
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xl4358/phase1_age_control_a1/logs/age-%j.out
#SBATCH --error=/scratch/xl4358/phase1_age_control_a1/logs/age-%j.err
set -euo pipefail
cd /scratch/xl4358/phase1_age_control_a1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
export HF_HOME=/scratch/xl4358/phase1_hf_cache
printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
hostname
hostname -I
cat /etc/ssh/ssh_host_ed25519_key.pub
/scratch/xl4358/phase1_env/bin/python -u run.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/scratch/xl4358/phase1_env/bin/python -u independent_audit.py "/scratch/xl4358/phase1_age_control_a1/output/job_${SLURM_JOB_ID}"
