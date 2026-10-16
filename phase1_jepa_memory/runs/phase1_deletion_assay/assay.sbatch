#!/bin/bash
#SBATCH --job-name=cv-phase1-delete-assay
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/xl4358/phase1_deletion_assay/logs/assay-%j.out
#SBATCH --error=/scratch/xl4358/phase1_deletion_assay/logs/assay-%j.err
set -euo pipefail
cd /scratch/xl4358/phase1_deletion_assay
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
export HF_HOME=/scratch/xl4358/phase1_hf_cache
hostname
hostname -I
cat /etc/ssh/ssh_host_ed25519_key.pub
/scratch/xl4358/phase1_env/bin/python -u run_assay.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/scratch/xl4358/phase1_env/bin/python -u independent_audit.py "/scratch/xl4358/phase1_deletion_assay/output/job_${SLURM_JOB_ID}"
