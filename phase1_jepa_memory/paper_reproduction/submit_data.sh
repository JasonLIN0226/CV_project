#!/usr/bin/env bash
#SBATCH --job-name=cv-space-official-data
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xl4358/phase1_space_data_acquisition/logs/data-%j.out
#SBATCH --error=/scratch/xl4358/phase1_space_data_acquisition/logs/data-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME=/scratch/xl4358/phase1_hf_cache
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH=/scratch/xl4358/phase1_space_vendor
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
cat /etc/ssh/ssh_host_ed25519_key.pub
ROOT=/scratch/xl4358/phase1_space_data_acquisition
exec /scratch/xl4358/phase1_env/bin/python "$ROOT/acquire_official_tsm.py" \
 --root "$ROOT" --output "/scratch/xl4358/phase1_space_official_data/job_$SLURM_JOB_ID"
