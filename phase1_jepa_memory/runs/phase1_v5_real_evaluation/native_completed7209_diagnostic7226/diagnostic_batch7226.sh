#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-real-diagnostics
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/scratch/xl4358/phase1_v5_real_diagnostics_7209/logs/diagnostic-%j.out
#SBATCH --error=/scratch/xl4358/phase1_v5_real_diagnostics_7209/logs/diagnostic-%j.err
set -euo pipefail
export HF_HOME=/scratch/xl4358/phase1_hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
TASK_CONTROL=/scratch/xl4358/phase1_v5_real_diagnostics_7209
cd "$TASK_CONTROL"
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python scripts/v5_run_frozen_real_diagnostics.py --control "$TASK_CONTROL" --source-lock-sha 8d00977db2202c22f303badc1d3008125426224e8147b6692238ddf39625419a --output /scratch/xl4358/phase1_v5_real_diagnostic_results/job_${SLURM_JOB_ID}
