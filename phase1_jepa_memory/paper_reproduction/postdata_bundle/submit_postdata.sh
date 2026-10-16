#!/usr/bin/env bash
#SBATCH --job-name=cv-space-data-binding
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/xl4358/phase1_space_postdata_handoff/logs/handoff-%j.out
#SBATCH --error=/scratch/xl4358/phase1_space_postdata_handoff/logs/handoff-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
cat /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python /scratch/xl4358/phase1_space_postdata_handoff/postdata_handoff.py
