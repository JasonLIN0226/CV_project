#!/usr/bin/env bash
#SBATCH --job-name=cv-space-model-binding
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/xl4358/phase1_space_postpredictor_handoff/logs/handoff-%j.out
#SBATCH --error=/scratch/xl4358/phase1_space_postpredictor_handoff/logs/handoff-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
ROOT=/scratch/xl4358/phase1_space_postpredictor_handoff
export PYTHONPATH=/scratch/xl4358/phase1_space_reference_vendor:$ROOT:/scratch/xl4358/phase1_space_paper_artifacts/src
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
cat /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python "$ROOT/postpredictor_handoff.py" --root "$ROOT"
