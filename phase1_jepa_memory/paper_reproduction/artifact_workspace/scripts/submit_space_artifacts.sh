#!/usr/bin/env bash
#SBATCH --job-name=cv-space-train-artifacts
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xl4358/phase1_space_paper_artifacts/logs/artifacts-%j.out
#SBATCH --error=/scratch/xl4358/phase1_space_paper_artifacts/logs/artifacts-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME=/scratch/xl4358/phase1_hf_cache
ROOT=/scratch/xl4358/phase1_space_paper_artifacts
export PYTHONPATH=/scratch/xl4358/phase1_space_reference_vendor:$ROOT/src:$ROOT/scripts
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
cat /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python "$ROOT/scripts/run_space_artifacts.py" \
 --root "$ROOT" --protocol "$ROOT/artifact_protocol.json" \
 --output "/scratch/xl4358/phase1_space_artifact_results/job_$SLURM_JOB_ID"
