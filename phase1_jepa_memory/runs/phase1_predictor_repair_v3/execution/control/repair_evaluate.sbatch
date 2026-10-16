#!/usr/bin/env bash
#SBATCH --job-name=cv-phase1-repair-audit
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xl4358/phase1_predictor_repair_v3/execution/logs/evaluate-%j.out
#SBATCH --error=/scratch/xl4358/phase1_predictor_repair_v3/execution/logs/evaluate-%j.err
set -euo pipefail
ROOT=/scratch/xl4358/phase1_predictor_repair_v3/execution
CONTROL="$ROOT/control"
: "${TRAINING_JOB:?Actual parent receipt required}"
export PYTHONPATH="$CONTROL/src"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=/scratch/xl4358/phase1_hf_cache
hostname
hostname -I
cat /etc/ssh/ssh_host_ed25519_key.pub
PARENT=$(/opt/slurm/bin/sacct -n -X -j "$TRAINING_JOB" --format=State,ExitCode -P)
case "$PARENT" in COMPLETED\|0:0*) ;; *) echo 'Parent is not actual COMPLETED0:0'; exit 3 ;; esac
/scratch/xl4358/phase1_env/bin/python "$CONTROL/run_repair.py" evaluate \
    --control "$CONTROL" --training-root "$ROOT/training/job_$TRAINING_JOB" \
    --output "$ROOT/evaluation/job_$SLURM_JOB_ID"
