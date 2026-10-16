#!/usr/bin/env bash
# One allocated course CPU job; never changes v3 checkpoints or evaluation files.
set -euo pipefail
if [[ "${1:-}" == --run ]]; then
  cd "${PROJECT_DIR:?Verified v4 project directory required}"
  export PYTHONPATH="$PROJECT_DIR/src"
  export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
  export HF_HOME=/scratch/xl4358/phase1_hf_cache
  printf 'Slurm job=%s host=%s\n' "$SLURM_JOB_ID" "$(hostname)"
  hostname -I
  ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
  exec /scratch/xl4358/phase1_env/bin/python scripts/v4_diagnose.py \
    --development-output /scratch/xl4358/phase1_v3_dev \
    --output /scratch/xl4358/phase1_v4_diagnostics
fi
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ "$TASK_ROOT" == /scratch/xl4358/phase1_jepa_memory_v4 ]] || { echo 'Submit from the isolated, verified v4 source'; exit 2; }
export PROJECT_DIR="$TASK_ROOT"
mkdir -p logs
export PATH=/opt/slurm/bin:$PATH
TASK_JOBS="$(squeue -u xl4358 -h -n cv-phase1,cv-phase1-v4-diag)"
[[ -z "$TASK_JOBS" ]] || { echo 'An existing science job must be reused'; exit 2; }
sbatch --parsable --export=ALL --job-name=cv-phase1-v4-diag --account=csci_ga_2271-2026fa \
  --partition=n2c48m24 --mem=16G --cpus-per-task=4 --time=00:45:00 \
  --output=logs/%x-%j.out --error=logs/%x-%j.err "$TASK_ROOT/scripts/submit_diagnostics.sh" --run
