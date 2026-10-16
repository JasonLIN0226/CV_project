#!/usr/bin/env bash
# One isolated course CPU allocation: frozen Motion mechanism, then conditional Gate trial.
set -euo pipefail
if [[ "${1:-}" == --run ]]; then
  cd "${PROJECT_DIR:?Verified isolated v4 development checkout required}"
  export PYTHONPATH="$PROJECT_DIR/src"
  export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
  export HF_HOME=/scratch/xl4358/phase1_hf_cache
  printf 'Slurm job=%s host=%s\n' "$SLURM_JOB_ID" "$(hostname)"
  hostname -I
  ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
  /scratch/xl4358/phase1_env/bin/python scripts/verify_v4_stage.py
  /scratch/xl4358/phase1_env/bin/python -m phase1.motion_mechanism \
    --reference-output /scratch/xl4358/phase1_v3_dev/motion/reference \
    --wide-output /scratch/xl4358/phase1_v3_dev/motion/motion_wide_span \
    --output /scratch/xl4358/phase1_v4_mechanism \
    --frozen-source-directory /scratch/xl4358/phase1_jepa_memory
  exec /scratch/xl4358/phase1_env/bin/python scripts/v4_develop_gate.py \
    --source-output /scratch/xl4358/phase1_v2 \
    --frozen-v3-root /scratch/xl4358/phase1_v3_dev \
    --frozen-source-directory /scratch/xl4358/phase1_jepa_memory \
    --diagnostic-root /scratch/xl4358/phase1_v4_diagnostics \
    --output /scratch/xl4358/phase1_v4_dev \
    --protocol configs/phase1_v4_gate_protocol.json
fi
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ "$TASK_ROOT" == /scratch/xl4358/phase1_jepa_memory_v4_development ]] || { echo 'Use the isolated verified development checkout'; exit 2; }
export PROJECT_DIR="$TASK_ROOT"
export PATH=/opt/slurm/bin:$PATH
mkdir -p logs
TASK_JOBS="$(squeue -u xl4358 -h -o '%j' | awk '/^cv-phase1($|-)/')"
[[ -z "$TASK_JOBS" ]] || { echo 'Reuse existing active Phase1 science'; exit 2; }
sbatch --parsable --export=ALL --job-name=cv-phase1-v4-dev --account=csci_ga_2271-2026fa \
  --partition=n2c48m24 --mem=16G --cpus-per-task=4 --time=04:00:00 \
  --output=logs/%x-%j.out --error=logs/%x-%j.err "$TASK_ROOT/scripts/submit_v4_development.sh" --run
