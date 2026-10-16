#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --run ]]; then
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
  TASK_ROOT="${PROJECT_DIR:?Slurm project directory missing}"
else
  TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "$TASK_ROOT"
: "${OUTPUT_DIR:?export OUTPUT_DIR=/scratch/your_netid/phase1_pilot}"
: "${PYTHON_BIN:?export PYTHON_BIN=/absolute/verified/python-or-launcher}"
[[ "$OUTPUT_DIR" = /* && "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || { echo 'Use absolute output and executable interpreter/launcher'; exit 2; }
export PROJECT_DIR="$TASK_ROOT" OUTPUT_DIR PYTHON_BIN
mkdir -p logs "$OUTPUT_DIR"
if [[ "${1:-}" == --run ]]; then
  export MPLCONFIGDIR="$OUTPUT_DIR/.matplotlib" HF_HOME="${HF_HOME:-$OUTPUT_DIR/huggingface}"
  printf 'Slurm job=%s host=%s\n' "${SLURM_JOB_ID:-unknown}" "$(hostname)"
  hostname -I 2>/dev/null || true
  if [[ -r /etc/ssh/ssh_host_ed25519_key.pub ]]; then ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub; fi
  export PYTHONPATH="$TASK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" CUBLAS_WORKSPACE_CONFIG=:4096:8
  # v3 builds its effective profile before preflight. A base-v2 preflight here
  # would create a conflicting resolved_config in the fresh confirmation root.
  if [[ "${TASK_COMMAND:-run-all}" != v3-* ]]; then
    "$PYTHON_BIN" -m phase1.cli preflight --config "${CONFIG_PATH:-configs/phase1.yaml}" --output "$OUTPUT_DIR" --device "${TASK_DEVICE:-cuda}" --check-checkpoint
  fi
  exec bash scripts/run.sh phase1 --output "$OUTPUT_DIR" --device "${TASK_DEVICE:-cuda}"
fi
case "${1:-l4}" in
  l4) TASK_PARTITION=g2-standard-12; TASK_RESOURCES=(--gres=gpu:1 --mem=60G) ;;
  a100) TASK_PARTITION=c12m85-a100-1; TASK_RESOURCES=(--gres=gpu:1 --mem=60G) ;;
  cpu) TASK_PARTITION=n2c48m24; TASK_RESOURCES=(--mem=16G); export TASK_COMMAND="${TASK_COMMAND:-finish}" TASK_DEVICE=cpu ;;
  *) echo 'Usage: bash scripts/submit.sh l4|a100|cpu'; exit 2 ;;
esac
export TASK_DEVICE="${TASK_DEVICE:-cuda}" TASK_COMMAND="${TASK_COMMAND:-run-all}"
TASK_DEPENDENCY=()
[[ -z "${AFTER_JOB:-}" ]] || TASK_DEPENDENCY=(--dependency="afterok:$AFTER_JOB")
sbatch --parsable --export=ALL --job-name=cv-phase1 --account=csci_ga_2271-2026fa \
  --partition="$TASK_PARTITION" "${TASK_RESOURCES[@]}" ${TASK_DEPENDENCY[@]+"${TASK_DEPENDENCY[@]}"} --time="${TIME_LIMIT:-02:00:00}" \
  --cpus-per-task=4 --output=logs/%x-%j.out --error=logs/%x-%j.err \
  --requeue --signal=B:TERM@120 "$TASK_ROOT/scripts/submit.sh" --run
