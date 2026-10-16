#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TASK_ROOT"
TASK_MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
case "$TASK_MODE" in
  smoke|phase1) TASK_COMMAND="${TASK_COMMAND:-run-all}"; TASK_CONFIG="${CONFIG_PATH:-configs/$TASK_MODE.yaml}" ;;
  preflight) TASK_COMMAND=preflight; TASK_CONFIG="${CONFIG_PATH:-configs/smoke.yaml}" ;;
  *) echo 'Usage: bash scripts/run.sh smoke|phase1|preflight [CLI options]'; exit 2 ;;
esac
export PYTHONPATH="$TASK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_EXTRA=()
if [[ "$TASK_COMMAND" == v3-develop ]]; then
  TASK_EXTRA=(--source-output "${SOURCE_OUTPUT:?Original v2 output required}" --task "${EXPERIMENT_TASK:?gate or motion required}")
elif [[ "$TASK_COMMAND" == v3-confirm* ]]; then
  TASK_EXTRA=(--development-output "${DEVELOPMENT_OUTPUT:?Actual development output required}")
fi
# exec preserves signal delivery through the Slurm -> shell -> Python chain.
exec "${PYTHON_BIN:-python3}" -m phase1.cli "$TASK_COMMAND" --config "$TASK_CONFIG" ${TASK_EXTRA[@]+"${TASK_EXTRA[@]}"} "$@"
