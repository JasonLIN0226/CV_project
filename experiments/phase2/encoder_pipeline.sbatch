#!/bin/bash
#SBATCH --job-name=cv-phase2-encode
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@60

set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: sbatch --chdir=TASK_ROOT --output=TASK_ROOT/slurm-%j.log encoder_pipeline.sbatch TASK_ROOT EXPECTED_COMMIT" >&2
  exit 2
fi
TASK_ROOT="$(cd -- "$1" && pwd -P)"
EXPECTED_COMMIT="$2"
PYTHON="$TASK_ROOT/venv/bin/python"
DRIVER="$TASK_ROOT/repo/experiments/phase2/run_hpc_pipeline.py"
[[ -x "$PYTHON" && -f "$DRIVER" ]] || { echo "Prepared venv/repo missing" >&2; exit 2; }
cd -- "$TASK_ROOT"
# exec gives the driver TERM directly; it records failure and stops child groups.
exec "$PYTHON" "$DRIVER" --task-root "$TASK_ROOT" --expected-commit "$EXPECTED_COMMIT"
