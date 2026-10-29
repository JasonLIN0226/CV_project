#!/bin/bash
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --job-name=clean-future-trainval
#SBATCH --output=/scratch/xl4358/clean_future_retention_audit_20261022/logs/clean-future-%j.out
#SBATCH --error=/scratch/xl4358/clean_future_retention_audit_20261022/logs/clean-future-%j.err

set -euo pipefail

: "${SLURM_JOB_ACCOUNT:?A valid course account must be supplied with sbatch --account}"

BASE=/scratch/xl4358/clean_future_retention_audit_20261022
CODE="$BASE/code/CV_project"
PYTHON="$BASE/venv/bin/python"
EXP="$CODE/experiments/retention_structured_noise_audit"
SOURCE=/scratch/xl4358/retention_structured_noise_audit_20261021/code/CV_project/results/retention_structured_noise_audit
CLEAN="$SOURCE/full_clean_v4"
LINES="$SOURCE/full_lines_v4"
DATA="$BASE/results/full_clean_future_v1"
REFERENCE="$BASE/results/reference"
CANDIDATE="$BASE/results/candidate"
FEATURES_READY="$BASE/results/features_ready"
EVALUATION="$BASE/results/validation_evaluation"
PREFLIGHT="$BASE/results/clean_future_zero_gpu_preflight.json"

test -f "$EXP/clean_future_source_lock.json"
test -f "$CLEAN/dataset/manifest.json"
test -f "$LINES/dataset/manifest.json"

cd "$CODE"
export HF_HOME="$BASE/hf_cache"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE/experiments/predictor/control/src:$EXP"

"$PYTHON" - "$EXP" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
lock = json.loads((root / "clean_future_source_lock.json").read_text())
if lock["status"] != "FROZEN_CLEAN_FUTURE_TRAIN_VALIDATION_V1":
    raise SystemExit("unexpected clean-future source lock")
for relative, expected in lock["files"].items():
    actual = hashlib.sha256((root.parents[1] / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"source mismatch: {relative}")
print("clean-future source lock verified", len(lock["files"]))
PY

if [ ! -f "$DATA/dataset/manifest.json" ]; then
  "$PYTHON" "$EXP/materialize_clean_future.py" \
    --clean-root "$CLEAN" \
    --lines-root "$LINES" \
    --output "$DATA" \
    --device cuda
fi

"$PYTHON" "$EXP/clean_future_preflight.py" \
  --config "$DATA/resolved_config.json" \
  --clean-root "$CLEAN" \
  --lines-root "$LINES" \
  --output "$PREFLIGHT"

"$PYTHON" "$EXP/fit_features.py" \
  --config "$DATA/resolved_config.json" \
  --output "$FEATURES_READY" \
  --probe-splits train,validation

"$PYTHON" "$EXP/train_reference.py" \
  --config "$DATA/resolved_config.json" \
  --output "$REFERENCE"

if [ ! -d "$CANDIDATE" ]; then
  "$PYTHON" "$EXP/prepare_candidate.py" \
    --source-config "$DATA/resolved_config.json" \
    --output "$CANDIDATE"
fi

"$PYTHON" "$EXP/train_candidate.py" \
  --config "$CANDIDATE/resolved_config.json" \
  --output "$CANDIDATE" \
  --source-lock "$EXP/clean_future_source_lock.json"

"$PYTHON" "$EXP/evaluate_clean_future_validation.py" \
  --reference-config "$DATA/resolved_config.json" \
  --candidate-config "$CANDIDATE/resolved_config.json" \
  --candidate-root "$CANDIDATE" \
  --output "$EVALUATION" \
  --device cuda
