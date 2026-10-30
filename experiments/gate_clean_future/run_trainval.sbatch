#!/bin/bash
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --job-name=clean-future-gate
#SBATCH --output=/scratch/xl4358/clean_future_gate_20261022/logs/slurm-%j.out
#SBATCH --error=/scratch/xl4358/clean_future_gate_20261022/logs/slurm-%j.err

set -euo pipefail

: "${SLURM_JOB_ACCOUNT:?A valid Torch project account must be supplied}"
printf 'slurm_account=%s\n' "$SLURM_JOB_ACCOUNT"

BASE=/scratch/xl4358/clean_future_gate_20261022
CODE="$BASE/code/CV_project"
RUN="$BASE/run-$SLURM_JOB_ID"
PYTHON=/scratch/xl4358/clean_future_retention_audit_20261022/venv/bin/python
DATASET="$CODE/results/gate_clean_future_20261022"
EXP="$CODE/experiments/gate_clean_future"

test -x "$PYTHON"
test ! -e "$RUN"
mkdir -p "$RUN"
cd "$CODE"
export HF_HOME="$BASE/hf_cache"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE/experiments/predictor/control/src:$CODE/experiments/retention_position_noise_audit:$CODE/experiments/retention_structured_noise_audit:$EXP"

"$PYTHON" - "$BASE/environment_ready.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
result = json.loads(path.read_text())
if result.get("status") != "ENVIRONMENT_READY":
    raise SystemExit("environment readiness failed")
print("environment readiness PASS", result["python_executable"])
PY

"$PYTHON" - "$EXP" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
lock = json.loads((root / "source_lock.json").read_text())
if lock.get("status") != "FROZEN_CLEAN_FUTURE_TRAINVAL_V1":
    raise SystemExit("unexpected source-lock status")
for name, expected in lock["files"].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit("frozen source changed: " + name)
print("clean-future source lock PASS")
PY

"$PYTHON" - "$DATASET" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config_path = root / "resolved_config.json"
config = json.loads(config_path.read_text())
reference_config = json.loads(
    Path(
        "/scratch/xl4358/retention_structured_noise_audit_20261021/"
        "code/CV_project/results/retention_structured_noise_audit/"
        "full_clean_v4/resolved_config.json"
    ).read_text()
)
manifest = json.loads((root / "dataset/manifest.json").read_text())
audit = json.loads((root / "dataset/split_audit.json").read_text())
if {row["split"] for row in manifest["records"]} != {"train", "validation"}:
    raise SystemExit("unexpected generated splits")
if manifest["generation"]["test_generated"] or manifest["generation"]["test_accessed"]:
    raise SystemExit("test boundary violated")
if not audit["passed"]:
    raise SystemExit("dataset audit failed")
if any("test_" in path.relative_to(root).as_posix() for path in root.rglob("*")):
    raise SystemExit("test-named artifact exists")
if config["decision"] != reference_config["decision"]:
    raise SystemExit("Gate thresholds changed")
if config["training"] != reference_config["training"]:
    raise SystemExit("Training sample/seed/epoch protocol changed")
if config["model"] != reference_config["model"]:
    raise SystemExit("Predictor architecture changed")
if config["data"]["pairs"] != reference_config["data"]["pairs"]:
    raise SystemExit("Declared split sizes changed")
config["output"] = str(root)
config["device"] = "cuda"
config["threads"] = 4
config_path.write_text(json.dumps(config, indent=2) + "\n")
print("train/validation-only dataset audit PASS", len(manifest["records"]))
PY

"$PYTHON" -m pytest -q "$EXP/test_protocol.py"

"$PYTHON" "$EXP/fit_features.py" \
  --config "$DATASET/resolved_config.json" \
  --output "$RUN/features"

"$PYTHON" "$CODE/experiments/retention_structured_noise_audit/train_reference.py" \
  --config "$DATASET/resolved_config.json" \
  --output "$RUN/training"

"$PYTHON" "$EXP/evaluate_validation.py" \
  --config "$DATASET/resolved_config.json" \
  --output "$RUN/validation" \
  --device cuda

printf 'CLEAN_FUTURE_TRAINVAL_COMPLETE=%s\n' "$RUN"
