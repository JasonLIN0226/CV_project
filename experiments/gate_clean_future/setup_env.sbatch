#!/bin/bash
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --job-name=clean-future-env
#SBATCH --output=/home/xl4358/clean-future-setup-%j.out
#SBATCH --error=/home/xl4358/clean-future-setup-%j.err

set -euxo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin

: "${SLURM_JOB_ACCOUNT:?A valid Torch project account must be supplied}"
printf 'slurm_account=%s\n' "$SLURM_JOB_ACCOUNT"

BASE=/scratch/xl4358/clean_future_gate_20261022
CODE="$BASE/code/CV_project"
VENV=/scratch/xl4358/clean_future_retention_audit_20261022/venv
PYTHON="$VENV/bin/python"
UV="$VENV/bin/uv"
HF_HOME="$BASE/hf_cache"

write_stage() {
  local stage="$1"
  local temporary="$BASE/setup_env_state.json.tmp"
  printf '{"stage":"%s","time_utc":"%s"}\n' \
    "$stage" "$(/usr/bin/date -u +%Y-%m-%dT%H:%M:%SZ)" > "$temporary"
  mv "$temporary" "$BASE/setup_env_state.json"
}

trap 'rc=$?; write_stage "FAILED_${CURRENT_STAGE}_rc${rc}"; exit "$rc"' ERR

CURRENT_STAGE=check_paths
write_stage "$CURRENT_STAGE"
test -x "$PYTHON"
test -x "$UV"

# Reuse the intact worker-local virtual environment from the interrupted prior
# setup. Only the incomplete wheels are repaired; the frozen scientific source
# is untouched.
CURRENT_STAGE=sync_dependencies
write_stage "$CURRENT_STAGE"
UV_PROJECT_ENVIRONMENT="$VENV" \
  "$UV" sync \
  --project "$CODE" \
  --frozen \
  --extra model

CURRENT_STAGE=install_pytest
write_stage "$CURRENT_STAGE"
"$UV" pip install \
  --python "$PYTHON" \
  "pytest==7.4.4"

CURRENT_STAGE=pip_check
write_stage "$CURRENT_STAGE"
"$PYTHON" -m pip check

CURRENT_STAGE=download_ijepa
write_stage "$CURRENT_STAGE"
HF_HOME="$HF_HOME" \
  "$PYTHON" - <<'PY'
import json
import platform
from pathlib import Path

import huggingface_hub
import matplotlib
import numpy
import PIL
import pytest
import safetensors
import torch
import transformers
import yaml
from huggingface_hub import snapshot_download

code = Path("/scratch/xl4358/clean_future_gate_20261022/code/CV_project")
config = json.loads(
    (code / "results/gate_clean_future_20261022/resolved_config.json").read_text()
)["encoder"]
snapshot = snapshot_download(
    repo_id=config["repository"],
    revision=config["revision"],
)
snapshot_path = Path(snapshot)
if not snapshot_path.is_dir():
    raise RuntimeError("Pinned I-JEPA snapshot was not downloaded")

result = {
    "status": "ENVIRONMENT_READY",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "numpy": numpy.__version__,
    "Pillow": PIL.__version__,
    "PyYAML": yaml.__version__,
    "matplotlib": matplotlib.__version__,
    "transformers": transformers.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "safetensors": safetensors.__version__,
    "pytest": pytest.__version__,
    "python_executable": str(Path(__import__("sys").executable).resolve()),
    "ijepa_cache_reused": True,
    "ijepa_repository": config["repository"],
    "ijepa_revision": config["revision"],
    "ijepa_snapshot": str(snapshot_path),
}
path = Path("/scratch/xl4358/clean_future_gate_20261022/environment_ready.json")
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(result, indent=2) + "\n")
temporary.replace(path)
print(path.read_text(), end="")
PY

CURRENT_STAGE=READY
write_stage "$CURRENT_STAGE"
