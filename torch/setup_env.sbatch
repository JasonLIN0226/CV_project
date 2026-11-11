#!/bin/bash
#SBATCH --job-name=cvmem_env
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
# Builds the conda environment used by every job of this repository and caches the pinned I-JEPA encoder.
# Submit with:  bash torch/submit_setup_env.sh   (it supplies the account, the log file and the settings of torch/cluster.sh)
set -euo pipefail
: "${BM_REPO:?submit with torch/submit_setup_env.sh}" "${BM_SCRATCH:?}" "${BM_CONDA_SH:?}" "${BM_CONDA_ENV:?}" "${HF_HOME:?}"
ENV=$BM_CONDA_ENV
ROOT=$BM_REPO
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/.pip-cache}"
echo "== host $(hostname) $(date -Is)"

source "$BM_CONDA_SH"
[ -x "$ENV/bin/python" ] || conda create -y -p "$ENV" -c conda-forge --override-channels python=3.11 pip
conda activate "$ENV"
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
python -m pip install -r "$ROOT/requirements.txt"      # same torch pin; adds everything else, including the PushT simulator
python -m pip check

python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
import numpy, torch, transformers, huggingface_hub
from huggingface_hub import snapshot_download

root = Path(sys.argv[1])
enc = json.loads((root / "configs/gate/dataset_config_template.json").read_text())["encoder"]
snapshot = snapshot_download(repo_id=enc["repository"], revision=enc["revision"])
print(json.dumps({"status": "ENVIRONMENT_READY", "python": sys.version.split()[0], "torch": torch.__version__,
                  "numpy": numpy.__version__, "transformers": transformers.__version__,
                  "huggingface_hub": huggingface_hub.__version__, "ijepa_snapshot": snapshot}, indent=2))
PY
echo "== done $(date -Is)"
