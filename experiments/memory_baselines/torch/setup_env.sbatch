#!/bin/bash
#SBATCH --job-name=cvmem_env
#SBATCH --account=torch_pr_147_courant
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ys4907/cv_project_runs/logs/%x_%j.log
# Builds the miniconda env used by every memory-baseline job and caches the pinned I-JEPA encoder.
set -euo pipefail
CONDA=/scratch/ys4907/miniconda3
ENV=$CONDA/envs/cvmem
ROOT=/scratch/ys4907/CV_project
export HF_HOME=/scratch/ys4907/hf_cache
export PIP_CACHE_DIR=/scratch/ys4907/.pip-cache
echo "== host $(hostname) $(date -Is)"

source "$CONDA/etc/profile.d/conda.sh"
[ -x "$ENV/bin/python" ] || conda create -y -p "$ENV" -c conda-forge --override-channels python=3.11 pip
conda activate "$ENV"
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
# Versions follow requirements-phase2-hpc.txt (PushT simulator packages left out: no PushT model exists yet).
python -m pip install numpy==1.26.4 transformers==5.5.4 huggingface-hub==1.11.0 safetensors==0.7.0 \
  PyYAML==6.0.3 Pillow==11.3.0 matplotlib==3.9.4 pytest==7.4.4
python -m pip check

python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
import numpy, torch, transformers, huggingface_hub
from huggingface_hub import snapshot_download

root = Path(sys.argv[1])
enc = json.loads((root / "results/gate_retention_comparison_20261029/dataset_config_template.json").read_text())["encoder"]
snapshot = snapshot_download(repo_id=enc["repository"], revision=enc["revision"])
print(json.dumps({"status": "ENVIRONMENT_READY", "python": sys.version.split()[0], "torch": torch.__version__,
                  "numpy": numpy.__version__, "transformers": transformers.__version__,
                  "huggingface_hub": huggingface_hub.__version__, "ijepa_snapshot": snapshot}, indent=2))
PY
echo "== done $(date -Is)"
