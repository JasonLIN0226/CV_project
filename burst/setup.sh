#!/bin/bash
# One-time setup on the course cluster. Run it in a terminal of an OOD session (a compute node
# with internet access); it takes about 15 minutes and can be run again safely.
#
#     bash burst/setup.sh
#
# Installs Miniforge if there is no conda, builds the Python 3.11 environment from
# requirements.txt, and caches the pinned I-JEPA encoder. Settings: burst/cluster.sh.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/cluster.sh"
mkdir -p "$BM_SCRATCH/logs" "$HF_HOME" || {
    echo "cannot write to $BM_SCRATCH or $HF_HOME: put BM_SCRATCH, BM_HF_HOME, BM_CONDA_SH and BM_CONDA_ENV lines for a folder you own into burst/.env" >&2; exit 1; }
echo "== $(hostname) $(date -Is)  repo: $BM_REPO  scratch: $BM_SCRATCH  env: $BM_CONDA_ENV"

if [ ! -f "$BM_CONDA_SH" ]; then
    root=${BM_CONDA_SH%/etc/profile.d/conda.sh}
    echo "== installing Miniforge into $root"
    installer=$(mktemp -d)/miniforge.sh
    curl -fsSL -o "$installer" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$installer" -b -p "$root"
fi
set +u; source "$BM_CONDA_SH"; set -u
[ -x "$BM_CONDA_ENV/bin/python" ] || conda create -y -p "$BM_CONDA_ENV" -c conda-forge --override-channels python=3.11 pip
set +u; conda activate "$BM_CONDA_ENV"; set -u
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
python -m pip install -r "$BM_REPO/requirements.txt"
python -m pip check || echo "pip check reported conflicts (see above)"

python - "$BM_REPO" <<'PY'
import json, sys
from pathlib import Path
import numpy, torch, transformers
from huggingface_hub import snapshot_download

root = Path(sys.argv[1])
enc = json.loads((root / "configs/gate/dataset_config_template.json").read_text())["encoder"]
snapshot = snapshot_download(repo_id=enc["repository"], revision=enc["revision"])
print(json.dumps({"status": "ENVIRONMENT_READY", "python": sys.version.split()[0], "torch": torch.__version__,
                  "numpy": numpy.__version__, "transformers": transformers.__version__,
                  "cuda_available_here": torch.cuda.is_available(), "ijepa_snapshot": snapshot}, indent=2))
PY
if [ ! -s "$HF_HOME/token" ]; then
    echo "== no Hugging Face token yet. The Gate bundle is private; log in once with:"
    echo "   bash burst/hf_login.sh"
fi
echo "== done $(date -Is)"
