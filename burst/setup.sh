#!/bin/bash
# One-time setup on the course cluster. Run it in a terminal of an OOD session (a compute node
# with internet access); it takes about 15 minutes and can be run again safely.
#
#     bash burst/setup.sh
#
# Installs uv if needed, builds the Python 3.11 environment from
# requirements.txt, and caches the pinned I-JEPA encoder. Settings: burst/cluster.sh.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/cluster.sh"
mkdir -p "$BM_SCRATCH/logs" "$HF_HOME" || {
    echo "Cannot write to $BM_SCRATCH or $HF_HOME. Set BM_SCRATCH and BM_HF_HOME to writable paths in burst/.env." >&2; exit 1; }
echo "== $(hostname) $(date -Is)  repo: $BM_REPO  scratch: $BM_SCRATCH  venv: $BM_VENV"

export PATH="$PATH:$HOME/.local/bin"
if ! command -v uv >/dev/null; then
    curl -fsSL https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL="$HOME/.local/bin" sh
fi
if [ ! -f "$BM_VENV/pyvenv.cfg" ]; then
    uv venv --managed-python --python 3.11 "$BM_VENV"
fi
bm_activate_venv
if ! python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))'; then
    echo "Burst needs Python 3.11. Set BM_VENV to a new venv path in burst/.env and rerun setup." >&2
    exit 1
fi
uv pip install --python "$BM_VENV/bin/python" --torch-backend=cu121 -r "$BM_REPO/requirements.txt" -e "$BM_REPO"
uv pip check --python "$BM_VENV/bin/python"

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
