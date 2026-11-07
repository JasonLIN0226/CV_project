#!/bin/bash
# Hidden-CoG PushT v4: frozen I-JEPA (ViT-H/14, pinned revision) encoding of train+validation,
# then class-readability probes. Submit from this directory:  sbatch run_encode_probe.sbatch
# First run creates .venv here (needs internet for pip); later runs reuse it.
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=pusht_cog_v4_ijepa
#SBATCH --output=pusht_cog_v4_ijepa_%j.out
set -euo pipefail
PKG=${SLURM_SUBMIT_DIR:-$PWD}
cd "$PKG"
test -f code/experiments/phase2/probe_pusht_cog_features.py || { echo "Submit from the package directory"; exit 1; }
hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [ ! -x .venv/bin/python ]; then
  PY=""
  for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null && "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
      PY=$cand; break
    fi
  done
  if [ -z "$PY" ] && command -v module >/dev/null; then
    module load python 2>/dev/null || module load anaconda3 2>/dev/null || true
    command -v python3 >/dev/null && python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' && PY=python3
  fi
  [ -n "$PY" ] || { echo "Need Python >= 3.10 on this node"; exit 1; }
  "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/python -c "import torch, transformers, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'numpy', numpy.__version__)"

export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONNOUSERSITE=1
DATA=$PKG/data/pusht-cog-v4-20261107
OUT=$PKG/output/pusht-cog-v4-20261107-ijepa
mkdir -p "$PKG/output"
if [ ! -f "$OUT/encoding_manifest.json" ]; then
  rm -rf -- "$OUT"   # partial output from an interrupted run
  .venv/bin/python -I code/experiments/phase2/probe_pusht_cog_features.py encode \
    --data "$DATA" --features "$OUT" --cache-dir "$PKG/hf_cache/ijepa" --batch-size 32
fi
.venv/bin/python -I code/experiments/phase2/probe_pusht_cog_features.py probe --data "$DATA" --features "$OUT"
echo "Done. Results: $OUT/probe/probe_results.json"
