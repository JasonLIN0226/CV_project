#!/bin/bash
#SBATCH --job-name=cvmem_gate_assets
#SBATCH --account=torch_pr_147_courant
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ys4907/cv_project_runs/logs/%x_%j.log
# Regenerates the clean-future Gate assets from the repo's own generator: dataset (seed 6301),
# I-JEPA features, train-only transforms and probes, and the three frozen predictors (seeds 17/29/43).
# The team's original copies live in a private HF bundle; this is a rebuild, not a download.
# Every stage resumes: rerun the same script after an interruption.
# CPU by default (the GPU queues are long and only the one-off encoding is slow on CPU). Six hours and
# 16 cores keep the job eligible for the cpu_short partition; resubmit to resume if it runs out. For a GPU:
#   sbatch --gres=gpu:1 --constraint=l40s --export=ALL,DEVICE=cuda gate_assets.sbatch
set -euo pipefail
ROOT=/scratch/ys4907/CV_project
RUNS=/scratch/ys4907/cv_project_runs
DATASET=$RUNS/gate_clean_future
SNAP=$RUNS/clean_future_control
EXP=$ROOT/experiments/gate_clean_future
export HF_HOME=/scratch/ys4907/hf_cache
DEVICE=${DEVICE:-cpu}
THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
source /scratch/ys4907/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/ys4907/miniconda3/envs/cvmem
cd "$ROOT"
export PYTHONPATH="$ROOT/experiments/predictor/control/src:$ROOT/experiments/retention_position_noise_audit:$ROOT/experiments/retention_structured_noise_audit:$EXP:$ROOT"
RUNPY="python $ROOT/experiments/memory_baselines/run_with_snapshot.py $SNAP/src"
echo "== host $(hostname) device $DEVICE threads $THREADS $(date -Is)"
[ "$DEVICE" = cpu ] || nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -d "$SNAP" ] || python "$EXP/frozen_control/materialize_snapshot.py" --output "$SNAP"

if [ ! -f "$DATASET/dataset/manifest.json" ]; then
  echo "== materialize dataset"
  [ ! -e "$DATASET" ] || { echo "partial dataset at $DATASET; remove it and resubmit"; exit 1; }
  $RUNPY "$EXP/materialize.py" \
    --source-config "$ROOT/results/gate_retention_comparison_20261029/dataset_config_template.json" \
    --output "$DATASET"
fi

python - "$DATASET" "$DEVICE" "$THREADS" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
path = root / "resolved_config.json"
c = json.loads(path.read_text())
m = json.loads((root / "dataset/manifest.json").read_text())
splits = {}
for r in m["records"]:
    splits[r["split"]] = splits.get(r["split"], 0) + 1
assert splits == {"train": 1024, "validation": 192}, splits
assert json.loads((root / "dataset/split_audit.json").read_text())["passed"]
c.update(output=str(root), device=sys.argv[2], threads=int(sys.argv[3]))
path.write_text(json.dumps(c, indent=2) + "\n")
print("dataset ok", splits, "frames", sum(r["frames"] for r in m["records"]))
PY

echo "== encode + transforms + probes $(date -Is)"
$RUNPY "$EXP/fit_features.py" --config "$DATASET/resolved_config.json" --output "$RUNS/gate_features_report"

echo "== train predictors $(date -Is)"
$RUNPY "$ROOT/experiments/retention_structured_noise_audit/train_reference.py" \
  --config "$DATASET/resolved_config.json" --output "$RUNS/gate_training_report"

echo "GATE_ASSETS_COMPLETE=$DATASET $(date -Is)"
