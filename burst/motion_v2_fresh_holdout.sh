#!/bin/bash
# Motion v2: the dev-chosen deletion-cost scorers on a second set of fresh scenes (pairs 608 to 703), paired with
# the attention policy (scripts/evaluate/motion_v2_fresh_holdout.py). Reads no validation data.
#
#     BM_CHAIN=2 bash burst/submit_gaps.sh motion_holdout
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/cluster.sh"
bm_activate_venv
cd "$BM_REPO"
export BM_MOTION_ROOT=${BM_MOTION_ROOT:-/scratch/$USER}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT="$BM_SCRATCH/motion_v2_fresh"
python scripts/evaluate/motion_v2_fresh_holdout.py --config configs/motion_v2/frozen.yaml --seed 0 --device cuda \
    --threads 4 --out "$ROOT/holdout" --fresh "$ROOT/scenes" --tables "$ROOT/scorer/tables" --variants "$ROOT/variants"
echo MOTION_HOLDOUT_DONE
