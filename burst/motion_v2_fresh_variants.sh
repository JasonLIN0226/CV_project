#!/bin/bash
# Motion v2: label and state-distribution variants of our deletion-cost scorer on the fresh scenes
# (scripts/train_scorer/motion_v2_fresh_variants.py), chosen on fresh dev. Needs burst/motion_v2_fresh.sh's
# scenes and loss tables. Validation is read only with BM_VALIDATE=1 (appended to the validation ledger).
#
#     BM_CHAIN=4 bash burst/submit_gaps.sh motion_variants
#     BM_VALIDATE=1 BM_CHAIN=2 bash burst/submit_gaps.sh motion_variants
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/cluster.sh"
bm_activate_venv
cd "$BM_REPO"
export BM_MOTION_ROOT=${BM_MOTION_ROOT:-/scratch/$USER}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT="$BM_SCRATCH/motion_v2_fresh"
python scripts/train_scorer/motion_v2_fresh_variants.py --config configs/motion_v2/frozen.yaml --seed 0 --device cuda \
    --threads 4 --out "$ROOT/variants" --fresh "$ROOT/scenes" --tables "$ROOT/scorer/tables" ${BM_VALIDATE:+--validate}
echo MOTION_VARIANTS_DONE
