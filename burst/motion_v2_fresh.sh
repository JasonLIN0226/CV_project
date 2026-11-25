#!/bin/bash
# Motion v2: our deletion-cost scorer (v2) trained and selected on fresh scenes the frozen readers never saw
# (scripts/train_scorer/motion_v2_deletion_cost_v2.py --fresh; environments/motion_v2/fresh.py).
# Generates and encodes 608 fresh scene pairs (512 for training states, 96 for checkpoint selection), builds the
# all-pairs loss tables, trains three scorer seeds. Every stage is cached, so a rerun continues.
# Validation is not read unless BM_VALIDATE=1 (each such run is appended to the validation ledger).
#
#     BM_CHAIN=4 bash burst/submit_gaps.sh motion_fresh              # train, dev selection only
#     BM_VALIDATE=1 BM_CHAIN=2 bash burst/submit_gaps.sh motion_fresh   # then the single validation replay
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/cluster.sh"
bm_activate_venv
cd "$BM_REPO"
export BM_MOTION_ROOT=${BM_MOTION_ROOT:-/scratch/$USER}     # the original Motion v2 deployment folders
export CUBLAS_WORKSPACE_CONFIG=:4096:8
OUT="$BM_SCRATCH/motion_v2_fresh"
python scripts/train_scorer/motion_v2_deletion_cost_v2.py --config configs/motion_v2/frozen.yaml --seed 0 \
    --device cuda --threads 4 --out "$OUT/scorer" --fresh "$OUT/scenes" ${BM_VALIDATE:+--validate}
echo MOTION_FRESH_DONE
