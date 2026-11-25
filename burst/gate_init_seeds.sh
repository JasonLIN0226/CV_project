#!/bin/bash
# Gate: our scorer and Compress to Remember (two slots) over extra initialisation seeds on the team's three
# frozen predictors (scripts/train_baselines/gate_init_seeds.py). Needs the private team bundle, so run
# `bash burst/hf_login.sh` once first. Run inside a Burst GPU session; finished units are skipped and the
# compressor resumes from its checkpoint, so rerunning after an interruption continues.
#
#     nohup bash burst/gate_init_seeds.sh [init seeds...] > /scratch/$USER/gate_init.log 2>&1 &
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/cluster.sh"
bm_activate_venv
cd "$BM_REPO"
INITS=${*:-"101 202 303"}
DEVICE=${BM_DEVICE:-cuda}                     # BM_DEVICE=cpu on a CPU session
METHODS=${BM_METHODS:-"ours pacc_recurrent"}  # BM_METHODS=ours: only our scorer (cheap on CPU)
[ -s "$BM_SCRATCH/gate_team/bundle_report.json" ] || python scripts/prepare_data/gate_team_bundle.py --work "$BM_SCRATCH/gate_team"
OUT="$BM_SCRATCH/reproduction/gate_init_seeds"
for M in $METHODS; do
    python scripts/train_baselines/gate_init_seeds.py --config configs/gate/team_bundle.yaml --seed 0 --device "$DEVICE" \
        --threads 4 --out "$OUT" --inits $INITS --methods "$M"
done
echo GATE_INIT_SEEDS_DONE
