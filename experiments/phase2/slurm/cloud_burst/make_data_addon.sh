#!/bin/bash
# Pack another dataset (+ current probe code and run script) as an add-on for an existing
# pusht_cog_v4_encode folder on the course cluster, reusing its .venv and weights.
# Usage: make_data_addon.sh DATASET_DIR OUT_DIR
# On the cluster, from the folder that contains pusht_cog_v4_encode/:
#   tar xzf <name>_addon.tar.gz && cd pusht_cog_v4_encode && \
#   DATA=$PWD/data/<name> SLURM_SUBMIT_DIR=$PWD nohup bash run_encode_probe.sbatch > run_<name>.log 2>&1 &
set -euo pipefail
DATA=$(realpath "$1"); OUT=$(realpath -m "$2"); name=$(basename "$DATA")
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
STAGE=$OUT/${name}_addon_stage; P=$STAGE/pusht_cog_v4_encode
test ! -e "$STAGE" || { echo "$STAGE exists"; exit 1; }
mkdir -p "$P/code/experiments/phase2" "$P/data/$name/observations" "$P/data/$name/evaluation_only"
cp "$REPO/experiments/phase2/probe_pusht_cog_features.py" "$P/code/experiments/phase2/"
cp "$HERE/run_encode_probe.sbatch" "$P/"
cp "$DATA"/{observation_index.json,protocol.json,preparation_manifest.json,structural_audit.json} "$P/data/$name/"
cp "$DATA/evaluation_only/sibling_classes.json" "$P/data/$name/evaluation_only/"
cp "$DATA"/observations/cog_train_*.npz "$DATA"/observations/cog_validation_*.npz "$P/data/$name/observations/"
# The probe diagnostic reads marker frames (privileged) for train/validation episodes only.
for f in "$DATA"/evaluation_only/cog_train_*_s?.json "$DATA"/evaluation_only/cog_validation_*_s?.json; do
  cp "$f" "$P/data/$name/evaluation_only/"
done
(cd "$STAGE" && tar -cf - pusht_cog_v4_encode | gzip -1 > "$OUT/${name}_addon.tar.gz")
rm -r -- "$STAGE"
sha256sum "$OUT/${name}_addon.tar.gz"
