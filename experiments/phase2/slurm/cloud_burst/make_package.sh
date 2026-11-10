#!/bin/bash
# Build the course cloud-bursting package for v4 I-JEPA encoding + probes (Nov 7 2026).
# Usage: make_package.sh DATASET_DIR HF_CACHE_DIR OUT_DIR
#   DATASET_DIR  e.g. build/pusht-cog-v4-20261107 (train+validation observations are copied;
#                the evaluation split is left out)
#   HF_CACHE_DIR cache holding models--facebook--ijepa_vith14_1k (pinned revision)
# Produces OUT_DIR/pusht_cog_v4_encode.tar.gz; run it with
#   tar xzf pusht_cog_v4_encode.tar.gz && cd pusht_cog_v4_encode && sbatch run_encode_probe.sbatch
set -euo pipefail
DATA=$(realpath "$1"); CACHE=$(realpath "$2"); OUT=$(realpath -m "$3")
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../../.." && pwd)
REV=f157467ea509bc356ff9f61fd3c0d840eec5e04e
P=$OUT/pusht_cog_v4_encode
test ! -e "$P" || { echo "$P exists"; exit 1; }
name=$(basename "$DATA")
mkdir -p "$P/code/experiments/phase2" "$P/code/experiments/predictor/control/src/phase1" \
  "$P/data/$name/observations" "$P/data/$name/evaluation_only" "$P/hf_cache/ijepa"
cp "$REPO/experiments/phase2/probe_pusht_cog_features.py" "$P/code/experiments/phase2/"
cp "$REPO"/experiments/predictor/control/src/phase1/{__init__,encoders,utils}.py "$P/code/experiments/predictor/control/src/phase1/"
cp "$DATA"/{observation_index.json,protocol.json,preparation_manifest.json,structural_audit.json} "$P/data/$name/"
cp "$DATA/evaluation_only/sibling_classes.json" "$P/data/$name/evaluation_only/"
cp "$DATA"/observations/cog_train_*.npz "$DATA"/observations/cog_validation_*.npz "$P/data/$name/observations/"
# Real files instead of HF symlinks, so the archive survives a laptop round trip.
SNAP=models--facebook--ijepa_vith14_1k/snapshots/$REV
mkdir -p "$P/hf_cache/ijepa/$SNAP"
for f in config.json model.safetensors preprocessor_config.json README.md; do
  cp -L "$CACHE/$SNAP/$f" "$P/hf_cache/ijepa/$SNAP/$f"
done
cp "$HERE"/{run_encode_probe.sbatch,requirements.txt,README.txt} "$P/"
(cd "$OUT" && tar -cf - pusht_cog_v4_encode | gzip -1 > pusht_cog_v4_encode.tar.gz)
sha256sum "$OUT/pusht_cog_v4_encode.tar.gz"
