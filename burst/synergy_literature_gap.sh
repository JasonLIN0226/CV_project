#!/bin/bash
# Synergy: the literature rows that were never rerun (Compress to Remember and FrameMorrow on the
# item_attention reader), three seeds each, plus the multihead Compress to Remember for the same paired
# analysis; then paired family-bootstrap comparisons with our released students on both readers.
# Run inside a Burst GPU session from the checkout; finished outputs are skipped, so rerunning continues.
#
#     nohup bash burst/synergy_literature_gap.sh > /scratch/$USER/syn_gap.log 2>&1 &
#
# Inputs: the public Synergy releases (no token) and TRAIN features re-encoded on this GPU
# (configs/synergy/gpu_features.yaml), the device class of the released features.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/cluster.sh"
bm_activate_venv
cd "$BM_REPO"
RUN="$BM_SCRATCH/reproduction/synergy_gap"
OUT="$RUN/synergy/replay"
mkdir -p "$OUT" "$RUN/synergy/stageA"
[ -s "$BM_SCRATCH/synergy/students_released/item_attention_standard/manifest.json" ] || \
    python scripts/prepare_data/synergy_release.py --work "$BM_SCRATCH/synergy"
[ -s "$BM_SCRATCH/synergy/trainvaldev_gpu/assets_report.json" ] || \
    python scripts/prepare_data/synergy_trainvaldev.py --release "$BM_SCRATCH/synergy/terminal/run" \
        --work "$BM_SCRATCH/synergy/trainvaldev_gpu" --hf-cache "$HF_HOME/hub" --device cuda
for R in item_attention multihead; do
    python scripts/generate_teacher_labels/synergy.py --config configs/synergy/release.yaml --seed 17 --device cuda \
        --out "$OUT" --reader "$R" --split confirmation --labels "$RUN/synergy/stageA"
done
for M in pacc framemorrow; do
    for S in 17 29 43; do
        python scripts/train_baselines/synergy.py --config configs/synergy/gpu_features.yaml --seed "$S" --device cuda \
            --out "$OUT" --reader item_attention --method "$M" --stage-a "$RUN/synergy/stageA"
    done
done
for S in 17 29 43; do
    python scripts/train_baselines/synergy.py --config configs/synergy/gpu_features.yaml --seed "$S" --device cuda \
        --out "$OUT" --reader multihead --method pacc --stage-a "$RUN/synergy/stageA"
done
OURS="$BM_REPO/runs/repro_20261111/synergy/replay"     # released students, replayed exactly on 11 Nov
for R in item_attention multihead; do
    for B in pacc:pacc framemorrow:fm_published; do
        M=${B%%:*}; P=${B##*:}
        [ "$R" = multihead ] && [ "$M" = framemorrow ] && continue
        python scripts/aggregate_results/synergy_paired.py --config configs/synergy/release.yaml --seed 0 \
            --out "$RUN/paired_${R}_ours_vs_${P}.json" --reader "$R" --stage-a "$RUN/synergy/stageA" \
            --a "$OURS"/students_ours_${R}_seed{17,29,43}.json --b "$OUT"/${M}_${R}_seed{17,29,43}.json --policy-b "$P"
    done
done
echo SYNERGY_GAP_DONE
