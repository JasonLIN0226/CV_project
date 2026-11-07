#!/bin/bash
# CV_project hidden-CoG PushT: frozen I-JEPA encoding of train+validation, then class probes.
# Default data: v2 continuous corpus (Nov 2); v1 (scene cut) ran as job 21615935 with DATA=...v1-20261101.
# Weights are prefetched in .cache/huggingface_cvproject/ijepa (pinned revision, offline here).
#SBATCH --account=torch_pr_147_courant
#SBATCH --partition=h200_public
#SBATCH --qos=gpu48
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
#SBATCH --job-name=pusht_cog_ijepa
#SBATCH --output=/projects/work/yang-lab/users/yz12129/logs/pusht_cog_ijepa_%j.out
set -euo pipefail
REPO=/projects/work/yang-lab/users/yz12129/CV_project
DATA=${DATA:-$REPO/build/pusht-cog-v2-20261102}
FEAT=${FEAT:-$DATA-ijepa}
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
"$REPO/.venv/bin/python" -I "$REPO/experiments/phase2/probe_pusht_cog_features.py" encode \
  --data "$DATA" --features "$FEAT" \
  --cache-dir /projects/work/yang-lab/users/yz12129/.cache/huggingface_cvproject/ijepa --batch-size 64
"$REPO/.venv/bin/python" -I "$REPO/experiments/phase2/probe_pusht_cog_features.py" probe \
  --data "$DATA" --features "$FEAT"
