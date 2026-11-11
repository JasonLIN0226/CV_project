#!/bin/bash
# Submit the job that builds the conda environment and caches the pinned I-JEPA encoder.
# Run on the cluster:  bash torch/submit_setup_env.sh      (settings: torch/cluster.sh)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cluster.sh"
[ -f "$BM_CONDA_SH" ] || { echo "no conda at $BM_CONDA_SH: install Miniconda there or set BM_CONDA_SH" >&2; exit 1; }
mkdir -p "$BM_SCRATCH/logs"
# shellcheck disable=SC2086
sbatch ${BM_ACCOUNT:+--account="$BM_ACCOUNT"} $BM_SBATCH_ARGS --output="$BM_SCRATCH/logs/%x_%j.log" --export=ALL "$BM_REPO/torch/setup_env.sbatch"
