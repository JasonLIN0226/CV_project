#!/bin/bash
#SBATCH --job-name=cvmem_gate_team
#SBATCH --account=torch_pr_147_courant
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ys4907/cv_project_runs/logs/%x_%j.log
# Downloads the team's frozen Gate bundles from the private HF org (token read from HF_HOME), verifies
# every published checksum and lays them out for gate_baselines.sbatch. No training, no encoding.
set -euo pipefail
ROOT=/scratch/ys4907/CV_project
RUNS=/scratch/ys4907/cv_project_runs
SNAP=$RUNS/clean_future_control
export HF_HOME=/scratch/ys4907/hf_cache PYTHONUNBUFFERED=1
source /scratch/ys4907/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/ys4907/miniconda3/envs/cvmem
cd "$ROOT"
echo "== host $(hostname) $(date -Is)"
python tools/verify_gate_clean_future_source.py
[ -d "$SNAP" ] || python experiments/gate_clean_future/frozen_control/materialize_snapshot.py --output "$SNAP"
python -m experiments.memory_baselines.gate_bundle --work "$RUNS/gate_team" --device cpu --threads 4
echo "GATE_TEAM_BUNDLE_COMPLETE $(date -Is)"
