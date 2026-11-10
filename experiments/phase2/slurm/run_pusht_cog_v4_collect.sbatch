#!/bin/bash
# CV_project hidden-CoG PushT v4 (natural episodes), Nov 7: dev recheck after tightening the
# start range / probe shift, then the full corpus (128/32/32 families). CPU only.
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96GB
#SBATCH --time=3:00:00
#SBATCH --job-name=pusht_cog_v4_collect
#SBATCH --output=/projects/work/yang-lab/users/yz12129/logs/pusht_cog_v4_collect_%j.out
set -euo pipefail
REPO=/projects/work/yang-lab/users/yz12129/CV_project
PY="$REPO/.venv/bin/python"
LIB="$REPO/build/pusht_cog_probe_library_v3.json"
export SDL_VIDEODRIVER=dummy PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -I "$REPO/experiments/phase2/dev_pusht_cog_natural.py" --library "$LIB" \
  --output "$REPO/build/pusht-cog-v4-dev2" --families 192 --processes "$SLURM_CPUS_PER_TASK"
"$PY" -I "$REPO/experiments/phase2/prepare_pusht_cog_v4_data.py" --library "$LIB" \
  --output "$REPO/build/pusht-cog-v4-20261107" --processes "$SLURM_CPUS_PER_TASK"
