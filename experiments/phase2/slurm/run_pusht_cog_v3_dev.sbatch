#!/bin/bash
# CV_project hidden-CoG PushT design v3 (matched probes), dev run on CPU (Nov 6).
# Reuses the cached motion library; designs + checks dev families 2026130000.. in a process pool.
# families.jsonl is appended per family, so a resubmit skips finished seeds.
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --job-name=pusht_cog_v3_dev
#SBATCH --output=/projects/work/yang-lab/users/yz12129/logs/pusht_cog_v3_dev_%j.out
set -euo pipefail
REPO=/projects/work/yang-lab/users/yz12129/CV_project
export SDL_VIDEODRIVER=dummy PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$REPO/.venv/bin/python" -I "$REPO/experiments/phase2/dev_pusht_cog_matched.py" \
  --library "$REPO/build/pusht_cog_probe_library_v3.json" \
  --output "${OUTPUT:-$REPO/build/pusht-cog-v3-dev}" --families ${FAMILIES:-32} --processes ${SLURM_CPUS_PER_TASK:-32}
