#!/bin/bash
#SBATCH --job-name=motion-v23
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --export=ALL
#SBATCH --open-mode=append
#SBATCH --output=/scratch/xy3257/motion-v2-20261031-deployment2/scientific-job-%j.stdout.log
#SBATCH --error=/scratch/xy3257/motion-v2-20261031-deployment2/scientific-job-%j.stderr.log

set -euo pipefail
set -o noclobber
export TZ=UTC
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1

motion_root=/scratch/xy3257/motion-v2-20261031-deployment2
motion_python=/scratch/xy3257/motion-v2-20261031-deployment1/venv/bin/python
cd "$motion_root/repo"

# A second execution fails even if the first stopped before creating runner output.
mkdir "$motion_root/scientific_job_started"
scontrol show job "$SLURM_JOB_ID" > "$motion_root/scientific_job_started/scontrol_start.txt"
date -u > "$motion_root/scientific_job_started/started_at_utc.txt"
git rev-parse HEAD > "$motion_root/scientific_job_started/git_head.txt"

# Cache and temporary writes are confined to deployment2. The runner places its
# own HF/Torch caches in its fresh scientific output before model imports.
mkdir "$motion_root/scientific_job_started/tmp"
export TMPDIR="$motion_root/scientific_job_started/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$motion_root/scientific_job_started/cache"
export CUDA_CACHE_PATH="$motion_root/scientific_job_started/cuda-cache"
export NUMBA_CACHE_DIR="$motion_root/scientific_job_started/numba-cache"
export MPLCONFIGDIR="$motion_root/scientific_job_started/matplotlib-cache"
export HF_HOME="$motion_root/scientific_job_started/hf-cache"
export TORCH_HOME="$motion_root/scientific_job_started/torch-cache"
export TRITON_CACHE_DIR="$motion_root/scientific_job_started/triton-cache"
export TORCH_EXTENSIONS_DIR="$motion_root/scientific_job_started/torch-extensions"

# Exactly one invocation; no retry, resume, cleanup or additional experiment.
set +e
"$motion_python" -I -B experiments/motion_v2/run_step2.py \
  --config experiments/motion_v2/config_step2_v2_3.json \
  --protocol experiments/motion_v2/protocol_step2_v2_3.json \
  --dataset results/motion_v2/v2_2/step1/data/dataset \
  --direction-bound results/motion_v2/v2_3/direction_bound \
  --output results/motion_v2/v2_3/step2_gpu_attempt1
motion_exit=$?
set -e
printf '%s\n' "$motion_exit" > "$motion_root/scientific_job_started/runner_exit_code.txt"
date -u > "$motion_root/scientific_job_started/finished_at_utc.txt"
exit "$motion_exit"
