#!/bin/bash
#SBATCH --job-name=motion-v2-step34
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:55:00
#SBATCH --no-requeue
#SBATCH --export=ALL
#SBATCH --open-mode=append
#SBATCH --output=/scratch/xy3257/motion-v2-20261101-step34-deployment1/scientific-job-%j.stdout.log
#SBATCH --error=/scratch/xy3257/motion-v2-20261101-step34-deployment1/scientific-job-%j.stderr.log

set -euo pipefail
set -o noclobber
export TZ=UTC PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
motion_root=/scratch/xy3257/motion-v2-20261101-step34-deployment1
motion_old=/scratch/xy3257/motion-v2-20261031-deployment3/repo
motion_python=/scratch/xy3257/motion-v2-20261031-deployment1/venv/bin/python
cd "$motion_root/repo"
mkdir "$motion_root/scientific_job_started"
scontrol show job "$SLURM_JOB_ID" > "$motion_root/scientific_job_started/scontrol_start.txt"
date -u > "$motion_root/scientific_job_started/started_at_utc.txt"
git rev-parse HEAD > "$motion_root/scientific_job_started/git_head.txt"
mkdir "$motion_root/scientific_job_started/tmp"
export TMPDIR="$motion_root/scientific_job_started/tmp" TMP="$motion_root/scientific_job_started/tmp" TEMP="$motion_root/scientific_job_started/tmp"
export XDG_CACHE_HOME="$motion_root/scientific_job_started/cache" CUDA_CACHE_PATH="$motion_root/scientific_job_started/cuda-cache"
export HF_HOME="$motion_root/scientific_job_started/hf-cache" TORCH_HOME="$motion_root/scientific_job_started/torch-cache"
export MPLCONFIGDIR="$motion_root/scientific_job_started/matplotlib-cache" TRITON_CACHE_DIR="$motion_root/scientific_job_started/triton-cache"
set +e
"$motion_python" -I -B experiments/motion_v2/run_step34.py \
  --mode gpu --protocol experiments/motion_v2/protocol_step34.json \
  --execution experiments/motion_v2/protocol_step34_execution.json \
  --dataset "$motion_old/results/motion_v2/v2_2/step1/data/dataset" \
  --assets "$motion_old/results/motion_v2/v2_3_1/step2_gpu_attempt1" \
  --cpu-receipt results/motion_v2/step3_4/cpu \
  --output results/motion_v2/step3_4/gpu
motion_exit=$?
set -e
printf '%s\n' "$motion_exit" > "$motion_root/scientific_job_started/runner_exit_code.txt"
date -u > "$motion_root/scientific_job_started/finished_at_utc.txt"
exit "$motion_exit"
