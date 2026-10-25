#!/bin/bash
set -euo pipefail
umask 077
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
/scratch/xy3257/cv-phase2-20261020/venv/bin/python /scratch/xy3257/cv-phase2-20261020/terminal-structures-20261025/bundle/experiments/phase2/submit_terminal_structures.py --execute --bundle /scratch/xy3257/cv-phase2-20261020/terminal-structures-20261025/bundle --run /scratch/xy3257/cv-phase2-20261020/terminal-structures-20261025/run
