#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/scratch/xy3257/cv-phase2-20261020/venv/bin/python -u /scratch/xy3257/cv-phase2-20261020/reader-scale-probe-20261024/bundle/experiments/phase2/run_scaled_use_time_probe.py --work /scratch/xy3257/cv-phase2-20261020/reader-scale-probe-20261024
/scratch/xy3257/cv-phase2-20261020/venv/bin/python /scratch/xy3257/cv-phase2-20261020/reader-scale-probe-20261024/pack_scaled_probe.py --work /scratch/xy3257/cv-phase2-20261020/reader-scale-probe-20261024
