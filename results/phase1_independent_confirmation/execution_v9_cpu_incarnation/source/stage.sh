#!/bin/bash
set -euo pipefail
export TZ=UTC OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=/scratch/xl4358/phase1_hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TASK_ROOT="/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation"
export PYTHONPATH="$TASK_ROOT/original_src:$TASK_ROOT"
case "$1" in
  preflight_gpu|encode_train|train_gpu|encode_test) ;;
  audit_cpu|evidence_cpu)
    export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ;;
  *) export CUDA_VISIBLE_DEVICES='' ;;
esac
cd "$TASK_ROOT"
exec /scratch/xl4358/phase1_env/bin/python -u "$TASK_ROOT/orchestrator.py" "$1"
