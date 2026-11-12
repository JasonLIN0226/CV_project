#!/bin/bash
#SBATCH --job-name=bm_pack
# One GPU job that runs many tasks (scripts/run_tasks.py). Submit it with torch/submit_pack.sh,
# which supplies the account, the GPU, the size of the job, the log file and the code snapshot.
set -euo pipefail
: "${BM_CODE:?submit with torch/submit_pack.sh}" "${BM_REPO:?}" "${BM_SCRATCH:?}" "${HF_HOME:?}" "${BM_CONDA_SH:?}" "${BM_CONDA_ENV:?}" "${BM_PACK_ARGS:?}"
echo "== $(hostname) $(date -Is) job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
set +u; source "$BM_CONDA_SH"; conda activate "$BM_CONDA_ENV"; set -u
cd "$BM_CODE"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# shellcheck disable=SC2086
python scripts/run_tasks.py $BM_PACK_ARGS && code=0 || code=$?
echo "== finished $(date -Is) with exit code $code"
exit $code
