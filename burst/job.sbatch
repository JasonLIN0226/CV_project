#!/bin/bash
#SBATCH --job-name=bm_burst
# One GPU job that runs many tasks (scripts/run_tasks.py). Submit it with burst/submit.sh, which
# supplies the account, the partition, the size of the job and the log file (burst/cluster.sh).
set -euo pipefail
: "${BM_REPO:?submit with burst/submit.sh}" "${BM_PACK_ARGS:?}"
source "$BM_REPO/burst/cluster.sh"
bm_activate_venv
echo "== $(hostname) $(date -Is) job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
cd "$BM_REPO"
git rev-parse HEAD 2>/dev/null | sed 's/^/code: /' || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# shellcheck disable=SC2086
python scripts/run_tasks.py $BM_PACK_ARGS && code=0 || code=$?
RUN_ID=$(echo "$BM_PACK_ARGS" | sed -n 's/.*--run-id \([^ ]*\).*/\1/p')
python scripts/aggregate_results/aggregate.py --run "runs/$RUN_ID" || echo "aggregate failed (the run may be incomplete)"
echo "== finished $(date -Is) with exit code $code; status: $BM_REPO/runs/$RUN_ID/pack_status.txt"
exit $code
