#!/bin/bash
# Run the selected tasks in this terminal, on the GPU of the OOD session you are in, instead of
# submitting a job. Same arguments as burst/submit.sh. Keep it alive after the browser closes with:
#
#     nohup bash burst/run_here.sh --run-id burst_a100 --select <task prefixes> > run_here.log 2>&1 &
#
# OOD sessions run in a container that shows the GPU device but not the NVIDIA libraries;
# burst/gpu_driver.sh fetches the matching libraries into your home folder the first time.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/cluster.sh"
set +u; source "$BM_CONDA_SH"; conda activate "$BM_CONDA_ENV"; set -u
source "$HERE/gpu_driver.sh"
cd "$BM_REPO"
python scripts/run_tasks.py "$@" && code=0 || code=$?
RUN_ID=$(echo "$*" | sed -n 's/.*--run-id \([^ ]*\).*/\1/p')
python scripts/aggregate_results/aggregate.py --run "runs/$RUN_ID" || echo "aggregate failed (the run may be incomplete)"
exit $code
