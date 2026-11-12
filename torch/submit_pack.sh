#!/bin/bash
# Submit one GPU job that runs the selected manifest tasks, several at a time on the one GPU
# (scripts/run_tasks.py). For GPU work when GPUs queue for days: one job instead of one per seed.
# Run it on the cluster, with plain sbatch and no Python on the login node:
#
#     ssh torch 'bash /scratch/$USER/CV_project/torch/submit_pack.sh --run-id <run_id> --select <task prefixes> [--skip <prefixes>] [--parallel 3]'
#
# The arguments go to scripts/run_tasks.py. The job runs from a copy of src/, scripts/, configs/
# and torch/ taken now, so later edits cannot change it. Running the same command again continues
# an unfinished run. Status: runs/<run_id>/pack_status.txt, which Mutagen syncs back.
# To list what would run, use the runner locally: python scripts/run_tasks.py --dry-run ...
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/cluster.sh"
RUN_ID=$(echo " $* " | sed -n 's/.* --run-id \([^ ]*\) .*/\1/p')
[ -n "$RUN_ID" ] || { echo "usage: bash torch/submit_pack.sh --run-id <id> --select <task prefixes> ..." >&2; exit 1; }
[ -x "$BM_CONDA_ENV/bin/python" ] || { echo "no environment at $BM_CONDA_ENV: run bash torch/submit_setup_env.sh first" >&2; exit 1; }
RUN="$BM_SCRATCH/reproduction/$RUN_ID"
mkdir -p "$RUN/logs" "$BM_REPO/runs/$RUN_ID"
export BM_CODE="$RUN/code_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$BM_CODE"
cp -a "$BM_REPO/src" "$BM_REPO/scripts" "$BM_REPO/configs" "$BM_REPO/torch" "$BM_CODE/"
find "$BM_CODE" -name __pycache__ -prune -exec rm -rf {} +
export BM_PACK_ARGS="$*"            # --parallel defaults to BM_PARALLEL
# shellcheck disable=SC2086
job=$(sbatch --parsable ${BM_ACCOUNT:+--account="$BM_ACCOUNT"} $BM_SBATCH_ARGS \
      --gres="gpu:${BM_GPU_TYPE:+$BM_GPU_TYPE:}1" $BM_GPU_SBATCH_ARGS \
      --cpus-per-task="$BM_PACK_CPUS" --mem="$BM_PACK_MEM" --time="$BM_PACK_TIME" \
      --output="$RUN/logs/pack_%j.log" --export=ALL "$HERE/pack.sbatch")
echo -e "pack\t$job\t$(date -Is)" | tee -a "$RUN/jobs.tsv"
echo "account: ${BM_ACCOUNT:-(Slurm default)}  gpu: ${BM_GPU_TYPE:-any}  $BM_PACK_CPUS CPUs, $BM_PACK_MEM, $BM_PACK_TIME, $BM_PARALLEL tasks at once"
echo "code snapshot: $BM_CODE"
echo "log:     $RUN/logs/pack_$job.log"
echo "status:  $BM_REPO/runs/$RUN_ID/pack_status.txt"
