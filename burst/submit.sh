#!/bin/bash
# Submit one Slurm job for the selected tasks, using the GPU allocation set in cluster.sh.
# Run it in a terminal of an OOD session, from the checkout:
#
#     bash burst/submit.sh --run-id burst_a100 --select <task prefixes> [--skip <prefixes>] [--parallel 3] [--dry-run]
#
# The arguments go to scripts/run_tasks.py. Running the same command again continues an unfinished run.
# With --dry-run nothing is submitted: it lists the tasks the job would run.
#
# Burst reclaims nodes, and a reclaimed job ends CANCELLED without being requeued. BM_CHAIN=N also queues N
# standby copies of the job: copy k starts only if copy k-1 did not finish successfully (afternotok), so the run
# continues where it stopped, and the copy that finishes cancels the standby copies still pending.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/cluster.sh"
bm_activate_venv
case " $* " in *" --run-id "*) ;; *) echo "usage: bash burst/submit.sh --run-id <id> --select <task prefixes> ..." >&2; exit 1 ;; esac
case " $* " in *" --dry-run "*)         # only list what the job would run
    cd "$BM_REPO" && exec python scripts/run_tasks.py "$@" ;;
esac
mkdir -p "$BM_SCRATCH/logs"
export BM_PACK_ARGS="$*"            # --parallel defaults to BM_PARALLEL
RUN_ID=$(echo "$*" | sed -n 's/.*--run-id \([^ ]*\).*/\1/p')
CHAIN=${BM_CHAIN:-0}
naming=() name=bm_burst
if [ "$CHAIN" -gt 0 ]; then         # one name per run, so a finished copy cancels only this run's standbys
    name="bm_burst_$RUN_ID"
    naming=(--job-name="$name")
    export BM_CHAIN_NAME="$name"
fi
dependency=()
for copy in $(seq 0 "$CHAIN"); do
    # shellcheck disable=SC2086
    job=$(sbatch --parsable --account="$BM_ACCOUNT" --partition="$BM_GPU_PARTITION" --gres="$BM_GPU_GRES" \
          --cpus-per-task="$BM_CPUS" --mem="$BM_MEM" --time="$BM_TIME" $BM_SBATCH_ARGS ${naming[@]+"${naming[@]}"} ${dependency[@]+"${dependency[@]}"} \
          --output="$BM_SCRATCH/logs/%x_%j.log" --export=ALL "$HERE/job.sbatch") || {
        echo "sbatch refused the job. If it is the size, try a smaller one, for example: BM_MEM=60G BM_CPUS=4 BM_TIME=04:00:00 bash burst/submit.sh ..." >&2
        echo "If sbatch is not available in this session, run the tasks here instead: bash burst/run_here.sh ..." >&2
        exit 1; }
    if [ "$copy" -eq 0 ]; then
        first=$job
        echo "submitted job $job  (account $BM_ACCOUNT, partition $BM_GPU_PARTITION, $BM_GPU_GRES, $BM_CPUS CPUs, $BM_MEM, $BM_TIME)"
    else
        echo "standby job $job: starts only if job ${dependency[0]#--dependency=afternotok:} does not finish"
    fi
    dependency=(--dependency="afternotok:$job")
done
echo "queue:   squeue -u $USER"
echo "log:     tail -f $BM_SCRATCH/logs/${name}_$first.log"
echo "status:  cat $BM_REPO/runs/$RUN_ID/pack_status.txt"
