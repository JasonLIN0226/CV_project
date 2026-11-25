#!/bin/bash
# Submit the comparison-gap runs as Slurm batch jobs on the course A100 partition, so nothing has to be
# watched: when Burst reclaims a node the job is requeued, and both drivers skip finished units and resume
# training from checkpoints.
#
#     bash burst/submit_gaps.sh            # both: synergy (public data) and gate (needs a Read HF token)
#     bash burst/submit_gaps.sh synergy    # only the Synergy literature reruns and paired comparisons
#     BM_CHAIN=4 bash burst/submit_gaps.sh synergy   # plus 4 standby copies (see below)
#
# Burst reclaims nodes and the reclaimed job ends CANCELLED without being requeued. BM_CHAIN=N queues N standby
# copies of each run: copy k starts only if copy k-1 ended in a failed or cancelled state (afternotok) and is
# removed by Slurm once an earlier copy succeeds (kill-on-invalid-dep), so a reclaimed run resumes by itself.
#
# Run it from any Burst shell that has sbatch, for example a terminal in an OOD Jupyter session.
# Logs: $BM_SCRATCH/logs/bm-<what>-<job>.log ; check with: squeue -u $USER
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PATH="/opt/slurm/bin:$PATH"      # OOD Jupyter sessions have the Slurm client here but not on PATH
source "$HERE/cluster.sh"
mkdir -p "$BM_SCRATCH/logs"
CHAIN=${BM_CHAIN:-0}
for what in ${*:-synergy gate}; do
    case "$what" in
        synergy) script="$HERE/synergy_literature_gap.sh" ;;
        gate)    script="$HERE/gate_init_seeds.sh" ;;
        *) echo "unknown run '$what' (use synergy or gate)" >&2; exit 2 ;;
    esac
    dependency=()
    for copy in $(seq 0 "$CHAIN"); do
        job=$(sbatch --parsable --job-name="bm-$what" --account="$BM_ACCOUNT" --partition="$BM_GPU_PARTITION" \
            --gres="$BM_GPU_GRES" --cpus-per-task=8 --mem=48G --time=06:00:00 --requeue --export=ALL \
            "${dependency[@]}" --output="$BM_SCRATCH/logs/bm-$what-%j.log" --wrap "bash $script")
        echo "submitted bm-$what as job $job${dependency:+ (standby after ${dependency[0]#--dependency=afternotok:})}"
        dependency=(--dependency="afternotok:$job" --kill-on-invalid-dep=yes)
    done
done
squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %R" || true
