# Settings for the course cluster (NYU cloud bursting, reached through https://ood.burst.hpc.nyu.edu).
# Sourced by the scripts in this folder. To change a value, export it before the command or put
# KEY=value lines (no quotes) in burst/.env, which git ignores. An exported variable wins over
# burst/.env, which wins over the default below.
#
#   BM_ACCOUNT        Slurm account                               default: csci_ga_2271-2026fa (the course account)
#   BM_GPU_PARTITION  Slurm partition                             default: c12m85-a100-1 or c24m170-a100-2
#   BM_GPU_COUNT      GPUs for the runner and Slurm request       default: 1
#   BM_GPU_GRES       Slurm GPU request override                  default: gpu:$BM_GPU_COUNT
#   BM_CPUS, BM_MEM, BM_TIME   size of the GPU job                default: 12/80G or 24/128G, 05:00:00
#   BM_PARALLEL       total tasks running at once                 default: 3
#   BM_SCRATCH        datasets, features, checkpoints, logs       default: /scratch/$USER/cv_project_runs
#   BM_REPO           this checkout                               default: detected
#   BM_HF_HOME        Hugging Face cache and token                default: /scratch/$USER/hf_cache
#   BM_VENV           Python 3.11 venv shared with jobs           default: $BM_REPO/.venv
#   UV_CACHE_DIR      uv package cache                            default: $BM_SCRATCH/.cache/uv
#   UV_PYTHON_INSTALL_DIR   uv-managed Python installations       default: $BM_SCRATCH/uv-python
#   BM_SBATCH_ARGS    extra sbatch options                        default: none

_bm_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "$_bm_here/.env" ]; then
    while IFS='=' read -r _bm_key _bm_value || [ -n "$_bm_key" ]; do
        _bm_key=${_bm_key#export }
        case "$_bm_key" in ''|'#'*|*[!A-Za-z0-9_]*) continue ;; esac
        [ -n "${!_bm_key:-}" ] || export "$_bm_key=$_bm_value"
    done < "$_bm_here/.env"
fi
export BM_ACCOUNT="${BM_ACCOUNT:-csci_ga_2271-2026fa}"
export BM_GPU_COUNT="${BM_GPU_COUNT:-1}"
if [[ ! "$BM_GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "BM_GPU_COUNT must be a positive integer" >&2
    return 1
fi
_bm_partition=c12m85-a100-1 _bm_cpus=12 _bm_mem=80G
if [[ "$BM_GPU_COUNT" == 2 ]]; then
    _bm_partition=c24m170-a100-2 _bm_cpus=24 _bm_mem=128G
fi
export BM_GPU_PARTITION="${BM_GPU_PARTITION:-$_bm_partition}"
export BM_GPU_GRES="${BM_GPU_GRES:-gpu:$BM_GPU_COUNT}"
export BM_CPUS="${BM_CPUS:-$_bm_cpus}" BM_MEM="${BM_MEM:-$_bm_mem}" BM_TIME="${BM_TIME:-05:00:00}"
export BM_PARALLEL="${BM_PARALLEL:-3}"
export BM_SCRATCH="${BM_SCRATCH:-/scratch/$USER/cv_project_runs}"
export BM_REPO="${BM_REPO:-$(dirname "$_bm_here")}"
export BM_HF_HOME="${BM_HF_HOME:-/scratch/$USER/hf_cache}"
export BM_VENV="${BM_VENV:-$BM_REPO/.venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$BM_SCRATCH/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$BM_SCRATCH/uv-python}"
export BM_SBATCH_ARGS="${BM_SBATCH_ARGS:-}"
export HF_HOME="$BM_HF_HOME"
unset _bm_here _bm_key _bm_value _bm_partition _bm_cpus _bm_mem

bm_activate_venv() {
    if [[ ! -x "$BM_VENV/bin/python" || ! -r "$BM_VENV/bin/activate" ]]; then
        echo "No usable venv at $BM_VENV. Run bash burst/setup.sh first." >&2
        return 1
    fi
    source "$BM_VENV/bin/activate"
}
