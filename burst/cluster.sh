# Settings for the course cluster (NYU cloud bursting, reached through https://ood.burst.hpc.nyu.edu).
# Sourced by the scripts in this folder. To change a value, export it before the command or put
# KEY=value lines (no quotes) in burst/.env, which git ignores. An exported variable wins over
# burst/.env, which wins over the default below.
#
#   BM_ACCOUNT        Slurm account                               default: csci_ga_2271-2026fa (the course account)
#   BM_GPU_PARTITION  partition of the GPU job                    default: c12m85-a100-1 (one A100, 12 CPUs, 85 GB)
#   BM_GPU_GRES       GPU request                                 default: gpu:1
#   BM_CPUS, BM_MEM, BM_TIME   size of the GPU job                default: 12, 80G, 05:00:00
#   BM_PARALLEL       tasks running at once on the one GPU        default: 3
#   BM_SCRATCH        datasets, features, checkpoints, logs       default: /scratch/$USER/cv_project_runs
#   BM_REPO           this checkout                               default: detected
#   BM_HF_HOME        Hugging Face cache and token                default: /scratch/$USER/hf_cache
#   BM_CONDA_SH       conda's activation script                   default: /scratch/$USER/miniforge3/etc/profile.d/conda.sh
#   BM_CONDA_ENV      the environment the jobs activate           default: /scratch/$USER/miniforge3/envs/cvmem
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
export BM_GPU_PARTITION="${BM_GPU_PARTITION:-c12m85-a100-1}"
export BM_GPU_GRES="${BM_GPU_GRES:-gpu:1}"
export BM_CPUS="${BM_CPUS:-12}" BM_MEM="${BM_MEM:-80G}" BM_TIME="${BM_TIME:-05:00:00}"
export BM_PARALLEL="${BM_PARALLEL:-3}"
export BM_SCRATCH="${BM_SCRATCH:-/scratch/$USER/cv_project_runs}"
export BM_REPO="${BM_REPO:-$(dirname "$_bm_here")}"
export BM_HF_HOME="${BM_HF_HOME:-/scratch/$USER/hf_cache}"
export BM_CONDA_SH="${BM_CONDA_SH:-/scratch/$USER/miniforge3/etc/profile.d/conda.sh}"
export BM_CONDA_ENV="${BM_CONDA_ENV:-/scratch/$USER/miniforge3/envs/cvmem}"
export BM_SBATCH_ARGS="${BM_SBATCH_ARGS:-}"
export HF_HOME="$BM_HF_HOME"
unset _bm_here _bm_key _bm_value
