#!/bin/bash
#SBATCH --job-name=cvmem_synergy_assets
#SBATCH --account=torch_pr_147_courant
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ys4907/cv_project_runs/logs/%x_%j.log
# Fetches the team's public release assets for the Synergy (terminal R/D/S) study and re-encodes the
# train/validation/development frames, which were never published in encoded form.
# CPU by default. For a GPU: sbatch --gres=gpu:1 --constraint=l40s --export=ALL,DEVICE=cuda synergy_assets.sbatch
set -euo pipefail
ROOT=/scratch/ys4907/CV_project
RUNS=/scratch/ys4907/cv_project_runs
SYN=$RUNS/synergy
REL=https://github.com/JasonLIN0226/CV_project/releases/download
export HF_HOME=/scratch/ys4907/hf_cache
DEVICE=${DEVICE:-cpu}
THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
source /scratch/ys4907/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/ys4907/miniconda3/envs/cvmem
cd "$ROOT"
echo "== host $(hostname) device $DEVICE threads $THREADS $(date -Is)"
[ "$DEVICE" = cpu ] || nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$SYN/dl"
fetch() {  # name, url, sha256 ("-" to skip)
  if [ ! -f "$SYN/dl/$1" ]; then
    curl -fsSL -o "$SYN/dl/$1.part" "$2"
    mv "$SYN/dl/$1.part" "$SYN/dl/$1"
  fi
  if [ "$3" != "-" ]; then echo "$3  $SYN/dl/$1" | sha256sum -c -; fi
}
fetch terminal_results.zip "$REL/terminal-structures-20261025/terminal-structures-results.zip" -
fetch conf_full.zip "$REL/qualified-reader-retention-confirmation-20261026/qualified-reader-retention-confirmation-20261026.zip" -
fetch conf_transfer.zip "$REL/qualified-reader-retention-confirmation-20261026/conf_transfer.zip" \
  ffe0b67e43e090ab32419046c62435b92495eb39e6dbee75b936f88d8a090b1e
[ -d "$SYN/terminal/run" ] || python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$SYN/dl/terminal_results.zip" "$SYN/terminal"
[ -d "$SYN/conf_full/build" ] || python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$SYN/dl/conf_full.zip" "$SYN/conf_full"
[ -f "$SYN/conf_xfer/records.json" ] || python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$SYN/dl/conf_transfer.zip" "$SYN/conf_xfer"

python -m experiments.memory_baselines.synergy_assets --release "$SYN/terminal/run" --work "$SYN/trainvaldev" \
  --hf-cache "$HF_HOME/hub" --device "$DEVICE"
echo "SYNERGY_ASSETS_COMPLETE=$SYN $(date -Is)"
