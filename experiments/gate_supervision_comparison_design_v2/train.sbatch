#!/bin/bash
#SBATCH --job-name=gate-v2-paired
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85a1001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:59:00
#SBATCH --no-requeue
#SBATCH --export=ALL
#SBATCH --output=/scratch/xl4358/clean_future_gate_20261022/gate_supervision_v2_20261029/run_20261029/train-%j.out
#SBATCH --error=/scratch/xl4358/clean_future_gate_20261022/gate_supervision_v2_20261029/run_20261029/train-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4
scontrol update JobId="$SLURM_JOB_ID" Requeue=0 || true
info=$(scontrol show job "$SLURM_JOB_ID" -o)
printf '%s\n' "$info"
case " $info " in *' Requeue=0 '*) ;; *) echo 'Cannot verify no-requeue; stopping before scientific work' >&2; exit 2;; esac
case " $info " in *' Restarts=0 '*) ;; *) echo 'Unexpected restart; inspect accounting before any resume' >&2; exit 2;; esac
cd /scratch/xl4358/clean_future_gate_20261022/gate_supervision_v2_20261029/run_20261029/source
sha256sum -c SHA256SUMS
exec /scratch/xl4358/clean_future_retention_audit_20261022/venv/bin/python experiments/gate_supervision_comparison_design_v2/run_trainval.py train
