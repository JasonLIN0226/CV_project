#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-reader-bind
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=1
#SBATCH --mem=1500M
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/xl4358/phase1_v5_frame_reader_handoff/logs/handoff-%j.out
#SBATCH --error=/scratch/xl4358/phase1_v5_frame_reader_handoff/logs/handoff-%j.err
set -euo pipefail
export HF_HOME=/scratch/xl4358/phase1_hf_cache
TASK_CONTROL=/scratch/xl4358/phase1_v5_frame_reader_handoff
cd "$TASK_CONTROL"
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python bind_and_submit_frame_reader.py --control "$TASK_CONTROL" --source-lock-sha 0ba2495d03106fdf7f97131d42c5a168831c57b40a3564e1bf2b4e134a054f2f
