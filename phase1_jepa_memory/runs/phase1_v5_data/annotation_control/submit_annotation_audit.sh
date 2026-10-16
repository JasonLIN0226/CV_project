#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-labels
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/scratch/xl4358/phase1_v5_data/annotation_control/logs/labels-%j.out
#SBATCH --error=/scratch/xl4358/phase1_v5_data/annotation_control/logs/labels-%j.err
set -euo pipefail
export HF_HOME=/scratch/xl4358/phase1_hf_cache
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_DISABLE_PROGRESS_BARS=1
cd /scratch/xl4358/phase1_v5_data/annotation_control
/scratch/xl4358/phase1_env/bin/python - <<'LOCKPY'
import json,hashlib,pathlib
root=pathlib.Path('.')
lock=root/'source_lock.json'
assert hashlib.sha256(lock.read_bytes()).hexdigest() == '1ef5ab0d944f53c5ed39f8a313b59470a4615a9e8f3c571f5982b00d22f52fd8'
for name,expected in json.loads(lock.read_text())['files'].items():
 p=root/name
 assert not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==expected
print('ALL7SOURCE_AND_INPUT_BYTES_VERIFIED_BEFORE_IMPORT',flush=True)
LOCKPY
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python audit_v5_annotations.py --manifest manifest.json --pts-report pts_report.json --output /scratch/xl4358/phase1_v5_data/annotation_runs/job_${SLURM_JOB_ID}
