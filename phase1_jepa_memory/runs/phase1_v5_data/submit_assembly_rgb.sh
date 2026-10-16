#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-data
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/scratch/xl4358/phase1_v5_data/control/logs/data-%j.out
#SBATCH --error=/scratch/xl4358/phase1_v5_data/control/logs/data-%j.err
set -euo pipefail
export HF_HOME=/scratch/xl4358/phase1_hf_cache
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
TASK_CONTROL=/scratch/xl4358/phase1_v5_data/control
cd "$TASK_CONTROL"
sha256sum -c <<'HASHES'
eb4ff098e79b4bf1038db1115c8aa9a7b6872c390ef38347c9e2ad84752b4d01  assembly101_development_acquisition.json
526a11e14bf74c6f76ea4b45e7a82d663feb956c3fea683f38a25eead0cb44ea  fetch_v5_assembly_rgb.py
HASHES
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python fetch_v5_assembly_rgb.py --manifest "$TASK_CONTROL/assembly101_development_acquisition.json" --output /scratch/xl4358/phase1_v5_data/rgb --mode fetch
