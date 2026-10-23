#!/bin/bash
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=motion-context-review
#SBATCH --output=/scratch/xl4358/phase1_motion_context_locked_test/review-%j.out
#SBATCH --error=/scratch/xl4358/phase1_motion_context_locked_test/review-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
cd /scratch/xl4358/phase1_motion_context_locked_test
/scratch/xl4358/phase1_env/bin/python -c "import os,socket,pathlib;print('JOB',os.environ['SLURM_JOB_ID']);print('HOST',socket.gethostname());print('IP',socket.gethostbyname(socket.gethostname()));print(pathlib.Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text(),flush=True)"
exec /scratch/xl4358/phase1_env/bin/python -u execute.py "$SLURM_JOB_ID"
