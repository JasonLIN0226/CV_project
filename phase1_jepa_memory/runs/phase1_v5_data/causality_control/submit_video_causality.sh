#!/usr/bin/env bash
#SBATCH --job-name=cv-v5-pts
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/scratch/xl4358/phase1_v5_data/causality_control/logs/pts-%j.out
#SBATCH --error=/scratch/xl4358/phase1_v5_data/causality_control/logs/pts-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
TASK_CONTROL=/scratch/xl4358/phase1_v5_data/causality_control
cd "$TASK_CONTROL"
sha256sum -c <<'LOCK_HASH'
d72d5531a89d83fbe376c65de7a986507586e37950e7117c18ba20deb6066d2f  source_lock.json
LOCK_HASH
sha256sum -c <<'SOURCE_HASHES'
071b0f1844492669c630219d3cca8a68ce1b415c12471cb9c4e4b7d092bc1801  audit_v5_video_data.py
ae56b40b6f8b067a8ad2dac664fbfbabac7f7a55b9a7bb031eb99289252bc017  av-18.0.0-cp311-abi3-manylinux_2_28_x86_64.whl
28c4b7b748db530db5a8fd341603a8ab1e0245f166a43999290a4f134e730fe3  design.json
eb4ff098e79b4bf1038db1115c8aa9a7b6872c390ef38347c9e2ad84752b4d01  manifest.json
641dd7c540513e04e80cce214fdd275a296bf77149db1a55caa559e6b0987616  protocol.json
5489738c68287f8c48186fd2db9adeaadd41405751bd0c0c82db7afb1762813d  test_v5_video_causality.py
6ec55db658e204879e1bbcc2d28e8e42a2a53152219e3f45bbbdb520ddbd11f7  v5_video_causality.py
SOURCE_HASHES
hostname
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /scratch/xl4358/phase1_env/bin/python audit_v5_video_data.py --control "$TASK_CONTROL" --rgb-root /scratch/xl4358/phase1_v5_data/rgb --output /scratch/xl4358/phase1_v5_data/causality_runs/job_${SLURM_JOB_ID} --vendor /scratch/xl4358/phase1_v5_data/av_vendor_18_0_0
