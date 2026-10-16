#!/usr/bin/env bash
# Operational recovery only: the completed Motion diagnostic is not rerun.
# Prepared, not submitted while the user's model-switch clarification is pending.
set -euo pipefail
PROJECT_DIR=/scratch/xl4358/phase1_jepa_memory_v4_development
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME=/scratch/xl4358/phase1_hf_cache
printf 'Slurm job=%s host=%s\n' "$SLURM_JOB_ID" "$(hostname)"
hostname -I
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
/scratch/xl4358/phase1_env/bin/python - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('/scratch/xl4358/phase1_jepa_memory_v4_development')
proof=json.loads((root/'deployment_verification.json').read_text())
assert proof['zip_sha256']=='2cadc27ac68819808049ed46325719c4497cf6ffd117539f5846ea712197ed59'
assert len(proof['source_files'])==53
for name,value in proof['source_files'].items():
    path=root/Path(name).relative_to('phase1_jepa_memory_v4_development')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==value,(name,'source changed')
view=Path('/scratch/xl4358/phase1_v3_verified_view')
assert hashlib.sha256((view/'verification.json').read_bytes()).hexdigest()=='a36baa1372b736e0cddebde99cbc6d49202a10b4bd9e35b86c20728c6bedaedf'
print('All53 frozen source files and exact archived verification match.',flush=True)
PY
/scratch/xl4358/phase1_env/bin/python scripts/verify_v4_stage.py
exec /scratch/xl4358/phase1_env/bin/python scripts/v4_develop_gate.py \
  --source-output /scratch/xl4358/phase1_v2 \
  --frozen-v3-root /scratch/xl4358/phase1_v3_verified_view \
  --frozen-source-directory /scratch/xl4358/phase1_jepa_memory \
  --diagnostic-root /scratch/xl4358/phase1_v4_diagnostics \
  --output /scratch/xl4358/phase1_v4_dev \
  --protocol configs/phase1_v4_gate_protocol.json
