#!/usr/bin/env bash
# Human: NYU sign-in. Agent: verified Cloud host, transfer, Slurm, results.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK_SSH_DIR="${PHASE1_SSH_DIR:-/private/tmp/phase1-hpc-control}"
TASK_CONFIG="$TASK_SSH_DIR/config"
TASK_REMOTE=/scratch/xl4358/phase1_jepa_memory
TASK_RUN="${PHASE1_RUN:-phase1_v3_dev}"
case "$TASK_RUN" in phase1_v3_dev|phase1_v3_confirm|phase1_v2) ;; *) echo 'Unsupported run'; exit 2 ;; esac
TASK_OUTPUT="/scratch/xl4358/$TASK_RUN"
TASK_PYTHON=/scratch/xl4358/phase1_env/bin/python
TASK_ZIP="$TASK_ROOT/../phase1_jepa_memory_upload.zip"
TASK_SSH=(ssh -F "$TASK_CONFIG" -o StrictHostKeyChecking=yes -o ForwardAgent=no)
[[ -r "$TASK_CONFIG" ]] || { echo 'Connection not prepared: ask the agent to verify the NYU hosts.'; exit 2; }

case "${1:-status}" in
  login)
    # The human enters the existing Mac key passphrase locally, once per session.
    if ! ssh-add -l 2>/dev/null | grep -Fq 'SHA256:IwMZLvgoThP4PQmKHd62JVcdIsmc7r3F0IhP0ddYDeg'; then
      ssh-add -t 8h /Users/linxiangyu/.ssh/id_ed25519
    fi
    "${TASK_SSH[@]}" -O check phase1-torch 2>/dev/null && exit 0
    exec "${TASK_SSH[@]}" -o ControlMaster=yes -o ControlPersist=8h \
      -o ServerAliveInterval=60 -Nf phase1-torch
    ;;
  status)
    "${TASK_SSH[@]}" -O check phase1-torch
    "${TASK_SSH[@]}" -o BatchMode=yes phase1-cloud \
      "set -e; hostname; /opt/slurm/bin/squeue -u xl4358 -o '%.10i %.24j %.12T %.10M %.24R'; test ! -f '$TASK_OUTPUT/report.md' || cat '$TASK_OUTPUT/report.md'"
    ;;
  push)
    # Transfer source only; run-all deterministically generates/audits the data.
    "${TASK_SSH[@]}" -o BatchMode=yes phase1-cloud 'TASK_JOBS=$(/opt/slurm/bin/squeue -u xl4358 -h -n cv-phase1) && test -z "$TASK_JOBS"' || {
      echo 'Do not change source while cv-phase1 is active (or connection failed).'; exit 2;
    }
    python3 "$TASK_ROOT/scripts/package.py"
    TASK_SHA="$(shasum -a 256 "$TASK_ZIP" | awk '{print $1}')"
    scp -F "$TASK_CONFIG" -o BatchMode=yes -o StrictHostKeyChecking=yes \
      "$TASK_ZIP" phase1-cloud:/scratch/xl4358/phase1_jepa_memory_upload.zip
    "${TASK_SSH[@]}" -o BatchMode=yes phase1-cloud "python3 - $TASK_SHA" <<'PY'
from pathlib import Path
import hashlib
import sys
import zipfile
p = Path('/scratch/xl4358/phase1_jepa_memory_upload.zip')
if hashlib.sha256(p.read_bytes()).hexdigest() != sys.argv[1]:
    raise SystemExit('Source archive checksum mismatch')
with zipfile.ZipFile(p) as z:
    if z.testzip() is not None:
        raise SystemExit('Corrupt source archive')
    if any(not n.startswith('phase1_jepa_memory/') or '..' in Path(n).parts for n in z.namelist()):
        raise SystemExit('Unexpected archive path')
    z.extractall(p.parent)
    print('Verified source archive:', len(z.namelist()), 'files, SHA256', sys.argv[1])
p.unlink()
PY
    ;;
  run)
    [[ "$TASK_RUN" == phase1_v3_dev ]] || { echo 'Development run requires PHASE1_RUN=phase1_v3_dev'; exit 2; }
    # Slurm owns the training process; closing the local terminal does not stop it.
    "${TASK_SSH[@]}" -o BatchMode=yes phase1-cloud 'bash -s' <<EOF
set -euo pipefail
export PATH=/opt/slurm/bin:\$PATH
TASK_JOBS=\$(squeue -u xl4358 -h -o '%i %j %b')
if printf '%s\\n' "\$TASK_JOBS" | awk '/gpu|cv-phase1/ {found=1} END {exit !found}'; then
  echo 'An experiment allocation is already active; reuse it instead of submitting another.'
  exit 2
fi
cd '$TASK_REMOTE'
export OUTPUT_DIR='$TASK_OUTPUT' PYTHON_BIN='$TASK_PYTHON'
export HF_HOME=/scratch/xl4358/phase1_hf_cache
export SOURCE_OUTPUT=/scratch/xl4358/phase1_v2
export TASK_COMMAND=v3-develop TASK_DEVICE=cpu TIME_LIMIT=02:00:00
export EXPERIMENT_TASK=gate
TASK_GATE=\$(bash scripts/submit.sh cpu)
[[ "\$TASK_GATE" =~ ^[0-9]+$ ]] || { echo "Unexpected gate submission: \$TASK_GATE"; exit 2; }
echo "Gate development job: \$TASK_GATE"
export EXPERIMENT_TASK=motion
AFTER_JOB="\$TASK_GATE" bash scripts/submit.sh cpu
EOF
    ;;
  run-confirm)
    [[ "$TASK_RUN" == phase1_v3_confirm ]] || { echo 'Use PHASE1_RUN=phase1_v3_confirm'; exit 2; }
    "${TASK_SSH[@]}" -o BatchMode=yes phase1-cloud 'bash -s' <<EOF
set -euo pipefail
export PATH=/opt/slurm/bin:\$PATH
test -z "\$(squeue -u xl4358 -h -n cv-phase1)"
cd '$TASK_REMOTE'
export PYTHONPATH='$TASK_REMOTE/src' HF_HOME=/scratch/xl4358/phase1_hf_cache
'$TASK_PYTHON' - <<'PY'
import json
from pathlib import Path
from phase1.experiment import development_report
result = development_report('/scratch/xl4358/phase1_v3_dev', verify=True)
if not result['confirmation_ready']:
    raise SystemExit('Actual development evidence rejected or incomplete; no confirmation allocation')
print('Both actual development comparisons supported; fixed confirmation authorized')
PY
export OUTPUT_DIR='$TASK_OUTPUT' PYTHON_BIN='$TASK_PYTHON' DEVELOPMENT_OUTPUT=/scratch/xl4358/phase1_v3_dev
export TASK_COMMAND=v3-confirm-features TASK_DEVICE=cuda TIME_LIMIT=02:00:00
TASK_ENCODE=\$(bash scripts/submit.sh a100)
echo "Confirmation encoder job: \$TASK_ENCODE"
export TASK_COMMAND=v3-confirm TASK_DEVICE=cpu
AFTER_JOB="\$TASK_ENCODE" bash scripts/submit.sh cpu
EOF
    ;;
  fetch)
    mkdir -p "$TASK_ROOT/runs/$TASK_RUN"
    rsync -ace "ssh -F $TASK_CONFIG -o BatchMode=yes -o StrictHostKeyChecking=yes -o ForwardAgent=no" \
      --exclude='observations/' --exclude='evaluation_only/' --exclude='features/' --exclude='*.pt' --exclude='*.npz' \
      --exclude='huggingface/' --exclude='.matplotlib/' \
      --include='*/' --include='*.md' --include='*.json' --include='*.jsonl' \
      --include='*.csv' --include='*.png' --include='*.log' --exclude='*' \
      "phase1-cloud:$TASK_OUTPUT/" "$TASK_ROOT/runs/$TASK_RUN/"
    ;;
  *) echo 'Usage: bash scripts/hpc.sh login|status|push|run|fetch|run-confirm'; exit 2 ;;
esac
