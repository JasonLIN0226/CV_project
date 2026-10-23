#!/bin/bash
set -euo pipefail
export TZ=UTC
cd /scratch/xl4358/phase1_independent_confirmation_execution_v10_checkpoint_resume
exec /scratch/xl4358/phase1_env/bin/python -u guardian_peer.py /scratch/xl4358/phase1_independent_confirmation_execution_v10_checkpoint_resume
