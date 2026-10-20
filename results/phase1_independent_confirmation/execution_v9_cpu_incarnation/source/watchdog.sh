#!/bin/bash
set -euo pipefail
export TZ=UTC
cd /scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation
exec /scratch/xl4358/phase1_env/bin/python -u watchdog.py /scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation
