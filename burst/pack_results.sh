#!/bin/bash
# Pack the small outputs of a run (JSON files, tables, status) into one archive in your home
# folder, to download through the OOD file browser:   bash burst/pack_results.sh <run_id>
# On your laptop, unpack it inside the checkout (tar xzf <run_id>_results.tar.gz) and rebuild
# the tables with: python scripts/aggregate_results/aggregate.py --run runs/<run_id>
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/cluster.sh"
RUN_ID=${1:?usage: bash burst/pack_results.sh <run_id>}
cd "$BM_REPO"
[ -d "runs/$RUN_ID" ] || { echo "no runs/$RUN_ID in $BM_REPO" >&2; exit 1; }
tar czf "$HOME/${RUN_ID}_results.tar.gz" "runs/$RUN_ID"
ls -lh "$HOME/${RUN_ID}_results.tar.gz"
