#!/bin/bash
# Store your Hugging Face token on the course cluster (once):   bash burst/hf_login.sh
# The Gate team bundle is private: the token needs read access to the team organisation.
# It asks for the token and writes it to $HF_HOME/token; nothing is printed or committed.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/cluster.sh"
set +u; source "$BM_CONDA_SH"; conda activate "$BM_CONDA_ENV"; set -u
python -c "from huggingface_hub import login, whoami; login(); print('logged in as', whoami()['name'])"
