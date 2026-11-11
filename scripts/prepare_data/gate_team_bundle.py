"""Download and verify the team's frozen Gate bundles and lay them out as a Gate asset tree.

    python scripts/prepare_data/gate_team_bundle.py --work <dir>

Needs read access to the private Hugging Face org (token in HF_HOME). See
src/bounded_memory/environments/gate/bundle.py for what is verified.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory.environments.gate.bundle import main  # noqa: E402

if __name__ == "__main__":
    main()
