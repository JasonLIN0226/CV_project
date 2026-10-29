from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
FILES = (
    "materialize.py",
    "fit_features.py",
    "evaluate_validation.py",
    "run_trainval.sbatch",
    "setup_env.sbatch",
    "test_protocol.py",
    "protocol.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    payload = {
        "status": "FROZEN_CLEAN_FUTURE_TRAINVAL_V1",
        "files": {name: digest(HERE / name) for name in FILES},
    }
    (HERE / "source_lock.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
