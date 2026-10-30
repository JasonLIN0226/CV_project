"""Train the frozen-architecture Gate reference on a structured arm."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONTROL_SOURCE = HERE.parent / "predictor" / "control" / "src"
if str(CONTROL_SOURCE) not in sys.path:
    sys.path.insert(0, str(CONTROL_SOURCE))

from phase1.evaluation import training_completion  # noqa: E402
from phase1.inputs import FeatureStore  # noqa: E402
from phase1.training import train  # noqa: E402
from phase1.utils import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    if c["data"]["tasks"] != ["gate"]:
        raise ValueError("Gate-only reference training required")
    if c["experiment"] != {
        "stage": "development",
        "arm": "reference",
        "tasks": ["gate"],
    }:
        raise ValueError("Reference experiment wrapper changed")
    result = train(c, resume=True)
    if result.get("interrupted"):
        raise RuntimeError("Reference training interrupted")
    store = FeatureStore(c)
    completion = training_completion(c, store)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "reference_completion.json", completion)
    if not completion["complete"]:
        raise RuntimeError("; ".join(completion["issues"]))
    print((output / "reference_completion.json").read_text(), end="")


if __name__ == "__main__":
    main()
