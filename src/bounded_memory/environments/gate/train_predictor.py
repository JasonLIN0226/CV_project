# Moved from experiments/retention_structured_noise_audit/train_reference.py; only the import block changed (no search-path edits).
"""Train the frozen-architecture Gate reference on a structured arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...predictors.phase1.evaluation import training_completion
from ...predictors.phase1.inputs import FeatureStore
from ...predictors.phase1.training import train
from ...predictors.phase1.utils import write_json


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
