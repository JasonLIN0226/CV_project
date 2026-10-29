"""Fast local checks for clean-future materialization and validation-only scope."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def test_generated_dataset_if_present():
    dataset = ROOT / "results/gate_clean_future_20261022"
    manifest_path = dataset / "dataset/manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads((dataset / "dataset/split_audit.json").read_text())
    assert manifest["generation"]["test_generated"] is False
    assert manifest["generation"]["test_accessed"] is False
    assert {row["split"] for row in manifest["records"]} == {
        "train",
        "validation",
    }
    assert audit["passed"] is True
    for row in manifest["records"][:20]:
        with np.load(dataset / "dataset" / row["obs"], allow_pickle=False) as handle:
            rgb = handle["rgb"]
            query = int(handle["query"])
        with np.load(dataset / "dataset" / row["eval"], allow_pickle=False) as handle:
            counts = handle["vertical_line_counts"]
        assert rgb.shape[0] == len(counts)
        assert np.all((counts[: query + 1] >= 7) & (counts[: query + 1] <= 14))
        assert np.all(counts[query + 1 :] == 0)
    assert not any(
        "test_" in path.relative_to(dataset).as_posix()
        for path in dataset.rglob("*")
    )


if __name__ == "__main__":
    test_generated_dataset_if_present()
    print("clean-future protocol check PASS")
