"""Verify the published Phase 1 snapshot; no model calls, downloads or GPU."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/phase1_latest"


def read(name):
    return json.loads((DATA / name).read_text())


def main():
    manifest = read("manifest.json")
    for name, record in manifest["files"].items():
        actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        assert actual == record["sha256"], name
    counts = {}
    for task in ("gate", "motion"):
        result = read(f"{task}_summary.json")
        prediction = result["prediction"]
        original = prediction["original_checks"]
        fixed = prediction["fixed_reference_checks"]
        assert len(original) == 51 and all(c["passed"] for c in original)
        assert len(fixed) == 32 and all(c["passed"] for c in fixed)
        if task == "gate":
            deletion = result["deletion"]["cells"]
            assert len(deletion) == 72 and all(c["passed"] for c in deletion)
        else:
            deletion = [c for c in result["deletion"]["oracle_baseline_comparisons"]
                        if c["choice"] == "oracle_transfer"]
            assert len(deletion) == 72 and all(c["positive_both_metrics"] for c in deletion)
        assert prediction["phase1_go"] is False
        counts[task] = {"prediction": "51/51", "fixed_reference": "32/32",
                        "task_specific_deletion": "72/72"}
    with (DATA / "startup_allocations.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 57
    assert sum(r["state"] == "REQUEUED" for r in rows) == 56
    elapsed = 0
    previous_end = None
    for row in rows:
        start, end = map(datetime.fromisoformat, (row["start_UTC"], row["end_UTC"]))
        seconds = int(row["elapsed_seconds"])
        assert (end - start).total_seconds() == seconds
        assert previous_end is None or start >= previous_end
        assert row["account"] == "csci_ga_2271-2026fa"
        assert "gres/gpu=1" in row["AllocTRES"]
        previous_end = end
        elapsed += seconds
    budget = read("budget_boundary.json")
    assert elapsed == budget["startup_GPU_allocated_seconds"] == 26520
    assert budget["prior_GPU_seconds"] + elapsed == budget["total_GPU_allocated_seconds"] == 39651
    assert budget["GPU_cap_seconds"] == 28800
    assert budget["excess_seconds"] == 39651 - 28800 == 10851
    assert budget["no_fresh9137_data"] is True
    assert budget["fresh_models"] == budget["fresh_maps"] == 0
    assert budget["phase1_go"] is False
    print(json.dumps({"snapshot_verification": "PASS", "development": counts,
                      "GPU_allocation_seconds": 39651, "Phase1_GO": False}, indent=2))


if __name__ == "__main__":
    main()
