#!/usr/bin/env python3
"""Verify published diagnostic receipts and arithmetic; never rerun ML models.

Standard library only. Checkpoints, RGB/features and prediction vectors remain
outside Git. Their hashes describe raw artifacts, not files this tool verifies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/phase2_history/20261020"
SOURCE_SNAPSHOTS = ROOT / "results/phase2_history/source_snapshots"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message, tolerance=1e-12):
    require(math.isfinite(actual) and math.isfinite(expected)
            and abs(actual - expected) <= tolerance * max(1., abs(actual), abs(expected)), message)


def verify_source(relative, expected, archived_sources):
    """Verify the source used by the historical receipt, not a later revision."""
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), "Source path escapes repository: " + relative)
    if path.is_file() and sha(path) == expected:
        return
    require(len(expected) == 64 and all(c in "0123456789abcdef" for c in expected),
            "Invalid source digest: " + relative)
    snapshot = (SOURCE_SNAPSHOTS / (expected + ".py")).resolve()
    require(snapshot.is_relative_to(SOURCE_SNAPSHOTS.resolve()) and snapshot.is_relative_to(ROOT),
            "Source snapshot escapes archive: " + relative)
    require(snapshot.is_file() and sha(snapshot) == expected,
            "Historical source missing or changed: " + relative)
    archived_sources[relative] = {
        "sha256": expected, "snapshot": str(snapshot.relative_to(ROOT)),
        "current_working_sha256": sha(path) if path.is_file() else None,
        "current_working_source_matches_historical_receipt": False,
    }


def verify(root=DEFAULT):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["schema_version"] == 1, "Unknown receipt schema")
    files = manifest["published_files"]
    archived_sources = {}
    for relative, record in files.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "Published path escapes receipt directory")
        require(path.stat().st_size == record["published_bytes"] and sha(path) == record["published_sha256"],
                "Published artifact changed: " + relative)
        require(record["raw_sources"] and all(len(s["sha256"]) == 64 for s in record["raw_sources"]),
                "Raw-source provenance missing: " + relative)
    for relative, expected in manifest["source_file_sha256"].items():
        verify_source(relative, expected, archived_sources)
    for relative, expected in manifest.get("related_repository_artifact_sha256", {}).items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == expected, "Related figure changed: " + relative)

    # Other workstreams own their own manifests. Extra files/directories are
    # allowed; when a compatible submanifest exists, verify its declared bytes.
    subordinate_counts = {}
    for name in ("gate_continuation", "software_checks"):
        path = root / name / "manifest.json"
        if path.exists():
            child = json.loads(path.read_text())
            require(child["schema_version"] == 1, "Unknown child receipt schema")
            for relative, record in child["published_files"].items():
                artifact = (path.parent / relative).resolve()
                require(artifact.is_relative_to(path.parent), "Child receipt escapes its directory")
                require(sha(artifact) == record["published_sha256"] and artifact.stat().st_size == record["published_bytes"],
                        "Child artifact changed: " + name + "/" + relative)
            subordinate_counts[name] = len(child["published_files"])

    pusht_root = root / "pusht_preparation"
    if (pusht_root / "portable_evidence.json").exists():
        receipt = json.loads((pusht_root / "portable_evidence.json").read_text())
        for relative, record in receipt["copied_raw_json"].items():
            artifact = (pusht_root / relative).resolve()
            require(artifact.is_relative_to(pusht_root), "PushT receipt escapes its directory")
            require(sha(artifact) == record["sha256"] and artifact.stat().st_size == record["bytes"],
                    "PushT published artifact changed: " + relative)
        for relative, expected in receipt["current_source_sha256"].items():
            verify_source(relative, expected, archived_sources)
        subordinate_counts["pusht_preparation"] = len(receipt["copied_raw_json"])

    software_path = root / "software_checks/run_manifest.json"
    if software_path.exists():
        software = json.loads(software_path.read_text())
        require(software["independent_checks_passed"] and
                software["software_test_counts"] == {"tests": 229, "failures": 0, "errors": 0, "skipped": 0},
                "Final software check result differs")
        for relative, expected in software["source_sha256"].items():
            verify_source(relative, expected, archived_sources)

    def read(name):
        require(name in files, "Required receipt missing from published manifest: " + name)
        return json.loads((root / name).read_text())

    history = read("history/summary.json")
    require(history["status"] == "completed_read_only_history_diagnostic", "History run incomplete")
    require(history["artifact_audit_after"]["test_arrays_opened"] == 0, "History diagnostic read test arrays")
    require(len(history["aggregate_metrics"]) == 120, "Unexpected history metric count")
    require(sum(row["mse"]["episodes"] for row in history["aggregate_metrics"]) == 2400,
            "Unexpected episode-horizon evaluation count")
    aggregates = {}
    for row in history["aggregate_metrics"]:
        values = row["mse"]["pair_values"]
        require(len(values) == row["mse"]["pairs"] and 2 * len(values) == row["mse"]["episodes"],
                "History analysis unit is not a complete sibling pair")
        close(row["mse"]["mean"], math.fsum(values.values()) / len(values), "Pair mean differs")
        aggregates[(row["task"], row["version"], row["split"], row["condition"], row["horizon"])] = row["mse"]["mean"]
    for row in history["contrasts"]:
        key = (row["task"], row["version"], row["split"])
        first = aggregates[key + (row["first"], row["horizon"])]
        second = aggregates[key + (row["second"], row["horizon"])]
        close(row["error_second_minus_first"]["mean"], second - first, "History contrast arithmetic differs")
    for task, replay in history["initialization_replay"].items():
        differences = [abs(replay["reconstructed"][m] - replay["logged"][m]) for m in replay["logged"]]
        require(max(differences) == replay["maximum_absolute_difference"] == 0., "Original initialization replay differs")
        require(replay["bit_exact_scores"], "Original scalar initialization replay was not exact")

    objective = read("objective/objective_audit.json")
    unwhitened = read("objective/no_whitening_ablation.json")
    for task in ("gate", "motion"):
        require(unwhitened["tasks"][task]["original_whitened_initial"] == objective["tasks"][task]["initial"],
                "Unit-column diagnostic omitted/misbound original initialization")
        require(unwhitened["tasks"][task]["initial"] != objective["tasks"][task]["initial"],
                "Unit-column own initialization was substituted with old initialization")
        require(len(unwhitened["tasks"][task]["logs"]) == 4, "Unit-column diagnostic must retain all four epochs")
        require(set(objective["tasks"][task]["four_epoch_arms"]) == {"original", "no_rank", "mse_only"},
                "Objective diagnostic arm mismatch")
    mask = read("objective/mask_inactive_diagnostic.json")
    require("posthoc" in mask["purpose"].lower() and all(v["model_state_bitwise_unchanged"] for v in mask["tasks"].values()),
            "Masking intervention is not clearly posthoc and read-only")
    reproduction = read("objective/reproduction_comparison.json")
    require(sum(v["numeric_values"] for v in reproduction["comparisons"].values()) == 1850,
            "Reproduction numerical comparison count differs")
    require(all(v["maximum_absolute_difference"] == 0 and v["different_numeric_values"] == 0
                for v in reproduction["comparisons"].values()), "Consolidated diagnostic replay differs")

    followup = read("followup/summary.json")
    protocol = read("followup/predeclared_protocol.json")["protocol"]
    curves = read("followup/selection_curves.json")
    require(followup["status"] == "completed_three_fixed_projection_arms" and len(followup["arms"]) == 6,
            "Followup did not finish all six task/arm combinations")
    require(protocol["adopt_best_arm"] is False and protocol["all_arms_exploratory"] is True,
            "Exploratory/adoption boundary changed")
    require(followup["artifact_audit"]["test_arrays_opened"] == 0, "Followup read test arrays")
    for entry in followup["arms"]:
        task, arm = entry["task"], entry["arm"]
        curve = curves["arms"][task + "/" + arm]
        require([row["epoch"] for row in curve] == list(range(41)), "Epoch-zero/40-epoch curve incomplete")
        scores = [row["validation_mse"] for row in curve]
        require(all(math.isfinite(value) for value in scores), "Nonfinite selection curve")
        best_epoch = min(range(len(scores)), key=lambda i: scores[i])
        require(best_epoch == entry["selected_epoch"], "Selection is not earliest minimum including epoch zero")
        close(scores[best_epoch], entry["selected_validation_mse"], "Selected MSE differs")
        close(scores[0], entry["initial_validation_mse"], "Own-arm initial baseline differs")
        require(curve[-1]["optimizer_steps"] == entry["optimizer_steps"] == 240, "Optimizer step count differs")
        for kind in ("selected", "terminal"):
            for split in ("train", "validation"):
                diagnostic = entry["diagnostics"][kind][split]
                require(diagnostic["pairs"] == (16 if split == "train" else 4), "Independent pair count differs")
                metrics = diagnostic["metrics"]
                for condition, benefit in diagnostic["history_benefit"].items():
                    close(benefit["mse_minus_recent"], metrics[condition]["mse"] - metrics["recent_only"]["mse"],
                          "History benefit arithmetic differs")
                    close(benefit["mse_minus_current_plus_mean"], metrics[condition]["mse"] - metrics["current_plus_train_target_mean"]["mse"],
                          "Fitted-baseline benefit arithmetic differs")
                    require(benefit["beats_recent_and_current_plus_mean"] ==
                            (metrics[condition]["mse"] < min(metrics["recent_only"]["mse"], metrics["current_plus_train_target_mean"]["mse"])),
                            "Useful-history flag differs")
                close(diagnostic["wrong_minus_correct"]["privileged"],
                      metrics["wrong_evidence"]["mse"] - metrics["privileged_small_budget"]["mse"], "Swap contrast differs")
    audit = read("followup/independent_artifact_audit.json")
    require(audit["metric_values_recomputed_exactly"] == 720 and audit["test_arrays_opened"] == 0,
            "Saved independent artifact audit differs")
    require(all(value is False for value in manifest["claim_boundaries"].values()), "Scientific claim boundary changed")
    return {"status": "passed_published_receipts_and_arithmetic", "published_files_verified": len(files),
            "source_files_verified": len(manifest["source_file_sha256"]), "history_episode_horizon_rows": 2400,
            "standalone_child_receipt_file_counts": subordinate_counts,
            "historical_source_snapshots_verified": archived_sources,
            "followup_arms": 6, "model_inference_rerun": False, "raw_binary_hashes_reverified": False,
            "scope": "Published bytes, historical source, paired means, contrasts and epoch-zero-inclusive selection only; archived source does not validate later working revisions."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
