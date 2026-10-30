"""Check CONF metadata/transfer integrity before any prediction or scoring.

This checks identity, hashes, shapes, frozen banks, and appearance regeneration;
it never evaluates a reader, student, forecast error, or retention decision.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(transfer, train_run, dev_transfer, appearance, appearance_log):
    receipt = json.loads((transfer / "receipt.json").read_text())
    if "A100" not in (receipt.get("cuda_device") or ""):
        raise ValueError("Encoding was not verified on A100")
    members = {}
    expected_members = {"records.json", "actions.npz", "rgb_index.json", "generation.json", "encoding.json",
                        "projected_features.npy", "output_basis_coords.npy", "target_mean_coords.npy"}
    if set(receipt["transfer_sha256"]) != expected_members:
        raise ValueError("Transfer receipt must bind every expected data file")
    for name, expected in receipt["transfer_sha256"].items():
        actual = sha(transfer / name)
        if actual != expected:
            raise ValueError("Transfer digest mismatch: " + name)
        members[name] = actual
    if receipt["basis_sha256"] != sha(train_run / "basis.npz"):
        raise ValueError("Frozen TRAIN basis mismatch")
    if receipt["normalization_sha256"] != sha(train_run / "data/normalization.npz"):
        raise ValueError("Frozen TRAIN normalization mismatch")
    generation = json.loads((transfer / "generation.json").read_text())
    if generation["records_sha256"] != sha(transfer / "records.json"):
        raise ValueError("Generation record binding mismatch")
    if generation["action_banks_sha256"] != sha(train_run / "data/action_banks.npz"):
        raise ValueError("Generated action bank digest differs from frozen TRAIN bank; stop and inspect actual arrays")
    if generation.get("profile_only") is not False:
        raise ValueError("Profile data cannot be confirmation")
    encoding = json.loads((transfer / "encoding.json").read_text())
    if encoding["features_sha256"] != receipt["features_sha256"]:
        raise ValueError("Encoded feature binding mismatch")
    recs = json.loads((transfer / "records.json").read_text())
    if {r["split"] for r in recs} != {"confirmation"}:
        raise ValueError("Transfer contains non-CONF records")
    if len({r["family"] for r in recs}) != 48 or len({r["id"] for r in recs}) != len(recs):
        raise ValueError("Expected 48 families and unique record IDs")
    if {r["scene"]["seed"] for r in recs} != {20261104}:
        raise ValueError("Unexpected scene seed")
    counts = dict(Counter(r["task_family"] for r in recs))
    if counts != {"R": 576, "D": 576, "S": 2304}:
        raise ValueError("Task counts differ from 48-family factorial design")
    for family in {r["family"] for r in recs}:
        if dict(Counter(r["task_family"] for r in recs if r["family"] == family)) != {"R": 12, "D": 12, "S": 48}:
            raise ValueError("Incomplete factorial family")
    projected = np.load(transfer / "projected_features.npy", mmap_mode="r", allow_pickle=False)
    output = np.load(transfer / "output_basis_coords.npy", mmap_mode="r", allow_pickle=False)
    if projected.dtype != np.float32 or output.dtype != np.float32 or projected.shape != output.shape:
        raise ValueError("Expected float32 projected and output coordinates of matching shape")
    if projected.shape != (receipt["unique_frames"], 128):
        raise ValueError("Feature array shape disagrees with encoding receipt")
    if not np.isfinite(projected).all() or not np.isfinite(output).all():
        raise ValueError("Nonfinite coordinate arrays")
    tm = np.load(transfer / "target_mean_coords.npy", allow_pickle=False)
    dev_tm = np.load(dev_transfer / "target_mean_coords.npy", allow_pickle=False)
    if tm.shape != (128,) or dev_tm.shape != (128,) or not np.isfinite(tm).all() or not np.isfinite(dev_tm).all():
        raise ValueError("Invalid target-mean coordinate arrays")
    delta = float(np.max(abs(tm - dev_tm)))
    if delta > 1e-5:
        raise ValueError("Target-mean coordinates differ from frozen DEV coordinates")
    banks = np.load(train_run / "data/action_banks.npz", allow_pickle=False)
    actions = np.load(transfer / "actions.npz", allow_pickle=False)
    if set(actions.files) != {r["id"] for r in recs}:
        raise ValueError("Action record IDs mismatch")
    for name, n in (("reference", 16), ("evaluation", 32)):
        if banks[name].shape != (n, 6, 2):
            raise ValueError("Frozen action bank dimensions mismatch")
        for r in recs:
            targets = np.asarray(r["plan_targets"][name])
            if targets.shape != (n, 2) or not np.issubdtype(targets.dtype, np.integer):
                raise ValueError("Missing/malformed plan targets")
            if targets.min() < 0 or targets.max() >= len(projected):
                raise ValueError("Target-frame index is out of bounds")
    for r in recs:
        a = actions[r["id"]]
        if a.shape != (len(r["rgb_indices"]) - 1, 2) or not np.isfinite(a).all():
            raise ValueError("Malformed per-record actions")
        if np.any(a[:r["query"]]) or not np.array_equal(a, np.asarray(r["scene"]["actions"], np.float32)):
            raise ValueError("Actions differ from frozen scene metadata")
    app = np.load(appearance, allow_pickle=False)
    if app["ids"].tolist() != [r["id"] for r in recs] or app["items"].shape != (len(recs), 3, 30):
        raise ValueError("Appearance rows/shape mismatch")
    lines = [json.loads(line) for line in appearance_log.read_text().splitlines() if line.startswith("{")]
    summaries = [line for line in lines if "hash_mismatches" in line]
    if len(summaries) != 1 or summaries[0]["hash_mismatches"] != 0 or summaries[0]["saved"] != len(recs):
        raise ValueError("Appearance regeneration did not produce exactly one zero-mismatch receipt")
    return {"status": "passed", "prediction_or_scoring_performed": False, "seed": 20261104,
            "families": 48, "records": len(recs), "task_counts": counts, "cuda_device": receipt["cuda_device"],
            "transfer_hashes": members, "target_mean_max_abs_difference": delta,
            "action_banks_sha256": sha(train_run / "data/action_banks.npz"),
            "action_bank_check": "CONF generation digest equals frozen TRAIN archive digest; shapes and plan target indices validated",
            "appearance_regeneration": summaries[0], "appearance_sha256": sha(appearance),
            "appearance_log_sha256": sha(appearance_log), "integrity_source_sha256": sha(__file__)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transfer", type=Path, required=True)
    ap.add_argument("--train-run", type=Path, required=True)
    ap.add_argument("--dev-transfer", type=Path, required=True)
    ap.add_argument("--appearance", type=Path, required=True)
    ap.add_argument("--appearance-log", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to replace the original integrity receipt")
    result = check(args.transfer, args.train_run, args.dev_transfer, args.appearance, args.appearance_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": result["status"], "records": result["records"], "families": result["families"]}))


if __name__ == "__main__":
    main()
