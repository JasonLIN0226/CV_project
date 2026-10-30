"""Zero-GPU verification of the clean-future train/validation dataset."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONTROL_SOURCE = HERE.parent / "predictor" / "control" / "src"
if str(CONTROL_SOURCE) not in sys.path:
    sys.path.insert(0, str(CONTROL_SOURCE))

from phase1.data import dataset_signature  # noqa: E402
from phase1.utils import file_hash, write_json  # noqa: E402


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--lines-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    c = json.loads(config_path.read_text())
    root = config_path.parent
    manifest = json.loads((root / "dataset/manifest.json").read_text())
    audit = json.loads((root / "dataset/split_audit.json").read_text())
    clean_root = args.clean_root.expanduser().resolve()
    lines_root = args.lines_root.expanduser().resolve()
    clean_manifest = json.loads((clean_root / "dataset/manifest.json").read_text())
    lines_manifest = json.loads((lines_root / "dataset/manifest.json").read_text())
    clean = {row["id"]: row for row in clean_manifest["records"]}
    lines = {row["id"]: row for row in lines_manifest["records"]}
    records = manifest["records"]
    if manifest["signature"] != dataset_signature(c):
        raise ValueError("Dataset signature does not match resolved config")
    if not audit.get("passed") or audit.get("signature") != manifest["signature"]:
        raise ValueError("Materialization audit is missing or stale")
    if {row["split"] for row in records} != {"train", "validation"}:
        raise ValueError("Only train and validation are authorized")
    if any((root / "dataset/observations" / split).exists() for split in (
        "test_id", "test_long_gap", "test_distractor_shift"
    )):
        raise ValueError("A test split directory exists")

    counts = Counter(row["split"] for row in records)
    pairs = {split: {row["pair"] for row in records if row["split"] == split} for split in counts}
    if counts != {"train": 1024, "validation": 192} or pairs["train"].__len__() != 512 or pairs["validation"].__len__() != 96:
        raise ValueError("Unexpected train/validation episode or pair count")

    cleared_targets = 0
    historical_pixels = 0
    future_pixels = 0
    sibling_checks = 0
    by_id = {row["id"]: row for row in records}
    for rec in records:
        noisy_rec, clean_rec = lines[rec["id"]], clean[rec["id"]]
        if file_hash(root / "dataset" / rec["obs"]) != rec["obs_sha256"]:
            raise ValueError(f"Observation hash mismatch: {rec['id']}")
        if file_hash(root / "dataset" / rec["eval"]) != rec["eval_sha256"]:
            raise ValueError(f"Evaluation hash mismatch: {rec['id']}")
        mixed = _load(root / "dataset" / rec["obs"])
        noisy = _load(lines_root / "dataset" / noisy_rec["obs"])
        clean_obs = _load(clean_root / "dataset" / clean_rec["obs"])
        q = int(rec["query"])
        if not np.array_equal(mixed["rgb"][: q + 1], noisy["rgb"][: q + 1]):
            raise ValueError(f"History splice differs: {rec['id']}")
        if not np.array_equal(mixed["rgb"][q + 1 :], clean_obs["rgb"][q + 1 :]):
            raise ValueError(f"Future splice differs: {rec['id']}")
        for horizon in c["data"]["horizons"]:
            target = _load(root / "dataset" / rec["eval"])
            assert target["visible"][q + horizon]
            if not np.array_equal(clean_obs["rgb"][q + horizon], mixed["rgb"][q + horizon]):
                raise ValueError(f"Target is not clean at h={horizon}: {rec['id']}")
            cleared_targets += 1
        sibling = by_id[rec["sibling"]]
        sibling_mixed = _load(root / "dataset" / sibling["obs"])
        recent = int(c["data"]["recent"])
        if not np.array_equal(
            mixed["rgb"][q - recent + 1 : q + 1],
            sibling_mixed["rgb"][q - recent + 1 : q + 1],
        ):
            raise ValueError(f"Recent sibling frames differ: {rec['id']}")
        sibling_checks += 1
        historical_pixels += int(mixed["rgb"][: q + 1].size)
        future_pixels += int(mixed["rgb"][q + 1 :].size)

    result = {
        "status": "PASS_ZERO_GPU_CLEAN_FUTURE_PREFLIGHT",
        "dataset_signature": manifest["signature"],
        "splits": audit["splits"],
        "episode_counts": dict(counts),
        "pair_counts": {split: len(values) for split, values in pairs.items()},
        "records_checked": len(records),
        "target_frames_checked": cleared_targets,
        "sibling_recent_checks": sibling_checks,
        "historical_pixels_checked": historical_pixels,
        "future_pixels_checked": future_pixels,
        "history_lines_preserved": True,
        "future_targets_clean": True,
        "clean_root_manifest_sha256": file_hash(clean_root / "dataset/manifest.json"),
        "lines_root_manifest_sha256": file_hash(lines_root / "dataset/manifest.json"),
        "test_data_read": False,
        "test_data_materialized": False,
        "thresholds_changed": False,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
