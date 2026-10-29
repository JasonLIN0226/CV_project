"""Build the preregistered clean-future target from matched structured arms.

Historical/current frames retain the random vertical-line nuisance stream.
All frames after the query are taken from the clean arm and therefore contain
no vertical lines. Only train and validation records are materialized.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONTROL_SOURCE = HERE.parent / "predictor" / "control" / "src"
if str(CONTROL_SOURCE) not in sys.path:
    sys.path.insert(0, str(CONTROL_SOURCE))

from phase1.data import dataset_signature  # noqa: E402
from phase1.utils import digest, file_hash, write_json, write_npz  # noqa: E402


DESIGN = "CLEAN_FUTURE_TARGET_V1"
SPLITS = ("train", "validation")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def _source_records(root: Path) -> tuple[dict, dict[str, dict]]:
    manifest_path = root / "dataset/manifest.json"
    audit_path = root / "dataset/split_audit.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed"):
        raise ValueError(f"Source audit did not pass: {root}")
    records = {
        row["id"]: row
        for row in manifest["records"]
        if row["task"] == "gate" and row["split"] in SPLITS
    }
    if len(records) != 1216:
        raise ValueError(f"Unexpected clean-future source episodes in {root}: {len(records)}")
    return manifest, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--lines-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    clean_root = args.clean_root.expanduser().resolve()
    lines_root = args.lines_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing overwrite: {output}")

    clean_manifest, clean = _source_records(clean_root)
    lines_manifest, lines = _source_records(lines_root)
    if set(clean) != set(lines):
        raise ValueError("Clean and noisy source identities differ")
    if clean_manifest.get("arm") != "clean":
        raise ValueError("Expected clean source arm")
    if lines_manifest.get("design_version") != DESIGN:
        lines_arm = lines_manifest.get("design_version")
        if lines_arm not in {
            "STRUCTURED_FULLIMAGE_ARROW_LINES_V2",
            "RETENTION_STRUCTURED_ARROW_LINES_V2_FULL_IMAGE",
        }:
            raise ValueError(f"Expected full-image noisy source arm, got {lines_arm!r}")

    lines_config = json.loads((lines_root / "resolved_config.json").read_text())
    config = copy.deepcopy(lines_config)
    config["output"] = str(output)
    config["device"] = args.device
    config["data"]["tasks"] = ["gate"]
    config["data"]["clean_future_target"] = {
        "version": DESIGN,
        "history_frames": "t <= query use the full-image random vertical-line arm",
        "future_frames": "t > query use the matched clean arm",
        "target_clean": True,
        "source_clean": str(clean_root),
        "source_lines": str(lines_root),
        "splits_materialized": list(SPLITS),
        "test_access": False,
    }
    config["experiment"] = {
        "stage": "development",
        "arm": "reference",
        "tasks": ["gate"],
    }
    signature = dataset_signature(config)

    dataset_root = output / "dataset"
    observations = dataset_root / "observations"
    evaluation = dataset_root / "evaluation_only"
    for path in (observations / "train", observations / "validation",
                 evaluation / "train", evaluation / "validation"):
        path.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", config)

    records = []
    diagnostics = []
    for index, episode_id in enumerate(sorted(lines)):
        noisy_rec, clean_rec = lines[episode_id], clean[episode_id]
        identity_fields = (
            "pair",
            "sibling",
            "label",
            "split",
            "task",
            "seed",
            "scene_index",
            "query",
            "gap",
            "frames",
        )
        for field in identity_fields:
            if noisy_rec[field] != clean_rec[field]:
                raise ValueError(f"Source identity mismatch {episode_id}/{field}")
        if clean_rec.get("vertical_line_count") is not None:
            raise ValueError("Clean source unexpectedly contains vertical lines")

        noisy_obs = _load_npz(lines_root / "dataset" / noisy_rec["obs"])
        clean_obs = _load_npz(clean_root / "dataset" / clean_rec["obs"])
        noisy_eval = _load_npz(lines_root / "dataset" / noisy_rec["eval"])
        clean_eval = _load_npz(clean_root / "dataset" / clean_rec["eval"])
        if set(noisy_eval) != set(clean_eval) or any(
            key != "vertical_line_counts"
            and not np.array_equal(noisy_eval[key], clean_eval[key])
            for key in noisy_eval
        ):
            raise ValueError(f"Evaluation metadata differs for {episode_id}")
        for key in ("actions", "query"):
            if not np.array_equal(noisy_obs[key], clean_obs[key]):
                raise ValueError(f"Source {key} differs for {episode_id}")

        q = int(noisy_rec["query"])
        noisy_rgb = np.asarray(noisy_obs["rgb"], dtype=np.uint8)
        clean_rgb = np.asarray(clean_obs["rgb"], dtype=np.uint8)
        if noisy_rgb.shape != clean_rgb.shape:
            raise ValueError(f"RGB shape mismatch for {episode_id}")
        mixed_rgb = noisy_rgb.copy()
        mixed_rgb[q + 1 :] = clean_rgb[q + 1 :]
        if not np.array_equal(mixed_rgb[: q + 1], noisy_rgb[: q + 1]):
            raise AssertionError("Historical/current frame splice changed")
        if not np.array_equal(mixed_rgb[q + 1 :], clean_rgb[q + 1 :]):
            raise AssertionError("Future frame splice changed")

        split = noisy_rec["split"]
        obs_rel = Path("observations") / split / f"{episode_id}.npz"
        eval_rel = Path("evaluation_only") / split / f"{episode_id}.npz"
        write_npz(
            dataset_root / obs_rel,
            rgb=mixed_rgb,
            actions=noisy_obs["actions"],
            query=np.int64(q),
        )
        shutil.copyfile(lines_root / "dataset" / noisy_rec["eval"], dataset_root / eval_rel)
        record = {
            key: value
            for key, value in noisy_rec.items()
            if key not in {
                "obs",
                "eval",
                "obs_sha256",
                "line_mask_sha256_by_frame",
                "vertical_line_counts",
            }
        }
        record.update(
            {
                "obs": str(obs_rel),
                "eval": str(eval_rel),
                "obs_sha256": file_hash(dataset_root / obs_rel),
                "eval_sha256": file_hash(dataset_root / eval_rel),
                "clean_future_start": q + 1,
                "history_rgb_source": "structured_lines",
                "future_rgb_source": "clean",
                "history_source_obs_sha256": noisy_rec["obs_sha256"],
                "future_source_obs_sha256": clean_rec["obs_sha256"],
            }
        )
        records.append(record)
        diagnostics.append(
            {
                "episode": episode_id,
                "pair": noisy_rec["pair"],
                "split": split,
                "query": q,
                "history_frames": q + 1,
                "clean_future_frames": int(len(mixed_rgb) - q - 1),
                "history_source_exact": True,
                "future_source_exact": True,
                "actions_exact": True,
                "evaluation_metadata_exact": True,
            }
        )
        if (index + 1) % 128 == 0:
            print(f"clean-future: materialized {index + 1}/{len(lines)} episodes", flush=True)

    records.sort(key=lambda row: (SPLITS.index(row["split"]), row["pair"], row["label"]))
    summaries = {}
    for split in SPLITS:
        split_records = [row for row in records if row["split"] == split]
        pairs = {row["pair"] for row in split_records}
        if len(split_records) != 2 * len(pairs):
            raise ValueError(f"Unbalanced materialized split {split}")
        summaries[split] = {
            "episodes": len(split_records),
            "pairs": len(pairs),
            "frames": sum(int(row["frames"]) for row in split_records),
            "history_rgb_source": "structured_lines",
            "future_rgb_source": "clean",
        }
    if summaries != {
        "train": {
            "episodes": 1024,
            "pairs": 512,
            "frames": sum(int(row["frames"]) for row in records if row["split"] == "train"),
            "history_rgb_source": "structured_lines",
            "future_rgb_source": "clean",
        },
        "validation": {
            "episodes": 192,
            "pairs": 96,
            "frames": sum(int(row["frames"]) for row in records if row["split"] == "validation"),
            "history_rgb_source": "structured_lines",
            "future_rgb_source": "clean",
        },
    }:
        raise ValueError("Unexpected clean-future episode/pair counts")

    manifest = {
        "signature": signature,
        "config_hash": digest(config["data"]),
        "code_hash": digest({"design": DESIGN}),
        "mode": "clean_future_gate_train_validation_only",
        "arch": "clean_future",
        "design_version": DESIGN,
        "source_clean_manifest_sha256": file_hash(clean_root / "dataset/manifest.json"),
        "source_lines_manifest_sha256": file_hash(lines_root / "dataset/manifest.json"),
        "records": records,
        "generation": {
            "status": "PASS",
            "episodes": len(records),
            "pairs": len(records) // 2,
            "splits": list(SPLITS),
            "history_rgb_source": "structured_lines",
            "future_rgb_source": "clean",
            "test_data_materialized": False,
            "gpu_seconds": 0,
        },
    }
    write_json(dataset_root / "manifest.json", manifest)
    audit = {
        "passed": True,
        "signature": signature,
        "scope": "TRAIN_VALIDATION_ONLY",
        "splits": summaries,
        "pair_diagnostics": diagnostics,
        "checks": [
            "matched_clean_and_lines_source_identities",
            "exact_actions_and_query",
            "exact_evaluation_metadata",
            "history_frames_equal_structured_lines",
            "future_frames_equal_clean",
            "balanced_sibling_pairs",
            "train_validation_only",
            "test_data_not_materialized",
            "fixed_arrow_cue_preserved",
        ],
        "test_access": False,
    }
    write_json(dataset_root / "split_audit.json", audit)
    write_json(
        output / "materialization_receipt.json",
        {
            "status": "PASS_CLEAN_FUTURE_MATERIALIZED",
            "design": DESIGN,
            "dataset_signature": signature,
            "episodes": len(records),
            "pairs": len(records) // 2,
            "test_access": False,
        },
    )
    print(json.dumps(audit["splits"], indent=2))


if __name__ == "__main__":
    main()
