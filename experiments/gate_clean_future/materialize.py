"""Materialize a train/validation-only noisy-input, clean-target Gate dataset.

The deterministic source scenes are unchanged. Frames at or before the query
keep the 7-14 full-image nuisance lines; only frames used as future targets are
rendered without those lines. Test records are never generated.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STRUCTURED = REPO / "experiments/retention_structured_noise_audit"
CONTROL = REPO / "experiments/predictor/control/src"
for source in (STRUCTURED, CONTROL):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from phase1.data import dataset_signature  # noqa: E402
from phase1.utils import digest, file_hash, write_json, write_npz  # noqa: E402
from structured_design import (  # noqa: E402
    ARROW_PANEL,
    FULL_IMAGE_LINE_PLACEMENT,
    STRUCTURED_LINE,
    structured_pair_bundle,
)


DESIGN_VERSION = "CLEAN_FUTURE_NOISY_INPUT_V1"
ALLOWED_SPLITS = ("train", "validation")


def _record_id(pair_id: str, label: int) -> str:
    return digest([pair_id, label, DESIGN_VERSION.lower()])[:24]


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _panel_box(size: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = ARROW_PANEL
    return (
        int(round(left * size)),
        int(round(top * size)),
        int(round(right * size)) + 1,
        int(round(bottom * size)) + 1,
    )


def _outside_panel(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    output = rgb.copy()
    output[y0:y1, x0:x1] = 0
    return output


def _inside_panel(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return rgb[y0:y1, x0:x1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing overwrite: {output}")

    c = copy.deepcopy(json.loads(args.source_config.expanduser().resolve().read_text()))
    c["output"] = str(output)
    c["device"] = "cpu"
    c["data"]["tasks"] = ["gate"]
    c["data"]["clean_future_protocol"] = {
        "version": DESIGN_VERSION,
        "input_frames": "frames at or before query retain 7-14 full-image nuisance lines",
        "future_target_frames": "frames after query contain no nuisance lines",
        "sibling_line_masks": "bit-identical for every historical frame",
        "generated_splits": list(ALLOWED_SPLITS),
        "test_generated": False,
        "test_accessed": False,
        "thresholds_changed": False,
        "line_color_rgb": list(STRUCTURED_LINE),
        "line_placement": FULL_IMAGE_LINE_PLACEMENT,
    }
    c["experiment"] = {
        "stage": "development",
        "arm": "reference",
        "tasks": ["gate"],
    }

    output.mkdir(parents=True)
    dataset_root = output / "dataset"
    write_json(output / "resolved_config.json", c)
    signature = dataset_signature(c)
    image_size = int(c["data"]["image_size"])
    panel_box = _panel_box(image_size)
    records = []
    pair_diagnostics = []
    split_summary = {}

    for split in ALLOWED_SPLITS:
        pair_count = int(c["data"]["pairs"][split])
        split_rows = []
        split_pairs = []
        for pair_index in range(pair_count):
            if pair_index % 32 == 0:
                print(f"{split}: {pair_index}/{pair_count}", flush=True)

            scene_lines, design_lines, bundle_lines = structured_pair_bundle(
                c,
                split,
                pair_index,
                include_vertical_lines=True,
                full_image_lines=True,
            )
            scene_clean, design_clean, bundle_clean = structured_pair_bundle(
                c,
                split,
                pair_index,
                include_vertical_lines=False,
                full_image_lines=True,
            )
            if scene_lines["seed"] != scene_clean["seed"]:
                raise ValueError("Line/clean source scene seed differs")
            if (
                design_lines.cue_start,
                design_lines.cue_end,
                design_clean.cue_start,
                design_clean.cue_end,
            ) != (
                design_clean.cue_start,
                design_clean.cue_end,
                design_lines.cue_start,
                design_lines.cue_end,
            ):
                raise ValueError("Line/clean cue interval differs")

            query = int(scene_lines["query"])
            pair_id = digest(
                [c["data"]["seed"], "gate", split, pair_index, DESIGN_VERSION.lower()]
            )[:24]
            ids = {label: _record_id(pair_id, label) for label in (0, 1)}
            if len(set(ids.values())) != 2:
                raise ValueError("Opaque sibling IDs collided")

            mixed_frames: dict[int, list] = {}
            for label in (0, 1):
                line_episode = bundle_lines[label]["episode"]
                clean_episode = bundle_clean[label]["episode"]
                for key in ("actions", "states", "classes", "visible", "evidence"):
                    if not np.array_equal(line_episode[key], clean_episode[key]):
                        raise ValueError(f"Line/clean source differs in {key}")
                input_frames = bundle_lines[label]["frames"][: query + 1]
                target_frames = bundle_clean[label]["frames"][query + 1 :]
                frames = [*input_frames, *target_frames]
                if len(frames) != int(scene_lines["length"]):
                    raise ValueError("Mixed frame count differs from source scene")
                if any(
                    not (7 <= frame.vertical_line_count <= 14)
                    for frame in input_frames
                ):
                    raise ValueError("Historical input is missing the approved nuisance lines")
                if any(frame.vertical_line_count != 0 for frame in target_frames):
                    raise ValueError("Clean future target contains nuisance lines")
                mixed_frames[label] = frames

            if not np.array_equal(
                bundle_lines[0]["episode"]["actions"],
                bundle_lines[1]["episode"]["actions"],
            ):
                raise ValueError("Sibling actions differ")
            if not np.array_equal(
                bundle_lines[0]["episode"]["evidence"],
                bundle_lines[1]["episode"]["evidence"],
            ):
                raise ValueError("Sibling evidence differs")
            evidence = np.asarray(bundle_lines[0]["episode"]["evidence"], np.int64)
            if np.any(evidence >= query - int(c["data"]["recent"]) + 1):
                raise ValueError("Evidence overlaps the recent buffer")
            if any(
                not np.array_equal(
                    mixed_frames[0][index].line_mask,
                    mixed_frames[1][index].line_mask,
                )
                for index in range(len(mixed_frames[0]))
            ):
                raise ValueError("Sibling nuisance line masks differ")

            cue_differences = 0
            neutral_equal = 0
            for index in range(query + 1):
                left = mixed_frames[0][index].rgb
                right = mixed_frames[1][index].rgb
                if not np.array_equal(
                    _outside_panel(left, panel_box),
                    _outside_panel(right, panel_box),
                ):
                    raise ValueError("Historical label leakage outside the arrow panel")
                if mixed_frames[0][index].is_cue:
                    if np.array_equal(
                        _inside_panel(left, panel_box),
                        _inside_panel(right, panel_box),
                    ):
                        raise ValueError("Cue direction is absent")
                    cue_differences += 1
                else:
                    if not np.array_equal(
                        _inside_panel(left, panel_box),
                        _inside_panel(right, panel_box),
                    ):
                        raise ValueError("Neutral arrow panel leaks the label")
                    neutral_equal += 1
            for horizon in c["data"]["horizons"]:
                index = query + int(horizon)
                if np.array_equal(
                    mixed_frames[0][index].rgb,
                    mixed_frames[1][index].rgb,
                ):
                    raise ValueError("Clean future target is identical across siblings")

            for label in (0, 1):
                episode = bundle_lines[label]["episode"]
                frames = mixed_frames[label]
                obs_rel = Path("observations") / split / f"{ids[label]}.npz"
                eval_rel = Path("evaluation_only") / split / f"{ids[label]}.npz"
                write_npz(
                    dataset_root / obs_rel,
                    rgb=np.stack([frame.rgb for frame in frames]),
                    actions=episode["actions"],
                    query=np.int64(query),
                )
                write_npz(
                    dataset_root / eval_rel,
                    states=episode["states"],
                    classes=episode["classes"],
                    visible=episode["visible"],
                    label=np.int64(label),
                    evidence=evidence,
                    cue_mask=np.asarray([frame.is_cue for frame in frames], dtype=np.bool_),
                    vertical_line_counts=np.asarray(
                        [frame.vertical_line_count for frame in frames],
                        dtype=np.int64,
                    ),
                )
                record = {
                    "id": ids[label],
                    "pair": pair_id,
                    "sibling": ids[1 - label],
                    "label": label,
                    "split": split,
                    "task": "gate",
                    "seed": scene_lines["seed"],
                    "scene_index": pair_index,
                    "query": query,
                    "gap": int(design_lines.post_cue_history_frames),
                    "frames": len(frames),
                    "obs": str(obs_rel),
                    "eval": str(eval_rel),
                    "obs_sha256": file_hash(dataset_root / obs_rel),
                    "eval_sha256": file_hash(dataset_root / eval_rel),
                    "cue_start": int(design_lines.cue_start),
                    "cue_end": int(design_lines.cue_end),
                    "history_depth": int(design_lines.history_depth),
                    "post_cue_history_frames": int(
                        design_lines.post_cue_history_frames
                    ),
                    "line_mask_sha256_by_frame": [
                        _sha(frame.line_mask) for frame in frames
                    ],
                    "vertical_line_counts": [
                        frame.vertical_line_count for frame in frames
                    ],
                    "input_frame_count": query + 1,
                    "clean_future_frame_count": len(frames) - query - 1,
                }
                records.append(record)
                split_rows.append(record)

            split_pairs.append(
                {
                    "pair": pair_id,
                    "scene_index": pair_index,
                    "cue_start": int(design_lines.cue_start),
                    "cue_end": int(design_lines.cue_end),
                    "query": query,
                    "input_frames_with_lines": query + 1,
                    "future_frames_without_lines": int(scene_lines["length"])
                    - query
                    - 1,
                    "cue_panel_differences": cue_differences,
                    "neutral_panel_equal_checks": neutral_equal,
                    "sibling_line_masks_exact": True,
                    "sibling_actions_exact": True,
                    "sibling_evidence_exact": True,
                    "sibling_future_targets_clean_and_distinct": True,
                }
            )
            pair_diagnostics.append(split_pairs[-1])

        split_summary[split] = {
            "pairs": pair_count,
            "records": len(split_rows),
            "frames_per_record": sorted({int(row["frames"]) for row in split_rows}),
            "total_frames": sum(int(row["frames"]) for row in split_rows),
            "input_frames": sum(int(row["input_frame_count"]) for row in split_rows),
            "clean_future_frames": sum(
                int(row["clean_future_frame_count"]) for row in split_rows
            ),
            "post_cue_min": min(
                int(row["post_cue_history_frames"]) for row in split_rows
            ),
            "post_cue_max": max(
                int(row["post_cue_history_frames"]) for row in split_rows
            ),
            "labels": {
                str(label): sum(int(row["label"]) == label for row in split_rows)
                for label in (0, 1)
            },
        }

    manifest = {
        "signature": signature,
        "config_hash": digest(c["data"]),
        "code_hash": digest(
            {
                "design": DESIGN_VERSION,
                "panel": ARROW_PANEL,
                "line_color": STRUCTURED_LINE,
                "allowed_splits": ALLOWED_SPLITS,
            }
        ),
        "mode": "clean_future_gate_trainval",
        "design_version": DESIGN_VERSION,
        "records": records,
        "generation": {
            "status": "PASS",
            "pairs": len(pair_diagnostics),
            "records": len(records),
            "frames": sum(int(row["frames"]) for row in records),
            "input_frames_with_lines": sum(
                int(row["input_frame_count"]) for row in records
            ),
            "clean_future_frames": sum(
                int(row["clean_future_frame_count"]) for row in records
            ),
            "generated_splits": list(ALLOWED_SPLITS),
            "test_generated": False,
            "test_accessed": False,
        },
    }
    write_json(dataset_root / "manifest.json", manifest)
    write_json(
        dataset_root / "split_audit.json",
        {
            "passed": True,
            "signature": signature,
            "design_version": DESIGN_VERSION,
            "generated_splits": list(ALLOWED_SPLITS),
            "test_generated": False,
            "test_accessed": False,
            "splits": split_summary,
            "pair_diagnostics": pair_diagnostics,
            "checks": [
                "only_train_and_validation_generated",
                "no_test_files_created",
                "historical_input_frames_have_7_to_14_lines",
                "future_target_frames_have_zero_lines",
                "sibling_historical_line_masks_identical",
                "cue_direction_only_inside_fixed_panel",
                "neutral_panel_does_not_encode_label",
                "sibling_actions_and_evidence_identical",
                "sibling_clean_future_targets_distinct",
                "thresholds_unchanged",
            ],
        },
    )
    print(
        json.dumps(
            {
                "status": "CLEAN_FUTURE_TRAINVAL_MATERIALIZED",
                "output": str(output),
                "records": len(records),
                "frames": sum(int(row["frames"]) for row in records),
                "splits": split_summary,
                "test_generated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
