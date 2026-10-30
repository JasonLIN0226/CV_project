"""Fit train-only transforms and probes for the clean-future Gate dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONTROL = REPO / "experiments/predictor/control/src"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

from phase1.cache import encode, fit_normalization, load_index  # noqa: E402
from phase1.data import load_episode, require_audit  # noqa: E402
from phase1.inputs import FeatureStore  # noqa: E402
from phase1.model import fit_basis  # noqa: E402
from phase1.probes import (  # noqa: E402
    choose_readout,
    load_decoder,
    probe_inputs,
)
from phase1.utils import code_hash, file_hash, write_json  # noqa: E402


SPLITS = ("train", "validation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    manifest = require_audit(c)
    observed_splits = {row["split"] for row in manifest["records"]}
    if observed_splits != set(SPLITS):
        raise ValueError(f"Expected only train/validation, observed {observed_splits}")
    if c["data"]["clean_future_protocol"]["test_accessed"]:
        raise ValueError("Test access must remain false")
    if c["experiment"] != {
        "stage": "development",
        "arm": "reference",
        "tasks": ["gate"],
    }:
        raise ValueError("Unexpected experiment configuration")

    ix = encode(c)
    fit_normalization(c)
    store = FeatureStore(c)
    basis = fit_basis(c, store, "gate")
    store.bases = {"gate": basis}
    root = Path(c["output"]) / "probes"
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    checkpoints = {}

    for variant in ("gate_cue", "recent_only"):
        sets = {}
        for split in SPLITS:
            rows = [
                row
                for row in manifest["records"]
                if row["task"] == "gate" and row["split"] == split
            ]
            x = np.stack([probe_inputs(store, row, variant, basis) for row in rows])
            y = np.array(
                [
                    [
                        2
                        * int(load_episode(c, row, True, keys=("label",))["label"])
                        - 1
                    ]
                    for row in rows
                ],
                np.float64,
            )
            sets[split] = (x, y, [row["pair"] for row in rows])
        model = choose_readout(
            c,
            sets["train"],
            sets["validation"],
            True,
        )
        path = root / f"gate_{variant}.npz"
        model.save(path)
        checkpoints[path.name] = file_hash(path)
        validation_x, validation_y, _ = sets["validation"]
        validation_prediction = model.predict(validation_x)[:, 0] > 0
        validation_accuracy = float(
            np.mean(validation_prediction == (validation_y[:, 0] > 0))
        )
        results[variant] = {
            "validation_accuracy": validation_accuracy,
            "validation_examples": int(len(validation_y)),
            "ridge": float(model.ridge),
        }

    decoder_sets = {}
    for split in SPLITS:
        x, y, groups, horizons = [], [], [], []
        for row in manifest["records"]:
            if row["task"] != "gate" or row["split"] != split:
                continue
            meta = load_episode(c, row, True)
            features = store.features(row)
            for horizon in c["data"]["horizons"]:
                target_index = int(row["query"]) + int(horizon)
                if not meta["visible"][target_index]:
                    raise ValueError("Visible future target expected")
                x.append(features[target_index])
                y.append(
                    [
                        2 * int(meta["classes"][target_index]) - 1,
                        *meta["states"][target_index],
                    ]
                )
                groups.append(row["pair"])
                horizons.append(int(horizon))
        decoder_sets[split] = (
            np.stack(x),
            np.asarray(y),
            groups,
            np.asarray(horizons),
        )

    decoder = choose_readout(
        c,
        decoder_sets["train"][:2],
        decoder_sets["validation"][:2],
        False,
    )
    decoder_path = root / "gate_state_decoder.npz"
    decoder.save(decoder_path)
    checkpoints[decoder_path.name] = file_hash(decoder_path)
    decoded = decoder.predict(decoder_sets["validation"][0])
    decoded_y = decoder_sets["validation"][1]
    decoded_horizons = decoder_sets["validation"][3]
    readout = {}
    for horizon in c["data"]["horizons"]:
        keep = decoded_horizons == int(horizon)
        readout[str(horizon)] = {
            "validation_accuracy": float(
                np.mean((decoded[keep, 0] > 0) == (decoded_y[keep, 0] > 0))
            ),
            "validation_position_rmse": float(
                np.sqrt(np.mean((decoded[keep, 1:] - decoded_y[keep, 1:]) ** 2))
            ),
            "validation_examples": int(keep.sum()),
        }

    write_json(
        root / "metrics.json",
        {
            "mode": c["mode"],
            "scope": "train_and_validation_only",
            "tasks": ["gate"],
            "cache_signature": ix["signature"],
            "results": results,
            "readout": readout,
            "checkpoints": checkpoints,
            "code_hash": code_hash("probes.py"),
            "input_basis_sha256": {"gate": file_hash(root.parent / "predictor/gate/basis.npz")},
            "normalization_sha256": file_hash(root.parent / "features/normalization.npz"),
            "test_accessed": False,
        },
    )
    load_decoder(c, "gate")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "features_ready.json",
        {
            "status": "PASS_CLEAN_FUTURE_TRAIN_ONLY_FEATURES",
            "cache_signature": ix["signature"],
            "episodes": len(manifest["records"]),
            "splits": sorted(observed_splits),
            "encoder": ix["encoder"],
            "encoder_hash": ix["encoder_hash"],
            "normalization_sha256": file_hash(
                Path(c["output"]) / "features/normalization.npz"
            ),
            "basis_sha256": file_hash(
                Path(c["output"]) / "predictor/gate/basis.npz"
            ),
            "probe_metrics_sha256": file_hash(root / "metrics.json"),
            "train_only_transforms": True,
            "held_out_used_for_fitting": False,
            "test_accessed": False,
        },
    )
    print((output / "features_ready.json").read_text(), end="")


if __name__ == "__main__":
    main()
