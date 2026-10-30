"""Fit the new train-only transforms and probes and encode one structured arm."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from structured_design import STRUCTURED_LINE


HERE = Path(__file__).resolve().parent
CONTROL_SOURCE = HERE.parent / "predictor" / "control" / "src"
if str(CONTROL_SOURCE) not in sys.path:
    sys.path.insert(0, str(CONTROL_SOURCE))

from phase1.cache import encode, fit_normalization, load_index  # noqa: E402
from phase1.data import require_audit  # noqa: E402
from phase1.inputs import FeatureStore  # noqa: E402
from phase1.model import fit_basis  # noqa: E402
from phase1.probes import load_decoder, probe  # noqa: E402
from phase1.utils import digest, file_hash, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--probe-splits",
        default="train,validation",
        help="Comma-separated probe splits; this experiment uses train,validation only",
    )
    args = parser.parse_args()
    probe_splits = tuple(item.strip() for item in args.probe_splits.split(",") if item.strip())
    if probe_splits != ("train", "validation"):
        raise ValueError("The clean-future experiment is strictly train/validation-only")
    c = json.loads(args.config.read_text())
    if c["data"]["tasks"] != ["gate"]:
        raise ValueError("Structured model experiment is Gate-only")
    if c["experiment"] != {
        "stage": "development",
        "arm": "reference",
        "tasks": ["gate"],
    }:
        raise ValueError("Use the allowed reference experiment wrapper")
    if c["device"] == "cuda":
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA device is required")
        if "A100" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Course A100 required")
    ix = encode(c)
    fit_normalization(c)
    store = FeatureStore(c)
    basis = fit_basis(c, store, "gate")
    store.bases = {"gate": basis}
    store.projected_features(store.manifest["records"][0])
    probe_metrics = probe(c, splits=probe_splits)
    load_decoder(c, "gate")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "features_ready.json",
        {
            "status": "PASS_NEW_TRAIN_ONLY_FEATURES_TRANSFORMS_PROBES",
            "dataset_signature": store.manifest["signature"],
            "cache_signature": ix["signature"],
            "episodes": len(store.manifest["records"]),
            "encoder": ix["encoder"],
            "encoder_hash": ix["encoder_hash"],
            "normalization_sha256": file_hash(Path(c["output"]) / "features/normalization.npz"),
            "basis_sha256": file_hash(Path(c["output"]) / "predictor/gate/basis.npz"),
            "probe_metrics_sha256": file_hash(
                Path(c["output"]) / "probes/metrics.json"
            ),
            "probe_results": len(probe_metrics["results"]),
            "probe_splits": list(probe_splits),
            "test_access": False,
            "train_only_transforms": True,
            "held_out_used_for_fitting": False,
            "line_color_rgb": list(STRUCTURED_LINE),
            "training": False,
        },
    )
    print((output / "features_ready.json").read_text(), end="")


if __name__ == "__main__":
    main()
