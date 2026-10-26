"""Predict CONF with frozen student archives; never call training functions.

Configure QR_* paths as in code/path_config.py. The parent workflow must pass
the DEV replay and integrity gates and create its one-shot opening receipt first.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import pipe
from experiments.phase2 import fit_terminal_students as fts


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reader", choices=("multihead", "item_attention"), required=True)
    ap.add_argument("--students", type=Path, required=True, help="Parent of reader_standard / reader_distilled archive directories")
    ap.add_argument("--appearance", type=Path, required=True)
    ap.add_argument("--opening-receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="Prediction parent, separate from frozen student archives")
    args = ap.parse_args()
    opening = json.loads(args.opening_receipt.read_text())
    if opening.get("split") != "confirmation" or opening.get("seed") != 20261104:
        raise ValueError("Require a recorded CONF opening, after all gates pass")
    if opening.get("dev_reproduction_passed") is not True or opening.get("integrity_passed") is not True:
        raise ValueError("DEV replay and integrity must pass before predictions")
    st = pipe.Store()
    recs = st.records
    if not recs or {r["split"] for r in recs} != {"confirmation"} or len({r["family"] for r in recs}) != 48:
        raise ValueError("Expected CONF-only 48-family store")
    ids = [r["id"] for r in recs]
    appearances = np.load(args.appearance, allow_pickle=False)
    if appearances["ids"].tolist() != ids:
        raise ValueError("Appearance rows do not align to CONF")
    features = [st.student_features(r) for r in recs]
    inputs = {"items": np.stack([x[0] for x in features]), "context": np.stack([x[1] for x in features])}
    app = {"items": appearances["items"], "context": np.empty((len(recs), 0), np.float32)}
    for mode in ("standard", "distilled"):
        output = args.output / f"{args.reader}_{mode}"
        if output.exists() and any(output.iterdir()):
            raise ValueError("Refusing to overwrite saved CONF predictions")
    for mode in ("standard", "distilled"):
        model_dir = args.students / f"{args.reader}_{mode}"
        prediction = fts.predict_students(model_dir, inputs, app if mode == "standard" else None)
        # The immutable predictor loads every archived arm; only cost is reported
        # for distilled groups, exactly as the frozen DEV evaluation specifies.
        report = prediction if mode == "standard" else {k: v for k, v in prediction.items() if "_cost_seed" in k}
        output = args.output / f"{args.reader}_{mode}"
        output.mkdir(parents=True, exist_ok=True)
        file = output / "confirmation_predictions.npz"
        np.savez_compressed(file, ids=np.asarray(ids),
                            **{k + "__drop": v["drop"] for k, v in report.items()},
                            **{k + "__logits": v["logits"] for k, v in report.items()})
        receipt = {"reader": args.reader, "mode": mode, "split": "confirmation", "records": len(recs),
                   "reported_arms": sorted(report), "training_called": False,
                   "source_sha256": sha(__file__), "student_manifest_sha256": sha(model_dir / "manifest.json"),
                   "opening_receipt_sha256": sha(args.opening_receipt), "predictions_sha256": sha(file),
                   "timing": {k: v["timing"] for k, v in report.items()}}
        (output / "prediction_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"reader": args.reader, "mode": mode, "records": len(recs), "reported_arms": len(report)}), flush=True)


if __name__ == "__main__":
    main()
