"""Retrain the Synergy readers (multihead, item_attention) on re-encoded TRAIN/VAL features.

    python scripts/train_predictor/synergy.py --terminal <terminal/run> --data <trainvaldev/data> \
        --out <dir> --device cpu [--released <reader-aggregation run folder>]

Runs environments/synergy/train_readers.py::train_one with the frozen config
(configs/synergy/reader_aggregation_20261025.json): seed 17, 1,024 updates, pure feature MSE,
memory sizes 0/1/2/3 in rotation, checkpoint with the lowest validation MSE. The original study
also verified that its inputs were the team's original feature cache, byte for byte; that cache
was never published, so the readers are trained on features re-encoded from regenerated frames
(scripts/prepare_data/synergy_trainvaldev.py).

Writes <out>/readers/<variant>/reader_selected.pt and reader_locked.json (checksum, schedule hash,
validation loss), and <out>/retraining_report.json comparing them with the released readers.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/synergy/reader_aggregation_20261025.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terminal", type=Path, required=True, help="terminal-study release folder (basis.npz, protocol.json)")
    ap.add_argument("--data", type=Path, required=True, help="re-encoded data folder (records.json, features.npy, actions.npz, ...)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=["multihead", "item_attention"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--released", type=Path, help="released reader folder, to compare schedule hash and validation loss")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import torch
    from bounded_memory.environments.synergy import train_readers as study
    from bounded_memory.predictors.readers import build_reader

    out = args.out.resolve()
    source = out / "source"                       # the layout train_one expects: basis.npz, protocol.json, data/
    source.mkdir(parents=True, exist_ok=True)
    for name, target in (("basis.npz", args.terminal / "basis.npz"), ("protocol.json", args.terminal / "protocol.json"),
                         ("data", args.data)):
        if not (source / name).exists():
            (source / name).symlink_to(target.resolve(), target_is_directory=target.is_dir())
    c = json.loads(CONFIG.read_text())
    base_c = json.loads((source / "protocol.json").read_text())["config"]
    for key in ("steps", "validate_every", "groups_per_task", "learning_rate", "weight_decay", "reader", "history_sizes", "horizons"):
        if c[key] != base_c[key]:
            sys.exit(f"reader config and terminal-study config disagree on {key}")
    basis = dict(np.load(source / "basis.npz", allow_pickle=False))
    torch.manual_seed(c["seed"])
    common = build_reader(len(basis["input_mean"]), base_c["reader"], study.legacy.H, basis, "original", c["heads"]).state_dict()
    report = {}
    for variant in args.variants:
        locked = out / "readers" / variant / "reader_locked.json"
        if not locked.exists():
            if locked.parent.exists():
                sys.exit(f"partial reader at {locked.parent}; remove it and rerun")
            study.train_one(c, base_c, source, out, variant, args.device, common)
        mine = json.loads(locked.read_text())
        entry = {"selected_step": mine["selected_step"], "validation_mse": mine["validation"]["equal_task_size_mse"],
                 "training_schedule_sha256": mine["training_schedule_sha256"], "checkpoint_sha256": mine["checkpoint_sha256"]}
        if args.released:
            theirs = json.loads((args.released / "readers" / variant / "reader_locked.json").read_text())
            entry["released"] = {"selected_step": theirs["selected_step"], "validation_mse": theirs["validation"]["equal_task_size_mse"],
                                 "training_schedule_sha256": theirs["training_schedule_sha256"]}
            entry["same_training_schedule"] = mine["training_schedule_sha256"] == theirs["training_schedule_sha256"]
        report[variant] = entry
        print(variant, json.dumps(entry), flush=True)
    (out / "retraining_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
