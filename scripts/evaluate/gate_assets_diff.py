"""Compare two Gate asset trees stage by stage: dataset, feature cache, normalisation, bases, decoder.

    python scripts/evaluate/gate_assets_diff.py --reference <team tree> --candidate <rebuilt tree> --out <dir> [--weights]

Used to attribute a difference in retrained results: if the regenerated dataset is byte-identical
and the re-encoded features agree closely, a differing number comes from predictor training, not
from data or encoding. Predictor training is compared through the per-epoch training logs (over
the epochs both trees have, every column except wall-clock time). --weights also loads the
checkpoints of both trees and compares every tensor; checkpoints are pickles, so use it only on
trees built by your own runs.
Writes <out>/gate_assets_diff.json.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bounded_memory.data.io import file_hash, read_json, write_json  # noqa: E402


def subspace_overlap(a, b):
    """Mean squared cosine between the column spaces of two bases (1.0 = same subspace)."""
    qa, qb = np.linalg.qr(a.astype(np.float64))[0], np.linalg.qr(b.astype(np.float64))[0]
    return float(np.linalg.norm(qa.T @ qb) ** 2 / min(qa.shape[1], qb.shape[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, required=True); ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", action="store_true", help="compare predictor tensors (loads pickles: own trees only)")
    args = ap.parse_args()
    ref, new = args.reference.resolve(), args.candidate.resolve()
    report = {"reference": str(ref), "candidate": str(new)}

    a, b = read_json(ref / "dataset/manifest.json")["records"], read_json(new / "dataset/manifest.json")["records"]
    same_ids = [r["id"] for r in a] == [r["id"] for r in b]
    report["dataset"] = {"episodes": [len(a), len(b)], "same_episode_ids_and_order": same_ids,
                         "byte_identical_episode_files": int(sum(x["obs_sha256"] == y["obs_sha256"] and x["eval_sha256"] == y["eval_sha256"]
                                                                 for x, y in zip(a, b))) if same_ids else None}

    ia, ib = read_json(ref / "features/index.json"), read_json(new / "features/index.json")
    worst, total, count, identical = 0.0, 0.0, 0, 0
    for rec in a if same_ids else []:
        with np.load(ref / "features" / ia["episodes"][rec["id"]]["path"]) as f, np.load(new / "features" / ib["episodes"][rec["id"]]["path"]) as g:
            x, y = f["z"].astype(np.float64), g["z"].astype(np.float64)
        d = np.abs(x - y)
        worst, total, count = max(worst, float(d.max())), total + float(d.sum()), count + d.size
        identical += int(ia["episodes"][rec["id"]]["sha256"] == ib["episodes"][rec["id"]]["sha256"])
    report["features"] = {"encoder_hash_equal": ia["encoder_hash"] == ib["encoder_hash"], "byte_identical_files": identical,
                          "max_abs_diff": worst, "mean_abs_diff": total / max(count, 1)}

    with np.load(ref / "features/normalization.npz") as f, np.load(new / "features/normalization.npz") as g:
        report["normalization"] = {"max_abs_diff_mean": float(np.abs(f["mean"] - g["mean"]).max()),
                                   "max_abs_diff_std": float(np.abs(f["std"] - g["std"]).max()),
                                   "byte_identical": file_hash(ref / "features/normalization.npz") == file_hash(new / "features/normalization.npz")}
    with np.load(ref / "predictor/gate/basis.npz") as f, np.load(new / "predictor/gate/basis.npz") as g:
        report["basis"] = {"byte_identical": file_hash(ref / "predictor/gate/basis.npz") == file_hash(new / "predictor/gate/basis.npz"),
                           "projection_subspace_overlap": subspace_overlap(f["projection"], g["projection"]),
                           "output_basis_subspace_overlap": subspace_overlap(f["output_basis"], g["output_basis"]),
                           "max_abs_diff_input_mean": float(np.abs(f["input_mean"] - g["input_mean"]).max())}
    with np.load(ref / "probes/gate_state_decoder.npz") as f, np.load(new / "probes/gate_state_decoder.npz") as g:
        report["state_decoder"] = {"max_abs_diff_weight": float(np.abs(f["weight"] - g["weight"]).max()),
                                   "ridge": [float(f["ridge"]), float(g["ridge"])]}
    report["predictors"] = {}
    for seed in (17, 29, 43):
        entry = report["predictors"][str(seed)] = {}
        logs = [tree / f"predictor/gate/seed_{seed}/training.csv" for tree in (ref, new)]
        if not all(path.exists() for path in logs):
            entry["status"] = "not trained in " + " and ".join(name for name, path in zip(("reference", "candidate"), logs) if not path.exists())
            continue
        rows = [list(csv.DictReader(path.open())) for path in logs]
        shared = min(len(rows[0]), len(rows[1]))
        columns = [k for k in rows[0][0] if k != "seconds"]          # wall-clock time is the only column allowed to differ
        numeric = [k for k in columns if k not in ("selection_metric", "mix_counts")]
        entry.update(epochs=[len(rows[0]), len(rows[1])], epochs_compared=shared,
                     epochs_with_identical_log=int(sum(all(x[k] == y[k] for k in columns) for x, y in zip(rows[0], rows[1]))),
                     max_abs_log_diff=max((abs(float(x[k]) - float(y[k])) for x, y in zip(rows[0], rows[1]) for k in numeric), default=None))
        files = [tree / f"predictor/gate/seed_{seed}/best.pt" for tree in (ref, new)]
        if len(rows[0]) != len(rows[1]):
            entry["status"] = "training stopped early in one tree; logs compared over the shared epochs, weights not compared"
            continue
        entry["byte_identical"] = file_hash(files[0]) == file_hash(files[1])
        if args.weights:
            import torch
            x, y = (torch.load(path, map_location="cpu", weights_only=False) for path in files)
            x, y = x.get("model", x), y.get("model", y)
            entry.update(same_tensor_names=sorted(x) == sorted(y),
                         max_abs_weight_diff=max(float((x[k].double() - y[k].double()).abs().max()) for k in x if k in y),
                         all_tensors_equal=sorted(x) == sorted(y) and all(torch.equal(x[k], y[k]) for k in x))
    write_json(args.out / "gate_assets_diff.json", report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
