"""Encode the hidden-CoG PushT corpus with frozen I-JEPA and probe class readability.

Stage `encode` (GPU) runs the repository's FrozenIJepa (pinned ViT-H/14, grid2
pooling, 5120 features) on every frame and saves one float32 array per episode.
Stage `probe` (CPU) asks whether the CoG class is linearly / nonlinearly readable
from chosen frames (designs v2/v4: continuous trajectories; v5: marker, see probe_inputs). Siblings share the
block pose after the probe and every frame from `converge_frame` on; they start
at angles that differ by the probe turn. The main probes pair frames:

  recent_window    frames query-3..query       control, must be chance (siblings identical)
  reset_only       reset frame                 single history frame; start angle is random
  reset_vs_query   reset frame + query frame   the relative turn: what a memory must keep
  probe_end_only   last retreat frame          pusher placement relative to the turned block
  target_only      query + HORIZON
  target_vs_query  query frame + target frame  does the class change the future?

Normalization, PCA(128) and the classifiers are fit on TRAIN only; the inverse
regularization is chosen by family-grouped 5-fold CV on TRAIN; VALIDATION is
reported. The evaluation split is never read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PHASE1_SRC = ROOT / "experiments/predictor/control/src"
ENCODER = {"backend": "ijepa", "repository": "facebook/ijepa_vith14_1k",
           "revision": "f157467ea509bc356ff9f61fd3c0d840eec5e04e",
           "dtype": "float32", "pooling": "grid2", "local_files_only": True}
SPLITS = ("train", "validation")  # evaluation is reserved
RECENT_LENGTH = 4
PCA_DIM = 128
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_index(data: Path):
    index = json.loads((data / "observation_index.json").read_text())
    return [r for r in index["records"] if r["split"] in SPLITS], index


def encode(data: Path, out: Path, cache_dir: Path, batch_size: int):
    import torch
    sys.path.insert(0, str(PHASE1_SRC))
    from phase1.encoders import FrozenIJepa

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for encoding")
    out.mkdir(parents=True, exist_ok=False)
    (out / "features").mkdir()
    config = {**ENCODER, "cache_dir": str(cache_dir), "batch_size": batch_size}
    encoder = FrozenIJepa(config, torch.device("cuda"))
    if any(p.requires_grad for p in encoder.model.parameters()) or encoder.model.training:
        raise AssertionError("Encoder must be frozen and in eval mode")
    records, _ = load_index(data)
    started = time.perf_counter()
    artifacts = {}
    for k, record in enumerate(records):
        with np.load(data / record["observations"], allow_pickle=False) as archive:
            rgb = archive["rgb"]
        z = np.concatenate([encoder.encode(rgb[i:i + batch_size]) for i in range(0, len(rgb), batch_size)])
        if z.shape != (len(rgb), 5120) or not np.isfinite(z).all():
            raise AssertionError(f"Bad features for {record['id']}: {z.shape}")
        path = out / "features" / f"{record['id']}.npy"
        np.save(path, z.astype(np.float32))
        artifacts[path.name] = sha256(path)
        if k % 50 == 0:
            print(f"{k}/{len(records)} episodes, {time.perf_counter() - started:.0f}s", flush=True)
    meta = {k: v for k, v in encoder.metadata.items() if k not in {"model_config", "processor"}}
    write_json(out / "encoding_manifest.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.perf_counter() - started, "data": str(data),
        "data_protocol_sha256": sha256(data / "protocol.json"),
        "splits_encoded": list(SPLITS), "episodes": len(records), "encoder": meta,
        "config": config, "gpu": torch.cuda.get_device_name(0), "feature_sha256": artifacts,
    })


def probe_inputs(features: dict, record: dict, marker_frames=None):
    """Probe designs (v2/v4) use the probe end; the marker design (v5) uses the middle marker
    frame, which is privileged (evaluation_only) and read here only for this diagnostic."""
    z = features[record["id"]]
    q, t = record["query_frame"], record["target_frame"]
    inputs = {
        "recent_window": z[q - RECENT_LENGTH + 1:q + 1].ravel(),
        "reset_only": z[0],
        "reset_vs_query": np.concatenate([z[0], z[q] - z[0]]),
    }
    if marker_frames is None:
        inputs["probe_end_only"] = z[record["probe_end_frame"]]
    else:
        m = marker_frames[len(marker_frames) // 2]
        inputs["marker_only"] = z[m]
        inputs["marker_vs_query"] = np.concatenate([z[m], z[q] - z[m]])
    inputs["target_only"] = z[t]
    inputs["target_vs_query"] = np.concatenate([z[q], z[t] - z[q]])
    return inputs


def fit_probe(x_train, y_train, groups, x_val, y_val, kind):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def model(c):
        head = (LogisticRegression(C=c, max_iter=5000) if kind == "linear" else
                MLPClassifier(hidden_layer_sizes=(256,), alpha=1.0 / c, max_iter=2000, random_state=0))
        return make_pipeline(StandardScaler(), PCA(PCA_DIM, random_state=0), head)

    cv = {}
    for c in C_GRID:
        scores = []
        for tr, te in GroupKFold(5).split(x_train, y_train, groups):
            scores.append(model(c).fit(x_train[tr], y_train[tr]).score(x_train[te], y_train[te]))
        cv[c] = float(np.mean(scores))
    best = max(cv, key=cv.get)
    final = model(best).fit(x_train, y_train)
    return {"C": best, "train_cv_accuracy": cv[best], "train_accuracy": float(final.score(x_train, y_train)),
            "validation_accuracy": float(final.score(x_val, y_val)), "cv_by_C": {str(k): v for k, v in cv.items()}}


def probe(data: Path, feat_dir: Path, out: Path):
    records, _ = load_index(data)
    classes = json.loads((data / "evaluation_only" / "sibling_classes.json").read_text())
    names = sorted({c for v in classes.values() for c in v})
    features = {r["id"]: np.load(feat_dir / "features" / f"{r['id']}.npy") for r in records}

    def label(r):
        return names.index(classes[r["family_id"]][int(r["id"].rsplit("_s", 1)[1])])

    # Sibling feature distances: zero where pixels match, scale against 12-step scene change.
    by_family = {}
    for r in records:
        by_family.setdefault(r["family_id"], []).append(r)
    sib = {"reset": [], "query": [], "target": [], "same_episode_query_to_target": []}
    for fam in by_family.values():
        z = [features[r["id"]] for r in fam]
        q, t = fam[0]["query_frame"], fam[0]["target_frame"]
        for key, i in (("reset", 0), ("query", q), ("target", t)):
            sib[key] += [float(np.linalg.norm(z[a][i] - z[b][i])) for a in range(3) for b in range(a + 1, 3)]
        sib["same_episode_query_to_target"] += [float(np.linalg.norm(x[t] - x[q])) for x in z]
    distances = {k: {"median": float(np.median(v)), "max": float(np.max(v))} for k, v in sib.items()}

    results = {}
    split = {s: [r for r in records if r["split"] == s] for s in SPLITS}
    y = {s: np.array([label(r) for r in split[s]]) for s in SPLITS}
    groups = np.array([r["family_id"] for r in split["train"]])
    marker = None
    if "probe_end_frame" not in records[0]:
        marker = {r["id"]: json.loads((data / "evaluation_only" / f"{r['id']}.json").read_text())["marker_frames"]
                  for r in records}
    inputs = {s: [probe_inputs(features, r, None if marker is None else marker[r["id"]]) for r in split[s]]
              for s in SPLITS}
    for name in inputs["train"][0]:
        x = {s: np.stack([d[name] for d in inputs[s]]) for s in SPLITS}
        results[name] = {kind: fit_probe(x["train"], y["train"], groups, x["validation"], y["validation"], kind)
                         for kind in ("linear", "mlp")}
        print(name, {k: round(v["validation_accuracy"], 3) for k, v in results[name].items()}, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "probe_results.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "classes": names, "chance": 1 / len(names),
        "counts": {s: len(split[s]) for s in SPLITS}, "pca_dim": PCA_DIM, "C_grid": list(C_GRID),
        "selection": "family-grouped 5-fold CV on train; validation reported; evaluation not read",
        "sibling_feature_l2": distances, "probes": results,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=("encode", "probe"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "encode":
        encode(args.data, args.features, args.cache_dir, args.batch_size)
    else:
        probe(args.data, args.features, args.output or args.features / "probe")


if __name__ == "__main__":
    main()
