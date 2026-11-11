"""Diagnostic: does the CoG cue survive frozen I-JEPA encoding? (hidden-CoG PushT v4)

End-to-end class probes on I-JEPA features were at chance for every history input
(384 labelled train episodes, 102 probe types). This asks whether that is a data
problem rather than missing information, in two supervised stages:

  1. Pose decoder (per frame): I-JEPA grid2 features (5120) -> StandardScaler -> PCA(256)
     -> Ridge or MLP regressor -> simulator block centre (x, y), (sin, cos) of the block
     angle and the pusher (x, y). Every train frame is a labelled example (~34k).
  2. Class from poses: decoded poses at the first frame, the end of the probe and the
     query -> the hand-built pose features of dev_pusht_cog_natural.py -> gradient-boosted
     trees, trained on decoded train poses, tested on decoded validation poses. The same
     classifier on true poses is the upper bound.

Stage 1 uses privileged simulator labels from evaluation_only/: this measures what the
features contain; it is not an input any retention method or predictor may use.
Transforms and models are fit on train only; the evaluation split is never read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pusht_cog_cue as v1
from dev_pusht_cog_natural import pose_features, relative_features

SPLITS = ("train", "validation")


def load(data: Path, features: Path):
    index = [r for r in json.loads((data / "observation_index.json").read_text())["records"]
             if r["split"] in SPLITS]
    z, y, label = {}, {}, {}
    for r in index:
        d = json.loads((data / "evaluation_only" / f"{r['id']}.json").read_text())
        pose = np.array(d["block_pose"])
        center = pose[:, :2] + np.stack([v1.rotate(v1.BLOCK_CENTER, a) for a in pose[:, 2]])
        y[r["id"]] = np.column_stack([center, np.sin(pose[:, 2]), np.cos(pose[:, 2]),
                                      np.array(d["pusher_position"])])
        z[r["id"]] = np.load(features / "features" / f"{r['id']}.npy")
        label[r["id"]] = d["cog_class"]
    return index, z, y, label


def pose_errors(pred, true):
    angle = np.arctan2(pred[:, 2], pred[:, 3]) - np.arctan2(true[:, 2], true[:, 3])
    angle = np.degrees(np.abs(np.angle(np.exp(1j * angle))))
    centre = np.linalg.norm(pred[:, :2] - true[:, :2], axis=1)
    pusher = np.linalg.norm(pred[:, 4:] - true[:, 4:], axis=1)
    q = lambda v: {"median": float(np.median(v)), "q90": float(np.quantile(v, .9))}
    return {"block_centre_px": q(centre), "block_angle_deg": q(angle), "pusher_px": q(pusher)}


def state_at(p, t):
    return {"center": p[t, :2].tolist(), "angle": float(np.arctan2(p[t, 2], p[t, 3])), "pusher": p[t, 4:].tolist()}


def classify(index, poses, label, names):
    from sklearn.ensemble import HistGradientBoostingClassifier
    rows = {s: [r for r in index if r["split"] == s] for s in SPLITS}

    def inputs(r):
        p = poses[r["id"]]
        s0, s1, q = state_at(p, 0), state_at(p, r["probe_end_frame"]), state_at(p, r["query_frame"])
        return {"present": pose_features(q),
                "first_frame+present": pose_features(s0) + pose_features(q) + relative_features(s0, q),
                "probe_end+present": pose_features(s1) + pose_features(q) + relative_features(s1, q)}

    X = {s: [inputs(r) for r in rows[s]] for s in SPLITS}
    y = {s: np.array([names.index(label[r["id"]]) for r in rows[s]]) for s in SPLITS}
    out = {}
    for key in X["train"][0]:
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
        model.fit([x[key] for x in X["train"]], y["train"])
        out[key] = float(model.score([x[key] for x in X["validation"]], y["validation"]))
    return out


def main():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    index, z, y, label = load(args.data, args.features)
    names = sorted(set(label.values()))
    train = [r for r in index if r["split"] == "train"]
    val = [r for r in index if r["split"] == "validation"]
    x_train = np.concatenate([z[r["id"]] for r in train])
    y_train = np.concatenate([y[r["id"]] for r in train])
    scaler = StandardScaler().fit(x_train)
    pca = PCA(256, random_state=0).fit(scaler.transform(x_train))
    reduce = lambda x: pca.transform(scaler.transform(x))
    y_scaler = StandardScaler().fit(y_train)

    results = {"true_poses": {"class_accuracy": classify(index, y, label, names)}}
    print("true poses", results["true_poses"], flush=True)
    regressors = {
        "ridge": Ridge(alpha=10.0),
        "mlp": MLPRegressor((512, 256), alpha=1e-3, max_iter=200, early_stopping=True, random_state=0),
    }
    for name, model in regressors.items():
        model.fit(reduce(x_train), y_scaler.transform(y_train))
        decoded = {r["id"]: y_scaler.inverse_transform(model.predict(reduce(z[r["id"]]))) for r in index}
        errors = pose_errors(np.concatenate([decoded[r["id"]] for r in val]),
                             np.concatenate([y[r["id"]] for r in val]))
        results[f"decoded_{name}"] = {"validation_pose_error": errors,
                                      "class_accuracy": classify(index, decoded, label, names)}
        print(name, results[f"decoded_{name}"], flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "data": args.data.name, "features": args.features.name,
        "counts": {"train_episodes": len(train), "validation_episodes": len(val),
                   "train_frames": int(len(x_train))},
        "classes": names, "chance": 1 / len(names),
        "pose_targets": ["block_centre_x", "block_centre_y", "sin_angle", "cos_angle", "pusher_x", "pusher_y"],
        "feature_reduction": "StandardScaler + PCA(256), fit on train frames",
        "classifier": "HistGradientBoosting(max_iter=300, lr=0.05) on dev_pusht_cog_natural pose features",
        "privileged_labels": "stage 1 uses simulator poses; diagnostic only",
        "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
