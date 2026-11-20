"""Diagnostic: does a predictor-style input basis keep the marker cue? (hidden-CoG PushT v5/v5b)

The repo predictor first projects every frame onto a whitened PCA basis fitted on all train
frames up to the query (phase1/model.py fit_basis, rank 128). PCA ranks directions by variance,
so a cue that is a small change in few frames can fall outside the basis. This measures, on
train/validation only:

  footprint     feature distance between siblings' marker frames (they differ only by the dot),
                against the feature norm and the change between consecutive frames
  kept_energy   fraction of that sibling difference lying inside the basis
  class probes  CoG class from one marker frame, through (a) the predictor-style basis and
                (b) a basis fitted on marker frames (as probe_pusht_cog_features.py does)

Marker frame indices come from evaluation_only/ (privileged); this is a diagnostic only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

SPLITS = ("train", "validation")
RANK = 128


def main():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    index = [r for r in json.loads((args.data / "observation_index.json").read_text())["records"]
             if r["split"] in SPLITS]
    ev = {r["id"]: json.loads((args.data / "evaluation_only" / f"{r['id']}.json").read_text()) for r in index}
    names = sorted({e["cog_class"] for e in ev.values()})
    z = {r["id"]: np.load(args.features / "features" / f"{r['id']}.npy") for r in index}
    train = [r for r in index if r["split"] == "train"]
    val = [r for r in index if r["split"] == "validation"]

    # Predictor-style basis: all train frames up to the query, standardized, whitened PCA.
    frames = np.concatenate([z[r["id"]][:r["query_frame"] + 1] for r in train])
    scaler = StandardScaler().fit(frames)
    pca = PCA(RANK, whiten=True, random_state=0).fit(scaler.transform(frames))
    components = pca.components_  # orthonormal rows in standardized space

    families = {}
    for r in index:
        families.setdefault((r["split"], r["family_id"]), []).append(r["id"])
    sib, kept, consecutive = [], [], []
    for (split, _), ids in families.items():
        for t in ev[ids[0]]["marker_frames"]:
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    d = z[ids[a]][t] - z[ids[b]][t]
                    sib.append(float(np.linalg.norm(d)))
                    ds = d / scaler.scale_
                    kept.append(float(np.linalg.norm(components @ ds) ** 2 / max(np.linalg.norm(ds) ** 2, 1e-12)))
        x = z[ids[0]]
        consecutive += [float(np.linalg.norm(x[t + 1] - x[t])) for t in range(1, ev[ids[0]]["marker_frames"][0] - 1)]
    norms = [float(np.linalg.norm(v[0])) for v in z.values()]
    q = lambda v: {"median": float(np.median(v)), "q10": float(np.quantile(v, .1)), "q90": float(np.quantile(v, .9))}

    def marker_frame(r):
        return z[r["id"]][ev[r["id"]]["marker_frames"][1]]

    y = {s: np.array([names.index(ev[r["id"]]["cog_class"]) for r in rows]) for s, rows in (("train", train), ("validation", val))}
    groups = np.array([r["family_id"] for r in train])
    inputs = {
        "predictor_style_basis": (lambda rows: pca.transform(scaler.transform(np.stack([marker_frame(r) for r in rows])))),
    }
    marker_train = np.stack([marker_frame(r) for r in train])
    ms = StandardScaler().fit(marker_train)
    mp = PCA(RANK, random_state=0).fit(ms.transform(marker_train))
    inputs["marker_fitted_basis"] = lambda rows: mp.transform(ms.transform(np.stack([marker_frame(r) for r in rows])))
    probes = {}
    for name, f in inputs.items():
        xt, xv = f(train), f(val)
        probes[name] = {}
        for kind, model in (("linear", make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))),
                            ("mlp", make_pipeline(StandardScaler(), MLPClassifier((256,), alpha=1.0, max_iter=2000, random_state=0)))):
            cv = float(cross_val_score(model, xt, y["train"], groups=groups, cv=GroupKFold(5)).mean())
            model.fit(xt, y["train"])
            probes[name][kind] = {"train_cv_accuracy": cv, "validation_accuracy": float(model.score(xv, y["validation"]))}

    result = {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "data": args.data.name,
        "features": str(args.features), "marker_radius_px": json.loads((args.data / "protocol.json").read_text())["marker"]["radius_px_at_96"],
        "classes": names, "chance": 1 / len(names), "basis_rank": RANK,
        "basis": "StandardScaler + whitened PCA on all train frames up to the query (predictor-style)",
        "footprint": {"sibling_marker_l2": q(sib), "feature_norm": q(norms), "consecutive_frame_l2": q(consecutive)},
        "kept_energy_of_sibling_marker_difference": q(kept),
        "class_probes_one_marker_frame": probes,
        "privileged": "marker frame indices from evaluation_only; diagnostic only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("data", "marker_radius_px", "footprint",
                                             "kept_energy_of_sibling_marker_difference", "class_probes_one_marker_frame")}, indent=1))


if __name__ == "__main__":
    main()
