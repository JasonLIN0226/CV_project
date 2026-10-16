"""Validate the complete scalar battery; reproduce declared, paired summaries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(16 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def interval(v):
    rng = np.random.default_rng(0)
    means = np.empty(2000)
    for i in range(2000):
        means[i] = v[rng.integers(0, len(v), len(v))].mean()
    return np.percentile(means, [2.5, 97.5]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evaluation_dir", type=Path)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    out = args.evaluation_dir
    protocol = json.loads((root / "reference_protocol.json").read_text())
    declared = json.loads((root / "aggregation_protocol.json").read_text())
    if not (out / "completion.json").is_file():
        raise RuntimeError("partial battery; no main-result table")
    completion = json.loads((out / "completion.json").read_text())
    if completion["records"] != 3796 * 8:
        raise RuntimeError("incomplete official record count")
    scalar = out / "scalars"
    elig = json.loads((scalar / "eligible_clips.json").read_text())
    if elig["scanned"] != 3840 or len(elig["clips"]) != 3796:
        raise RuntimeError("test-set coverage differs")
    ids = {int(x[0]) for x in elig["clips"]}
    expected = {int(x[0]): (int(x[1]), x[2]) for x in elig["clips"]}
    arrays, file_proof = {}, []
    for arm in protocol["arms"]:
        found = {int(p.stem): p for p in (scalar / arm).glob("*.npz")}
        if set(found) != ids:
            raise RuntimeError("missing/extra records; never use a smaller intersection")
        arrays[arm] = {}
        for idx, p in found.items():
            with np.load(p, allow_pickle=False) as z:
                rec = {k: z[k].copy() for k in z.files}
            steps, name = expected[idx]
            if int(rec["n_steps"]) != steps or str(rec["video_id"]) != name:
                raise RuntimeError("record/eligible trajectory identity differs")
            for metric in ("psnr", "lpips", "dreamsim"):
                if rec[metric].shape != (steps,) or not np.isfinite(rec[metric]).all():
                    raise RuntimeError("nonfinite/incomplete metric")
            if len(rec["frac"]) != steps or not np.all(np.diff(rec["frame_idx"]) > 0):
                raise RuntimeError("record timeline differs")
            arrays[arm][idx] = rec
            file_proof.append({"path": str(p.relative_to(out)), "sha256": digest(p)})
    grid = np.linspace(.02, 1, 25)
    routes = {"ABA": [], "ABCA": [], "both": sorted(ids)}
    for idx in sorted(ids):
        name = expected[idx][1]
        route = "ABCA" if "/ABCA/" in name else "ABA" if "/ABA/" in name else None
        if route is None:
            raise RuntimeError("unknown route")
        routes[route].append(idx)
    rows, paired = [], []
    for route, group in routes.items():
        for window in declared["windows"]:
            chosen = [i for i in group if expected[i][0] >= 25] if window == "LAST_5_SECONDS_ELIGIBLE_ONLY" else group
            if not chosen:
                raise RuntimeError("empty declared evaluation group")
            values = {}
            for arm in protocol["arms"]:
                for metric in ("psnr", "lpips", "dreamsim"):
                    v = []
                    for idx in chosen:
                        rec = arrays[arm][idx]
                        if window == "FULL_TRAJECTORY":
                            score = rec[metric].mean()
                        elif window == "FINAL_QUARTER":
                            score = np.interp(grid, rec["frac"], rec[metric])[grid >= .75].mean()
                        else:
                            score = rec[metric][-25:].mean()
                        v.append(score)
                    v = np.asarray(v)
                    values[(arm, metric)] = v
                    ci = interval(v)
                    rows.append({"route": route, "window": window, "arm": arm,
                                 "metric": metric, "clips": len(chosen), "mean": float(v.mean()),
                                 "ci_low": ci[0], "ci_high": ci[1]})
            for baseline in ("B_posefa_temporal", "B_posefa_worldmem", "pnp_longlive"):
                for metric in ("psnr", "lpips", "dreamsim"):
                    a = values[("pnp_learned_vmz", metric)]
                    b = values[(baseline, metric)]
                    improvement = a - b if metric == "psnr" else b - a
                    ci = interval(improvement)
                    paired.append({"route": route, "window": window, "baseline": baseline,
                                   "metric": metric, "clips": len(chosen),
                                   "absolute_improvement": float(improvement.mean()),
                                   "ci_low": ci[0], "ci_high": ci[1],
                                   "win_rate": float((improvement > 0).mean()),
                                   "two_sided_wilcoxon_p": float(wilcoxon(improvement).pvalue) if np.any(improvement) else 1.0})
    for name, data in [("summary_scores.csv", rows), ("paired_improvements.csv", paired)]:
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    (out / "raw_scalar_files_proof.json").write_text(json.dumps(file_proof, indent=2) + "\n")
    report = {"status": "COMPLETE_SCALAR_BATTERY_VALIDATED_AND_AGGREGATED",
              "records": len(file_proof), "eligible_clips": len(ids), "test_clips_scanned": 3840,
              "arms": protocol["arms"], "routes": {k: len(v) for k, v in routes.items()},
              "primary_author_notebook_route": "ABCA", "primary_window": "FINAL_QUARTER",
              "full_trajectory_and_both_routes_reported": True,
              "summary_rows": len(rows), "paired_rows": len(paired),
              "aggregation_protocol_sha256": digest(root / "aggregation_protocol.json"),
              "source_sha256": digest(Path(__file__)),
              "current_models_and_data_final_native_audit_still_required": True,
              "Phase1_pass_claimed": False}
    (out / "aggregation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
