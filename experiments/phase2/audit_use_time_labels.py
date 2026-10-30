"""Audit saved TRAIN deletion/attention labels without fitting or model calls.

Privileged cue annotations are diagnostic strata only. This script neither
changes labels nor uses confirmation/test data. The source label column j is
the loss after deleting sorted candidate j (or that candidate's attention).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOL = 1e-8


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0}
    return {"n": len(values), "mean": float(values.mean()),
            "min": float(values.min()), "p10": float(np.quantile(values, .1)),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9)),
            "max": float(values.max())}


def minimum_summary(values, cue, destroys):
    values = np.asarray(values, dtype=np.float64)
    optimal = values <= values.min() + TOL
    best = np.flatnonzero(optimal)
    cue, destroys = np.asarray(cue, bool), np.asarray(destroys, bool)
    return {"minimum": float(values.min()), "maximum": float(values.max()),
            "range": float(np.ptp(values)),
            "centered_rms": float(np.sqrt(np.mean((values-values.mean())**2))),
            "minimum_delete_indices": best.tolist(),
            "exact_argmin_index": int(np.argmin(values)),
            "tied_minimum": len(best) > 1,
            "minimum_can_delete_cue": bool(cue[best].any()),
            "every_minimum_deletes_cue": bool(cue[best].all()),
            "minimum_can_destroy_cue": bool(destroys[best].any()),
            "every_minimum_destroys_cue": bool(destroys[best].all()),
            # These compare deletion choices, not adding/removing a cue while
            # holding timestamps/actions fixed. Positive means protect cue.
            "delete_cue_minus_best_noncue_delete":
                float(values[cue].min()-values[~cue].min()) if cue.any() and (~cue).any() else None,
            "destroy_cue_minus_best_preserving_delete":
                float(values[destroys].min()-values[~destroys].min()) if destroys.any() else None}


def aggregate(rows):
    answer = {"parents": len(rows), "episodes": len({r["episode_id"] for r in rows}),
              "scene_pairs": len({r["pair"] for r in rows}),
              "candidate_cue_counts": dict(Counter(str(r["candidate_cue_count"]) for r in rows)),
              "complete_loss_possible_parents": sum(r["complete_loss_possible"] for r in rows),
              "targets_equal_each_other": sum(r["targets_equal_each_other"] for r in rows),
              "minimum_sets_disjoint": sum(r["minimum_sets_disjoint"] for r in rows)}
    for name in ("cost", "attention"):
        answer[name] = {"values": distribution([v for r in rows for v in r[name + "_labels"]]),
                        "ranges": distribution([r[name]["range"] for r in rows]),
                        "centered_rms": distribution([r[name]["centered_rms"] for r in rows])}
        for key in ("tied_minimum", "minimum_can_delete_cue", "every_minimum_deletes_cue",
                    "minimum_can_destroy_cue", "every_minimum_destroys_cue"):
            answer[name][key] = sum(r[name][key] for r in rows)
        for key in ("delete_cue_minus_best_noncue_delete", "destroy_cue_minus_best_preserving_delete"):
            margins = [r[name][key] for r in rows if r[name][key] is not None]
            answer[name][key] = {**distribution(margins),
                "negative_below_tolerance": sum(v < -TOL for v in margins),
                "within_tolerance": sum(abs(v) <= TOL for v in margins),
                "positive_above_tolerance": sum(v > TOL for v in margins)}
    return answer


def run(args):
    output, snapshot, labels = args.output.resolve(), args.snapshot.resolve(), args.labels.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; previous audits are never overwritten")
    sourcehashes, loaded = {}, {}
    def read_json(path):
        path = Path(path).resolve()
        sourcehashes[str(path.relative_to(ROOT))] = sha256(path)
        return json.loads(path.read_text())
    backup = read_json(snapshot / "backup-index.json")["files"]
    def verified(relative):
        path = (snapshot / relative).resolve()
        if not path.is_relative_to(snapshot) or relative not in backup:
            raise ValueError("Input absent from backup index: " + relative)
        digest = sha256(path)
        if digest != backup[relative]["sha256"] or path.stat().st_size != backup[relative]["bytes"]:
            raise ValueError("Backup input checksum mismatch: " + relative)
        sourcehashes[str(path.relative_to(ROOT))] = digest
        return path
    manifest = read_json(verified("data/dataset/manifest.json"))
    by_id = {r["id"]: r for r in manifest["records"] if r["task"] == "gate" and r["split"] == "train"}
    ix = read_json(verified("data/features/index.json"))
    with np.load(verified("data/features/normalization.npz"), allow_pickle=False) as archive:
        mean, std = archive["mean"].copy(), archive["std"].copy()
    metadata = read_json(labels / "training_metadata.json")
    original = read_json(ROOT / "results/literature_diagnosis/learnability-20261022/training_metadata.json")
    if metadata != original:
        raise ValueError("Training metadata differs from the exported frozen probe metadata")
    protocol = read_json(labels / "protocol.json")
    npz = labels / "training_inputs_labels.npz"
    sourcehashes[str(npz.relative_to(ROOT))] = sha256(npz)
    with np.load(npz, allow_pickle=False) as archive:
        cost, attention = archive["cost"].copy(), archive["attention"].copy()
        input_shape = list(archive["inputs"].shape)
    if len(metadata) != 1282 or cost.shape != (1282, 3) or attention.shape != cost.shape:
        raise ValueError("Unexpected fixed training label population")
    if not np.isfinite(cost).all() or not np.isfinite(attention).all():
        raise ValueError("Non-finite saved labels")
    teacher_dir = ROOT / "build/literature-baselines-20261022/dev-v3"
    teacher_meta = read_json(teacher_dir / "attention_training_metadata.json")
    teacher_path = teacher_dir / "attention_training_targets.npz"
    sourcehashes[str(teacher_path.relative_to(ROOT))] = sha256(teacher_path)
    with np.load(teacher_path, allow_pickle=False) as archive:
        teacher_labels = archive["targets"].copy()
    if len(teacher_meta) != len(metadata):
        raise ValueError("Teacher/probe metadata length mismatch")
    for i, (meta, teacher) in enumerate(zip(metadata, teacher_meta)):
        order = np.argsort(teacher["candidate_times"])
        if ({k: v for k, v in meta.items() if k != "candidate_times"} !=
                {k: v for k, v in teacher.items() if k != "candidate_times"}
                or meta["candidate_times"] != [teacher["candidate_times"][j] for j in order]
                or not np.array_equal(attention[i], teacher_labels[i][order])):
            raise ValueError("Candidate columns are not aligned with original teacher labels")
    source_paths = [Path(__file__), ROOT / "experiments/phase2/diagnose_oracle_learnability.py",
                    ROOT / "experiments/phase2/run_literature_baselines.py",
                    ROOT / "experiments/phase2/run_gate_confirmation.py"]
    for path in source_paths:
        sourcehashes[str(path.relative_to(ROOT))] = sha256(path)
    # The unchanged label-producing sources are checked against their saved run.
    for name in ("diagnose_oracle_learnability.py", "run_literature_baselines.py"):
        relative = "experiments/phase2/" + name
        if sourcehashes[relative] != protocol["source_sha256"][relative]:
            raise ValueError("Label producer source drift: " + relative)
    rows = []
    for i, meta in enumerate(metadata):
        if meta["split"] != "train" or meta["episode_id"] not in by_id:
            raise ValueError("Only old TRAIN parents may be audited")
        rec = by_id[meta["episode_id"]]
        if rec["id"] not in loaded:
            with np.load(verified("data/dataset/" + rec["eval"]), allow_pickle=False) as archive:
                evidence, classes = archive["evidence"].copy(), archive["classes"].copy()
            if evidence.shape != (1,):
                raise ValueError("This audit expects one Gate evidence group")
            info = ix["episodes"][rec["id"]]
            feature_path = verified("data/features/" + info["path"])
            if sha256(feature_path) != info["sha256"]:
                raise ValueError("Feature index checksum mismatch")
            with np.load(feature_path, allow_pickle=False) as archive:
                z = ((archive["z"]-mean)/std).astype(np.float32)
            if not np.all(classes[:rec["query"]+1] == -1) or not np.all(classes[rec["query"]+1:] != -1):
                raise ValueError("Expected first outcome class at query+1")
            # Exact normalized-feature equivalents include the duplicated cue
            # frame. Compare full vectors, never a task-state proxy.
            equivalent = [j for j in range(rec["query"]+1)
                          if np.array_equal(z[j], z[int(evidence[0])])]
            loaded[rec["id"]] = (z, evidence.tolist(), equivalent)
        z, evidence, equivalent = loaded[rec["id"]]
        t, q, times = int(meta["time"]), int(rec["query"]), meta["candidate_times"]
        if times != sorted(times) or len(set(times)) != 3 or times[-1] >= t-3:
            raise ValueError("Saved label columns require sorted three-candidate parents")
        recent = list(range(t-3, t+1))
        cue = [j in equivalent for j in times]
        recent_cue = [j for j in recent if j in equivalent]
        destroys = [is_cue and sum(cue) == 1 and not recent_cue for is_cue in cue]
        target_times = [t+3, t+6]
        regime = ("both_at_or_before_query" if target_times[1] <= q else
                  "straddles_query" if target_times[0] <= q else "both_after_query")
        row = {"parent_index": i, **meta, "episode_query": q,
               "outcome_first_time": q+1, "target_times": target_times,
               "target_query_relations": ["before" if x < q else "at" if x == q else "after" for x in target_times],
               "timing_regime": regime, "both_targets_strictly_before_query": target_times[1] < q,
               "last_target_at_query": target_times[1] == q,
               "targets_equal_each_other": bool(np.array_equal(z[target_times[0]], z[target_times[1]])),
               "evidence_annotation_times": evidence, "evidence_equivalent_feature_times": equivalent,
               "candidate_cue_mask": cue, "candidate_cue_count": sum(cue),
               "recent_times": recent, "recent_cue_times": recent_cue,
               "deleting_candidate_completely_loses_resident_cue": destroys,
               "complete_loss_possible": bool(any(destroys)),
               "cost_labels": cost[i].tolist(), "attention_labels": attention[i].astype(float).tolist(),
               "cost": minimum_summary(cost[i], cue, destroys),
               "attention": minimum_summary(attention[i], cue, destroys)}
        row["minimum_sets_disjoint"] = not bool(set(row["cost"]["minimum_delete_indices"]) &
                                                 set(row["attention"]["minimum_delete_indices"]))
        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        for key in ("timing="+row["timing_regime"], "candidate_cues="+str(row["candidate_cue_count"]),
                    "complete_loss_possible="+str(row["complete_loss_possible"]).lower(),
                    "timing="+row["timing_regime"]+";candidate_cues="+str(row["candidate_cue_count"])+
                    ";complete_loss_possible="+str(row["complete_loss_possible"]).lower()):
            grouped[key].append(row)
    summary = {"all": aggregate(rows), "strata": {k: aggregate(v) for k, v in sorted(grouped.items())},
               "strictly_before_query_parents": sum(r["both_targets_strictly_before_query"] for r in rows),
               "last_target_at_query_parents": sum(r["last_target_at_query"] for r in rows),
               "target_query_relation_counts": dict(Counter("/".join(r["target_query_relations"]) for r in rows))}
    if summary["strictly_before_query_parents"] != 418 or summary["last_target_at_query_parents"] != 102:
        raise ValueError("Unexpected audited timing counts")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "parent_records.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    write(output / "summary.json", summary)
    write(output / "protocol.json", {"version": "use-time-saved-label-audit-v1",
          "created_utc": datetime.now(timezone.utc).isoformat(), "absolute_tie_tolerance": TOL,
          "relative_tie_tolerance": 0., "input_shape": input_shape,
          "teacher_column_alignment": "all 1282 saved attention rows exactly equal original teacher labels after candidate sorting",
          "source_labels": "saved per-decision mean h3/h6 deletion MSE and saved future-attention salience",
          "delete_preference": "minimum cost / minimum attention; all actions within absolute tolerance included",
          "cue_definition": "exact normalized-feature equality to the one privileged Gate evidence frame",
          "complete_loss": "deleting the only cue-equivalent candidate with no cue in recent R=4",
          "timing_boundary": "query frame is pre-outcome; first outcome class is query+1, checked for all TRAIN episodes",
          "target_equality": "exact observed feature equality; never implies equal reader losses",
          "unit": "parent rows are correlated within 32 episodes / 16 scene pairs; descriptive counts, no independent-sample inference",
          "scope": "old TRAIN saved label audit only; no model calls, training, confirmation data, test arrays, or causal claim",
          "source_labels_directory": str(labels.relative_to(ROOT))})
    write(output / "sourcehashes.json", {"sha256": dict(sorted(sourcehashes.items())),
          "outputs": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({"output": str(output), "timing": {k: len(v) for k, v in grouped.items() if k.startswith("timing=") and ";" not in k},
                      "all": summary["all"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT/"build/hpc-20261020/development-assets")
    parser.add_argument("--labels", type=Path, default=ROOT/"build/literature-diagnosis-20261022/learnability-v2")
    parser.add_argument("--output", type=Path, default=ROOT/"results/literature_diagnosis/use-time-label-audit-20261024")
    run(parser.parse_args())
