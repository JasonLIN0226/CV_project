"""Independent NumPy arithmetic/provenance audit of a saved Gate2 pilot.

No experiment modules, torch, encoder or reader are imported. This does not
rerun a model. It checks archived predictions against archived target features,
and distinguishes a completed geometry stop from completed reader evaluation.
Family bootstrap intervals describe this development pilot, not confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

BITS = ((0, 0), (0, 1), (1, 0), (1, 1))
STYLES = ("benign_duplicate", "distant_style_duplicate")
ARMS = ("AB", "AprimeB", "AAprime", "Ablank", "Bblank", "wrongA", "wrongB", "blank2")
TOL = 1e-6
SPLITS = {"train": (20261028, 240), "validation": (20261029, 48),
          "geometry": (20261030, 24), "development": (20261031, 48)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


class Checks:
    def __init__(self):
        self.count = 0
        self.failures = []
        self.warnings = []
        self.max_abs_errors = {}

    def require(self, ok, label):
        self.count += 1
        if not bool(ok):
            self.failures.append(label)

    def close(self, actual, expected, label, atol=1e-10):
        a, b = np.asarray(actual), np.asarray(expected)
        valid = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        error = float(np.max(np.abs(a.astype(np.float64) - b))) if valid and a.size else (0. if valid else float("inf"))
        key = label.split("/")[0]
        if np.isfinite(error):
            self.max_abs_errors[key] = max(error, self.max_abs_errors.get(key, 0.))
        self.require(valid and error <= atol, label)


def classify(pred, truth, bits, cf_bits):
    """Joint horizon objective: one class/deletion, never per-horizon mixing."""
    delta = np.asarray(pred, np.float64)[None] - np.asarray(truth, np.float64)
    per_h = np.mean(delta * delta, axis=2)
    distances = np.mean(delta * delta, axis=(1, 2))
    own, cf = BITS.index(tuple(bits)), BITS.index(tuple(cf_bits))
    nearest = np.flatnonzero(distances <= distances.min() + TOL).tolist()
    cls = int(distances.argmin())
    return {"mse": float(distances[own]), "class_distances": distances.tolist(),
            "mse_by_horizon": per_h[own].tolist(), "predicted_class": cls,
            "nearest_class_set": nearest, "predicted_bits": list(BITS[cls]),
            "feature_class_correct": nearest == [own], "counterfactual_class_correct": nearest == [cf],
            "per_horizon_class": per_h.argmin(0).tolist(),
            "per_horizon_nearest_class_sets": [np.flatnonzero(per_h[:, h] <= per_h[:, h].min() + TOL).tolist() for h in range(2)]}


def diversity(candidates, times):
    """Independent literal enumeration of retained-pair cosine objective."""
    z = np.asarray(candidates, np.float64)
    norm = np.linalg.norm(z, axis=1)
    unit = z / np.maximum(norm[:, None], np.finfo(np.float64).tiny)
    similarity = np.clip(unit @ unit.T, -1., 1.)
    similarity[np.logical_and(norm[:, None] == 0, norm[None, :] == 0)] = 1.
    distances = 1. - similarity
    retained, objectives = [], []
    for drop in range(3):
        keep = [j for j in range(3) if j != drop]
        # Use a two-vector matmul as the native memory implementation does.
        pair = z[keep]
        lengths = np.linalg.norm(pair, axis=1)
        u = pair / np.maximum(lengths[:, None], np.finfo(np.float64).tiny)
        s = np.clip(u @ u.T, -1., 1.)
        s[np.logical_and(lengths[:, None] == 0, lengths[None, :] == 0)] = 1.
        distance = float(1. - s[0, 1])
        retained.append(float(distances[keep[0], keep[1]]))
        objectives.append((distance, distance, sum(times[j] for j in keep)))
    return max(range(3), key=lambda i: objectives[i]), distances, np.asarray(retained)


def family_interval(values, families, *, seed=20261025, repetitions=2000):
    grouped = defaultdict(list)
    for value, family in zip(values, families):
        grouped[family].append(float(value))
    means = np.asarray([np.mean(grouped[k]) for k in sorted(grouped)])
    if not len(means):
        raise ValueError("Cannot bootstrap an empty set")
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, len(means), (repetitions, len(means)))].mean(1)
    return {"family_count": len(means), "episode_count": len(values),
            "episode_mean": float(np.mean(values)), "equal_family_mean": float(means.mean()),
            "family_cluster_percentile_95": np.quantile(draws, [.025, .975]).tolist(),
            "bootstrap_repetitions": repetitions, "bootstrap_seed": seed}


def check_protocol(protocol, checks):
    """Check declared parameters against the frozen pilot, independently."""
    expected = {
        "name": "gate2-terminal-pilot-v1", "status": "pre_execution",
        "splits": [{"split": s, "seed": seed, "quartets": count,
                    "paired_style_families": count//2} for s, (seed, count) in SPLITS.items()],
        "styles": list(STYLES), "horizons": [3, 6],
        "model": {"hidden": 128, "heads": 4, "layers": 2, "dropout": 0., "rank": 128},
        "steps": 1024, "validation_every": 128, "quartets_per_update": 8,
        "model_seed": 17, "memory_dropout": .15,
        "history_modes": ["randomK", "latestK", "latest3", "random_size"],
        "origin_training": "query, preuse, and uniform all origins 3..q",
        "selection": "earliest minimum equal-origin/equal-history validation MSE",
        "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .0001, "gradient_norm_clip": 1.},
        "scheduler": {"name": "MultiStepLR", "milestones": [1024//3, 2*1024//3], "gamma": .5},
        "normalization": "TRAIN prefix frames only, episode-frame multiplicity, std floor .05",
        "basis": "TRAIN only; all prefix origins and all six quartet future contrasts",
        "content_slots": "fixed ages [6,4], all actions and current/recent identical",
        "terminal_candidates": "common FIFO until q-1; q-6,q-5,q-4; compare one deletion at q",
        "terminal_uniform": "uniform one-of-three deletion; NumPy default_rng seeded from SHA256 JSON [17,family] first16hex; shared styles and siblings, not full-stream Algorithm R",
        "geometry_gate": "at least one GEOMETRY quartet has a diversity redundant-pair selection",
        "oracle_tolerance": TOL,
        "reader_gate": {"applied_to": "all and each style separately", "classification_ties": "within1e-6; ties fail accuracy",
                        "AB_accuracy_min": .8, "AprimeB_accuracy_min": .8,
                        "wrongA_counterfactual_accuracy_min": .8, "wrongB_counterfactual_accuracy_min": .8,
                        "each_complementary_arm_mse_below_each_single_by": TOL},
        "no_student_training": True, "development_only": True}
    for key, value in expected.items():
        checks.require(protocol.get(key) == value, "protocol/" + key)


def check_hashes(run, checks, source_root=None):
    m = read(run / "manifest.json")
    checks.require(m["status"] == "completed", "manifest/completed")
    for relative, expected in m["files"].items():
        path = (run / relative).resolve()
        safe = path.is_relative_to(run.resolve())
        checks.require(safe and path.is_file(), "manifest/file/" + relative)
        if safe and path.is_file():
            checks.require(sha(path) == expected, "manifest/sha256/" + relative)
    protocol = read(run / "protocol.json")
    check_protocol(protocol, checks)
    checked_sources = 0
    if source_root is not None:
        for relative, expected in protocol["source_sha256"].items():
            path = (source_root / relative).resolve()
            safe = path.is_relative_to(source_root.resolve())
            checks.require(safe and path.is_file(), "source/file/" + relative)
            if safe and path.is_file():
                checks.require(sha(path) == expected, "source/sha256/" + relative)
                checked_sources += 1
    return {"manifest_sha256": sha(run / "manifest.json"), "files_checked": len(m["files"]),
            "source_files_checked": checked_sources,
            "source_verification": "checked local snapshot" if source_root else "not requested; archived protocol hashes retained"}


def check_records(run, checks):
    records = read(run / "data/records.json")
    by_id = {r["id"]: r for r in records}
    checks.require(len(by_id) == len(records), "records/unique_ids")
    quartets, families = defaultdict(list), defaultdict(list)
    actions = np.load(run / "data/actions.npz", allow_pickle=False)
    for r in records:
        q = r["query"]
        checks.require(r["evidence"] == [q-6, q-5, q-4], "records/cue_schedule/" + r["id"])
        axes = r["semantic_axes"]
        checks.require(sorted(Counter(axes).values()) == [1, 2], "records/independent_axes/" + r["id"])
        checks.require(sorted(r["scene"]["roles"]) == ["A", "Aprime", "B"], "records/roles/" + r["id"])
        duplicate = r["scene"]["duplicate_axis"]
        checks.require(all(axis == duplicate if role != "B" else axis != duplicate
                           for role, axis in zip(r["scene"]["roles"], axes)), "records/roles_to_axes/" + r["id"])
        ac = actions[r["id"]]
        checks.require(ac.shape == (len(r["rgb_indices"])-1, 2) and np.all(ac[:q] == 0)
                       and np.all(ac[q:] > 0), "records/control_schedule/" + r["id"])
        quartets[r["quartet"]].append(r)
        families[r["family"]].append(r)
    for split, (seed, n) in SPLITS.items():
        rr = [r for r in records if r["split"] == split]
        checks.require(len(rr) == 4*n and len({r["quartet"] for r in rr}) == n, "records/count/" + split)
        cells = Counter((r["condition"], r["scene"]["duplicate_axis"], r["scene"]["role_permutation_index"]) for r in rr)
        checks.require(len(cells) == 24 and set(cells.values()) == {n//6}, "records/balanced_cells/" + split)
        checks.require(all(r["scene"]["seed"] == seed for r in rr), "records/seed/" + split)
    for key, rr in quartets.items():
        rr.sort(key=lambda r: r["bits"])
        checks.require(tuple(tuple(r["bits"]) for r in rr) == BITS, "records/complete_quartet/" + key)
        base, q = rr[0], rr[0]["query"]
        cue = set(base["evidence"])
        for r in rr[1:]:
            checks.require(np.array_equal(actions[r["id"]], actions[base["id"]]), "records/shared_actions/" + r["id"])
            checks.require(r["rgb_indices"][q-3:q+1] == base["rgb_indices"][q-3:q+1], "records/shared_R4/" + r["id"])
            checks.require(all(r["rgb_indices"][t] == base["rgb_indices"][t] for t in range(q+1) if t not in cue), "records/no_precue_bit_leak/" + r["id"])
        checks.require(all(len({r["rgb_indices"][q+h] for r in rr}) == 4 for h in (3, 6)), "records/four_targets/" + key)
        bit_map = {tuple(r["bits"]): r for r in rr}
        for r in rr:
            for ai, axis in enumerate(("a", "b")):
                bits = r["bits"].copy(); bits[ai] ^= 1
                sibling = bit_map[tuple(bits)]
                for t, semantic in zip(r["evidence"], r["semantic_axes"]):
                    equal = r["rgb_indices"][t] == sibling["rgb_indices"][t]
                    checks.require(equal == (semantic != axis), "records/bit_visual_channel/" + r["id"] + "/" + str(t))
    for family, rr in families.items():
        checks.require(len(rr) == 8 and len({r["split"] for r in rr}) == 1,
                       "records/family_unit/" + family)
        for bits in BITS:
            pair = [r for r in rr if tuple(r["bits"]) == bits]
            checks.require(len(pair) == 2 and {r["condition"] for r in pair} == set(STYLES), "records/matched_styles/" + family)
            if len(pair) != 2:
                continue
            a, b = pair
            changed = a["evidence"][a["scene"]["roles"].index("Aprime")]
            checks.require([i for i, (x, y) in enumerate(zip(a["rgb_indices"], b["rgb_indices"])) if x != y] == [changed], "records/style_only_Aprime/" + family)
            checks.require(np.array_equal(actions[a["id"]], actions[b["id"]]), "records/style_same_actions/" + family)
    actions.close()
    return records, by_id, quartets


def check_features(run, records, checks):
    with np.load(run / "data/normalization.npz", allow_pickle=False) as norm:
        mean, std, train_indices = norm["mean"], norm["std"], norm["train_indices"]
    expected = np.concatenate([r["rgb_indices"][:r["query"]+1] for r in records if r["split"] == "train"])
    checks.require(np.array_equal(train_indices, expected), "normalization/TRAIN_prefix_only")
    with np.load(run / "data/unique_features.npz", allow_pickle=False) as f:
        raw, normalized = f["raw"], f["normalized"]
    checks.require(raw.dtype == normalized.dtype == np.float32 and raw.shape == normalized.shape,
                   "features/dtype_shape")
    sums = np.zeros(raw.shape[1], np.float64); squares = sums.copy()
    for start in range(0, len(expected), 128):
        batch = raw[expected[start:start+128]].astype(np.float64)
        sums += batch.sum(0); squares += (batch*batch).sum(0)
    mean64 = sums/len(expected)
    std64 = np.sqrt(np.maximum(squares/len(expected)-mean64*mean64, 0.))
    checks.close(mean, mean64.astype(np.float32), "normalization/mean", atol=1e-6)
    checks.close(std, np.maximum(std64, .05).astype(np.float32), "normalization/std", atol=1e-6)
    for start in range(0, len(raw), 128):
        checks.close(normalized[start:start+128], ((raw[start:start+128]-mean)/std).astype(np.float32),
                     "features/normalized/" + str(start), atol=1e-6)
    return normalized


def check_geometry(run, records, z, checks):
    archive = read(run / "geometry_locked.json")
    by_id = {r["id"]: r for r in records if r["split"] == "geometry"}
    checks.require({r["id"] for r in archive["rows"]} == set(by_id), "geometry/all_prespecified_rows")
    checks.require(len(archive["rows"]) == len(by_id), "geometry/no_duplicate_rows")
    rows = []
    for old in archive["rows"]:
        r = by_id[old["id"]]
        cand = z[np.asarray(r["rgb_indices"])[r["evidence"]]]
        drop, distances, retained = diversity(cand, r["evidence"])
        checks.close(old["pairwise_cosine_distance"], distances, "geometry/distances/" + r["id"])
        checks.close(old["retained_distance_by_drop"], retained, "geometry/retained_distances/" + r["id"])
        actual = old["predicted_drop"]
        uniform_seed = int(hashlib.sha256(json.dumps([17, r["family"]]).encode()).hexdigest()[:16], 16)
        uniform = int(np.random.default_rng(uniform_seed).integers(3))
        checks.require(old["frozen_choices"] == {"fifo": 0, "terminal_uniform": uniform, "visual_diversity": actual},
                       "geometry/frozen_choices/" + r["id"])
        if actual != drop and abs(retained[actual]-retained[drop]) < 1e-12:
            checks.warnings.append("Near-exact floating tie in geometry native drop: " + r["id"])
        else:
            checks.require(actual == drop, "geometry/actual_drop/" + r["id"])
        single = next(i for i, a in enumerate(r["semantic_axes"]) if r["semantic_axes"].count(a) == 1)
        margin = float(retained[single]-np.delete(retained, single).max())
        checks.close(old["redundant_margin_over_best_complementary"], margin, "geometry/margin/" + r["id"])
        checks.require(old["max_distance_drop_set_within_tolerance"] == np.flatnonzero(retained >= retained.max()-TOL).tolist(), "geometry/tie_set/" + r["id"])
        checks.require(old["redundant_pair_retained"] == (actual == single), "geometry/semantic_pair/" + r["id"])
        checks.require(old["redundant_strictly_farthest_over_tolerance"] == (margin > TOL), "geometry/strict_margin/" + r["id"])
        rows.append(old)
    summary = {style: {"episodes": sum(r["condition"] == style for r in rows),
                      "redundant_episodes": sum(r["condition"] == style and r["redundant_pair_retained"] for r in rows),
                      "redundant_quartets": len({r["quartet"] for r in rows if r["condition"] == style and r["redundant_pair_retained"]})} for style in STYLES}
    checks.require(summary == archive["by_condition"], "geometry/summary")
    passed = any(r["redundant_pair_retained"] for r in rows)
    checks.require(archive["feasibility_pass"] == passed, "geometry/feasibility")
    checks.require(archive["before_reader_basis_or_fit"] and not archive["selection_on_development"], "geometry/protocol_flags")
    return {"feasibility_pass": passed, "by_condition": summary}


def check_preuse(run, records_by_id, z, summary, checks):
    rows = read(run / "preuse_rows.json")
    with np.load(run / "preuse_predictions.npz", allow_pickle=False) as f:
        predictions, targets, controls = f["predictions"], f["targets"], f["causal_previous_frame_prediction"]
    checks.require(len(rows) == len(predictions) == len(targets) == len(controls), "preuse/shapes")
    for row, p, y, control in zip(rows, predictions, targets, controls):
        r = records_by_id[row["id"]]; t = r["query"]-6
        checks.require(row["origin"] == t and row["history_times"] == [t-5, t-4], "preuse/origin/" + row["id"])
        expected = z[np.asarray(r["rgb_indices"])[[t+3, t+6]]]
        checks.close(y, expected, "preuse/target_provenance/" + row["id"], atol=0.)
        checks.close(control, np.repeat(z[r["rgb_indices"][t-1]][None], 2, axis=0), "preuse/control/" + row["id"], atol=0.)
        error = (p.astype(np.float64)-y)**2
        checks.close(row["mse"], error.mean(), "preuse/mse/" + row["id"])
        checks.close(row["mse_by_horizon"], error.mean(1), "preuse/horizons/" + row["id"])
        checks.close(row["causal_previous_frame_mse"], ((control.astype(np.float64)-y)**2).mean(), "preuse/control_mse/" + row["id"])
    for style in ("all",)+STYLES:
        rr = [r for r in rows if style == "all" or r["condition"] == style]
        for key in ("mse", "causal_previous_frame_mse"):
            checks.close(summary["preuse_static_diagnostic"]["by_condition"][style][key], np.mean([r[key] for r in rr]), "preuse/summary/" + style + "/" + key)


def check_evaluation(run, records, by_id, quartets, z, summary, checks):
    rows = read(run / "evaluation_rows.json")
    with np.load(run / "evaluation_predictions.npz", allow_pickle=False) as f:
        predictions, targets = f["predictions"], f["quartet_targets"]
    checks.require(sha(run / "evaluation_predictions.npz") == summary["saved_prediction_sha256"], "evaluation/prediction_sha")
    dev = {r["id"]: r for r in records if r["split"] == "development"}
    checks.require(predictions.shape == (len(dev)*11, 2, z.shape[1]) and targets.shape == (len(dev), 4, 2, z.shape[1]), "evaluation/shapes")
    checks.require(len(rows) == len(predictions), "evaluation/row_count")
    per_id = defaultdict(dict)
    for row, pred in zip(rows, predictions):
        r = dev[row["id"]]
        key = row["id"] + "/" + row["arm"]
        checks.require(row["arm"] not in per_id[row["id"]], "evaluation/no_duplicate_arm/" + key)
        for field, source in (("quartet", "quartet"), ("family", "family"), ("condition", "condition"), ("bits", "bits")):
            checks.require(row[field] == r[source], "evaluation/metadata/" + field + "/" + key)
        truth = targets[row["episode_index"]]
        expected = np.stack([z[np.asarray(s["rgb_indices"])[[s["query"]+3, s["query"]+6]]] for s in quartets[r["quartet"]]])
        checks.close(truth, expected, "evaluation/target_provenance/" + key, atol=0.)
        cf = r["bits"].copy()
        if row["arm"] in ("wrongA", "wrongB"):
            a = 0 if r["scene"]["duplicate_axis"] == "a" else 1
            cf[a if row["arm"] == "wrongA" else 1-a] ^= 1
        checks.require(row["counterfactual_bits"] == cf, "evaluation/counterfactual_axis/" + key)
        new = classify(pred, truth, r["bits"], cf)
        for field, value in new.items():
            if field in ("mse", "class_distances", "mse_by_horizon"):
                checks.close(row[field], value, "evaluation/arithmetic/" + field + "/" + key)
            else:
                checks.require(row[field] == value, "evaluation/classification/" + field + "/" + key)
        per_id[row["id"]][row["arm"]] = dict(row, **new)
    expected_arms = set(ARMS) | {"native_drop_"+str(i) for i in range(3)}
    checks.require(set(per_id) == set(dev), "evaluation/all_development_episodes")
    for rid, rr in per_id.items():
        checks.require(set(rr) == expected_arms, "evaluation/all_arms/" + rid)
    check_preuse(run, by_id, z, summary, checks)
    return rows, per_id


def check_decisions(run, by_id, per_id, z, summary, checks):
    decisions = read(run / "terminal_decisions.json")
    checks.require({d["id"] for d in decisions} == set(per_id) and len(decisions) == len(per_id), "decisions/episode_set")
    for d in decisions:
        r = by_id[d["id"]]; axes = r["semantic_axes"]
        native = [per_id[d["id"]]["native_drop_"+str(i)] for i in range(3)]
        losses = np.asarray([n["mse"] for n in native])
        semantic = [i for i, axis in enumerate(axes) if axes.count(axis) == 2]
        minimum = np.flatnonzero(losses <= losses.min()+TOL).tolist()
        exact_minimum = np.flatnonzero(losses == losses.min()).tolist()
        checks.close(d["losses"], losses, "decisions/losses/" + d["id"])
        checks.require(d["semantic_correct_delete_set"] == semantic, "decisions/semantic_mask/" + d["id"])
        checks.require(d["oracle_min_delete_set"] == minimum, "decisions/minimum_set/" + d["id"])
        checks.require(d["oracle_exact_min_delete_set"] == exact_minimum, "decisions/exact_minimum_set/" + d["id"])
        checks.require(d["oracle_all_semantically_correct"] == (set(minimum) <= set(semantic)), "decisions/all_semantic/" + d["id"])
        checks.require(d["oracle_any_semantically_correct"] == bool(set(minimum)&set(semantic)), "decisions/any_semantic/" + d["id"])
        attention = np.asarray(d["attention"])
        checks.require(attention.shape == (3,) and np.all(attention >= 0) and abs(attention.sum()-1) < 1e-6, "decisions/attention_probability/" + d["id"])
        checks.require(d["choices"]["fifo"] == 0 and d["choices"]["reader_attention"] == int(attention.argmin()), "decisions/policy_choices/" + d["id"])
        uniform_seed = int(hashlib.sha256(json.dumps([17, r["family"]]).encode()).hexdigest()[:16], 16)
        uniform = int(np.random.default_rng(uniform_seed).integers(3))
        checks.require(d["choices"]["terminal_uniform"] == uniform, "decisions/uniform_choice/" + d["id"])
        cand = z[np.asarray(r["rgb_indices"])[r["evidence"]]]
        drop, _, distances = diversity(cand, r["evidence"])
        actual = d["choices"]["visual_diversity"]
        checks.require(actual == drop or abs(distances[actual]-distances[drop]) < 1e-12, "decisions/diversity/" + d["id"])
        for policy, p in d["policies"].items():
            chosen = p["drop"]
            checks.require(chosen in (0, 1, 2) and chosen == d["choices"][policy], "decisions/legal_action/" + policy + "/" + d["id"])
            checks.close(p["mse"], losses[chosen], "decisions/policy_mse/" + d["id"])
            checks.close(p["regret"], losses[chosen]-losses.min(), "decisions/regret/" + d["id"])
            checks.require(p["semantic_correct"] == (chosen in semantic), "decisions/policy_semantic/" + d["id"])
            checks.require(p["feature_class_correct"] == native[chosen]["feature_class_correct"], "decisions/policy_class/" + d["id"])
    by_quartet = defaultdict(list)
    for d in decisions:
        by_quartet[d["quartet"]].append(d)
    archived = {c["quartet"]: c for c in summary["quartet_consistency"]}
    checks.require(set(archived) == set(by_quartet), "consistency/quartet_set")
    for quartet, rr in by_quartet.items():
        sets = [set(d["oracle_min_delete_set"]) for d in rr]
        exact_sets = [set(d["oracle_exact_min_delete_set"]) for d in rr]
        c = archived[quartet]
        checks.require(c["same_exact_min_set"] == all(s == exact_sets[0] for s in exact_sets), "consistency/exact_sets/" + quartet)
        checks.require(c["same_tolerant_min_set"] == all(s == sets[0] for s in sets), "consistency/tolerant_sets/" + quartet)
        checks.require(c["common_exact_optimal_action"] == sorted(set.intersection(*exact_sets)), "consistency/common_exact_action/" + quartet)
        checks.require(c["common_optimal_action"] == sorted(set.intersection(*sets)), "consistency/common_action/" + quartet)
        checks.require(c["all_four_oracles_semantic_subset"] == all(d["oracle_all_semantically_correct"] for d in rr), "consistency/semantic_subset/" + quartet)
    return decisions


def check_summaries(rows, decisions, summary, checks):
    for condition, group in summary["summary"].items():
        rr = [r for r in rows if condition == "all" or r["condition"] == condition or r["cell"] == condition]
        dd = [d for d in decisions if condition == "all" or d["condition"] == condition or d["cell"] == condition]
        for arm, old in group["arms"].items():
            ar = [r for r in rr if r["arm"] == arm]
            expected = {"episodes": len(ar), "mse": np.mean([r["mse"] for r in ar]),
                        "feature_class_accuracy": np.mean([r["feature_class_correct"] for r in ar]),
                        "counterfactual_class_accuracy": np.mean([r["counterfactual_class_correct"] for r in ar]),
                        "classification_tie_fraction": np.mean([len(r["nearest_class_set"]) > 1 for r in ar]),
                        "bit_accuracy": [np.mean([len(r["nearest_class_set"]) == 1 and r["predicted_bits"][a] == r["bits"][a] for r in ar]) for a in range(2)]}
            for key, value in expected.items():
                checks.close(old[key], value, "summary/arm/" + condition + "/" + arm + "/" + key)
        for policy, old in group["policies"].items():
            pp = [d["policies"][policy] for d in dd]
            regrets = np.asarray([p["regret"] for p in pp])
            expected = {"mse": np.mean([p["mse"] for p in pp]), "mean_regret": regrets.mean(),
                        "positive_regret_fraction": np.mean(regrets > TOL),
                        "regret_quantiles_0_50_95_100": np.quantile(regrets, [0, .5, .95, 1]),
                        "semantic_correct_fraction": np.mean([p["semantic_correct"] for p in pp]),
                        "feature_class_accuracy": np.mean([p["feature_class_correct"] for p in pp])}
            for key, value in expected.items():
                checks.close(old[key], value, "summary/policy/" + condition + "/" + policy + "/" + key)
        checks.close(group["oracle_mse"], np.mean([min(d["losses"]) for d in dd]), "summary/oracle_mse/" + condition)
        checks.close(group["oracle_semantic_subset_fraction"], np.mean([d["oracle_all_semantically_correct"] for d in dd]), "summary/oracle_semantic/" + condition)
    gate = {}
    for condition in ("all",)+STYLES:
        ar = summary["summary"][condition]["arms"]
        gate.update({condition+"/AB_accuracy": ar["AB"]["feature_class_accuracy"] >= .8,
                     condition+"/AprimeB_accuracy": ar["AprimeB"]["feature_class_accuracy"] >= .8,
                     condition+"/wrongA_counterfactual": ar["wrongA"]["counterfactual_class_accuracy"] >= .8,
                     condition+"/wrongB_counterfactual": ar["wrongB"]["counterfactual_class_accuracy"] >= .8})
        for complete in ("AB", "AprimeB"):
            for single in ("Ablank", "Bblank"):
                gate[condition+"/"+complete+"_below_"+single] = ar[complete]["mse"] < ar[single]["mse"]-TOL
    checks.require(gate == summary["reader_gate_checks"], "summary/reader_gate_checks")
    checks.require(all(gate.values()) == summary["reader_gate_pass"], "summary/reader_gate_pass")
    strong = all(summary["summary"]["all"]["policies"][p]["semantic_correct_fraction"] == 1. for p in ("reader_attention", "visual_diversity"))
    checks.require(strong == summary["strong_baselines_all_semantically_correct"], "summary/strong_baselines")
    strong_gaps = {p: summary["summary"]["all"]["policies"][p]["positive_regret_fraction"] > 0 for p in ("reader_attention", "visual_diversity")}
    checks.require(strong_gaps == summary["strong_baselines_have_positive_local_regret"], "summary/strong_regret_flags")
    if not all(gate.values()):
        expected_decision = "stop_reader_gate_failed"
    elif not all(strong_gaps.values()):
        expected_decision = "stop_a_strong_baseline_closes_local_oracle_gap"
    elif strong:
        expected_decision = "stop_semantic_problem_solved_residual_reader_gap"
    else:
        expected_decision = "candidate_terminal_gap_requires_independent_confirmation"
    checks.require(expected_decision == summary["decision"], "summary/decision")


def uncertainty(per_id, decisions):
    out = {}
    for style in ("all",)+STYLES:
        dd = [d for d in decisions if style == "all" or d["condition"] == style]
        families = [d["family"] for d in dd]
        metrics = {}
        for policy in dd[0]["policies"]:
            metrics[policy + "_minus_oracle_mse"] = [d["policies"][policy]["regret"] for d in dd]
            metrics[policy + "_semantic_error"] = [not d["policies"][policy]["semantic_correct"] for d in dd]
        for a, b in (("AB", "AAprime"), ("AprimeB", "AAprime"), ("AB", "Ablank"), ("AB", "Bblank")):
            metrics[a+"_minus_"+b+"_mse"] = [per_id[d["id"]][a]["mse"]-per_id[d["id"]][b]["mse"] for d in dd]
        for arm in ("AB", "AprimeB", "wrongA", "wrongB"):
            field = "counterfactual_class_correct" if arm.startswith("wrong") else "feature_class_correct"
            metrics[arm+"_"+field] = [per_id[d["id"]][arm][field] for d in dd]
        out[style] = {name: family_interval(values, families) for name, values in metrics.items()}
    return out


def audit(run, output, source_root=None):
    checks = Checks()
    provenance = check_hashes(run, checks, source_root)
    records, by_id, quartets = check_records(run, checks)
    z = check_features(run, records, checks)
    geometry = check_geometry(run, records, z, checks)
    summary = read(run / "summary.json")
    evaluation = {}
    if not geometry["feasibility_pass"]:
        checks.require(summary["decision"] == "stop_geometry_no_competition" and summary["reader_fitted"] is False,
                       "geometry_stop/no_reader_claim")
        checks.require(not (run / "evaluation_predictions.npz").exists() and not (run / "reader_selected.pt").exists(),
                       "geometry_stop/no_reader_outputs")
        scope = "completed_geometry_stop; reader and retention benefit not evaluated"
    else:
        rows, per_id = check_evaluation(run, records, by_id, quartets, z, summary, checks)
        decisions = check_decisions(run, by_id, per_id, z, summary, checks)
        check_summaries(rows, decisions, summary, checks)
        evaluation = {"prediction_rows": len(rows), "episodes": len(per_id),
                      "quartets": len({d["quartet"] for d in decisions}), "families": len({d["family"] for d in decisions}),
                      "joint_horizon_objective": "mean of h3 and h6 feature MSE, one deletion for both",
                      "oracle_tied_episodes": sum(len(d["oracle_min_delete_set"]) > 1 for d in decisions),
                      "oracle_all_semantic_episodes": sum(d["oracle_all_semantically_correct"] for d in decisions),
                      "oracle_any_semantic_episodes": sum(d["oracle_any_semantically_correct"] for d in decisions),
                      "quartets_with_common_optimal_action": sum(bool(c["common_optimal_action"]) for c in summary["quartet_consistency"]),
                      "family_cluster_uncertainty": uncertainty(per_id, decisions)}
        scope = "saved evaluation arithmetic, structural metadata and family uncertainty"
    checks.require(summary.get("students_trained") is False, "scope/no_student_training")
    result = {"status": "passed" if not checks.failures else "failed", "scope": scope,
              "experiment_decision": summary["decision"], "check_count": checks.count,
              "failure_count": len(checks.failures), "failures": checks.failures,
              "warnings": checks.warnings, "max_abs_errors": checks.max_abs_errors,
              "provenance": provenance, "geometry": geometry, "evaluation": evaluation,
              "audit_source_sha256": sha(__file__), "numpy_version": np.__version__,
              "uncertainty_protocol": {"cluster_unit": "family", "bootstrap_repetitions": 2000, "bootstrap_seed": 20261025},
              "model_calls": 0, "new_training": False,
              "limits": ["Archived prediction arithmetic is checked; model inference is not replayed.",
                         "An oracle common action across siblings does not establish causal learnability.",
                         "Four siblings and two styles share one independent family; intervals cluster by family.",
                         "Development-only uncertainty is descriptive, not independent confirmation.",
                         "Feature nearest-sibling classification is not a physical decoder."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "scope", "experiment_decision", "check_count", "failure_count")}), flush=True)
    return result


def self_test():
    # Exact class, a genuine joint-horizon tie, and counterfactual axis target.
    targets = np.array([[[0., 0.], [0., 0.]], [[0., 1.], [0., 1.]],
                        [[1., 0.], [1., 0.]], [[1., 1.], [1., 1.]]])
    c = classify(targets[2], targets, (0, 0), (1, 0))
    assert c["mse"] == .5 and c["counterfactual_class_correct"] and not c["feature_class_correct"]
    tied = classify(np.full((2, 2), .5), targets, (0, 0), (0, 0))
    assert tied["nearest_class_set"] == [0, 1, 2, 3] and not tied["feature_class_correct"]
    # Duplicate pair has zero cosine distance; native tie prefers newer pair.
    choice, _, distance = diversity([[1., 0.], [1., 0.], [0., 1.]], [3, 4, 5])
    assert choice == 0 and np.array_equal(distance, [1., 1., 0.])
    result = family_interval([1., 1., 3., 3.], ["f1", "f1", "f2", "f2"], repetitions=200)
    assert result["family_count"] == 2 and result["equal_family_mean"] == 2.
    assert result["family_cluster_percentile_95"] == [1., 3.]
    print(json.dumps({"status": "passed_numpy_synthetic_sanity", "checks": 4, "model_calls": 0}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
    elif args.run is None or args.output is None:
        p.error("--run and --output are required unless --self-test is used")
    else:
        result = audit(args.run.resolve(), args.output.resolve(), args.source_root.resolve() if args.source_root else None)
        raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
