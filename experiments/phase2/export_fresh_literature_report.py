"""Audit and publish an already completed fresh Gate run; never execute a model.

Uses NumPy/Matplotlib only. Pair bootstrap combines siblings and frozen student
seeds before sampling 32 independent scene pairs within each condition. RGB
figures are actual saved observations, never decoded or generated predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.phase2.export_literature_report import (
    COLORS, LABELS, ORDER, read, relative, save, scrub, sha, source_info, style, write,
    wr_trace,
)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ("id", "long_gap", "distractor_shift")
TITLES = {"id": "ID", "long_gap": "Long gap", "distractor_shift": "Distractor appearance shift"}
BOOTSTRAPS = 10000
SEED = 20261022
EPS = 1e-8


def close(a, b, message, tol=1e-12):
    if not np.allclose(a, b, rtol=0, atol=tol):
        raise ValueError(message)


def hashed_files(directory, records):
    for name, expected in records.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha(path) != expected:
            raise ValueError("Artifact identity failed: " + name)


def cue_oracle_split(alternatives, groups):
    keeps = lambda key: all(set(group).intersection(key) for group in groups)
    best_loss, best_key = min((loss, key) for key, loss in alternatives.items())
    with_cue = [loss for key, loss in alternatives.items() if keeps(key)]
    without_cue = [loss for key, loss in alternatives.items() if not keeps(key)]
    best_with = min(with_cue) if with_cue else None
    best_without = min(without_cue) if without_cue else None
    return {"selected_keeps_cue": keeps(best_key), "best_with_cue_mse": best_with,
            "best_without_cue_mse": best_without,
            "strictly_prefers_no_cue": best_with is not None and best_without is not None and best_without < best_with-EPS,
            "requires_no_cue_within_tolerance": best_with is not None and best_with > best_loss+EPS,
            "strictly_prefers_cue": best_with is not None and best_without is not None and best_with < best_without-EPS,
            "tie_tolerance": EPS}


def audit_run(source, data, summary, rows, records, events, subsets):
    """Independent scalar/shape/set checks; does not import the experiment."""
    manifest = read(data / "manifest.json")
    if (summary.get("status") != "completed_fresh_development_diagnostic"
            or summary.get("student_training_calls") != 0 or summary.get("normalization_fitting_calls") != 0
            or summary.get("old_test_arrays_read") != 0 or manifest.get("status") != "completed"):
        raise ValueError("Require the complete frozen fresh experiment")
    if (len(rows), len(records), summary["independent_pairs"]) != (1728, 192, 96):
        raise ValueError("Incomplete experiment")
    if summary["protocol_sha256"] != sha(source / "protocol.json"):
        raise ValueError("Protocol checksum failed")
    hashed_files(source, summary["result_artifact_sha256"])
    hashed_files(data, manifest["artifact_sha256"])
    for name, expected in summary["fresh_source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen evaluation source changed: " + name)
    if summary["fresh_input_audit"]["manifest_sha256"] != sha(data / "manifest.json"):
        raise ValueError("Fresh manifest binding failed")
    recs = {r["id"]: r for r in records}
    for condition in CONDITIONS:
        part = [r for r in records if r["condition"] == condition]
        if len(part) != 64 or len({r["pair"] for r in part}) != 32:
            raise ValueError("Scene-pair count mismatch")
    trace = defaultdict(list)
    for event in events:
        trace[event["episode_id"], event["policy"], event["seed"]].append(event)
        t = event["time"]
        recent, retained = event["recent_timestamps"], event["retained_timestamps"]
        if (recent != list(range(max(0, t-3), t+1)) or len(retained) > 2
                or len(set(recent + retained)) != len(recent + retained)
                or any(k < 0 or k >= max(0, t-3) for k in retained)):
            raise ValueError("Noncausal or over-capacity trace")
    cache = defaultdict(dict)
    for r in subsets:
        key = tuple(r["kept_times"])
        if key in cache[r["episode_id"]]:
            raise ValueError("Duplicate oracle cache key")
        cache[r["episode_id"]][key] = float(np.mean(r["feature_mse"]))
    with np.load(source / "prediction_vectors.npz", allow_pickle=False) as arrays:
        predictions, targets = arrays["predictions"], arrays["targets"]
    if predictions.shape != targets.shape or predictions.shape != (1728, 2, 5120):
        raise ValueError("Unexpected prediction-vector shape")
    mse = ((predictions.astype(np.float64)-targets.astype(np.float64))**2).mean(axis=2)
    close(mse, [r["feature_mse"] for r in rows], "Saved per-horizon MSE mismatch")
    close(mse.mean(1), [r["mean_mse"] for r in rows], "Saved mean MSE mismatch")
    by_episode = defaultdict(list)
    for index, row in enumerate(rows):
        by_episode[row["episode_id"]].append((index, row))
    oracle_audit, local_choices = [], {}
    for eid, indexed_rows in by_episode.items():
        record = recs[eid]
        q = record["query"]
        expected = set(combinations(range(q-3), 2))
        actual = {k for k in cache[eid] if len(k) == 2}
        if actual != expected or set(cache[eid]) != expected | {()}:
            raise ValueError("Hindsight oracle did not enumerate exactly all historical K2 subsets")
        best_loss, best_key = min((cache[eid][key], key) for key in expected)
        with np.load(data / record["feature_path"], allow_pickle=False) as arrays:
            z = arrays["z"]
        with np.load(data / record["eval_path"], allow_pickle=False) as arrays:
            evidence = arrays["evidence"]
        groups = [[t for t in range(q-3) if np.array_equal(z[t], z[int(e)])] for e in evidence]
        if len(indexed_rows) != 9:
            raise ValueError("Missing episode arm")
        for index, row in indexed_rows:
            key = eid, row["policy"], row["seed"]
            timeline = trace[key]
            if [x["time"] for x in timeline] != list(range(q+1)):
                raise ValueError("Incomplete online prefix")
            if sorted(row["kept_times"]) != sorted(timeline[-1]["retained_timestamps"]) or len(row["kept_times"]) != 2:
                raise ValueError("Wrong online query subset")
            if not np.array_equal(targets[index], z[q + np.array([3, 6])]):
                raise ValueError("Prediction target does not match fresh feature bank")
            before = timeline[-2]
            candidates = before["retained_timestamps"] + before["recent_timestamps"][:1]
            alternatives = {tuple(sorted(k)): cache[eid][tuple(sorted(k))] for k in combinations(candidates, 2)}
            local_loss, local_key = min((value, key2) for key2, value in alternatives.items())
            if (row["local_oracle"]["kept_times"] != list(local_key)
                    or row["local_oracle"]["subsets_evaluated"] != 3
                    or row["hindsight_oracle"]["kept_times"] != list(best_key)
                    or row["hindsight_oracle"]["subsets_evaluated"] != math.comb(q-3, 2)):
                raise ValueError("Oracle identity/count mismatch")
            close(row["local_oracle"]["mean_mse"], local_loss, "Local optimum mismatch")
            close(row["hindsight_oracle"]["mean_mse"], best_loss, "Hindsight minimum mismatch")
            close(row["local_regret"], row["mean_mse"]-local_loss, "Local regret mismatch")
            close(row["hindsight_regret"], row["mean_mse"]-best_loss, "Hindsight regret mismatch")
            close(row["mean_mse"], cache[eid][tuple(sorted(row["kept_times"]))], "Online/cache loss mismatch")
            survived = all(set(group).intersection(row["kept_times"]) for group in groups)
            lost = [next((event["time"] for event in timeline if event["time"] > max(group)
                    and not set(group).intersection(event["recent_timestamps"] + event["retained_timestamps"])), None)
                    for group in groups]
            if (groups != row["evidence_equivalent_feature_times"] or survived != row["evidence_survived"]
                    or lost != row["first_evidence_information_loss"]):
                raise ValueError("Exact-cue equivalence/survival/first-loss mismatch")
            choices = [{"kept_times": list(k), "mean_mse": v} for k, v in sorted(alternatives.items())]
            local_choices[key] = {"candidate_times": candidates, "alternatives": choices,
                                  "cue_oracle": cue_oracle_split(alternatives, groups)}
            teacher = row["teacher_diagnostic"]
            if teacher is not None:
                kept = [t for i, t in enumerate(candidates) if i != int(np.argmin(teacher["probabilities"]))]
                teacher_loss = cache[eid][tuple(sorted(kept))]
                if (teacher["kept_times"] != kept or teacher["future_query_count"] != 6
                        or teacher["offline_only"] is not True
                        or teacher["same_deletion_as_student"] != (set(kept) == set(row["kept_times"]))):
                    raise ValueError("Teacher candidate/action/timestamp mismatch")
                close(sum(teacher["probabilities"]), 1, "Teacher mass", tol=2e-6)
                close(teacher["mean_mse"], teacher_loss, "Teacher loss mismatch")
                close(teacher["teacher_local_regret"], teacher_loss-local_loss, "Teacher regret mismatch")
                close(teacher["student_minus_teacher_mse"], row["mean_mse"]-teacher_loss, "Teacher/student delta mismatch")
        oracle_audit.append({"episode_id": eid, "condition": record["condition"], "query": q,
            "historical_candidates": q-3, "exact_K2_subsets": len(expected),
            "hindsight_kept_times": list(best_key), "hindsight_mse": best_loss,
            "cue_oracle": cue_oracle_split({k: cache[eid][k] for k in expected}, groups),
            "all_online_subsets_feasible": True})
    controls = read(source / "history_controls.json")
    with np.load(source / "wrong_history_predictions.npz", allow_pickle=False) as arrays:
        wrong_predictions = arrays["predictions"]
    control_targets = np.stack([targets[by_episode[r["episode_id"]][0][0]] for r in controls])
    wrong_mse = ((wrong_predictions.astype(np.float64)-control_targets.astype(np.float64))**2).mean(axis=(1, 2))
    close(wrong_mse, [r["wrong_sibling_evidence_mse"] for r in controls], "Wrong-sibling prediction MSE mismatch")
    for control in controls:
        lookup = cache[control["episode_id"]]
        correct = lookup[tuple(sorted(control["privileged_kept_times"]))]
        close(control["privileged_evidence_mse"], correct, "Privileged control loss mismatch")
        close(control["recent_only_mse"], lookup[()], "Recent-only control loss mismatch")
        close(control["privileged_minus_recent"], correct-lookup[()], "History gain mismatch")
        close(control["wrong_minus_privileged"], control["wrong_sibling_evidence_mse"]-correct, "Wrong-history delta mismatch")
    result = {"status": "passed", "audited_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_rows": len(rows), "max_saved_mse_discrepancy": float(np.max(np.abs(mse-np.array([r["feature_mse"] for r in rows])))),
        "local_oracle_parents": len(rows), "local_alternatives_each": 3,
        "hindsight_episodes": len(oracle_audit), "total_hindsight_K2_subsets": sum(r["exact_K2_subsets"] for r in oracle_audit),
        "causal_trace_events": len(events), "evidence_equivalence_recomputed_from_features": True,
        "fresh_data_artifacts_verified": len(manifest["artifact_sha256"]),
        "wrong_sibling_prediction_rows": len(controls),
        "max_wrong_history_mse_discrepancy": float(np.max(np.abs(wrong_mse-np.array([r["wrong_sibling_evidence_mse"] for r in controls])))),
        "checks": ["completed frozen protocol", "all saved artifact/source hashes", "exact prediction target and MSE",
                   "same-parent K2 deletion enumeration", "exhaustive historical K2 enumeration",
                   "teacher action/loss arithmetic", "exact cue copies, survival and first loss", "resident trace capacity/timestamps",
                   "wrong-sibling saved vectors/control differences", "oracle preference for cue/no-cue robust to ties"],
        "not_claimed": "This is independent artifact arithmetic, not a second model run or fresh-data training."}
    return result, oracle_audit, local_choices


def pair_ci(rows, value):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair"]].append(float(value(row)))
    values = np.array([np.mean(pairs[p]) for p in sorted(pairs)])
    if len(values) != 32:
        raise ValueError("CI requires 32 independent pairs within a condition")
    rng = np.random.default_rng(SEED)
    bootstrap = values[rng.integers(len(values), size=(BOOTSTRAPS, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "independent_pairs": len(values), "episode_seed_rows": len(rows)}


def statistics(rows, controls, oracle_checks, choices):
    global_cue = {r["episode_id"]: r["cue_oracle"] for r in oracle_checks}
    output = {"method": "10,000 percentile bootstrap resamples of 32 scene-pair means within each condition; siblings and frozen PaFu seeds averaged first",
              "seed": SEED, "ci_scope": "Scene-sampling uncertainty conditional on the three frozen students; exploratory, no multiplicity correction",
              "by_condition": {}}
    for condition in CONDITIONS:
        part = [r for r in rows if r["condition"] == condition]
        policies = {}
        for policy in ORDER:
            arm = [r for r in part if r["policy"] == policy]
            fields = {name: pair_ci(arm, lambda r, name=name: r[name]) for name in
                      ("mean_mse", "local_regret", "hindsight_regret", "evidence_survived")}
            fields.update({"delta_vs_recent": pair_ci(arm, lambda r: r["mean_mse"]-r["recent_only_mse"]),
                "local_positive_regret_fraction": pair_ci(arm, lambda r: r["local_regret"] > EPS),
                "hindsight_positive_regret_fraction": pair_ci(arm, lambda r: r["hindsight_regret"] > EPS),
                "worse_than_recent_fraction": pair_ci(arm, lambda r: r["mean_mse"]-r["recent_only_mse"] > EPS),
                "cue_loss_rows": sum(not r["evidence_survived"] for r in arm),
                "cue_retained_with_positive_local_regret_rows": sum(r["evidence_survived"] and r["local_regret"] > EPS for r in arm),
                "max_dynamic_logical_bytes": max(r["logical_memory_maxima"]["total_logical_bytes"] for r in arm),
                "controller_parameter_bytes": max(r["controller_parameter_bytes"] for r in arm),
                "mean_online_seconds_per_episode": float(np.mean([r["online_seconds"] for r in arm]))})
            local_cue = lambda r: choices[r["episode_id"], r["policy"], r["seed"]]["cue_oracle"]
            fields["cue_oracle"] = {
                "baseline_retains_cue_rows": sum(r["evidence_survived"] for r in arm),
                "baseline_retains_but_selected_hindsight_drops_cue_rows": sum(r["evidence_survived"] and not global_cue[r["episode_id"]]["selected_keeps_cue"] for r in arm),
                "baseline_retains_but_hindsight_strictly_prefers_no_cue_rows": sum(r["evidence_survived"] and global_cue[r["episode_id"]]["strictly_prefers_no_cue"] for r in arm),
                "baseline_retains_but_local_strictly_prefers_no_cue_rows": sum(r["evidence_survived"] and local_cue(r)["strictly_prefers_no_cue"] for r in arm),
                "joint_retained_and_hindsight_prefers_no_cue": pair_ci(arm, lambda r: r["evidence_survived"] and global_cue[r["episode_id"]]["strictly_prefers_no_cue"]),
                "joint_retained_and_local_prefers_no_cue": pair_ci(arm, lambda r: r["evidence_survived"] and local_cue(r)["strictly_prefers_no_cue"])}
            policies[policy] = fields
        pafu = [r for r in part if r["policy"] == "pafu"]
        teacher = {name: pair_ci(pafu, lambda r, name=name: r["teacher_diagnostic"][name]) for name in
                   ("mean_mse", "teacher_local_regret", "student_minus_teacher_mse", "same_deletion_as_student")}
        teacher["same_deletion_rows"] = sum(r["teacher_diagnostic"]["same_deletion_as_student"] for r in pafu)
        teacher["rows"] = len(pafu)
        teacher["student_vs_controls"] = {}
        for policy in ("attention", "visual_diversity", "wr"):
            comparison = {r["episode_id"]: r["mean_mse"] for r in part if r["policy"] == policy}
            teacher["student_vs_controls"][policy] = pair_ci(pafu, lambda r: r["mean_mse"]-comparison[r["episode_id"]])
        refs = [r for r in part if r["policy"] == "fifo"]
        control = [r for r in controls if r["condition"] == condition]
        output["by_condition"][condition] = {"policies": policies, "pafu_teacher": teacher,
            "hindsight_cue": {"episodes": len(refs),
                "selected_keeps_cue": sum(global_cue[r["episode_id"]]["selected_keeps_cue"] for r in refs),
                "strictly_prefers_cue": sum(global_cue[r["episode_id"]]["strictly_prefers_cue"] for r in refs),
                "strictly_prefers_no_cue": sum(global_cue[r["episode_id"]]["strictly_prefers_no_cue"] for r in refs),
                "no_cue_preference_pair_fraction": pair_ci(refs, lambda r: global_cue[r["episode_id"]]["strictly_prefers_no_cue"])},
            "references": {"recent_only": pair_ci(refs, lambda r: r["recent_only_mse"]),
                "privileged_evidence": pair_ci(refs, lambda r: r["privileged_evidence_mse"]),
                "hindsight_K2": pair_ci(refs, lambda r: r["hindsight_oracle"]["mean_mse"])},
            "history_controls": {name: pair_ci(control, lambda r, name=name: r[name]) for name in
                ("recent_only_mse", "privileged_evidence_mse", "wrong_sibling_evidence_mse", "privileged_minus_recent", "wrong_minus_privileged")}}
    return output


def warmup_classification(rows, events):
    cases = wr_trace(rows, events)
    by_id = {r["episode_id"]: r for r in rows if r["policy"] == "wr"}
    for case in cases:
        row = by_id[case["episode_id"]]
        onset = min(row["evidence_times"])
        onset_record = next(r for r in case["scalar_records"] if r["time"] == onset)
        cue = set(sum(case["evidence_equivalent_feature_times"], []))
        confirmed = [r["time"] for r in case["scalar_records"] if r["time"] in cue and r["confirmed_at"] is not None]
        available = [r["time"] for r in case["scalar_records"] if r["threshold"] is not None]
        label = ("retained" if case["evidence_survived"] else
                 "lost_after_cue_confirmation" if confirmed else
                 "lost_with_onset_threshold_unavailable" if onset_record["threshold"] is None else
                 "lost_without_cue_confirmation_with_onset_threshold_available")
        case.update(condition=row["condition"], onset=onset, onset_score=onset_record["score"],
                    onset_threshold=onset_record["threshold"], first_threshold_time=min(available) if available else None,
                    confirmed_cue_times=confirmed, classification=label,
                    local_regret=row["local_regret"], hindsight_regret=row["hindsight_regret"])
    return cases


def select_badcases(rows, warmup, choices):
    wr_lookup = {r["episode_id"]: r for r in rows if r["policy"] == "wr"}
    failed = [r for r in warmup if r["classification"] == "lost_with_onset_threshold_unavailable"]
    if not failed:
        failed = [r for r in warmup if not r["evidence_survived"]] or warmup
    selected_wr = max(failed, key=lambda r: (r["hindsight_regret"], r["episode_id"]))
    pafu = [r for r in rows if r["policy"] == "pafu" and r["seed"] == 17 and r["evidence_survived"] and r["local_regret"] > EPS]
    if not pafu:
        pafu = [r for r in rows if r["policy"] == "pafu" and r["seed"] == 17]
    selected_pafu = max(pafu, key=lambda r: (r["local_regret"], r["episode_id"]))
    selected = [dict(wr_lookup[selected_wr["episode_id"]]), dict(selected_pafu)]
    for row, rule in zip(selected, ("Largest hindsight regret within WR onset-threshold-unavailable cue-loss cases (fallback: other WR losses)",
                                    "Fixed PaFu seed17; largest local regret among cue-retained episodes (fallback: all seed17 episodes)")):
        row["selection_rule"] = rule
        row["local_parent"] = choices[row["episode_id"], row["policy"], row["seed"]]
    selected[0]["wr_classification"] = selected_wr
    return selected


def export_frames(data, output, selected, records):
    records_by_id = {r["id"]: r for r in records}
    with np.load(data / "unique_rgb.npz", allow_pickle=False) as arrays:
        rgb = arrays["rgb"]
    hashes = read(data / "unique_rgb_sha256.json")
    (output / "frames").mkdir(exist_ok=True)
    for case in selected:
        rec = records_by_id[case["episode_id"]]
        cue, q = min(case["evidence_times"]), case["query"]
        slots = [("Cue", cue), ("Query", q), ("Future h3", q+3), ("Future h6", q+6)]
        slots += [(f"Kept {i+1}", t) for i, t in enumerate(case["kept_times"])]
        oracle_name = "hindsight_oracle" if case["policy"] == "wr" else "local_oracle"
        label_prefix = "Hindsight K2" if case["policy"] == "wr" else "Local oracle"
        slots += [(f"{label_prefix} {i+1}", t) for i, t in enumerate(case[oracle_name]["kept_times"])]
        case["rgb_frames"] = []
        for label, time in slots:
            index = rec["rgb_indices"][time]
            import hashlib
            if hashlib.sha256(rgb[index].tobytes()).hexdigest() != hashes[index]:
                raise ValueError("RGB frame hash mismatch")
            filename = f"frames/{case['episode_id']}_t{time}.png"
            plt.imsave(output / filename, rgb[index])
            case["rgb_frames"].append({"label": label, "time": time, "unique_rgb_index": index,
                                      "rgb_sha256": hashes[index], "png": filename})
    return selected


def render(output):
    style()
    stats, cases = read(output / "statistics.json"), read(output / "bad_cases.json")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    for ax, condition in zip(axes, CONDITIONS):
        part = stats["by_condition"][condition]
        y = np.arange(len(ORDER))
        means = [part["policies"][p]["mean_mse"]["mean"] for p in ORDER]
        intervals = np.array([part["policies"][p]["mean_mse"]["ci95"] for p in ORDER])
        ax.barh(y, means, color=COLORS, alpha=.8, height=.6)
        ax.errorbar(means, y, xerr=np.maximum(0, np.stack([np.array(means)-intervals[:, 0], intervals[:, 1]-means])),
                    fmt="none", color="#29343a", capsize=3, lw=1)
        ax.axvline(part["references"]["recent_only"]["mean"], color="#b24b4b", ls="--", lw=1.2, label="Recent only")
        ax.axvline(part["references"]["hindsight_K2"]["mean"], color="#35664c", ls=":", lw=1.6, label="Hindsight K2*")
        ax.set(yticks=y, yticklabels=LABELS, xlabel="Feature MSE (h3/h6 mean)", title=TITLES[condition], xlim=(0, float(intervals.max())*1.12))
        ax.grid(axis="x", alpha=.15)
    axes[0].invert_yaxis()
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(.59, .94), ncol=2)
    fig.suptitle("Fresh Gate development: frozen predictor/students, R=4, K=2", fontsize=14, y=.98)
    fig.text(.5, .025, "Each condition: 32 new scene pairs. Bars: means; intervals: 95% pair-bootstrap CI (10,000 resamples).\n"
             "Siblings and 3 PaFu seeds are averaged inside each pair. *Privileged evaluation-only reference; equal K does not mean equal total bytes.",
             ha="center", va="bottom", fontsize=9, color="#50575b")
    fig.subplots_adjust(left=.15, right=.99, top=.82, bottom=.20, wspace=.14)
    save(fig, figures, "fresh_metrics")
    fig, axes = plt.subplots(2, 8, figsize=(16.6, 6.4))
    for i, case in enumerate(cases):
        scope = "hindsight" if case["policy"] == "wr" else "local"
        title = (f"{case['policy'].upper()} | {case['condition']} | {case['episode_id'][:10]} | "
                 f"MSE {case['mean_mse']:.4f}; {scope} regret {case[scope+'_regret']:.4f}; "
                 f"cue {'retained' if case['evidence_survived'] else 'lost'}")
        fig.text(.055, .88-i*.39, title, fontsize=11, weight="semibold")
        for ax, frame in zip(axes[i], case["rgb_frames"]):
            ax.imshow(plt.imread(output / frame["png"]), interpolation="nearest")
            ax.set_title(f"{frame['label']}\nt={frame['time']}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#ced3d6")
    fig.suptitle("Actual RGB observations and evaluator-only oracle alternatives", fontsize=14, y=.98)
    fig.text(.5, .025, "Future h3/h6 images are ground truth, not model predictions. Future frames and oracle choices are evaluator-only.\n"
             "WR hindsight can access discarded history; PaFu local oracle shares its three-item parent. Cases selected by largest gap within stated strata.",
             ha="center", va="bottom", fontsize=9, color="#50575b")
    fig.subplots_adjust(left=.035, right=.99, top=.81, bottom=.15, hspace=.70, wspace=.16)
    save(fig, figures, "fresh_rgb_badcases")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "build/literature-baselines-20261022"
    parser.add_argument("--source", type=Path, default=base / "fresh-eval")
    parser.add_argument("--data", type=Path, default=base / "fresh-data")
    parser.add_argument("--output", type=Path, default=ROOT / "results/literature_baselines/fresh-20261022")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    source, data, output = args.source.resolve(), args.data.resolve(), args.output.resolve()
    if args.render_only:
        provenance = read(output / "manifest.json")
        hashed_files(output, {name: r["published_sha256"] for name, r in provenance["published_files"].items()})
    else:
        if not (source / "summary.json").exists():
            raise ValueError("Fresh summary not complete yet; no results interpreted/exported")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Refuse to overwrite an existing archive")
        summary, rows, records = read(source / "summary.json"), read(source / "episode_results.json"), read(data / "records.json")
        events = [json.loads(line) for line in (source / "online_trace.jsonl").read_text().splitlines()]
        subsets, controls = read(source / "oracle_subsets.json"), read(source / "history_controls.json")
        audit, oracle_checks, choices = audit_run(source, data, summary, rows, records, events, subsets)
        stats = statistics(rows, controls, oracle_checks, choices)
        warmup = warmup_classification(rows, events)
        selected = select_badcases(rows, warmup, choices)
        output.mkdir(parents=True)
        raw = {}
        for name in ("summary.json", "protocol.json", "episode_results.json", "history_controls.json"):
            write(output / name, scrub(read(source / name)))
            raw["evaluation/"+name] = source_info(source / name)
        for original, destination in (("manifest.json", "data_manifest.json"), ("hardware.json", "hardware.json"),
                ("encoder_config.json", "encoder_config.json"), ("frozen_encoder_audit.json", "frozen_encoder_audit.json"),
                ("rgb_pair_audit.json", "rgb_pair_audit.json"), ("records.json", "data_records.json")):
            write(output / destination, scrub(read(data / original)))
            raw["encoding/"+original] = source_info(data / original)
        receipt = read(source.parent / "hpc-execution-receipt.json")
        screenshot = source.parent / "a100-completed.png"
        if (receipt["screenshot_sha256"] != sha(screenshot) or receipt["hardware"] != read(data / "hardware.json")
                or receipt["state"] != "COMPLETED" or receipt["exit_code"] != "0:0"):
            raise ValueError("HPC receipt/screenshots mismatch")
        write(output / "hpc-execution-receipt.json", scrub(receipt))
        shutil.copyfile(screenshot, output / "a100-completed.png")
        raw["hpc-execution-receipt.json"] = source_info(source.parent / "hpc-execution-receipt.json")
        raw["a100-completed.png"] = source_info(screenshot)
        write(output / "audit.json", audit)
        write(output / "oracle_enumeration_audit.json", oracle_checks)
        write(output / "statistics.json", stats)
        write(output / "wr_classification.json", warmup)
        pafu = [{k: row[k] for k in ("episode_id", "pair", "condition", "seed", "query", "kept_times", "mean_mse", "local_regret", "teacher_diagnostic")}
                | {"local_parent": choices[row["episode_id"], row["policy"], row["seed"]]} for row in rows if row["policy"] == "pafu"]
        write(output / "pafu_teacher_diagnostic.json", pafu)
        selected = export_frames(data, output, selected, records)
        write(output / "bad_cases.json", selected)
        excluded = {"evaluation/"+p.name: source_info(p) for p in source.iterdir() if p.suffix == ".npz" or p.name in ("online_trace.jsonl", "oracle_subsets.json")}
        excluded.update({"encoding/unique_rgb.npz": source_info(data / "unique_rgb.npz"),
                         "encoding/unique_features.npz": source_info(data / "unique_features.npz")})
        provenance = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Portable completed fresh development evidence; independent artifact arithmetic; no model execution",
            "raw_sources": raw, "excluded_large_artifacts": excluded,
            "original_evaluation_source_sha256": summary["fresh_source_sha256"],
            "original_training_source_sha256": summary["frozen_students"]["training_source_sha256"],
            "internal_hash_references": "Hashes inside exported JSON refer to original raw sources, not path-redacted copies",
            "transformations": {"copied_json": "Numerics preserved; machine repository prefix replaced by $REPOSITORY",
                "statistics": "Pair-block aggregation/bootstrap only; no refitting", "audit": "Recompute saved MSE, exhaustive K2 and local-parent oracle arithmetic, cue copies and teacher choices",
                "wr_classification": "Immutable WR scalar records plus candidate-ID traces and explicit onset/confirmation strata",
                "rgb": "Exact saved RGB frame indexed through records.json; PNG transport, no image synthesis"}}
        (output / "README.md").write_text("""# Fresh Gate literature-baseline development evidence

192 episodes / 96 new scene pairs; three conditions, 32 pairs each. Fixed old
predictor and frozen dev-v3 PaFu students; no fresh fitting. These are explicit
WR/PaFu mechanism adaptations, not full paper reproductions or a held-out benchmark claim.

- [Summary](summary.json), [protocol](protocol.json), [per-case rows](episode_results.json).
- [Independent arithmetic audit](audit.json), [oracle enumeration](oracle_enumeration_audit.json).
- [Statistics and pair CIs](statistics.json), [history controls](history_controls.json).
- [WR strata and traces](wr_classification.json), [PaFu teacher diagnostics](pafu_teacher_diagnostic.json).
- [Actual RGB bad cases](bad_cases.json), [data provenance](data_manifest.json), [export manifest](manifest.json).
- [Metrics PNG](figures/fresh_metrics.png) / [PDF](figures/fresh_metrics.pdf).
- [RGB cases PNG](figures/fresh_rgb_badcases.png) / [PDF](figures/fresh_rgb_badcases.pdf).
- [A100 receipt](hpc-execution-receipt.json), [scheduler screenshot](a100-completed.png).

A100 encoding job10097 completed with 143 allocated GPU-seconds; model prediction
and offline oracle enumeration ran on CPU. Encoding time alone is not allocated
GPU wall time. Equal K is not equal total bytes. CIs resample independent scene
pairs after averaging siblings and fixed student seeds, not individual rows.
Selected RGB examples are explicitly chosen diagnostic cases, not prevalence estimates.

Weights and large arrays are excluded; their raw hashes remain recorded. Replot
from these portable JSON/PNG files, without a model or the original dataset:

```sh
python experiments/phase2/export_fresh_literature_report.py --render-only
```
""")
    render(output)
    provenance["exporter"] = {"path": relative(__file__), "sha256": sha(__file__),
        "shared_export_helpers_sha256": sha(ROOT / "experiments/phase2/export_literature_report.py"),
        "matplotlib": matplotlib.__version__, "numpy": np.__version__, "model_calls": 0}
    provenance["published_files"] = {str(p.relative_to(output)): {"published_sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    write(output / "manifest.json", provenance)
    hashed_files(output, {name: r["published_sha256"] for name, r in provenance["published_files"].items()})
    if any(p.suffix in (".pt", ".npz", ".npy") for p in output.rglob("*")):
        raise ValueError("Unexpected large input/model array in archive")
    print(json.dumps({"output": relative(output), "published_files": len(provenance["published_files"]),
                      "published_bytes": sum(r["bytes"] for r in provenance["published_files"].values()),
                      "new_model_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
