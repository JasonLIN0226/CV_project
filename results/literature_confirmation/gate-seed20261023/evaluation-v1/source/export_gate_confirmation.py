"""Audit and publish a completed frozen confirmation; no fitting/model calls."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "experiments/predictor/control/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import numpy as np
from experiments.phase2.diagnose_oracle_learnability import causal_features, parent_from_arrays
from experiments.phase2.generate_deletion_pilot import sha256, write
from experiments.phase2.run_gate_confirmation import (
    PRIMARY_ARM, PRIMARY_CONTROLS, CONDITIONS, load_probes, primary_decision, evidence_summary, verify_lock,
)


def grouped_ci(rows, field, condition="all"):
    episodes, pairs = defaultdict(list), defaultdict(list)
    for row in rows:
        if condition == "all" or row["condition"] == condition:
            episodes[(row["pair"], row["episode_id"])].append(row[field])
    for (pair, _), values in episodes.items():
        pairs[pair].append(float(np.mean(values)))
    values = np.asarray([np.mean(v) for v in pairs.values()])
    rng = np.random.default_rng(20261022)
    bootstrap = values[rng.integers(len(values), size=(5000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "low": float(np.quantile(bootstrap, .025)),
            "high": float(np.quantile(bootstrap, .975)), "pairs": len(values)}


def audit_export(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Require an empty portable output")
    summary = json.loads((source/"summary.json").read_text())
    protocol = json.loads((source/"protocol.json").read_text())
    verify_lock(protocol)
    assert summary["protocol_sha256"] == sha256(source/"protocol.json")
    for name, digest in summary["result_sha256"].items():
        assert sha256(source/name) == digest, name
    receipt = json.loads((source/"run_receipt.json").read_text())
    data = ROOT / receipt["confirmation_directory"]
    assert sha256(data/"manifest.json") == receipt["confirmation_manifest_sha256"]
    manifest = json.loads((data/"manifest.json").read_text())
    for name, digest in manifest["artifact_sha256"].items():
        assert sha256(data/name) == digest, name
    records = {r["id"]: r for r in json.loads((data/"records.json").read_text())}
    rows = json.loads((source/"online_results.json").read_text())
    common = json.loads((source/"common_parent_results.json").read_text())
    statistics = json.loads((source/"statistics.json").read_text())
    decision = json.loads((source/"primary_decision.json").read_text())
    assert decision == primary_decision(statistics)
    cache = defaultdict(dict)
    for row in json.loads((source/"oracle_subsets.json").read_text()):
        cache[row["episode_id"]][tuple(row["kept_times"])] = float(np.mean(row["feature_mse"]))
    max_error = 0.
    with np.load(source/"prediction_vectors.npz") as a:
        vectors, cases = a["predictions"], a["cases"]
    assert len(rows) == len(vectors) == len(cases) == 192*13
    assert len(set(cases)) == len(cases)
    arrays = {}
    for rec in records.values():
        with np.load(data/rec["feature_path"]) as a:
            arrays[rec["id"]] = (a["z"].copy(), a["actions"].copy())
    for row, prediction, case in zip(rows, vectors, cases):
        assert case == row["episode_id"]+":"+row["arm"]
        rec = records[row["episode_id"]]; q = rec["query"]
        y = arrays[rec["id"]][0][q+np.array([3, 6])].astype(np.float64)
        other = arrays[rec["sibling"]][0][q+np.array([3, 6])].astype(np.float64)
        error = prediction.astype(np.float64)-y; direction = y-other
        coefficient = np.sum(error*direction, -1)/np.sum(direction*direction, -1)
        along = coefficient[:, None]*direction
        values = {"mse": float(np.mean(error**2)), "contrast_mse": float(np.mean(along**2)),
                  "orthogonal_mse": float(np.mean((error-along)**2)),
                  "feature_branch_correct": float(np.mean(np.sum((prediction-y)**2, -1) < np.sum((prediction-other)**2, -1)))}
        for field, value in values.items():
            delta = abs(row[field]-value); max_error = max(max_error, delta)
            assert delta < 1e-12, (case, field)
        assert abs(row["mse"]-cache[rec["id"]][tuple(row["kept_times"])] ) < 1e-12
        local = min(cache[rec["id"]][tuple(t for j, t in enumerate(row["candidate_times"]) if j != d)] for d in range(3))
        assert abs(row["local_regret"]-(row["mse"]-local)) < 1e-12
    probes = load_probes(ROOT/protocol["paths"]["parameters"])
    with np.load(ROOT/protocol["paths"]["followup"]/"gate/unit_columns/basis.npz") as a:
        basis = {k: a[k].copy() for k in ("input_mean", "projection")}
    for row in common:
        rec = records[row["episode_id"]]; z, actions = arrays[rec["id"]]; q = rec["query"]
        times = row["candidate_times"]; c, r, p = parent_from_arrays(rec, z, actions, q, times)
        x = causal_features(c, r, q, basis, p)
        losses = [cache[rec["id"]][tuple(t for j, t in enumerate(times) if j != d)] for d in range(3)]
        np.testing.assert_array_equal(row["losses"], losses)
        for arm, probe in probes.items():
            assert row["choices"][arm] == int(np.argmin(probe.predict(x)))
        for arm, choice in row["choices"].items():
            assert row[arm+"_mse"] == losses[choice]
            assert row[arm+"_regret"] == losses[choice]-min(losses)
    ci_error = 0.
    for chosen, groups in [(common, statistics["common_parent"])]+[
            ([r for r in rows if r["arm"] == arm], groups) for arm, groups in statistics["online"].items()]:
        for condition, fields in groups.items():
            for field, expected in fields.items():
                actual = grouped_ci(chosen, field, condition)
                delta = max(abs(actual[k]-expected[k]) for k in actual)
                assert delta < 1e-12; ci_error = max(ci_error, delta)
    events, deleted, last = defaultdict(list), defaultdict(set), {}
    for line in (source/"online_trace.jsonl").read_text().splitlines():
        row = json.loads(line); key = (row["episode_id"], row["arm"])
        assert row["time"] == last.get(key, -1)+1; last[key] = row["time"]
        resident = row["retained_timestamps"]+row["recent_timestamps"]
        assert not set(resident).intersection(deleted[key])
        assert len(row["retained_timestamps"]) <= 2 and len(row["recent_timestamps"]) <= 4
        if row["deleted_timestamp"] is not None:
            deleted[key].add(row["deleted_timestamp"])
        events[key].append(row)
    for row in rows:
        key = (row["episode_id"], row["arm"])
        assert last[key] == row["query"]
        check = evidence_summary(events[key], row["kept_times"], row["evidence_equivalent_feature_times"])
        assert all(check[k] == row[k] for k in check)
    by_id = defaultdict(dict)
    for row in rows:
        by_id[row["episode_id"]][row["arm"]] = row
    contrasts = []
    for episode, arms in by_id.items():
        primary = arms[PRIMARY_ARM]
        contrast = {k: primary[k] for k in ("episode_id", "pair", "condition")}
        for control in PRIMARY_CONTROLS:
            for field in ("contrast_mse", "orthogonal_mse", "feature_branch_correct", "cue_survival"):
                contrast[control+"__"+field] = primary[field]-arms[control][field]
        contrasts.append(contrast)
    fields = [k for k in contrasts[0] if "__" in k]
    task_comparisons = {condition: {field: grouped_ci(contrasts, field, condition) for field in fields} for condition in CONDITIONS}
    primary_rows = [r for r in rows if r["arm"] == PRIMARY_ARM]
    illustrative = {"selection_rule": "One largest positive and one largest negative masked-cost-minus-reader-attention episode; descriptive illustrations only, full population retained",
                    "worst": max(primary_rows, key=lambda r: (r["minus_attention"], r["episode_id"])),
                    "best": min(primary_rows, key=lambda r: (r["minus_attention"], r["episode_id"]))}
    audit = {"online_prediction_rows_checked": len(rows), "max_mse_contrast_recomputation_error": max_error,
             "common_parent_rows_checked": len(common), "fixed_probe_choices_recomputed": len(common)*len(probes),
             "trace_events_checked": sum(map(len, events.values())), "max_pair_CI_recomputation_error": ci_error,
             "all_online_first_loss_and_survival_recomputed": True, "fit_calls": 0, "world_model_calls": 0,
             "scope": "Independent saved-vector arithmetic, frozen-probe choices, oracle-cache consistency and trace/CI audit; no repeated world-model inference."}
    output.mkdir(parents=True)
    files = ("protocol.json", "run_receipt.json", "online_results.json", "common_parent_results.json",
             "oracle_subsets.json", "statistics.json", "primary_decision.json", "summary.json")
    for name in files:
        shutil.copy2(source/name, output/name)
    write(output/"audit.json", audit)
    write(output/"task_contrast_comparisons.json", task_comparisons)
    write(output/"illustrative_cases.json", illustrative)
    write(output/"data_manifest.json", manifest)
    shutil.copy2(data/"records.json", output/"data_records.json")
    (output/"source").mkdir()
    for name in ("run_gate_confirmation.py", "gate_confirmation_data.py", "export_gate_confirmation.py"):
        shutil.copy2(ROOT/"experiments/phase2"/name, output/"source"/name)
    render(output)
    (output/"README.md").write_text("""# Frozen Gate confirmation, seed 20261023

The locked protocol predates generation/readout of this separate 96-pair set.
All six existing probes, WR/PaFu, FIFO/diversity/attention, reader and normalization
are frozen. No training, support selection, hyperparameter selection or arm
selection uses these observations. See `primary_decision.json` for the single
conjunctive decision against shuffled cost, reader-attention and diversity.

`online_results.json` and `common_parent_results.json` retain every result.
`statistics.json` uses independent sibling pairs, not repeated parent rows.
`task_contrast_comparisons.json` is descriptive, evaluator-only, and cannot replace
the primary comparison. A positive feature-MSE result alone would not establish
physical planning utility. Same K does not mean equal total memory/compute.

`audit.json` reconstructs saved-vector arithmetic, probe choices, first information
loss/survival, irreversible traces and pair CIs. Large prediction vectors and full
traces remain under the manifest-bound ignored build directory. Data features/RGB
are not included here; `data_manifest.json` identifies all source bytes.
""")
    write(output/"manifest.json", {"raw_source_directory": str(source.relative_to(ROOT)),
          "raw_source_sha256": {p.name: sha256(p) for p in source.iterdir() if p.is_file()},
          "exporter_sha256": sha256(Path(__file__)),
          "published_files": {str(p.relative_to(output)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                              for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({"audit": audit, "decision": decision}, indent=2))


def render(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stats = json.loads((output/"statistics.json").read_text())["online"]
    arms = ["original_cost_ridge", "original_attention_ridge", "original_shuffled_cost_ridge",
            "masked_cost_ridge", "masked_attention_ridge", "masked_shuffled_cost_ridge",
            "fifo", "attention", "visual_diversity", "wr", "pafu_seed_17", "pafu_seed_23", "pafu_seed_41"]
    labels = [a.replace("_ridge", "").replace("_", " ") for a in arms]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    y = np.arange(len(arms))
    for ax, field, title in zip(axes, ("mse", "cue_survival"), ("Mean feature MSE", "Cue survival at query")):
        values = [stats[a]["all"][field]["mean"] for a in arms]
        low = [v-stats[a]["all"][field]["low"] for a, v in zip(arms, values)]
        high = [stats[a]["all"][field]["high"]-v for a, v in zip(arms, values)]
        colors = ["#c05e3f" if a == PRIMARY_ARM else "#3c8097" for a in arms]
        ax.barh(y, values, xerr=[low, high], color=colors, capsize=3)
        ax.set_xlabel(title); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(y, labels); axes[0].invert_yaxis(); axes[1].set_xlim(0, 1.05)
    fig.suptitle("Independent seed-20261023 confirmation: all frozen arms")
    fig.text(.5, .012, "96 sibling pairs; 95% pair-bootstrap intervals. Primary arm fixed before data. No new fits or selection.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96)); (output/"figures").mkdir()
    for ext in ("png", "pdf"):
        fig.savefig(output/"figures"/("frozen_confirmation."+ext), dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_export(args.source, args.output)
