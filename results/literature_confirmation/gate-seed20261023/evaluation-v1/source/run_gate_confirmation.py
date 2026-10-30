"""Pre-lock and evaluate frozen retention probes on seed-20261023 scenes.

`freeze` reads only existing development artifacts and source. `evaluate` refuses
to proceed if any locked source/parameter changed. No fitting or selection API is
called. Future observations, evidence and task-contrast labels stay in the harness.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "experiments/predictor/control/src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np
import torch
from experiments.phase2.causal_model_scores import CausalModelScores
from experiments.phase2.diagnose_gate_oracle_metric import paired_components
from experiments.phase2.diagnose_oracle_learnability import (
    ARMS, RidgeProbe, ProbeRetention, causal_features, parent_from_arrays, paired_summary,
)
from experiments.phase2.gate_confirmation_data import ConfirmationGateDataset, SEED, SCOPE
from experiments.phase2.generate_deletion_pilot import sha256, write
from experiments.phase2.memory import BoundedMemory, PolicyContext
from experiments.phase2.portable_snapshot import PortableSnapshot, state_hash
from experiments.phase2.run_fresh_literature_diagnostic import load_frozen_students
from experiments.phase2.run_gate_followup_phase2 import FollowupGateEvaluator
from experiments.phase2.run_literature_baselines import (
    QueryOracle, evidence_equivalence, online_policy, overflow_parent, student_inputs,
)

BASELINES = ("fifo", "visual_diversity", "attention", "wr")
CONDITIONS = ("all", "id", "long_gap", "distractor_shift")
PRIMARY_ARM = "masked_cost_ridge"
PRIMARY_CONTROLS = ("masked_shuffled_cost_ridge", "attention", "visual_diversity")


def load_probes(path):
    raw = json.loads(Path(path).read_text())
    original = {name: RidgeProbe(np.asarray(raw[name]["scale"], np.float64),
                                 np.asarray(raw[name]["weights"], np.float64),
                                 float(raw[name]["label_scale"]), name) for name in ARMS}
    scale = original["cost_ridge"].scale
    if scale.shape != (903,) or np.count_nonzero(scale <= 1e-6) != 710:
        raise ValueError("Unexpected frozen probe support")
    for probe in original.values():
        if not np.array_equal(probe.scale, scale) or probe.weights.shape != scale.shape:
            raise ValueError("Frozen probe architectures/scales differ")
    probes = {"original_" + name: probe for name, probe in original.items()}
    probes.update({"masked_" + name: RidgeProbe(probe.scale.copy(),
                  np.where(scale <= 1e-6, 0., probe.weights), probe.label_scale, name)
                   for name, probe in original.items()})
    return probes


def primary_decision(statistics):
    """Conjunctive preregistered criterion; never choose the best probe arm."""
    group = statistics["online"][PRIMARY_ARM]["all"]
    comparisons = {control: group["minus_" + control] for control in PRIMARY_CONTROLS}
    passed = all(row["high"] < 0 for row in comparisons.values())
    return {"primary_arm": PRIMARY_ARM, "comparisons": comparisons,
            "all_three_upper_95pct_bounds_below_zero": passed,
            "conclusion": ("Evidence of transfer for this frozen probe under the frozen feature-MSE reader; task-value validity still requires the separate contrast/content diagnosis."
                           if passed else "Independent confirmation did not establish retention gains over all three required controls; do not promote a new method."),
            "scope": "Conjunctive single-claim test, not a search over arms or conditions; descriptive secondary intervals are not multiplicity-corrected."}


def freeze(args):
    path = args.protocol.resolve()
    if path.exists():
        raise ValueError("Refuse to overwrite a confirmation protocol")
    old = PortableSnapshot(args.snapshot)
    evaluator = FollowupGateEvaluator(old, args.followup)
    probes = load_probes(args.parameters)
    files = [Path(__file__), args.encoder]
    files += [ROOT / "experiments/phase2" / name for name in (
        "gate_confirmation_data.py", "diagnose_oracle_learnability.py", "diagnose_gate_oracle_metric.py",
        "run_fresh_literature_diagnostic.py", "encode_fresh_gate_diagnostic.py", "run_literature_baselines.py",
        "run_retention_baseline_pilot.py", "run_gate_followup_phase2.py", "causal_model_scores.py",
        "literature_baselines.py", "memory.py", "portable_snapshot.py")]
    files += [ROOT / "experiments/predictor/control/src/phase1" / name for name in ("model.py", "inputs.py")]
    inputs = [args.parameters, args.prior_records, args.students / "protocol.json", args.students / "summary.json"]
    inputs += [args.students / f"pafu_seed_{seed}.pt" for seed in (17, 23, 41)]
    inputs += list(evaluator._artifacts)
    protocol = {
        "version": "frozen-gate-confirmation-v1", "locked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": SCOPE, "seed": SEED, "independent_pairs": 96, "episodes": 192,
        "conditions": ["id", "long_gap", "distractor_shift"], "pairs_per_condition": 32,
        "encoder_path": str(args.encoder.resolve().relative_to(ROOT)), "encoder_sha256": sha256(args.encoder),
        "source_sha256": {str(p.resolve().relative_to(ROOT)): sha256(p) for p in files},
        "input_sha256": {str(p.resolve().relative_to(ROOT)): sha256(p) for p in inputs},
        "predictor_binding": evaluator.binding,
        "paths": {name: str(getattr(args, name).resolve().relative_to(ROOT)) for name in
                  ("snapshot", "followup", "students", "parameters", "prior_records")},
        "recent": 4, "capacity": 2, "horizons": [3, 6],
        "objective": "equal mean h3/h6 feature MSE; one common deletion action",
        "primary_arm": PRIMARY_ARM, "primary_controls": list(PRIMARY_CONTROLS),
        "primary_rule": "All three paired online-MSE differences must have two-sided bootstrap95% upper bound strictly below zero, over all96 independent pairs. No arm/condition selection.",
        "secondary": ["task sibling-contrast/orthogonal error and branch correctness (evaluator only)",
                      "evidence first complete information loss and query survival",
                      "same-parent local regret and optimum-action rate", "R4/K2 numeric/controller bytes and latency"],
        "probes": list(probes), "mask": "Original coefficients set to zero iff existing TRAIN RMS <=1e-6; 710/903 columns; no refit",
        "baseline_arms": list(BASELINES) + [f"pafu_seed_{s}" for s in (17, 23, 41)],
        "common_parents": "Deduplicated union of actual FIFO, diversity, reader-attention final-query K+1 states; independent of fitted probe outputs. WR is stateful and evaluated only on its own online trajectory.",
        "oracles": "Local deletion at query; no exhaustive archive-selection oracle in this confirmation",
        "bootstrap": {"unit": "episode means then sibling pair means", "replicates": 5000, "seed": 20261022},
        "fitting_calls": 0, "support_or_hyperparameter_tuning": False, "new_reader_or_normalization": False,
        "test_arrays": False, "prior_development_metadata_sha256": sha256(args.prior_records),
        "limitations": ["Same generator family; independent seeds do not establish new-task generalization.",
                        "Fixed feature-MSE criterion can reward reader/timing effects; task contrast is descriptive, not physical success.",
                        "One frozen shuffled-label seed is a negative control, not a full permutation test.",
                        "Same K and paired probe capacity; different baseline total bytes/compute reported separately."]}
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, protocol)
    print(json.dumps({"status": "locked_before_confirmation_read", "protocol": str(path),
                      "sha256": sha256(path), "probe_parameters_per_arm": next(iter(probes.values())).bytes}, indent=2))


def verify_lock(protocol):
    for family in ("source_sha256", "input_sha256"):
        for relative, expected in protocol[family].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError("Frozen confirmation source/input changed: " + relative)
    if (protocol["seed"] != SEED or protocol["scope"] != SCOPE
            or protocol["primary_arm"] != PRIMARY_ARM or protocol["primary_controls"] != list(PRIMARY_CONTROLS)):
        raise ValueError("Confirmation decision rule changed")


def replay(online, episode_id, z, actions, q):
    """Only this harness has an episode; the online object sees one item/step."""
    events, parent = [], None
    for t in range(q + 1):
        if t == q:
            parent = overflow_parent(online.stream.snapshot, z[t], episode_id, t)
        snapshot, event = online.observe(z[t], executed_action=None if t == 0 else actions[t - 1])
        events.append(event)
    if parent is None or len(snapshot.retained) != 2:
        raise AssertionError("Missing R4/K2 final parent")
    return snapshot, parent, events


def evidence_summary(events, kept, groups):
    first_loss = [next((event["time"] for event in events
                       if event["time"] > max(group) and not set(group).intersection(
                           event["retained_timestamps"] + event["recent_timestamps"])), None)
                  for group in groups]
    return {"cue_survival": float(all(set(kept).intersection(group) for group in groups)),
            "first_evidence_information_loss": first_loss,
            "evidence_equivalent_feature_times": groups}


def evaluate(args):
    protocol_path = args.protocol.resolve()
    protocol_sha = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    verify_lock(protocol)  # BEFORE any confirmation directory is opened.
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty confirmation output")
    out.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    torch.set_num_threads(1)
    paths = {name: ROOT / value for name, value in protocol["paths"].items()}
    old = PortableSnapshot(paths["snapshot"])
    prior = json.loads(paths["prior_records"].read_text())
    fresh = ConfirmationGateDataset(args.data, encoder_sha256=protocol["encoder_sha256"],
                                    prior_records=old.records + prior)
    fresh.verify_old_normalization(old.norm)
    scores = CausalModelScores(FollowupGateEvaluator(old, paths["followup"]))
    if scores._evaluator.binding != protocol["predictor_binding"]:
        raise ValueError("Reader binding differs from locked reader")
    models, student_proof = load_frozen_students(paths["students"], old, fresh.manifest["started_utc"])
    probes = load_probes(paths["parameters"])
    write(out / "protocol.json", protocol)
    write(out / "run_receipt.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
          "locked_protocol_sha256": protocol_sha, "confirmation_manifest_sha256": fresh.manifest_sha256,
          "frozen_students": student_proof, "confirmation_directory": str(args.data.resolve().relative_to(ROOT))})
    rows, common, cache_rows, predictions, names = [], [], [], [], []
    with (out / "online_trace.jsonl").open("x") as trace:
        for index, rec in enumerate(fresh.records):
            z, actions = fresh.episode(rec)
            sibling = fresh.by_id[rec["sibling"]]
            sibling_z, sibling_actions = fresh.episode(sibling)
            q = rec["query"]
            if (not np.array_equal(actions, sibling_actions)
                    or not np.array_equal(z[q-3:q+1], sibling_z[q-3:q+1])):
                raise ValueError("Sibling causal recent/action equivalence failed")
            oracle = QueryOracle(rec["id"], z, actions, q, scores)
            # Evidence is evaluator-only, and is never attached to online/scores.
            groups = evidence_equivalence(z, fresh.evidence(rec), q - 3)
            target = z[q + np.array([3, 6])]
            sibling_target = sibling_z[q + np.array([3, 6])]
            pool = set()
            instances = [(name, online_policy(rec["id"], name, scores, seed=17), 0) for name in BASELINES]
            instances += [(f"pafu_seed_{seed}", online_policy(rec["id"], "pafu", scores, seed=seed, student=model),
                           sum(p.numel()*p.element_size() for p in model.parameters())) for seed, model in models.items()]
            instances += [(name, ProbeRetention(rec["id"], scores, probe), probe.bytes) for name, probe in probes.items()]
            for arm, online, parameter_bytes in instances:
                snap, parent, events = replay(online, rec["id"], z, actions, q)
                times = tuple(sorted(e.timestamp for e in parent[0]))
                kept = tuple(sorted(e.timestamp for e in snap.retained))
                if arm in ("fifo", "attention", "visual_diversity"):
                    pool.add(times)
                losses, prediction = oracle.evaluate(kept)
                local = oracle.best(times)
                component = paired_components(prediction, target, sibling_target)
                row = {"episode_id": rec["id"], "pair": rec["pair"], "condition": rec["condition"],
                       "arm": arm, "query": q, "kept_times": list(kept), "candidate_times": list(times),
                       "feature_mse": losses.tolist(), "mse": float(losses.mean()),
                       "local_oracle_mse": local["mean_mse"], "local_regret": float(losses.mean())-local["mean_mse"],
                       **component, **evidence_summary(events, kept, groups),
                       "controller_parameter_bytes": parameter_bytes,
                       "dynamic_logical_maxima": online.maxima, "online_seconds": online.seconds}
                rows.append(row); predictions.append(prediction); names.append(rec["id"] + ":" + arm)
                for event in events:
                    trace.write(json.dumps({"episode_id": rec["id"], "arm": arm, **event}, allow_nan=False) + "\n")
            for times in sorted(pool):
                c, r, past = parent_from_arrays(rec, z, actions, q, times)
                x = causal_features(c, r, q, scores._basis, past)
                loss = np.array([float(oracle.evaluate([t for j, t in enumerate(times) if j != d])[0].mean()) for d in range(3)])
                choices = {name: int(np.argmin(probe.predict(x))) for name, probe in probes.items()}
                context = PolicyContext(q, r)
                choices.update(fifo=0, attention=int(np.argmin(scores.attention(c, context))),
                               visual_diversity=int(BoundedMemory(2, "visual_diversity")._select_deletion(c, context, 3)))
                for seed, model in models.items():
                    with torch.inference_mode():
                        choices[f"pafu_seed_{seed}"] = int(torch.argmin(model(torch.tensor(student_inputs(c, r, q, scores._basis)))))
                row = {"episode_id": rec["id"], "pair": rec["pair"], "condition": rec["condition"],
                       "candidate_times": list(times), "losses": loss.tolist(), "choices": choices,
                       "margin": float(np.partition(loss, 1)[1]-loss.min())}
                for name, choice in choices.items():
                    row[name + "_mse"] = float(loss[choice])
                    row[name + "_regret"] = float(loss[choice]-loss.min())
                    row[name + "_optimal"] = float(loss[choice] <= loss.min()+1e-8)
                for control in PRIMARY_CONTROLS:
                    row["primary_minus_"+control] = row[PRIMARY_ARM+"_mse"]-row[control+"_mse"]
                common.append(row)
            cache_rows.extend({"episode_id": rec["id"], "kept_times": list(k), "feature_mse": v[0].tolist()}
                              for k, v in oracle.cache.items())
            if (index+1) % 16 == 0:
                print(json.dumps({"stage": "frozen_confirmation", "episodes": index+1, "total": 192}), flush=True)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["episode_id"]][row["arm"]] = row
    for arms in grouped.values():
        pafu = np.mean([arms[f"pafu_seed_{s}"]["mse"] for s in (17, 23, 41)])
        for row in arms.values():
            for control in (*PRIMARY_CONTROLS, "masked_attention_ridge", "fifo", "wr"):
                row["minus_"+control] = row["mse"]-arms[control]["mse"]
            row["minus_pafu_seed_mean"] = row["mse"]-pafu
    all_arms = list(grouped[next(iter(grouped))])
    online_fields = ["mse", "local_regret", "contrast_mse", "orthogonal_mse", "feature_branch_correct", "cue_survival"]
    online_fields += ["minus_"+c for c in (*PRIMARY_CONTROLS, "masked_attention_ridge", "fifo", "wr", "pafu_seed_mean")]
    common_arms = list(common[0]["choices"])
    common_fields = [a+"_"+f for a in common_arms for f in ("mse", "regret", "optimal")]
    common_fields += ["primary_minus_"+c for c in PRIMARY_CONTROLS]
    statistics = {"online": {a: paired_summary([r for r in rows if r["arm"] == a], online_fields, CONDITIONS) for a in all_arms},
                  "common_parent": paired_summary(common, common_fields, CONDITIONS)}
    decision = primary_decision(statistics)
    write(out / "online_results.json", rows); write(out / "common_parent_results.json", common)
    write(out / "oracle_subsets.json", cache_rows); write(out / "statistics.json", statistics)
    write(out / "primary_decision.json", decision)
    np.savez_compressed(out / "prediction_vectors.npz", predictions=np.stack(predictions), cases=np.array(names))
    verify_lock(protocol)
    if sha256(protocol_path) != protocol_sha:
        raise ValueError("Protocol changed during confirmation")
    for seed, model in models.items():
        if state_hash(model) != student_proof["model_state_sha256"][seed]:
            raise ValueError("Frozen PaFu student changed")
    summary = {"status": "completed_frozen_literature_confirmation", "elapsed_seconds": time.monotonic()-began,
               "seed": SEED, "episodes": 192, "independent_pairs": 96, "online_rows": len(rows),
               "common_parent_rows": len(common), "protocol_sha256": protocol_sha,
               "primary_decision": decision, "fitting_calls": 0, "new_GPU_seconds_for_evaluation": 0,
               "fresh_data_audit": fresh.audit(), "old_data_audit": old.audit(), "predictor_audit": scores.audit(),
               "limits": protocol["limitations"],
               "result_sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write(out / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"],
                      "primary_decision": decision}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("freeze")
    lock.add_argument("--protocol", type=Path, required=True)
    lock.add_argument("--encoder", type=Path, default=ROOT/"experiments/phase2/encode_gate_confirmation.py")
    lock.add_argument("--snapshot", type=Path, default=ROOT/"build/hpc-20261020/development-assets")
    lock.add_argument("--followup", type=Path, default=ROOT/"build/history-diagnosis-20261020/followup-blas1")
    lock.add_argument("--students", type=Path, default=ROOT/"build/literature-baselines-20261022/dev-v3")
    lock.add_argument("--parameters", type=Path, default=ROOT/"results/literature_diagnosis/learnability-20261022/probe_parameters.json")
    lock.add_argument("--prior-records", type=Path, default=ROOT/"build/literature-baselines-20261022/fresh-data/records.json")
    run = sub.add_parser("evaluate")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args) if args.command == "freeze" else evaluate(args)


if __name__ == "__main__":
    main()
