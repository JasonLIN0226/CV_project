"""No-fit Gate2 deletion consensus with a fixed TRAIN-only action reference bank.

Prepare locks the bank and sanitized causal capsules. The selector sees
only those capsules plus the frozen reader/bank. Benchmark also includes an
explicitly separate actual-plan loading-parity diagnostic. The evaluator opens
archived actual-plan outcomes only after every deployment choice is locked.
This is post-hoc development diagnosis, not a new independent confirmation.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/predictor/control/src"))
COUNT = 16
SEED = 20261025
TOL = 1e-6
FIELDS = ("z", "age", "past_actions", "current")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def capsule_digest(capsule):
    h = hashlib.sha256()
    for key in FIELDS:
        value = np.ascontiguousarray(capsule[key])
        h.update(key.encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def causal_capsule(features, indices, clock, executed_actions):
    """No future features/actions, IDs, labels, roles or query schedule argument.

    `clock` is the current stream timestamp. `indices` contains exactly the R4
    recent timestamps then the real K+1 candidate timestamps, all <= clock.
    This function rejects a future action array rather than silently slicing it.
    """
    indices = np.asarray(indices, np.int64)
    if indices.shape != (7,) or np.any(indices > clock) or len(set(indices)) != 7:
        raise ValueError("Require seven distinct already-observed timestamps")
    if executed_actions.shape != (clock, 2):
        raise ValueError("Capsule accepts executed actions only, shape[current_time,2]")
    if features.shape[0] != clock+1:
        raise ValueError("Capsule accepts observed feature prefix only")
    prefix = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(executed_actions, axis=0)])
    result = {"z": features[indices].copy(), "age": (clock-indices).astype(np.float32),
              "past_actions": (prefix[clock]-prefix[indices]).astype(np.float32), "current": features[clock].copy()}
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError("Nonfinite causal input")
    return result


def agreement(predictions):
    """Distance before expectation: [draw, deletion, horizon, feature]."""
    p = np.asarray(predictions, np.float64)
    if p.ndim != 4 or p.shape[0] != COUNT or p.shape[1:3] != (3, 2):
        raise ValueError("Invalid reference prediction shape")
    per_draw = np.zeros((COUNT, 3, 3), np.float64)
    for a in range(3):
        for b in range(a+1, 3):
            per_draw[:, a, b] = per_draw[:, b, a] = ((p[:, a]-p[:, b])**2).mean((1, 2))
    distances = per_draw.mean(0)
    scores = distances.sum(1)
    optimal = np.flatnonzero(scores <= scores.min()+TOL).tolist()
    return {"drop": int(scores.argmin()), "medoid_tolerance_set": optimal,
            "medoid_exact_tie_set": np.flatnonzero(scores == scores.min()).tolist(),
            "tolerance_tiebreak_would_change": int(scores.argmin()) != optimal[0], "sum_distances": scores.tolist(),
            "mean_pairwise_distances": distances.tolist(), "per_draw_pairwise_distances": per_draw.tolist()}


def self_test():
    p = np.zeros((COUNT, 3, 2, 1), np.float32)
    p[:, 1] = 1; p[:, 2] = 10
    assert agreement(p)["drop"] == 1
    assert agreement(p[:, [2, 0, 1]])["drop"] == 2
    p[:, 0] = np.array([1., -1.]*(COUNT//2))[:, None, None]
    p[:, 1] = 0
    a = agreement(p)
    assert a["mean_pairwise_distances"][0][1] == 1.
    assert float(((p[:, 0].mean(0)-p[:, 1].mean(0))**2).mean()) == 0.
    assert np.allclose(a["mean_pairwise_distances"], agreement(p[::-1])["mean_pairwise_distances"], atol=0, rtol=0)
    try:
        causal_capsule(np.zeros((8, 2), np.float32), [4, 5, 6, 7, 1, 2, 3], 7, np.zeros((8, 2), np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("Future action rejection failed")
    return {"status": "passed", "checks": ["medoid", "permutation", "anti_cancellation", "action_draw_order", "reject_future_action_field"]}


def prepare(args):
    run, out, artifacts = args.run.resolve(), args.output.resolve(), args.artifacts.resolve()
    for path in (out, artifacts):
        if path.exists() and any(path.iterdir()):
            raise ValueError("Require new empty directories")
        path.mkdir(parents=True, exist_ok=True)
    manifest, protocol, locked = (read(run/name) for name in ("manifest.json", "protocol.json", "reader_locked.json"))
    needed = ("reader_selected.pt", "basis.npz", "basis_provenance.json", "data/normalization.npz",
              "data/unique_features.npz", "data/actions.npz", "data/records.json", "data/encoding.json",
              "evaluation_predictions.npz", "evaluation_rows.json", "terminal_decisions.json")
    for name in needed:
        if sha(run/name) != manifest["files"][name]:
            raise ValueError("Source archive hash mismatch: "+name)
    if sha(run/"reader_selected.pt") != locked["selected_sha256"]:
        raise ValueError("Selected reader binding mismatch")
    model_source = ROOT / "experiments/predictor/control/src/phase1/model.py"
    if sha(model_source) != protocol["source_sha256"][str(model_source.relative_to(ROOT))]:
        raise ValueError("Frozen reader source differs")
    records = read(run/"data/records.json")
    train = defaultdict(list)
    for record in records:
        if record["split"] == "train":
            train[record["family"]].append(record)
    if len(train) != 120:
        raise ValueError("Expected120 TRAIN families")
    actions = np.load(run/"data/actions.npz", allow_pickle=False)
    selected = np.random.default_rng(SEED).choice(len(train), COUNT, replace=False)
    bank, bank_sources = [], []
    keys = sorted(train)
    for index in selected:
        family = keys[int(index)]
        rr = sorted(train[family], key=lambda r: r["id"])
        if len(rr) != 8:
            raise ValueError("Expected8 TRAIN episodes per family")
        plans = [actions[r["id"]][r["query"]:r["query"]+6].copy() for r in rr]
        if not all(np.array_equal(p, plans[0]) for p in plans):
            raise ValueError("TRAIN family plans must match across siblings/styles")
        bank.append(plans[0]); bank_sources.append({"train_family": family, "train_episode_id": rr[0]["id"], "origin": rr[0]["query"]})
    bank = np.stack(bank).astype(np.float32)
    if bank.shape != (COUNT, 6, 2):
        raise ValueError("Reference bank shape")
    np.save(out/"reference_action_bank.npy", bank, allow_pickle=False)
    # Bank file is locked before any DEV capsule construction or scoring.
    write(out/"reference_action_bank.json", {"seed": SEED, "plans": COUNT, "source_split": "train",
          "selection": "sorted120 TRAIN family IDs, default_rng(seed).choice(120,16,replace=False)",
          "same_for_every_parent": True, "persistent_bytes": bank.nbytes, "sources": bank_sources,
          "bank_sha256": sha(out/"reference_action_bank.npy")})
    with np.load(run/"data/unique_features.npz", allow_pickle=False) as f:
        features = f["normalized"]
    capsules, identities, mutation_checks = [], [], []
    for r in records:
        if r["split"] != "development":
            continue
        # Adapter uses only the current terminal timestamp; no semantic cue annotation.
        clock = r["query"]
        prefix_features = features[r["rgb_indices"][:clock+1]]
        indices = list(range(clock-3, clock+1))+list(range(clock-6, clock-3))
        ac = actions[r["id"]]
        capsule = causal_capsule(prefix_features, indices, clock, ac[:clock])
        # Prove future mutations cannot change any tensor reaching the selector.
        zero, poison = ac.copy(), ac.copy()
        zero[clock:] = 0
        poison[clock:] = np.float32(12345.)
        mutated = [causal_capsule(prefix_features, indices, clock, x[:clock]) for x in (zero, poison)]
        same = all(capsule_digest(capsule) == capsule_digest(m) for m in mutated)
        if not same:
            raise ValueError("Unexecuted-action mutation affected selector input")
        capsules.append(capsule)
        identities.append({"id": r["id"], "current_time": clock, "candidate_times": list(range(clock-6, clock-3)),
                           "causal_input_sha256": capsule_digest(capsule)})
        mutation_checks.append({"id": r["id"], "future_zero_and_poison_capsules_identical": same})
    actions.close()
    arrays = {k: np.stack([c[k] for c in capsules]) for k in FIELDS}
    np.savez_compressed(artifacts/"causal_inputs.npz", **arrays)
    write(out/"parent_ids.json", identities)
    write(out/"input_boundary_checks.json", {"parents": len(capsules), "selector_fields": FIELDS,
          "selector_receives_actual_future_actions": False, "selector_receives_semantic_metadata": False,
          "selector_receives_future_targets_or_prototypes": False, "rows": mutation_checks})
    spec = {"name": "gate2-causal-consensus-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "post-hoc development no-fit diagnostic on already inspected DEV; not independent confirmation",
            "run": str(run), "artifacts": str(artifacts), "reference_count": COUNT, "seed": SEED,
            "horizons": [3, 6], "reported_set_tolerance": TOL,
            "selection_rule": "exact float64 argmin, earliest index for exact ties;1e-6 set is report-only",
            "forecast_rule": "E_a[MSE(pred_d(a),pred_e(a))], then medoid argmin_d sum_e; never MSE of mean predictions",
            "observational_input_boundary": FIELDS, "actual_future_plan_in_selector": False,
            "reference_bank_persistent_bytes": bank.nbytes, "fits": 0, "encoder_calls": 0,
            "models": protocol["model"], "training_steps_added": 0,
            "expected_forecast_examples_per_parent": COUNT*3,
            "source_sha256": {"runner": sha(__file__), "model": sha(model_source)},
            "archive_hashes": {name: sha(run/name) for name in needed},
            "bank_sha256": sha(out/"reference_action_bank.npy"),
            "causal_inputs_sha256": sha(artifacts/"causal_inputs.npz"),
            "parent_ids_sha256": sha(out/"parent_ids.json"), "synthetic_checks": self_test()}
    write(out/"protocol.json", spec)
    print(json.dumps({"stage": "prepared", "parents": len(capsules), "bank_bytes": bank.nbytes, "output": str(out)}), flush=True)


def load_locked(out):
    spec = read(out/"protocol.json")
    if sha(__file__) != spec["source_sha256"]["runner"]:
        raise ValueError("Runner changed after protocol lock")
    if sha(out/"reference_action_bank.npy") != spec["bank_sha256"] or sha(out/"parent_ids.json") != spec["parent_ids_sha256"]:
        raise ValueError("Locked bank/identity changed")
    artifacts = Path(spec["artifacts"])
    if sha(artifacts/"causal_inputs.npz") != spec["causal_inputs_sha256"]:
        raise ValueError("Causal capsule artifact changed")
    with np.load(artifacts/"causal_inputs.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in FIELDS}
    capsules = [{k: arrays[k][i] for k in FIELDS} for i in range(len(arrays["current"]))]
    ids = read(out/"parent_ids.json")
    for capsule, r in zip(capsules, ids):
        if capsule_digest(capsule) != r["causal_input_sha256"]:
            raise ValueError("Input capsule binding mismatch")
    return spec, capsules, ids


class Selector:
    """The deployed-facing score() method accepts only a causal capsule."""
    def __init__(self, spec, out):
        import torch
        from phase1.model import DynamicsPredictor
        self.torch = torch
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        run = Path(spec["run"])
        if sha(ROOT/"experiments/predictor/control/src/phase1/model.py") != spec["source_sha256"]["model"]:
            raise ValueError("Frozen reader source changed")
        for name in ("basis.npz", "reader_selected.pt", "data/normalization.npz"):
            if sha(run/name) != spec["archive_hashes"][name]:
                raise ValueError("Frozen reader/transform changed")
        with np.load(run/"basis.npz", allow_pickle=False) as f:
            self.basis = {k: f[k].copy() for k in f.files}
        with np.load(run/"data/normalization.npz", allow_pickle=False) as f:
            self.normalization = {k: f[k].copy() for k in ("mean", "std")}
        state = torch.load(run/"reader_selected.pt", map_location="cpu", weights_only=True)
        if state["basis_sha256"] != spec["archive_hashes"]["basis.npz"]:
            raise ValueError("Checkpoint basis binding mismatch")
        self.model = DynamicsPredictor(len(self.basis["input_mean"]), spec["models"], (3, 6), self.basis).cpu()
        self.model.load_state_dict(state["model"])
        self.model.eval().requires_grad_(False)
        self.bank = np.load(out/"reference_action_bank.npy", allow_pickle=False)
        self.bank.setflags(write=False)
        self.forward_examples = 0
        self.workspace = {}

    def _forecast(self, capsule, plans):
        """Internal kernel also used for explicitly separate parity diagnostics."""
        from phase1.inputs import collate
        if set(capsule) != set(FIELDS):
            raise ValueError("Selector input whitelist violation")
        z = capsule["z"]
        projected = ((z-self.basis["input_mean"])@self.basis["projection"]).astype(np.float32)
        current = ((capsule["current"]-self.basis["input_mean"])@self.basis["projection"]).astype(np.float32)
        samples = []
        for plan in plans:
            for drop in range(3):
                keep = list(range(4))+[4+j for j in range(3) if j != drop]
                samples.append({"z": z[keep], "age": capsule["age"][keep],
                                "past_actions": capsule["past_actions"][keep], "current": capsule["current"],
                                "type": np.asarray([0]*4+[1]*2, np.int64), "actions": plan,
                                "projected_z": projected[keep], "projected_current": current})
        batch = collate(samples, "cpu")
        with self.torch.inference_mode():
            predictions = self.model(batch).numpy().reshape(len(plans), 3, 2, -1)
        if not np.isfinite(predictions).all():
            raise ValueError("Nonfinite CPU reader prediction")
        self.forward_examples += len(samples)
        self.workspace = {"collated_input_tensor_bytes": sum(v.nelement()*v.element_size() for v in batch.values()),
                          "forecast_output_bytes": predictions.nbytes,
                          "causal_capsule_bytes": sum(v.nbytes for v in capsule.values()),
                          "hidden_BLAS_and_allocator_workspace_included": False}
        return predictions

    def score(self, capsule):
        predictions = self._forecast(capsule, self.bank)
        return agreement(predictions), predictions


def benchmark(out):
    spec, capsules, ids = load_locked(out)
    selector = Selector(spec, out)
    started = time.perf_counter()
    first, predictions = selector.score(capsules[0])
    latency = time.perf_counter()-started
    # Prepared zero/poison futures produce bitwise identical capsules for all192.
    # Repeat one identical capsule through the score boundary to validate output determinism.
    repeat, repeated = selector.score({k: v.copy() for k, v in capsules[0].items()})
    if first["drop"] != repeat["drop"] or not np.array_equal(predictions, repeated):
        raise ValueError("Identical causal inputs changed selection")
    run = Path(spec["run"])
    # Separate noncausal replay parity; its outputs are never used for selection.
    with np.load(run/"data/actions.npz", allow_pickle=False) as f:
        clock = ids[0]["current_time"]
        actual_plan = f[ids[0]["id"]][clock:clock+6]
    actual_predictions = selector._forecast(capsules[0], actual_plan[None])[0]
    rows = read(run/"evaluation_rows.json")
    positions = [next(i for i, r in enumerate(rows) if r["id"] == ids[0]["id"] and r["arm"] == "native_drop_"+str(d)) for d in range(3)]
    with np.load(run/"evaluation_predictions.npz", allow_pickle=False) as f:
        archived = f["predictions"][positions]
        truth = f["quartet_targets"][rows[positions[0]]["episode_index"]]
    bits = tuple(rows[positions[0]]["bits"])
    target = truth[((0, 0), (0, 1), (1, 0), (1, 1)).index(bits)]
    cpu_loss = ((actual_predictions.astype(np.float64)-target)**2).mean((1, 2))
    old_loss = np.array([rows[i]["mse"] for i in positions])
    result = {"stage": "benchmark", "parent_id": ids[0]["id"], "selector_seconds_first_parent": latency,
              "linear_192_parent_estimate_seconds": latency*len(capsules),
              "reference_forecast_examples_per_parent": COUNT*3, "reference_forward_example_total": COUNT*3*2,
              "separate_actual_plan_parity_examples": 3, "identical_capsule_predictions_bitwise_equal": True,
              "selected_drop": first["drop"], "medoid_tolerance_set": first["medoid_tolerance_set"],
              "actual_plan_replay_is_not_selector_input": True,
              "actual_plan_parity": {"maximum_feature_difference": float(np.max(abs(actual_predictions-archived))),
                                     "maximum_loss_difference": float(np.max(abs(cpu_loss-old_loss))),
                                     "original_loss": old_loss.tolist(), "cpu_loss": cpu_loss.tolist(),
                                     "argmin_unchanged": int(cpu_loss.argmin()) == int(old_loss.argmin()),
                                     "full_order_unchanged": bool(np.array_equal(np.argsort(cpu_loss), np.argsort(old_loss)))},
              "reference_bank_persistent_bytes": selector.bank.nbytes,
              "workspace_for_reference_batch": {"forecast_output_bytes": predictions.nbytes,
                     "pair_distance_float64_copy_bytes": predictions.size*8,
                     "note": "Tensor lower bound; process RSS and collated inputs recorded in full run"}}
    write(out/"benchmark.json", result)
    print(json.dumps(result), flush=True)


def score_all(out):
    spec, capsules, ids = load_locked(out)
    if not (out/"benchmark.json").exists():
        raise ValueError("Run benchmark and inspect its estimate before all-parent scoring")
    selector = Selector(spec, out)
    artifact = Path(spec["artifacts"])/"reference_predictions.npy"
    if artifact.exists() or (out/"choices_locked.json").exists():
        raise ValueError("Do not overwrite a completed/partial scored artifact")
    saved = np.lib.format.open_memmap(artifact, mode="w+", dtype=np.float32,
                                     shape=(len(capsules), COUNT, 3, 2, len(capsules[0]["current"])))
    rows, times = [], []
    for i, (capsule, identity) in enumerate(zip(capsules, ids)):
        start = time.perf_counter()
        decision, forecasts = selector.score(capsule)
        times.append(time.perf_counter()-start)
        saved[i] = forecasts
        rows.append({**identity, "prediction_index": i, **decision, "seconds": times[-1]})
        if (i+1) % 32 == 0:
            print(json.dumps({"stage": "score", "parents": i+1, "total": len(capsules)}), flush=True)
    saved.flush(); del saved
    write(out/"choices.json", rows)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak*1024
    lock = {"status": "choices_frozen_before_evaluation", "parents": len(rows),
            "choices_sha256": sha(out/"choices.json"), "reference_predictions_sha256": sha(artifact),
            "reference_predictions_bytes": artifact.stat().st_size,
            "reference_bank_persistent_bytes": selector.bank.nbytes,
            "reader_parameter_bytes": sum(p.nelement()*p.element_size() for p in selector.model.parameters()),
            "reader_buffer_bytes": sum(p.nelement()*p.element_size() for p in selector.model.buffers()),
            "forecast_examples": selector.forward_examples, "forecasts_per_parent": COUNT*3,
            "inference_seconds_total": sum(times), "inference_seconds_median": float(np.median(times)),
            "inference_seconds_p95": float(np.quantile(times, .95)), "workspace": selector.workspace,
            "float64_pair_distance_copy_bytes": COUNT*3*2*len(capsules[0]["current"])*8,
            "process_peak_RSS_bytes": peak_bytes, "RSS_is_not_persistent_policy_memory": True,
            "platform": platform.platform(), "torch_version": selector.torch.__version__,
            "numpy_version": np.__version__, "device": "cpu", "fits": 0}
    write(out/"choices_locked.json", lock)
    print(json.dumps({"stage": "choices_locked", "parents": len(rows), "inference_seconds": sum(times)}), flush=True)


def evaluate(out):
    spec = read(out/"protocol.json"); lock = read(out/"choices_locked.json")
    if sha(out/"choices.json") != lock["choices_sha256"]:
        raise ValueError("Locked choices changed")
    chosen = read(out/"choices.json")
    run = Path(spec["run"])
    # These outcome/semantic files are opened only after the choices lock.
    native = {r["id"]: r for r in read(run/"terminal_decisions.json")}
    archived = {(r["id"], r["arm"]): r for r in read(run/"evaluation_rows.json")}
    evaluated = []
    for row in chosen:
        n = native[row["id"]]; drop = row["drop"]
        result = archived[(row["id"], "native_drop_"+str(drop))]
        evaluated.append({**row, "family": n["family"], "quartet": n["quartet"], "condition": n["condition"],
                          "mse": n["losses"][drop], "oracle_mse": min(n["losses"]),
                          "regret": n["losses"][drop]-min(n["losses"]),
                          "semantic_correct": drop in n["semantic_correct_delete_set"],
                          "feature_class_correct": result["feature_class_correct"],
                          "baseline": n["policies"]})
    summary = {}
    for condition in ("all", "benign_duplicate", "distant_style_duplicate"):
        rr = [r for r in evaluated if condition == "all" or r["condition"] == condition]
        summary[condition] = {"episodes": len(rr), "families": len({r["family"] for r in rr}),
                              "mse": float(np.mean([r["mse"] for r in rr])),
                              "mean_regret": float(np.mean([r["regret"] for r in rr])),
                              "positive_regret_fraction": float(np.mean([r["regret"] > TOL for r in rr])),
                              "semantic_correct_count": sum(r["semantic_correct"] for r in rr),
                              "feature_class_correct_count": sum(r["feature_class_correct"] for r in rr),
                              "medoid_exact_tie_count": sum(len(r["medoid_exact_tie_set"]) > 1 for r in rr),
                              "medoid_tolerance_set_ambiguous_count": sum(len(r["medoid_tolerance_set"]) > 1 for r in rr),
                              "tolerance_tiebreak_would_change_count": sum(r["tolerance_tiebreak_would_change"] for r in rr),
                              "minus_baseline_mse": {p: float(np.mean([r["mse"]-r["baseline"][p]["mse"] for r in rr]))
                                                     for p in ("fifo", "terminal_uniform", "visual_diversity", "reader_attention")}}
    write(out/"evaluated_rows.json", evaluated)
    write(out/"summary.json", {"status": "completed", "scope": spec["scope"], "causal_selector_actual_plan_access": False,
          "actual_plan_evaluation": "Reuses saved native predictions/losses from the same frozen A100 reader; actual plan is evaluator-only",
          "summary": summary, "reference_forecast_examples": lock["forecast_examples"], "training_steps": 0,
          "limitations": ["Reference plans are a TRAIN-only prior, not actual unexecuted DEV actions.",
                         "These DEV outcomes were inspected before this heuristic was proposed; no independent confirmation.",
                         "A medoid of three deletion forecasts is not a deletion-cost oracle or causal proof.",
                         "768 additional persistent bank bytes and48 reader forecast examples per decision must be counted.",
                         "This terminal query diagnostic does not repair the early-reader problem."]})
    print(json.dumps({"stage": "evaluated", "summary": summary}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "benchmark", "score", "evaluate", "self-test"))
    p.add_argument("--run", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--artifacts", type=Path)
    a = p.parse_args()
    if a.stage == "self-test":
        print(json.dumps(self_test()))
    elif a.stage == "prepare":
        if None in (a.run, a.output, a.artifacts):
            p.error("prepare requires --run --output --artifacts")
        prepare(a)
    elif a.output is None:
        p.error("--output required")
    elif a.stage == "benchmark":
        benchmark(a.output.resolve())
    elif a.stage == "score":
        score_all(a.output.resolve())
    else:
        evaluate(a.output.resolve())


if __name__ == "__main__":
    main()
