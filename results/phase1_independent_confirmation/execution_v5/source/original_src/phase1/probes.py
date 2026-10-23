import copy
from pathlib import Path

import numpy as np
import torch

from .cache import fit_normalization
from .config import SPLITS
from .data import load_episode
from .inputs import FeatureStore, selected_evidence
from .model import fit_basis, project
from .stats import clustered_interval, wilson_pair_bound
from .utils import write_npz, write_json, file_hash, code_hash, digest


class LinearReadout:
    def fit(self, x, y, ridge):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if y.ndim == 1:
            y = y[:, None]
        self.mean, self.std = x.mean(0), np.maximum(x.std(0), .05)
        x = (x-self.mean)/self.std
        self.bias = y.mean(0)
        xx, yy = torch.from_numpy(x), torch.from_numpy(y-self.bias)
        if len(x) <= x.shape[1]:
            gram = xx @ xx.T
            gram.diagonal().add_(ridge)
            w = xx.T @ torch.linalg.solve(gram, yy)
        else:
            gram = xx.T @ xx
            gram.diagonal().add_(ridge)
            w = torch.linalg.solve(gram, xx.T @ yy)
        self.weight = w.numpy()
        self.ridge = ridge
        return self

    def predict(self, x):
        return ((np.asarray(x)-self.mean)/self.std) @ self.weight + self.bias

    def save(self, path):
        write_npz(path, mean=self.mean, std=self.std, weight=self.weight, bias=self.bias, ridge=np.float64(self.ridge))

    @classmethod
    def load(cls, path):
        obj = cls()
        with np.load(path, allow_pickle=False) as f:
            for k in f.files:
                setattr(obj, k, f[k])
        return obj


def choose_readout(c, train, val, classification):
    best, best_score = None, float("inf")
    for ridge in c["probe"]["ridge"]:
        obj = LinearReadout().fit(train[0], train[1], ridge)
        pred = obj.predict(val[0])
        if classification:
            accuracy = ((pred[:, 0] > 0) == (val[1][:, 0] > 0)).mean()
            score = -accuracy + 1e-6 * np.mean((pred-val[1])**2)
        else:
            score = np.mean((pred-val[1])**2)
        if score < best_score:
            best, best_score = obj, score
    return best


def probe_inputs(store, rec, variant, basis):
    z, q = store.features(rec), rec["query"]
    z = project(z, basis)
    raw = load_episode(store.c, rec, True, keys=("evidence",))["evidence"]
    ev = selected_evidence(store.c, rec, raw)
    if variant == "recent_only":
        return z[q-store.c["data"]["recent"]+1:q+1].reshape(-1)
    if variant == "motion_two_frame":
        dt = ev[1]-ev[0]
        return np.concatenate([z[ev[1]], (z[ev[1]]-z[ev[0]])/dt, np.array([dt], np.float32)])
    return z[ev[-1] if variant == "motion_single_end" else ev[0]].copy()


def probe(c):
    from .training import experiment_tasks
    tasks = experiment_tasks(c)
    fit_normalization(c)
    store = FeatureStore(c)
    root = Path(c["output"])/"probes"
    results, checkpoint_meta = [], {}
    for task in tasks:
        basis = fit_basis(c, store, task)
        variants = ["gate_cue", "recent_only"] if task == "gate" else ["motion_two_frame", "motion_single_start", "motion_single_end", "recent_only"]
        for variant in variants:
            sets = {}
            for split in SPLITS:
                rows = [r for r in store.manifest["records"] if r["task"] == task and r["split"] == split]
                x = np.stack([probe_inputs(store, r, variant, basis) for r in rows])
                y = np.array([[2*int(load_episode(c, r, True)["label"])-1] for r in rows], np.float64)
                sets[split] = (x, y, [r["pair"] for r in rows])
            model = choose_readout(c, sets["train"], sets["validation"], True)
            path = root/f"{task}_{variant}.npz"
            model.save(path)
            checkpoint_meta[path.name] = file_hash(path)
            for split in SPLITS:
                x, y, groups = sets[split]
                correct = (model.predict(x)[:, 0] > 0) == (y[:, 0] > 0)
                ci = clustered_interval(correct, groups, c["evaluation"]["bootstrap"])
                results.append({"task": task, "variant": variant, "split": split, "accuracy": ci,
                                "chance": .5, "wilson_effective_pairs": wilson_pair_bound(ci["mean"], ci["pairs"]),
                                "ridge_selected_on": "validation", "ridge": float(model.ridge)})
        # Independent decoder, trained ONLY on real visible train target features.
        sets = {}
        for split in SPLITS:
            x, y, groups, horizons = [], [], [], []
            for rec in store.manifest["records"]:
                if rec["task"] != task or rec["split"] != split:
                    continue
                meta = load_episode(c, rec, True)
                for h in c["data"]["horizons"]:
                    ti = rec["query"]+h
                    assert meta["visible"][ti] and meta["classes"][ti] >= 0
                    x.append(store.features(rec)[ti])
                    y.append([2*meta["classes"][ti]-1, *meta["states"][ti]])
                    groups.append(rec["pair"])
                    horizons.append(h)
            sets[split] = (np.stack(x), np.asarray(y), groups, np.array(horizons))
        decoder = choose_readout(c, sets["train"], sets["validation"], False)
        path = root/f"{task}_state_decoder.npz"
        decoder.save(path)
        checkpoint_meta[path.name] = file_hash(path)
        for split in SPLITS:
            x, y, groups, hs = sets[split]
            p = decoder.predict(x)
            for h in c["data"]["horizons"]:
                keep = hs == h
                g = np.array(groups)[keep]
                acc = clustered_interval((p[keep, 0]>0) == (y[keep, 0]>0), g, c["evaluation"]["bootstrap"])
                err = clustered_interval(np.mean((p[keep, 1:]-y[keep, 1:])**2, 1), g, c["evaluation"]["bootstrap"])
                results.append({"task": task, "variant": "state_decoder_on_true_target", "split": split,
                                "horizon": h, "accuracy": acc, "position_mse": err,
                                "chance": float(max(np.mean(y[keep, 0]>0), np.mean(y[keep, 0]<=0))),
                                "chance_definition": "empirical majority visible-state class", "ridge": float(decoder.ridge)})
    output = {"mode": c["mode"], "tasks": tasks, "cache_signature": store.ix["signature"], "results": results,
              "checkpoints": checkpoint_meta, "code_hash": code_hash("probes.py"),
              "input_basis_sha256": {task: file_hash(Path(c['output'])/'predictor'/task/'basis.npz') for task in tasks},
              "normalization_sha256": file_hash(Path(c["output"])/"features/normalization.npz")}
    write_json(root/"metrics.json", output)
    return output


def load_decoder(c, task):
    from .utils import read_json
    from .cache import load_index
    root = Path(c["output"])/"probes"
    meta = read_json(root/"metrics.json")
    p = root/f"{task}_state_decoder.npz"
    if meta["cache_signature"] != load_index(c)["signature"] or meta["code_hash"] != code_hash("probes.py") or file_hash(p) != meta["checkpoints"][p.name]:
        raise ValueError("State decoder provenance mismatch")
    return LinearReadout.load(p)


def motion_precision_input(store, rec, evidence, variant, basis):
    """Compressed images plus the same legal times/actions for all three controls."""
    evidence = np.asarray(evidence, np.int64)
    t0, t1 = evidence.tolist()
    q = rec["query"]
    if not 0 <= t0 < t1 < q-store.c["data"]["recent"]+1:
        raise ValueError("Motion precision evidence crosses the legal history boundary")
    # Compute just the two legal projected image observations.  Future feature
    # values cannot affect this readout input.
    p = project(store.features(rec)[evidence], basis)
    actions = load_episode(store.c, rec, keys=("actions",))["actions"]
    prefix = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(actions[:q], axis=0)])
    context = np.concatenate([np.asarray([t0, t1, q-t0, q-t1, t1-t0], np.float32),
                              (prefix[q]-prefix[evidence]).reshape(-1)])
    if variant == "two_frame":
        visual = np.concatenate([p[1], (p[1]-p[0])/(t1-t0)])
    elif variant == "single_start":
        visual = p[0]
    elif variant == "single_end":
        visual = p[1]
    else:
        raise ValueError(f"Unknown motion precision variant {variant}")
    return np.concatenate([visual, context]).astype(np.float32)


def visible_signed_speed(c, rec, metadata=None):
    """Diagnostic target only: signed x displacement of visible pre-occlusion state."""
    if rec["task"] != "motion":
        raise ValueError("Signed-speed diagnostic is motion-only")
    meta = load_episode(c, rec, True, keys=("evidence", "visible", "states")) if metadata is None else metadata
    ev = selected_evidence({k: v for k, v in c.items() if k != "experiment"}, rec, meta["evidence"])
    start, end = ev.tolist()
    if not np.all(meta["visible"][:end+1]):
        raise ValueError("Signed-speed target must use continuously visible pre-occlusion states")
    states = np.asarray(meta["states"], np.float64)
    speed = (states[end, 0]-states[start, 0])/(end-start)
    if not np.isfinite(speed):
        raise ValueError("Nonfinite signed-speed target")
    return np.array([speed], np.float64)


def motion_precision(c, splits=None):
    """One fixed signed-speed diagnostic; no simulator target enters predictor inputs.

    Each readout fits train only; ridge and the strongest single-frame control
    are chosen on validation.  Old held-out splits are development evidence,
    never used to choose these readouts or a confirmation checkpoint.
    """
    base = copy.deepcopy(c)
    base.pop("experiment", None)
    store = FeatureStore(base)
    basis = fit_basis(base, store, "motion")
    available = {r["split"] for r in store.manifest["records"] if r["task"] == "motion"}
    if splits is None:
        splits = ["validation"]
    splits = list(splits)
    if not splits or len(set(splits)) != len(splits) or not set(splits).issubset(available-{"train"}):
        raise ValueError("Motion precision requires distinct available non-training evaluation splits")
    root = Path(c["output"])/"probes"
    policies = {"current": {}, "wide_span": {"arm": "motion_wide_span"}}
    variants = ("two_frame", "single_start", "single_end")
    results, comparisons, strongest, checkpoints, errors, groups = [], [], {}, {}, {}, {}
    spans = {}
    for policy, overlay in policies.items():
        policy_c = dict(base, experiment=overlay)
        sets = {variant: {} for variant in variants}
        for split in ["train", "validation", *[s for s in splits if s != "validation"]]:
            rows = [r for r in store.manifest["records"] if r["task"] == "motion" and r["split"] == split]
            ys, es = [], []
            for rec in rows:
                meta = load_episode(base, rec, True, keys=("evidence", "visible", "states"))
                ev = selected_evidence(policy_c, rec, meta["evidence"])
                if not np.all(meta["visible"][ev]):
                    raise ValueError("Selected motion evidence must be visible")
                ys.append(visible_signed_speed(base, rec, meta))
                es.append(ev)
            y, g = np.stack(ys), [r["pair"] for r in rows]
            spans[(policy, split)] = [int(ev[1]-ev[0]) for ev in es]
            for variant in variants:
                x = np.stack([motion_precision_input(store, rec, ev, variant, basis) for rec, ev in zip(rows, es)])
                sets[variant][split] = (x, y, g)
        models = {}
        for variant in variants:
            model = choose_readout(base, sets[variant]["train"], sets[variant]["validation"], False)
            path = root/"motion_precision"/f"{policy}_{variant}.npz"
            model.save(path)
            models[variant] = model
            checkpoints[str(path.relative_to(root))] = file_hash(path)
        single_scores = {variant: float(np.mean((models[variant].predict(sets[variant]["validation"][0])-sets[variant]["validation"][1])**2)) for variant in variants[1:]}
        strongest[policy] = min(variants[1:], key=lambda variant: (single_scores[variant], variants.index(variant)))
        for variant in variants:
            for split in splits:
                x, y, g = sets[variant][split]
                err = np.mean((models[variant].predict(x)-y)**2, axis=1)
                if not np.isfinite(err).all():
                    raise ValueError("Nonfinite signed-speed probe error")
                errors[(policy, variant, split)], groups[split] = err, g
                span = spans[(policy, split)]
                results.append({"policy": policy, "variant": variant, "split": split,
                                "signed_speed_mse": clustered_interval(err, g, base["evaluation"]["bootstrap"]),
                                "span_steps": {"min": min(span), "max": max(span), "mean": float(np.mean(span))},
                                "ridge": float(models[variant].ridge), "ridge_selected_on": "validation"})
    gates = {}
    for split in splits:
        g = groups[split]
        def compare(name, baseline, alternative):
            ci = clustered_interval(errors[baseline]-errors[alternative], g, base["evaluation"]["bootstrap"])
            passed = ci["low"] > 0
            comparisons.append({"comparison": name, "split": split, "absolute_mse_reduction": ci, "passed": passed})
            return passed
        wide_vs_current = compare("wide_two_vs_current_two", ("current", "two_frame", split), ("wide_span", "two_frame", split))
        policy_gates = {}
        for policy in policies:
            singles = {single: compare(f"{policy}_two_vs_{single}", (policy, single, split), (policy, "two_frame", split)) for single in variants[1:]}
            policy_gates[policy] = {"strongest_single_selected_on_validation": strongest[policy],
                                    "better_than_matched_strongest_single": singles[strongest[policy]],
                                    "better_than_each_single": all(singles.values())}
        gates[split] = {"wide_better_than_current": wide_vs_current, "policies": policy_gates,
                        "passed": wide_vs_current and policy_gates["wide_span"]["better_than_each_single"]}
    basis_path = Path(base["output"])/"predictor/motion/basis.npz"
    output = {"mode": c["mode"], "scope": "validation_only" if splits == ["validation"] else "declared_development_splits",
              "fit_split": "train", "selection_split": "validation", "evaluation_splits": splits,
              "target": "visible_pre_occlusion_signed_x_velocity_in_normalized_coordinates_per_frame",
              "input": "train_only_compressed_visual_features_and_matched_legal_timestamps_past_actions",
              "cache_signature": store.ix["signature"], "input_basis_sha256": file_hash(basis_path),
              "normalization_sha256": file_hash(Path(base["output"])/"features/normalization.npz"),
              "results": results, "comparisons": comparisons, "gates": gates,
              "passed": all(row["passed"] for row in gates.values()),
              "strongest_single": strongest, "checkpoints": checkpoints,
              "code_hash": code_hash("probes.py", "inputs.py"),
              "policy_signature": digest({"current": "raw_start_end", "wide_span": "zero_same_end", "splits": splits})}
    write_json(root/"motion_precision.json", output)
    return output
