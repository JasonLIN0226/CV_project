"""Prespecified frozen motion mechanism tests, never deployable/Phase 1 results.

Primary: reference checkpoint, original start/end sampling.  The wide/native
arm is a replication only and cannot rescue a failed primary mechanism gate.
No original source or run is written, and no basis/decoder/model is fitted.
"""
import argparse
import copy
import hashlib
import importlib.util
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from .config import CONDITIONS, SPLITS, scientific_hash
from .data import load_episode
from .diagnostics import (ARMS, POLICIES, _hash_sources, _motion_pairs, _source_config,
    _verify_source_proof, _assert_wrong_selected, assert_exact, assert_sample_exact,
    verify_native_metrics)
from .evaluation import pair_diagnostic, summarize
from .inputs import FeatureStore, collate
from .probes import load_decoder
from .stats import clustered_interval, paired_improvement
from .training import TRAIN_CONDITIONS, load_predictor
from .utils import code_hash, digest, file_hash, read_json, write_csv, write_json
from . import diagnostics as diagnostic_helpers, utils as scientific_utils


VARIANTS = ("native", "pure_p_velocity")
REGISTERED_SCIENTIFIC_HASHES = {
    "reference": "1b9085e5e4d475d0ae7097563f565a5cef109162667f362fbcbc077d67dd08c8",
    "motion_wide_span": "d17e95f3243933be08f7782b62562f70dcef7d210f1769222ffc62a3b2491154"}
REGISTERED_CODE_HASHES = {
    "checkpoint": "6a190d8f29ebc82b16ec272335010da2379814bfdfdc7c44bbe6de5422eb2d0f",
    "evaluation": "e29f058b8219a245332353929166cfb3b96bffdb22d0d678d7208feed7bd1870"}


def registered_source_config(c, arm):
    if scientific_hash(c) != REGISTERED_SCIENTIFIC_HASHES[arm] or c["threads"] != 4:
        raise ValueError("Original motion scientific config/coefficients differ from preregistration")


def _scientific_sources(directory):
    return {p.name: file_hash(p) for p in sorted(Path(directory).glob("*.py"))
            if p.name not in {"diagnostics.py", "motion_mechanism.py", "gate_diagnostics.py"}}


def loaded_code_proof(directory):
    """Hash actual loaded files independently; never copy expected proof fields."""
    source = _scientific_sources(directory)
    groups = {"checkpoint": ("training.py", "model.py", "inputs.py"),
              "evaluation": ("evaluation.py", "stats.py")}
    actual = {kind: digest({name: source[name] for name in names}) for kind, names in groups.items()}
    for kind, names in groups.items():
        if actual[kind] != code_hash(*names) or actual[kind] != REGISTERED_CODE_HASHES[kind]:
            raise ValueError(f"Actual loaded {kind} scientific source differs from original v3; use its frozen source directory")
    return {"actual_code_hashes": actual, "file_sha256": {name: source[name] for names in groups.values() for name in names},
        "matches_registered_original_v3": True,
        "checkpoint_comparison": "training_completion/load_predictor independently compare these runtime signatures with every actual original best/latest checkpoint"}


@contextmanager
def frozen_runtime(frozen_source_directory):
    """Execute archived science, not current science with substituted hashes.

    A private package namespace preserves the original relative imports and
    code_hash roots. Only these two diagnostic modules come from the current
    checkout. No source copying, scientific source mutation or refitting occurs.
    """
    root = Path(frozen_source_directory).expanduser().resolve()
    candidates = [root/"src/phase1", root/"phase1", root]
    required = ("__init__.py", "utils.py", "config.py", "training.py", "model.py",
                "inputs.py", "evaluation.py", "stats.py", "data.py", "cache.py", "probes.py")
    package = next((p for p in candidates if all((p/name).is_file() for name in required)), None)
    if package is None:
        raise ValueError("Frozen source directory must contain the complete original phase1 package")
    before = _scientific_sources(package)
    prefix = "_phase1_mechanism_frozen_"+uuid4().hex
    saved_rng, write_bytecode = _rng_state(), sys.dont_write_bytecode
    sys.dont_write_bytecode = True

    def load(name, path, package_path=None):
        spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=package_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    try:
        load(prefix, package/"__init__.py", [str(package)])
        load(prefix+".diagnostics", Path(diagnostic_helpers.__file__).resolve())
        module = load(prefix+".motion_mechanism", Path(__file__).resolve())
        if Path(module.scientific_utils.__file__).resolve().parent != package:
            raise ValueError("Archived scientific package was not actually loaded")
        yield module
    finally:
        sys.dont_write_bytecode = write_bytecode
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix+"."):
                del sys.modules[name]
        _restore_rng(saved_rng)
        if before != _scientific_sources(package):
            raise ValueError("Archived scientific source changed during mechanism diagnostic")


def verify_basis_provenance(c, store, model=None):
    """Verify explicit train membership, persisted arrays and loaded buffers."""
    path = Path(c["output"])/"predictor/motion/basis.npz"
    meta = read_json(path.with_suffix(".json"))
    if meta.get("fit_split") != "train" or meta.get("task") != "motion" or meta.get("sha256") != file_hash(path):
        raise ValueError("Motion basis must be unchanged and fitted on train only")
    with np.load(path, allow_pickle=False) as f:
        basis = {key: f[key] for key in f.files}
    expected = [r["id"] for r in store.manifest["records"] if r["task"] == "motion" and r["split"] == "train"]
    actual = basis["train_ids"].tolist()
    if (actual != expected or len(actual) != 2*c["data"]["pairs"]["train"] or
            len(actual) != len(set(actual)) or meta.get("episodes") != len(actual)):
        raise ValueError("Motion basis training IDs differ from the exact original motion/train membership")
    if (meta.get("input_rank") != basis["projection"].shape[1] or
            meta.get("output_rank") != basis["output_basis"].shape[1]):
        raise ValueError("Persisted basis ranks differ from metadata")
    for key in ("input_mean", "projection", "target_mean", "output_basis"):
        if not np.isfinite(basis[key]).all():
            raise ValueError("Persisted basis contains nonfinite values")
        if model is not None:
            assert_exact(getattr(model, key).detach().cpu().numpy(), basis[key], f"checkpoint basis buffer/{key}")
    return {"fit_split": "train", "task": "motion", "episodes": len(actual),
        "train_ids_sha256": digest(actual), "npz_sha256": file_hash(path), "metadata_sha256": file_hash(path.with_suffix(".json")),
        "input_rank": meta["input_rank"], "output_rank": meta["output_rank"],
        "checkpoint_buffers_equal_persisted_basis": model is not None}


def _rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _rng_hash(state=None):
    py, np_state, cpu = _rng_state() if state is None else state
    h = hashlib.sha256(repr(py).encode())
    h.update(repr((np_state[0], *np_state[2:])).encode())
    h.update(np_state[1].tobytes())
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def _restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def model_state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().numpy()
        h.update(repr((name, array.shape, str(array.dtype))).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def velocity_forward(model, batch, pure_p_velocity=False):
    """Exact native arithmetic plus ONE intervention; no target/state argument.

    In the intervention only the dependent variable of the weighted velocity
    regression becomes p.  Weights, variance floor, aligned memory, past_mean,
    parameters and actions are unchanged.  Derived transported/interaction
    features consistently use that one changed velocity.
    """
    valid = batch['mask']
    z = batch['z'].masked_fill(~valid[..., None], 0)
    p = batch['projected_z'].masked_fill(~valid[..., None], 0) if 'projected_z' in batch else (z-model.input_mean)@model.projection
    current = batch['projected_current'] if 'projected_current' in batch else (batch['current']-model.input_mean)@model.projection
    history = valid & (batch['type'] == 1)
    score = model.score(torch.cat([p, p-current[:, None]], -1)).squeeze(-1)
    score = score+torch.log1p((p-current[:, None]).square().mean(-1))
    w = torch.softmax(score.masked_fill(~history, -1e4), 1)*history
    w = w/w.sum(1, keepdim=True).clamp_min(1e-8)
    past = model.past_action(batch['past_actions'].masked_fill(~valid[..., None], 0))
    aligned = p+past
    age = batch['age'].masked_fill(~valid, 0)/8
    mean_age = (w*age).sum(1)
    center = age-mean_age[:, None]
    memory = (w[..., None]*aligned).sum(1)
    past_mean = (w[..., None]*past).sum(1)
    variance = (w*center.square()).sum(1)
    native = -(w[..., None]*center[..., None]*(aligned-memory[:, None])).sum(1)/variance.clamp_min(.01)[:, None]
    p_mean = (w[..., None]*p).sum(1)
    pure = -(w[..., None]*center[..., None]*(p-p_mean[:, None])).sum(1)/variance.clamp_min(.01)[:, None]
    action = -(w[..., None]*center[..., None]*(past-past_mean[:, None])).sum(1)/variance.clamp_min(.01)[:, None]
    velocity = pure if pure_p_velocity else native
    outputs = []
    for h in model.horizons:
        actions = batch['actions'][:, :h]
        action_features = model.action(torch.cat([actions[:, 0], actions.sum(1), actions[:, -1]], -1))
        time = mean_age+h/8
        transported = memory+velocity*time[:, None]
        stats = torch.stack([mean_age, history.any(1).float(), torch.full_like(mean_age, h/8)], -1)
        features = torch.cat([current, memory, velocity, transported, action_features,
            memory*action_features, past_mean, velocity*action_features, stats], -1)
        coeff = model.output(features)
        outputs.append(batch['current']+model.target_mean+coeff@model.output_basis.T)
    trace = {"history": history, "weights": w, "p": p, "past": past, "age": age,
        "variance": variance, "kappa": variance/variance.clamp_min(.01), "memory": memory,
        "past_mean": past_mean, "native_velocity": native, "pure_velocity": pure,
        "action_velocity": action}
    return torch.stack(outputs, 1), trace


def frozen_forward_pair(model, clone, batch):
    """Fail closed unless clone/native math reproduce the actual original model."""
    before = _rng_hash()
    with torch.inference_mode():
        original = model(batch)
        native, trace = velocity_forward(clone, batch)
        pure, pure_trace = velocity_forward(clone, batch, pure_p_velocity=True)
    assert_exact(original.cpu().numpy(), native.cpu().numpy(), "native mechanism clone forward")
    if not torch.isfinite(pure).all():
        raise ValueError("Nonfinite pure-p mechanism prediction")
    for key in ("weights", "variance", "kappa", "memory", "past_mean"):
        assert_exact(trace[key].cpu().numpy(), pure_trace[key].cpu().numpy(), f"unchanged mechanism {key}")
    if before != _rng_hash():
        raise ValueError("Frozen mechanism forward changed an RNG state")
    return {"native": native.cpu().numpy(), "pure_p_velocity": pure.cpu().numpy()}, trace


def velocity_scalars(trace, index, require_two=False):
    """Compact legal-input-only decomposition; true physical speed is not used."""
    arrays = {k: v[index].detach().cpu().numpy().astype(np.float64) for k, v in trace.items()}
    history = arrays["history"].astype(bool)
    native, pure, action = (arrays[k] for k in ("native_velocity", "pure_velocity", "action_velocity"))
    if not np.allclose(native, pure+action, atol=2e-5, rtol=2e-4):
        raise ValueError("Actual velocity does not reproduce its visual/action decomposition")
    result = {"history_count": int(history.sum()), "variance": float(arrays["variance"]),
        "kappa": float(arrays["kappa"]), "variance_floor_active": bool(arrays["variance"] < .01),
        "native_velocity_mean_square": float(np.mean(native**2)),
        "pure_velocity_mean_square": float(np.mean(pure**2)),
        "action_velocity_mean_square": float(np.mean(action**2)),
        "decomposition_max_abs_error": float(np.max(np.abs(native-pure-action))),
        "span_steps": None, "raw_visual_secant_mean_square": None,
        "raw_action_secant_mean_square": None, "action_to_visual_energy_ratio": None,
        "visual_secant_degenerate": None, "two_frame_formula_max_abs_error": None}
    if require_two:
        positions = np.flatnonzero(history)
        if len(positions) != 2:
            raise ValueError("Privileged/wrong motion mechanism requires exactly two historical frames")
        first, last = positions
        dt = 8*(arrays["age"][first]-arrays["age"][last])
        if dt <= 0:
            raise ValueError("Motion mechanism frame order must have a positive span")
        visual = 8*(arrays["p"][last]-arrays["p"][first])/dt
        # past=end-to-query controls; their difference is minus start-to-end.
        contamination = 8*(arrays["past"][last]-arrays["past"][first])/dt
        expected = float(arrays["kappa"])*(visual+contamination)
        if not np.allclose(native, expected, atol=2e-5, rtol=2e-4):
            raise ValueError("Actual B=2 velocity does not reproduce kappa times visual/action secants")
        energy = float(np.mean(visual**2))
        contaminated_energy = float(np.mean(contamination**2))
        result.update(span_steps=float(dt), raw_visual_secant_mean_square=energy,
            raw_action_secant_mean_square=contaminated_energy,
            action_to_visual_energy_ratio=contaminated_energy/energy if energy > 1e-12 else None,
            visual_secant_degenerate=energy <= 1e-12,
            two_frame_formula_max_abs_error=float(np.max(np.abs(native-expected))))
    return result


class BasisOracle:
    """Target-aware lower bound, never a predictor or a reported model output."""
    def __init__(self, output_basis, target_mean):
        self.basis = np.asarray(output_basis, np.float64)
        self.mean = np.asarray(target_mean, np.float64)
        if (self.basis.ndim != 2 or self.mean.shape != (self.basis.shape[0],) or
                not np.isfinite(self.basis).all() or not np.isfinite(self.mean).all()):
            raise ValueError("Invalid fixed output basis/mean for oracle")
        self.pinv = np.linalg.pinv(self.basis, rcond=1e-12)
        self.rank = int(np.linalg.matrix_rank(self.basis, tol=np.linalg.norm(self.basis, 2)*1e-12))

    def error(self, current, target):
        residual = np.asarray(target, np.float64)-np.asarray(current, np.float64)-self.mean
        remaining = residual-(residual@self.pinv.T)@self.basis.T
        if not np.isfinite(remaining).all():
            raise ValueError("Nonfinite basis-oracle residual")
        return np.mean(remaining**2, axis=-1)


def _metric_rows(c, store, model, decoder, part, samples, predictions, seed, split, condition, variant, policy):
    rows = []
    for rec, sample, prediction in zip(part, samples, predictions):
        _, target, info = sample
        meta = load_episode(c, rec, True, keys=("classes", "states"))
        other = store.features(store.records[rec["sibling"]])[rec["query"]+np.asarray(c["data"]["horizons"])]
        state = decoder.predict(prediction)
        D, R, K = prediction.shape[-1], c["data"]["recent"], info["history_count"]
        for j, h in enumerate(c["data"]["horizons"]):
            p, y = prediction[j]*store.norm["std"]+store.norm["mean"], target[j]*store.norm["std"]+store.norm["mean"]
            pairacc, margin, separation = pair_diagnostic(prediction[j], target[j], other[j])
            ti = rec["query"]+h
            rows.append({"variant": variant, "policy": policy, "seed": seed, "task": "motion",
                "split": split, "condition": condition, "horizon": h, "episode": rec["id"], "pair": rec["pair"],
                "latent_mse": float(np.mean((p-y)**2)), "normalized_mse": float(np.mean((prediction[j]-target[j])**2)),
                "cosine_distance": 1-float(np.dot(p, y)/max(np.linalg.norm(p)*np.linalg.norm(y), 1e-12)),
                "pair_accuracy": pairacc, "pair_margin": margin, "target_pair_distance": separation,
                "state_accuracy": float((state[j, 0]>0) == (meta["classes"][ti]>0)),
                "state_position_mse": float(np.mean((state[j, 1:]-meta["states"][ti])**2)),
                "history_count": K, "history_feature_bytes": K*D*4,
                "projected_history_bytes": K*model.projection.shape[1]*4, "recent_feature_bytes": R*D*4,
                "timestamp_type_mask_bytes": (K+R)*13, "past_action_bytes": (K+R)*8,
                "evidence_hit": float(info["evidence_hit"])})
    return rows


def evaluate_mechanism_seed(c, store, model, decoder, seed, arm, collect_oracle=False):
    """Native all existing slices; intervention ONLY on validation, never fresh data."""
    device = torch.device("cpu")
    model.eval().requires_grad_(False)
    clone = copy.deepcopy(model).eval().requires_grad_(False)
    original_hash = model_state_hash(model)
    if original_hash != model_state_hash(clone):
        raise ValueError("Mechanism clone weights/buffers differ from source")
    policy = POLICIES[ARMS.index(arm)]
    oracle = BasisOracle(model.output_basis.cpu().numpy(), model.target_mean.cpu().numpy()) if collect_oracle else None
    raw, scalars, oracle_rows, recent_checks = [], [], [], []
    for split in SPLITS[1:]:
        records = _motion_pairs([r for r in store.manifest["records"] if r["task"] == "motion" and r["split"] == split],
                               c["data"]["pairs"][split])
        for condition in CONDITIONS:
            hashes = {v: hashlib.sha256() for v in VARIANTS}
            for start in range(0, len(records), c["training"]["batch_size"]):
                part = records[start:start+c["training"]["batch_size"]]
                samples = [store.sample(r, condition) for r in part]
                if condition == "wrong_evidence":
                    for rec, sample in zip(part, samples):
                        _assert_wrong_selected(store, rec, sample[0], sample[2])
                if condition == "recent_only":
                    for i in range(0, len(samples), 2):
                        assert_sample_exact(samples[i][0], samples[i+1][0], "mechanism paired recent inputs")
                batch = collate([s[0] for s in samples], device)
                if split == "validation":
                    predictions, trace = frozen_forward_pair(model, clone, batch)
                    for i, rec in enumerate(part):
                        scalars.append({"seed": seed, "split": split, "condition": condition, "pair": rec["pair"],
                            "episode": rec["id"], **velocity_scalars(trace, i,
                                require_two=condition in {"privileged_small_budget", "wrong_evidence"})})
                    if condition == "recent_only":
                        assert_exact(predictions["native"], predictions["pure_p_velocity"], "mechanism unchanged recent output")
                        for v in VARIANTS:
                            hashes[v].update(predictions[v].tobytes())
                else:
                    with torch.inference_mode():
                        predictions = {"native": model(batch).cpu().numpy()}
                for variant, prediction in predictions.items():
                    raw.extend(_metric_rows(c, store, model, decoder, part, samples, prediction, seed, split, condition, variant, policy))
                if oracle is not None and condition == "recent_only":
                    for rec, sample in zip(part, samples):
                        errors = oracle.error(sample[0]["current"], sample[1])
                        for j, h in enumerate(c["data"]["horizons"]):
                            oracle_rows.append({"split": split, "horizon": h, "episode": rec["id"], "pair": rec["pair"],
                                "oracle_normalized_mse": float(errors[j]), "target_aware_not_deployable": True})
            if split == "validation" and condition == "recent_only":
                recent_checks.append({"seed": seed, "byte_identical": True,
                    "prediction_sha256": {v: h.hexdigest() for v, h in hashes.items()}})
        print(f"motion mechanism frozen: {arm} seed={seed} split={split}", flush=True)
    if original_hash != model_state_hash(model) or original_hash != model_state_hash(clone):
        raise ValueError("Frozen mechanism weights/buffers changed during evaluation")
    proof = {"model_state_sha256": original_hash, "clone_exact": True, "weights_unchanged": True,
        "native_forward_byte_exact": True, "forward_rng_unchanged": True, "recent": recent_checks}
    if oracle is not None:
        proof["basis_oracle"] = {"rank": oracle.rank, "shape": list(oracle.basis.shape),
            "method": "float64 Moore-Penrose pseudoinverse, rcond=1e-12; no orthonormality assumption",
            "target_aware_not_deployable": True}
    return raw, scalars, oracle_rows, proof


def _matched_validation(c, raw, conditions, horizons, seed):
    seeds = c["training"]["seeds"] if seed == "all_seeds_mean" else [seed]
    chosen = [r for r in raw if r["split"] == "validation" and r["seed"] in seeds
              and r["condition"] in conditions and r["horizon"] in horizons]
    keyed = {}
    for row in chosen:
        key = (row["seed"], row["condition"], row["horizon"], row["variant"], row["episode"])
        if key in keyed:
            raise ValueError("Duplicate validation mechanism observation")
        keyed[key] = row
    episodes = sorted({r["episode"] for r in chosen})
    if len(episodes) != 2*c["data"]["pairs"]["validation"]:
        raise ValueError("Mechanism validation has incomplete episode coverage")
    aligned = []
    for eid in episodes:
        means, pair_ids = {}, set()
        for variant in VARIANTS:
            values = []
            for s in seeds:
                for condition in conditions:
                    for h in horizons:
                        row = keyed.get((s, condition, h, variant, eid))
                        if row is None:
                            raise ValueError("Missing matched seed/condition/horizon/variant in validation")
                        pair_ids.add(row["pair"])
                        values.append(row)
            means[variant] = {k: float(np.mean([r[k] for r in values]))
                for k in ("normalized_mse", "state_position_mse", "pair_margin")}
        if len(pair_ids) != 1:
            raise ValueError("Mechanism comparison changed base-pair identity")
        aligned.append((pair_ids.pop(), means))
    groups = [pair for pair, _ in aligned]
    if any(groups.count(g) != 2 for g in set(groups)):
        raise ValueError("Mechanism bootstrap requires complete sibling pairs")
    return aligned


def validation_decision(c, raw):
    checks = []
    for seed in c["training"]["seeds"]+["all_seeds_mean"]:
        for h in c["data"]["horizons"]:
            rows = _matched_validation(c, raw, ["privileged_small_budget"], [h], seed)
            ci = clustered_interval([m["native"]["normalized_mse"]-m["pure_p_velocity"]["normalized_mse"] for _, m in rows],
                [p for p, _ in rows], c["evaluation"]["bootstrap"])
            position = clustered_interval([m["native"]["state_position_mse"]-m["pure_p_velocity"]["state_position_mse"] for _, m in rows],
                [p for p, _ in rows], c["evaluation"]["bootstrap"])
            checks.append({"kind": "privileged_validation_future_mse", "seed": seed, "horizon": h,
                "absolute_mse_reduction": ci, "position_mse_reduction": position, "passed": ci["low"] > 0})
        rows = _matched_validation(c, raw, TRAIN_CONDITIONS, c["data"]["horizons"], seed)
        ci = clustered_interval([m["native"]["normalized_mse"]-m["pure_p_velocity"]["normalized_mse"] for _, m in rows],
            [p for p, _ in rows], c["evaluation"]["bootstrap"])
        checks.append({"kind": "four_condition_all_horizon_validation", "seed": seed,
            "absolute_mse_reduction": ci, "passed": ci["mean"] >= 0,
            "rule": "Prespecified equal-condition/equal-horizon mean must not worsen; interval reported"})
    return {"passed": all(r["passed"] for r in checks), "checks": checks,
        "selection": "No fitting/checkpoint selection; frozen primary validation evidence only"}


def basis_ceiling(c, native, oracle_rows):
    oracle = {(r["split"], r["horizon"], r["episode"]): r for r in oracle_rows}
    if len(oracle) != len(oracle_rows):
        raise ValueError("Duplicate basis oracle observation")
    checks = []
    for split in SPLITS[1:]:
        for h in c["data"]["horizons"]:
            selected = [r for r in native if r["split"] == split and r["horizon"] == h and r["condition"] == "recent_only"]
            by = {(r["seed"], r["episode"]): r for r in selected}
            if len(by) != len(selected):
                raise ValueError("Duplicate reference recent observation for basis ceiling")
            episodes = sorted({r["episode"] for r in selected})
            if len(episodes) != 2*c["data"]["pairs"][split]:
                raise ValueError("Incomplete reference recent/basis ceiling slice")
            for seed in c["training"]["seeds"]+["all_seeds_mean"]:
                seeds = c["training"]["seeds"] if seed == "all_seeds_mean" else [seed]
                base, floor, groups = [], [], []
                for eid in episodes:
                    item = oracle[(split, h, eid)]
                    values = [by[(s, eid)] for s in seeds]
                    if any(r["pair"] != item["pair"] for r in values):
                        raise ValueError("Oracle and reference recent pair mismatch")
                    base.append(float(np.mean([r["normalized_mse"] for r in values])))
                    floor.append(item["oracle_normalized_mse"])
                    groups.append(item["pair"])
                if any(groups.count(g) != 2 for g in set(groups)):
                    raise ValueError("Basis ceiling requires complete sibling pairs")
                if np.any(np.asarray(floor) > np.asarray(base)+1e-7+1e-6*np.abs(base)):
                    raise ValueError("Oracle lower bound exceeds an actual model in its fixed output space")
                maximum = paired_improvement(base, floor, groups, c["evaluation"]["bootstrap"],
                    c["decision"]["baseline_floor"], c["decision"]["absolute_latent_gain"])
                threshold = c["decision"]["latent_relative_improvement"] if maximum["scale"].startswith("relative") else c["decision"]["absolute_latent_gain"]
                checks.append({"split": split, "horizon": h, "seed": seed,
                    "fixed_reference_recent_mse": float(np.mean(base)), "oracle_mse": float(np.mean(floor)),
                    "maximum_possible_improvement": maximum, "required_original_threshold": threshold,
                    "point_ceiling_blocks": maximum["mean"] < threshold,
                    "target_aware_not_deployable": True})
    vetoes = [r for r in checks if r["split"] in SPLITS[2:] and r["point_ceiling_blocks"]]
    return {"velocity_training_veto": bool(vetoes), "vetoes": vetoes, "checks": checks,
        "interpretation": "Optimistic target-aware fixed-output-space ceiling. Passing this bound does not establish predictability or Phase 1 GO; blocking a required old-development slice vetoes velocity training."}


def _guard_destination(output, roots):
    output = Path(output).expanduser().resolve()
    for root in roots.values():
        protected = [root]+[(root/p).resolve() for p in ("dataset", "features", "predictor", "probes", "evaluation")]
        if any(output == p or output in p.parents or p in output.parents for p in protected):
            raise ValueError("Mechanism output must be outside original runs/shared artifacts")
    for name in ("motion_mechanism.json", "motion_mechanism.md", "motion_mechanism.raw.csv",
                 "motion_mechanism.velocity.csv", "motion_mechanism.oracle.csv"):
        if (output/name).exists():
            raise ValueError("Mechanism report already exists; choose a new output")
    return output


def motion_mechanism(reference_output, wide_output, output, device="cpu", frozen_source_directory=None):
    """Prepare evidence for at most ONE single-factor training trial, never run it."""
    if torch.device(device).type != "cpu":
        raise ValueError("Prespecified mechanism diagnostic uses allocated CPU only")
    if frozen_source_directory is not None:
        with frozen_runtime(frozen_source_directory) as runtime:
            return runtime.motion_mechanism(reference_output, wide_output, output, device=device)
    roots = {a: Path(p).expanduser().resolve() for a, p in zip(ARMS, (reference_output, wide_output))}
    if len(set(roots.values())) != 2:
        raise ValueError("Distinct original reference/wide runs required")
    scientific_directory = Path(scientific_utils.__file__).resolve().parent
    scientific_before = _scientific_sources(scientific_directory)
    source_code_proof = loaded_code_proof(scientific_directory)
    diagnostic_files = {"motion_mechanism.py": Path(__file__).resolve(),
                        "diagnostics.py": Path(diagnostic_helpers.__file__).resolve()}
    diagnostic_before = {name: file_hash(path) for name, path in diagnostic_files.items()}
    preregistration = Path(__file__).resolve().parents[2]/"configs/phase1_v4_motion_mechanism_protocol.json"
    preregistration_sha = file_hash(preregistration)
    output = _guard_destination(output, {**roots, "scientific_source": scientific_directory})
    before = _hash_sources(roots)
    configs = {a: _source_config(p, a) for a, p in roots.items()}
    for arm, c in configs.items():
        registered_source_config(c, arm)
    plain = [{k: v for k, v in c.items() if k != "experiment"} for c in configs.values()]
    if scientific_hash(plain[0]) != scientific_hash(plain[1]):
        raise ValueError("Original arms differ beyond the fixed frame policy")
    for name in ("dataset/manifest.json", "features/index.json", "features/normalization.npz",
                 "predictor/motion/basis.npz", "probes/motion_state_decoder.npz"):
        if before[ARMS[0]][name] != before[ARMS[1]][name]:
            raise ValueError(f"Original arms do not share fixed {name}")
    saved_rng, old_threads = _rng_state(), torch.get_num_threads()
    torch.set_num_threads(configs["reference"]["threads"])
    arms, all_raw, all_scalars, oracle_rows, primary_raw = {}, [], [], [], None
    try:
        for arm, c in configs.items():
            store = FeatureStore(c, max_cached=2*max(c["data"]["pairs"].values()))
            basis_provenance = verify_basis_provenance(c, store)
            completion = _verify_source_proof(c, store)
            decoder = load_decoder(c, "motion")
            raw, scalars, proofs = [], [], {}
            for seed in c["training"]["seeds"]:
                model, ck = load_predictor(c, seed, store, torch.device("cpu"), "motion")
                if ck["epoch"] != completion["models"][f"motion/{seed}"]["best_epoch"]:
                    raise ValueError("Mechanism checkpoint differs from fixed validation selection")
                verified_basis = verify_basis_provenance(c, store, model)
                values, trace, oracle, proof = evaluate_mechanism_seed(c, store, model, decoder, seed, arm,
                    collect_oracle=arm == "reference" and seed == c["training"]["seeds"][0])
                raw.extend(values)
                scalars.extend(trace)
                oracle_rows.extend(oracle)
                proofs[str(seed)] = proof
                proofs[str(seed)]["basis_provenance"] = verified_basis
                del model, ck
            native = [r for r in raw if r["variant"] == "native"]
            reproduction = verify_native_metrics(native, roots[arm]/"evaluation/raw_records.csv", arm)
            arms[arm] = {"role": "primary" if arm == "reference" else "replication_cannot_rescue_primary",
                "policy": POLICIES[ARMS.index(arm)], "source": str(roots[arm]), "training_completion": completion,
                "basis_provenance": basis_provenance,
                "native_reproduction": reproduction, "forward_proofs": proofs,
                "validation": validation_decision(c, raw), "native_summary": summarize(c, native)}
            if arm == "reference":
                primary_raw = native
            all_raw.extend({"checkpoint_arm": arm, **r} for r in raw)
            all_scalars.extend({"checkpoint_arm": arm, **r} for r in scalars)
    finally:
        torch.set_num_threads(old_threads)
        _restore_rng(saved_rng)
        if _rng_hash(saved_rng) != _rng_hash():
            raise ValueError("Caller RNG state was not restored")
        if before != _hash_sources(roots):
            raise ValueError("Original run changed during mechanism diagnostic")
        if scientific_before != _scientific_sources(scientific_directory):
            raise ValueError("Loaded scientific source changed during mechanism diagnostic")
        if preregistration_sha != file_hash(preregistration):
            raise ValueError("Motion mechanism preregistration changed during diagnostic")
        if diagnostic_before != {name: file_hash(path) for name, path in diagnostic_files.items()}:
            raise ValueError("Diagnostic implementation changed during evaluation")
    ceiling = basis_ceiling(configs["reference"], primary_raw, oracle_rows)
    authorized = not ceiling["velocity_training_veto"] and arms["reference"]["validation"]["passed"]
    result = {"status": "MECHANISM_DIAGNOSTIC_ONLY_NOT_GO", "source_unchanged": True, "source_sha256": before,
        "scientific_source_directory": str(scientific_directory), "scientific_source_sha256": scientific_before,
        "loaded_code_proof": source_code_proof, "preregistration_file": str(preregistration), "preregistration_sha256": preregistration_sha,
        "fitted_any_component": False, "weights_unchanged": True, "caller_rng_restored": True,
        "arms": arms, "basis_oracle": ceiling, "authorize_singlefactor_training": authorized,
        "allowed_candidate": "Only velocity regression dependent variable aligned -> p; same original start/end policy, aligned memory, actions, parameters, basis, targets, loss, seeds, epochs and validation selection.",
        "decision_rule": "Primary reference/old only: every seed/horizon privileged validation paired future-MSE CI low > 0; four-condition/all-horizon validation mean nonworse for each seed and pooled; any old-development fixed-basis point ceiling below original 10% rule vetoes. No automatic rank expansion or training submission.",
        "scope": "Existing development only; pure-p intervention validation only; true futures used solely for scoring/oracle; no fresh confirmation opened",
        "formula_tolerance": {"atol": 2e-5, "rtol": 2e-4, "reason": "float32 weighted-covariance cancellation; native forward itself must be byte exact"},
        "diagnostic_file_sha256": diagnostic_before,
        "diagnostic_code_hash": digest({**diagnostic_before,
            "scientific_code_hash": code_hash("model.py", "inputs.py", "evaluation.py", "stats.py")})}
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("raw", all_raw), ("velocity", all_scalars), ("oracle", oracle_rows)):
        path = output/f"motion_mechanism.{name}.csv"
        write_csv(path, rows)
        result[f"{name}_sha256"] = file_hash(path)
    write_json(output/"motion_mechanism.json", result)
    (output/"motion_mechanism.md").write_text("\n".join([
        "# Frozen motion mechanism diagnostic", "", "This is not a Phase 1 GO result.", "",
        f"Eligible for the one prespecified training contrast: **{authorized}**.",
        f"Fixed-basis old-development ceiling veto: **{ceiling['velocity_training_veto']}**.",
        f"Primary frozen validation mechanism checks passed: **{arms['reference']['validation']['passed']}**.", "",
        "The primary uses reference weights with the original start/end policy. Wide/native results are a replication and cannot rescue it.",
        "The basis oracle reads true futures and is an optimistic, nondeployable error lower bound, not a model prediction.",
        "All parameters/buffers remain frozen. Native outputs and original native metrics must reproduce; recent outputs are byte identical.",
        "Only velocity uses raw projected p instead of action-aligned features; aligned memory and all other model components remain unchanged.",
        "Eligibility does not launch training, prove causation beyond this intervention, or establish fresh-test success.", ""]), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-output", required=True)
    parser.add_argument("--wide-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen-source-directory", help="Original v3 checkout/package; execute its science in an isolated namespace")
    args = parser.parse_args(argv)
    print(motion_mechanism(args.reference_output, args.wide_output, args.output,
        frozen_source_directory=args.frozen_source_directory)["status"], flush=True)


if __name__ == "__main__":
    main()
