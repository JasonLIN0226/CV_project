"""Gate2 terminal-deletion pilot: geometry, reader qualification, then baselines.

This fresh task has four counterfactual siblings and three historical cues.
Only one terminal deletion is compared, after a common FIFO prefix. No student
is fitted. All outputs, including feasibility failures, are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "experiments/predictor/control/src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.phase2.run_reader_scale_probe import empty_directory, identity, sha, write

HORIZONS = (3, 6)
BITS = ((0, 0), (0, 1), (1, 0), (1, 1))
STYLES = ("benign_duplicate", "distant_style_duplicate")
SPLITS = (("train", 20261028, 240), ("validation", 20261029, 48),
          ("geometry", 20261030, 24), ("development", 20261031, 48))
MODEL_CONFIG = {"hidden": 128, "heads": 4, "layers": 2, "dropout": 0., "rank": 128}
CONTENT_ARMS = ("AB", "AprimeB", "AAprime", "Ablank", "Bblank", "wrongA", "wrongB", "blank2")
MODES = ("randomK", "latestK", "latest3", "random_size")
ORIGINS = ("query", "preuse", "middle")
TOL = 1e-6


def emit(stage, **kwargs):
    print(json.dumps({"stage": stage, **kwargs}), flush=True)


def protocol(args):
    files = [Path(__file__), ROOT / "experiments/phase2/gate2_scene.py",
             ROOT / "docs/GATE2_TERMINAL_PILOT_PROTOCOL.md",
             ROOT / "experiments/phase2/memory.py",
             ROOT / "experiments/phase2/run_reader_scale_probe.py",
             ROOT / "experiments/phase2/encode_fresh_gate_diagnostic.py"]
    files += [ROOT / "experiments/predictor/control/src/phase1" / name
              for name in ("model.py", "inputs.py", "encoders.py", "utils.py")]
    return {"name": "gate2-terminal-pilot-v1", "status": "pre_execution",
            "splits": [dict(split=s, seed=seed, quartets=n, paired_style_families=n // 2)
                       for s, seed, n in SPLITS],
            "styles": STYLES, "model": MODEL_CONFIG, "horizons": HORIZONS,
            "steps": args.steps, "validation_every": args.validate_every,
            "quartets_per_update": 8, "model_seed": 17, "memory_dropout": .15,
            "history_modes": MODES, "origin_training": "query, preuse, and uniform all origins 3..q",
            "selection": "earliest minimum equal-origin/equal-history validation MSE",
            "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .0001, "gradient_norm_clip": 1.},
            "scheduler": {"name": "MultiStepLR", "milestones": [args.steps//3, 2*args.steps//3], "gamma": .5},
            "training_loss": "feature MSE + 2*mean all-six quartet pair-difference MSE + .5*mean ordered-pair softplus(-4*(other_error-own_error)/target_separation); rank only where separation>1e-6",
            "normalization": "TRAIN prefix frames only, episode-frame multiplicity, std floor .05",
            "basis": "TRAIN only; all prefix origins and all six quartet future contrasts",
            "content_slots": "fixed ages [6,4], all actions and current/recent identical",
            "content_roles": "A and Aprime are duplicate views of the scene's duplicated physical axis; B is the other axis, not always y",
            "terminal_candidates": "common FIFO until q-1; q-6,q-5,q-4; compare one deletion at q",
            "terminal_uniform": "uniform one-of-three deletion; NumPy default_rng seeded from SHA256 JSON [17,family] first16hex; shared styles and siblings, not full-stream Algorithm R",
            "geometry_gate": "at least one GEOMETRY quartet has a diversity redundant-pair selection",
            "oracle_tolerance": TOL, "feature_classification": "nearest four actual sibling future features, joint h3/h6 MSE",
            "reader_gate": {"applied_to": "all and each style separately", "classification_ties": "within1e-6; ties fail accuracy",
                            "AB_accuracy_min": .8, "AprimeB_accuracy_min": .8,
                            "wrongA_counterfactual_accuracy_min": .8, "wrongB_counterfactual_accuracy_min": .8,
                            "each_complementary_arm_mse_below_each_single_by": TOL},
            "no_student_training": True, "development_only": True,
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files}}


def generate_encode(args, out, device):
    from experiments.phase2 import gate2_scene as env
    from experiments.phase2.encode_fresh_gate_diagnostic import (
        ENCODER_CONFIG, EXPECTED_ENCODER_METADATA_SHA256, digest)
    from phase1.encoders import FrozenIJepa
    import torch

    data = out / "data"
    data.mkdir()
    records, frames, lookup, actions, evaluator_only = [], [], {}, {}, {}
    for split, seed, count in SPLITS:
        for index in range(count // 2):
            for style in STYLES:
                scene = env.scene(seed, index, split, style)
                quartet = []
                for bits in BITS:
                    ep = env.episode(scene, bits)
                    q = int(ep["query"])
                    evidence = np.asarray(ep["evidence"], dtype=int)
                    axes = list(ep["semantic_axes"])
                    if evidence.tolist() != [q-6, q-5, q-4] or sorted(axes) not in (["a", "a", "b"], ["a", "b", "b"]):
                        raise ValueError("Gate2 cue schedule/axes contract failed")
                    if ep["rgb"].dtype != np.uint8 or ep["actions"].shape != (len(ep["rgb"])-1, 2):
                        raise ValueError("Gate2 RGB/actions shape contract failed")
                    if np.any(ep["actions"][:q]) or q < 9 or len(ep["rgb"]) <= q+6:
                        raise ValueError("Gate2 prefix controls/query/target contract failed")
                    rid = identity(["gate2-v1", scene["scene_id"], bits])
                    indices = []
                    for frame in ep["rgb"]:
                        key = hashlib.sha256(frame.tobytes()).hexdigest()
                        if key not in lookup:
                            lookup[key] = len(frames)
                            frames.append(frame.copy())
                        indices.append(lookup[key])
                    r = {"id": rid, "quartet": scene["scene_id"], "family": scene["family_id"],
                         "split": split, "condition": style, "query": q, "bits": list(bits),
                         "evidence": evidence.tolist(), "semantic_axes": axes, "rgb_indices": indices,
                         "scene": scene}
                    records.append(r)
                    actions[rid] = np.asarray(ep["actions"], np.float32)
                    evaluator_only["states__"+rid] = np.asarray(ep["states"])
                    evaluator_only["bits__"+rid] = np.asarray(bits, np.int64)
                    quartet.append(ep)
                base = quartet[0]
                for ep in quartet[1:]:
                    if not np.array_equal(base["actions"], ep["actions"]) or not np.array_equal(base["rgb"][q-3:q+1], ep["rgb"][q-3:q+1]):
                        raise ValueError("Sibling current/recent/actions differ")
                targets = [ep["rgb"][[q+3, q+6]].tobytes() for ep in quartet]
                if len(set(targets)) != 4:
                    raise ValueError("Four sibling targets must differ")
    write(data / "records.json", records)
    np.savez_compressed(data / "evaluator_only.npz", **evaluator_only)
    bank = np.stack(frames)
    np.savez_compressed(data / "unique_rgb.npz", rgb=bank)
    config = dict(ENCODER_CONFIG, cache_dir=str(args.cache_dir.resolve()), batch_size=args.encoder_batch_size)
    encoder = FrozenIJepa(config, torch.device(device))
    if digest(encoder.metadata) != EXPECTED_ENCODER_METADATA_SHA256:
        raise ValueError("Pinned I-JEPA metadata changed")
    start = time.monotonic()
    encoded = []
    for at in range(0, len(bank), args.encoder_batch_size):
        z = encoder.encode(bank[at:at+args.encoder_batch_size])
        if z.shape[1:] != (5120,) or z.dtype != np.float32 or not np.isfinite(z).all():
            raise ValueError("Invalid frozen encoder output")
        encoded.append(z)
        if at % (50*args.encoder_batch_size) == 0:
            emit("encoding", done=at+len(z), total=len(bank))
    raw = np.concatenate(encoded)
    # Count repeated TRAIN frames with their ordinary episode/time multiplicity.
    ti = np.concatenate([r["rgb_indices"][:r["query"]+1] for r in records if r["split"] == "train"])
    sx = raw[ti].astype(np.float64)
    mean = sx.mean(0).astype(np.float32)
    std = np.maximum(sx.std(0), .05).astype(np.float32)
    normalized = ((raw-mean)/std).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite normalized features")
    np.savez_compressed(data / "normalization.npz", mean=mean, std=std, train_indices=ti)
    np.savez_compressed(data / "unique_features.npz", raw=raw, normalized=normalized)
    np.savez_compressed(data / "actions.npz", **actions)
    write(data / "encoding.json", {"unique_frames": len(bank), "seconds": time.monotonic()-start,
          "metadata": encoder.metadata, "metadata_sha256": digest(encoder.metadata),
          "normalization_sha256": sha(data / "normalization.npz")})
    del encoder, encoded, bank, frames, sx
    torch.cuda.empty_cache()
    return records, normalized, actions


def diversity_choice(z, times, current):
    from experiments.phase2.memory import BoundedMemory, MemoryEntry, PolicyContext
    memory = BoundedMemory(2, "visual_diversity")
    transition = None
    for t, value in zip(times, z):
        transition = memory.observe(MemoryEntry("geometry", int(t), value), PolicyContext(current))
    return times.index(transition.deleted_id[1])


def uniform_choice(family):
    seed = int(hashlib.sha256(json.dumps([17, family]).encode()).hexdigest()[:16], 16)
    return int(np.random.default_rng(seed).integers(3))


def geometry_audit(records, z, out):
    rows = []
    for r in records:
        # The gate reads only its dedicated GEOMETRY split. All presets reported.
        if r["split"] != "geometry":
            continue
        cand = z[np.array(r["rgb_indices"])[r["evidence"]]]
        norm = np.linalg.norm(cand.astype(np.float64), axis=1)
        unit = cand / np.maximum(norm[:, None], np.finfo(np.float64).tiny)
        similarity = np.clip(unit@unit.T, -1, 1)
        similarity[np.logical_and(norm[:, None] == 0, norm[None, :] == 0)] = 1.
        distances = 1-similarity
        drop = diversity_choice(cand, r["evidence"], r["query"])
        kept_axes = [a for i, a in enumerate(r["semantic_axes"]) if i != drop]
        deletion_distances = np.array([distances[tuple(j for j in range(3) if j != i)] for i in range(3)])
        redundant_drop = next(i for i, axis in enumerate(r["semantic_axes"]) if r["semantic_axes"].count(axis) == 1)
        margin = deletion_distances[redundant_drop]-max(np.delete(deletion_distances, redundant_drop))
        rows.append({"id": r["id"], "quartet": r["quartet"], "family": r["family"],
                     "condition": r["condition"], "bits": r["bits"], "semantic_axes": r["semantic_axes"],
                     "pairwise_cosine_distance": distances.tolist(), "predicted_drop": drop,
                     "frozen_choices": {"fifo": 0, "terminal_uniform": uniform_choice(r["family"]), "visual_diversity": drop},
                     "retained_distance_by_drop": deletion_distances.tolist(),
                     "max_distance_drop_set_within_tolerance": np.flatnonzero(deletion_distances >= deletion_distances.max()-TOL).tolist(),
                     "actual_tie_break": "lexicographic min distance, mean distance, sum timestamps; exact floating comparisons",
                     "redundant_margin_over_best_complementary": float(margin),
                     "redundant_strictly_farthest_over_tolerance": bool(margin > TOL),
                     "redundant_pair_retained": len(set(kept_axes)) == 1})
    summary = {c: {"episodes": sum(r["condition"] == c for r in rows),
                  "redundant_episodes": sum(r["condition"] == c and r["redundant_pair_retained"] for r in rows),
                  "redundant_quartets": len({r["quartet"] for r in rows if r["condition"] == c and r["redundant_pair_retained"]})}
               for c in STYLES}
    result = {"before_reader_basis_or_fit": True, "selection_on_development": False,
              "rows": rows, "by_condition": summary,
              "feasibility_pass": any(r["redundant_pair_retained"] for r in rows),
              "metric": "actual BoundedMemory visual_diversity on TRAIN-normalized I-JEPA features"}
    write(out / "geometry_locked.json", result)
    emit("geometry", **summary, feasibility_pass=result["feasibility_pass"])
    return result


def groups(records, split):
    result = {}
    for r in records:
        if r["split"] == split:
            result.setdefault(r["quartet"], []).append(r)
    for rows in result.values():
        rows.sort(key=lambda r: r["bits"])
        if tuple(tuple(r["bits"]) for r in rows) != BITS:
            raise ValueError("Incomplete quartet")
    return list(result.values())


def fit_basis(records, z, out, device):
    import torch
    train = groups(records, "train")
    visual, residual, contrasts = [], [], []
    for quartet in train:
        ys = []
        for r in quartet:
            zz = z[r["rgb_indices"]]
            visual.append(zz[:r["query"]+1])
            origins = np.arange(3, r["query"]+1)
            y = zz[origins[:, None]+np.array(HORIZONS)]-zz[origins, None]
            ys.append(y)
            residual.append(y.reshape(-1, z.shape[1]))
        for a, b in itertools.combinations(range(4), 2):
            contrasts.append((2*(ys[a]-ys[b])).reshape(-1, z.shape[1]))
    x = np.concatenate(visual)
    y = np.concatenate(residual)
    im, tm = x.mean(0), y.mean(0)

    def pca(a, whiten=False):
        torch.manual_seed(17)
        value = torch.from_numpy(a).to(device)
        _, singular, vec = torch.pca_lowrank(value, q=min(128, min(a.shape)-1), center=False, niter=3)
        if whiten:
            vec = vec/(singular/np.sqrt(len(a))).clamp_min(.05)
        return vec.cpu().numpy()

    basis = {"input_mean": im, "projection": pca(x-im, True), "target_mean": tm,
             "output_basis": pca(np.concatenate([y-tm]+contrasts))}
    if not all(np.isfinite(v).all() for v in basis.values()):
        raise ValueError("Nonfinite fitted basis")
    np.savez_compressed(out / "basis.npz", **basis)
    write(out / "basis_provenance.json", {"train_ids": [r["id"] for rr in train for r in rr],
          "all_training_prefix_origins": True, "all_six_quartet_contrasts": True,
          "basis_sha256": sha(out / "basis.npz")})
    return basis


class Samples:
    def __init__(self, records, z, actions, basis, device):
        from phase1.model import project
        self.records, self.z, self.actions = records, z, actions
        self.p = project(z, basis)
        self.basis, self.device = basis, device
        self.by_quartet = {r[0]["quartet"]: r for split, _, _ in SPLITS for r in groups(records, split)}

    def sample(self, r, t, indices, memory_override=None, fixed_times=None):
        from phase1.model import project
        zz = self.z[r["rgb_indices"]]
        recent = np.arange(t-3, t+1)
        times = np.concatenate([recent, np.asarray(indices if fixed_times is None else fixed_times, int)])
        history = zz[indices].copy() if memory_override is None else np.asarray(memory_override, np.float32)
        values = np.concatenate([zz[recent], history])
        ac = self.actions[r["id"]]
        prefix = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(ac[:t], axis=0)])
        projected = self.p[np.array(r["rgb_indices"])[times]].copy() if memory_override is None else project(values, self.basis)
        sample = {"z": values, "age": (t-times).astype(np.float32),
                  "type": np.array([0]*4+[1]*len(history), np.int64),
                  "past_actions": prefix[t]-prefix[times], "actions": ac[t:t+6].copy(),
                  "current": zz[t].copy(), "projected_z": projected,
                  "projected_current": self.p[r["rgb_indices"][t]].copy()}
        return sample, zz[t+np.array(HORIZONS)].copy()

    def sampled_batch(self, quartets, mode, origin, rng, dropout=False):
        from phase1.inputs import collate
        import torch
        arrays, targets, metadata = [], [], []
        for quartet in quartets:
            q = quartet[0]["query"]
            t = q if origin == "query" else (int(rng.integers(3, q-5)) if origin == "preuse" else int(rng.integers(3, q+1)))
            valid = np.arange(max(t-3, 0))
            if mode == "latestK":
                idx = valid[-2:]
            elif mode == "latest3":
                pool = valid[-3:]
                idx = np.sort(rng.choice(pool, min(2, len(pool)), replace=False))
            else:
                n = min(2, len(valid)) if mode == "randomK" else int(rng.integers(0, min(2, len(valid))+1))
                idx = np.sort(rng.choice(valid, n, replace=False))
            if dropout and rng.random() < .15:
                idx = idx[:0]
            for r in quartet:
                a, y = self.sample(r, t, idx)
                arrays.append(a); targets.append(y)
                metadata.append({"id": r["id"], "origin": t, "indices": idx.tolist()})
        return collate(arrays, self.device), torch.from_numpy(np.stack(targets)).to(self.device), metadata


def quartet_loss(prediction, target):
    import torch
    p = prediction.reshape(-1, 4, *prediction.shape[1:])
    y = target.reshape_as(p)
    pairs, ranks = [], []
    for a, b in itertools.combinations(range(4), 2):
        pairs.append(((p[:, a]-p[:, b])-(y[:, a]-y[:, b])).square().mean())
        separation = (y[:, a]-y[:, b]).square().mean(-1)
        for own, other in ((a, b), (b, a)):
            own_error = (p[:, own]-y[:, own]).square().mean(-1)
            other_error = (p[:, own]-y[:, other]).square().mean(-1)
            valid = separation > 1e-6
            if valid.any():
                ranks.append(torch.nn.functional.softplus(-4*(other_error-own_error)[valid]/separation[valid]).mean())
    mse = (prediction-target).square().mean()
    return mse+2*torch.stack(pairs).mean()+(.5*torch.stack(ranks).mean() if ranks else 0.)


def train_reader(args, records, z, actions, basis, out, device):
    import torch
    from phase1.model import DynamicsPredictor
    from phase1.utils import seed_all
    seed_all(17, 4)
    samples = Samples(records, z, actions, basis, device)
    train, validation = groups(records, "train"), groups(records, "validation")
    model = DynamicsPredictor(z.shape[1], MODEL_CONFIG, HORIZONS, basis).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.steps//3, 2*args.steps//3], gamma=.5)
    rng = np.random.default_rng(17)
    best, selected, logs = float("inf"), None, []
    counts, seen = {}, set()
    began = time.monotonic()

    def validate():
        model.eval()
        values = {}
        with torch.inference_mode():
            for oi, origin in enumerate(ORIGINS):
                for mi, mode in enumerate(MODES):
                    vrng = np.random.default_rng(17000+100*oi+mi)
                    total = count = 0
                    for start in range(0, len(validation), 8):
                        batch, target, _ = samples.sampled_batch(validation[start:start+8], mode, origin, vrng)
                        n = len(target)
                        loss = (model(batch)-target).square().mean()
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite validation loss")
                        total += float(loss)*n
                        count += n
                    values[origin+"/"+mode] = total/count
        return {"equal_origin_mode_mse": float(np.mean(list(values.values()))), "cells": values}

    for step in range(args.steps+1):
        if step % args.validate_every == 0 or step == args.steps:
            v = validate()
            if v["equal_origin_mode_mse"] < best:
                best, selected = v["equal_origin_mode_mse"], step
                torch.save({"model": model.state_dict(), "step": step, "validation": v,
                            "basis_sha256": sha(out / "basis.npz"), "model_config": MODEL_CONFIG}, out / "reader_selected.pt")
            logs.append({"step": step, **v})
            write(out / "learning_curve.json", logs)
            emit("training", step=step, validation=v["equal_origin_mode_mse"])
        if step == args.steps:
            break
        chosen = rng.choice(len(train), 8, replace=False)
        seen.update(map(int, chosen))
        mode, origin = MODES[step % 4], ORIGINS[(step//4) % 3]
        batch, target, metadata = samples.sampled_batch([train[i] for i in chosen], mode, origin, rng, True)
        for r in metadata:
            key = str(r["origin"])
            counts[key] = counts.get(key, 0)+1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = quartet_loss(model(batch), target)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite reader training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
    state = torch.load(out / "reader_selected.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    write(out / "reader_locked.json", {"selected_step": selected, "best_validation_mse": best,
          "steps": args.steps, "seconds": time.monotonic()-began, "origin_example_counts": counts,
          "unique_training_quartets": len(seen), "selected_sha256": sha(out / "reader_selected.pt"),
          "geometry_sha256": sha(out / "geometry_locked.json"), "development_used_for_selection": False})
    return model, samples


def content_spec(samples, r, arm):
    axes = r["semantic_axes"]
    duplicate = max(set(axes), key=axes.count)
    roles = r["scene"]["roles"]
    ai, api, bi = (roles.index(role) for role in ("A", "Aprime", "B"))
    times = r["evidence"]
    zz = samples.z[r["rgb_indices"]]
    a, ap, b, blank = zz[times[ai]], zz[times[api]], zz[times[bi]], zz[r["query"]]
    counterfactual = r["bits"].copy()
    if arm in ("wrongA", "wrongB"):
        axis = (0 if duplicate == "a" else 1) if arm == "wrongA" else (1 if duplicate == "a" else 0)
        counterfactual[axis] ^= 1
        sibling = next(s for s in samples.by_quartet[r["quartet"]] if s["bits"] == counterfactual)
        sz = samples.z[sibling["rgb_indices"]]
        if arm == "wrongA":
            a = sz[times[ai]]
        else:
            b = sz[times[bi]]
    choices = {"AB": [a, b], "AprimeB": [ap, b], "AAprime": [a, ap],
               "Ablank": [a, blank], "Bblank": [blank, b], "wrongA": [a, b],
               "wrongB": [a, b], "blank2": [blank, blank]}
    return np.stack(choices[arm]), counterfactual


def predict_rows(model, samples, requests):
    import torch
    from phase1.inputs import collate
    values = []
    with torch.inference_mode():
        for start in range(0, len(requests), 64):
            batch = collate([x[0] for x in requests[start:start+64]], samples.device)
            pred = model(batch).cpu().numpy()
            if not np.isfinite(pred).all():
                raise ValueError("Nonfinite saved evaluation prediction")
            values.append(pred)
    return np.concatenate(values)


def evaluate_preuse(model, samples, dev, out):
    """Restricted static-target check, not general early-origin calibration."""
    requests, metadata, controls = [], [], []
    for r in dev:
        t = r["query"]-6
        idx = list(range(t-5, t-3))
        a, y = samples.sample(r, t, idx)
        requests.append((a, y))
        zz = samples.z[r["rgb_indices"]]
        controls.append(np.repeat(zz[t-1][None], 2, axis=0))
        metadata.append({"id": r["id"], "quartet": r["quartet"], "family": r["family"],
                         "condition": r["condition"], "origin": t, "history_times": idx,
                         "target_times": [t+h for h in HORIZONS]})
    predictions = predict_rows(model, samples, requests)
    targets = np.stack([r[1] for r in requests])
    control = np.stack(controls)
    for r, p, y, c in zip(metadata, predictions, targets, control):
        r["mse"] = float(np.mean((p.astype(np.float64)-y)**2))
        r["mse_by_horizon"] = np.mean((p.astype(np.float64)-y)**2, axis=1).tolist()
        r["causal_previous_frame_mse"] = float(np.mean((c.astype(np.float64)-y)**2))
    np.savez_compressed(out / "preuse_predictions.npz", predictions=predictions, targets=targets,
                        causal_previous_frame_prediction=control)
    write(out / "preuse_rows.json", metadata)
    return {"origin": "q-6", "targets": "q-3 and q; both after final cue and before motion",
            "history": "latest two eligible frames, same original architecture",
            "limitation": "Restricted static target diagnostic; original reader ignores noncurrent recent tokens",
            "by_condition": {c: {"mse": float(np.mean([r["mse"] for r in metadata if c == "all" or r["condition"] == c])),
                                  "causal_previous_frame_mse": float(np.mean([r["causal_previous_frame_mse"] for r in metadata if c == "all" or r["condition"] == c]))}
                             for c in ("all",)+STYLES}}


def evaluate(model, samples, records, out):
    import torch
    dev = [r for quartet in groups(records, "development") for r in quartet]
    requests, rows, targets = [], [], []
    preuse = evaluate_preuse(model, samples, dev, out)
    for r in dev:
        q = r["query"]
        truth = np.stack([samples.z[s["rgb_indices"]][q+np.array(HORIZONS)] for s in samples.by_quartet[r["quartet"]]])
        targets.append(truth)
        for arm in CONTENT_ARMS:
            content, cf = content_spec(samples, r, arm)
            a, y = samples.sample(r, q, [], content, [q-6, q-4])
            requests.append((a, y))
            rows.append({"id": r["id"], "quartet": r["quartet"], "family": r["family"], "condition": r["condition"],
                         "cell": r["condition"]+"/"+r["scene"]["duplicate_axis"]+"/"+str(r["scene"]["role_permutation_index"]),
                         "bits": r["bits"], "arm": arm, "counterfactual_bits": cf, "episode_index": len(targets)-1})
        for drop in range(3):
            indices = [t for i, t in enumerate(r["evidence"]) if i != drop]
            a, y = samples.sample(r, q, indices)
            requests.append((a, y))
            rows.append({"id": r["id"], "quartet": r["quartet"], "family": r["family"], "condition": r["condition"],
                         "cell": r["condition"]+"/"+r["scene"]["duplicate_axis"]+"/"+str(r["scene"]["role_permutation_index"]),
                         "bits": r["bits"], "arm": "native_drop_"+str(drop), "drop": drop,
                         "counterfactual_bits": r["bits"], "episode_index": len(targets)-1})
    predictions = predict_rows(model, samples, requests)
    targets = np.stack(targets)
    for row, pred in zip(rows, predictions):
        truth = targets[row["episode_index"]].astype(np.float64)
        dist = np.mean((pred.astype(np.float64)[None]-truth)**2, axis=(1, 2))
        per_h = np.mean((pred.astype(np.float64)[None]-truth)**2, axis=2)
        own = BITS.index(tuple(row["bits"]))
        cf = BITS.index(tuple(row["counterfactual_bits"]))
        cls = int(dist.argmin())
        tied = np.flatnonzero(dist <= dist.min()+TOL).tolist()
        horizon_sets = [np.flatnonzero(per_h[:, h] <= per_h[:, h].min()+TOL).tolist() for h in range(2)]
        row.update(mse=float(dist[own]), class_distances=dist.tolist(), predicted_class=cls,
                   nearest_class_set=tied, predicted_bits=list(BITS[cls]), feature_class_correct=tied == [own],
                   counterfactual_class_correct=tied == [cf],
                   per_horizon_class=per_h.argmin(0).tolist(),
                   per_horizon_nearest_class_sets=horizon_sets,
                   mse_by_horizon=per_h[own].tolist())
    np.savez_compressed(out / "evaluation_predictions.npz", predictions=predictions, quartet_targets=targets)
    write(out / "evaluation_rows.json", rows)
    by_id = {r["id"]: {x["arm"]: x for x in rows if x["id"] == r["id"]} for r in dev}
    decisions = []
    for r in dev:
        zz = samples.z[r["rgb_indices"]]
        cand = zz[r["evidence"]]
        native = [by_id[r["id"]]["native_drop_"+str(i)] for i in range(3)]
        losses = np.array([n["mse"] for n in native])
        oracle = np.flatnonzero(losses <= losses.min()+TOL).tolist()
        exact_oracle = np.flatnonzero(losses == losses.min()).tolist()
        semantic = [i for i, axis in enumerate(r["semantic_axes"]) if r["semantic_axes"].count(axis) == 2]
        dp = diversity_choice(cand, r["evidence"], r["query"])
        with torch.inference_mode():
            p = torch.from_numpy(samples.p[np.array(r["rgb_indices"])[r["evidence"]]]).to(samples.device)
            current = torch.from_numpy(samples.p[r["rgb_indices"][r["query"]]]).to(samples.device)
            score = model.score(torch.cat([p, p-current], -1)).squeeze(-1)+torch.log1p((p-current).square().mean(-1))
            attention = torch.softmax(score, 0).cpu().numpy()
        picks = {"fifo": 0, "terminal_uniform": uniform_choice(r["family"]),
                 "visual_diversity": dp, "reader_attention": int(attention.argmin())}
        decisions.append({"id": r["id"], "quartet": r["quartet"], "family": r["family"], "condition": r["condition"],
                          "bits": r["bits"], "semantic_axes": r["semantic_axes"], "losses": losses.tolist(),
                          "cell": r["condition"]+"/"+r["scene"]["duplicate_axis"]+"/"+str(r["scene"]["role_permutation_index"]),
                          "oracle_min_delete_set": oracle, "semantic_correct_delete_set": semantic,
                          "oracle_exact_min_delete_set": exact_oracle,
                          "oracle_all_semantically_correct": set(oracle) <= set(semantic),
                          "oracle_any_semantically_correct": bool(set(oracle) & set(semantic)),
                          "attention": attention.tolist(), "choices": picks,
                          "policies": {k: {"drop": v, "regret": float(losses[v]-losses.min()),
                                             "mse": float(losses[v]), "semantic_correct": v in semantic,
                                             "feature_class_correct": native[v]["feature_class_correct"]} for k, v in picks.items()}})
    consistency = []
    for quartet in groups(records, "development"):
        rr = [d for d in decisions if d["quartet"] == quartet[0]["quartet"]]
        sets = [set(d["oracle_min_delete_set"]) for d in rr]
        exact_sets = [set(d["oracle_exact_min_delete_set"]) for d in rr]
        consistency.append({"quartet": quartet[0]["quartet"], "family": quartet[0]["family"],
                            "condition": quartet[0]["condition"], "same_exact_min_set": all(s == exact_sets[0] for s in exact_sets),
                            "same_tolerant_min_set": all(s == sets[0] for s in sets),
                            "common_exact_optimal_action": sorted(set.intersection(*exact_sets)),
                            "common_optimal_action": sorted(set.intersection(*sets)),
                            "all_four_oracles_semantic_subset": all(d["oracle_all_semantically_correct"] for d in rr)})
    summary = {}
    for condition in ("all",)+STYLES+tuple(sorted({r["cell"] for r in rows})):
        rr = [r for r in rows if condition == "all" or r["condition"] == condition or r["cell"] == condition]
        arms = {}
        for arm in CONTENT_ARMS:
            ar = [r for r in rr if r["arm"] == arm]
            arms[arm] = {"episodes": len(ar), "mse": float(np.mean([r["mse"] for r in ar])),
                         "feature_class_accuracy": float(np.mean([r["feature_class_correct"] for r in ar])),
                         "counterfactual_class_accuracy": float(np.mean([r["counterfactual_class_correct"] for r in ar])),
                         "classification_tie_fraction": float(np.mean([len(r["nearest_class_set"])>1 for r in ar])),
                         "bit_accuracy": [float(np.mean([len(r["nearest_class_set"]) == 1 and r["predicted_bits"][a] == r["bits"][a] for r in ar])) for a in range(2)]}
        dd = [r for r in decisions if condition == "all" or r["condition"] == condition or r["cell"] == condition]
        policies = {}
        for policy in ("fifo", "terminal_uniform", "visual_diversity", "reader_attention"):
            pp = [d["policies"][policy] for d in dd]
            regrets = [p["regret"] for p in pp]
            policies[policy] = {"mse": float(np.mean([p["mse"] for p in pp])),
                                "mean_regret": float(np.mean(regrets)), "positive_regret_fraction": float(np.mean(np.array(regrets)>TOL)),
                                "regret_quantiles_0_50_95_100": np.quantile(regrets, [0, .5, .95, 1]).tolist(),
                                "semantic_correct_fraction": float(np.mean([p["semantic_correct"] for p in pp])),
                                "feature_class_accuracy": float(np.mean([p["feature_class_correct"] for p in pp]))}
        summary[condition] = {"arms": arms, "policies": policies,
                              "oracle_mse": float(np.mean([min(d["losses"]) for d in dd])),
                              "oracle_semantic_subset_fraction": float(np.mean([d["oracle_all_semantically_correct"] for d in dd]))}
    checks = {}
    for condition in ("all",)+STYLES:
        ar = summary[condition]["arms"]
        checks.update({condition+"/AB_accuracy": ar["AB"]["feature_class_accuracy"] >= .8,
                       condition+"/AprimeB_accuracy": ar["AprimeB"]["feature_class_accuracy"] >= .8,
                       condition+"/wrongA_counterfactual": ar["wrongA"]["counterfactual_class_accuracy"] >= .8,
                       condition+"/wrongB_counterfactual": ar["wrongB"]["counterfactual_class_accuracy"] >= .8})
        for complete in ("AB", "AprimeB"):
            for single in ("Ablank", "Bblank"):
                checks[condition+"/"+complete+"_below_"+single] = ar[complete]["mse"] < ar[single]["mse"]-TOL
    qualified = all(checks.values())
    strong = {k: summary["all"]["policies"][k] for k in ("visual_diversity", "reader_attention")}
    strong_all_correct = all(p["semantic_correct_fraction"] == 1. for p in strong.values())
    strong_gaps = {k: p["positive_regret_fraction"] > 0 for k, p in strong.items()}
    if not qualified:
        decision = "stop_reader_gate_failed"
    elif not all(strong_gaps.values()):
        decision = "stop_a_strong_baseline_closes_local_oracle_gap"
    elif strong_all_correct:
        decision = "stop_semantic_problem_solved_residual_reader_gap"
    else:
        decision = "candidate_terminal_gap_requires_independent_confirmation"
    result = {"scope": "development_pilot", "summary": summary, "reader_gate_checks": checks,
              "reader_gate_pass": qualified, "strong_baselines_all_semantically_correct": strong_all_correct,
              "strong_baselines_have_positive_local_regret": strong_gaps,
              "decision": decision,
              "students_trained": False, "feature_classification_is_not_physical_decoding": True,
              "preuse_static_diagnostic": preuse,
              "quartet_consistency": consistency, "independent_cluster_unit": "family (both styles and all four siblings)",
              "saved_prediction_sha256": sha(out / "evaluation_predictions.npz")}
    write(out / "terminal_decisions.json", decisions)
    write(out / "summary.json", result)
    emit("completed", decision=result["decision"], reader_gate=checks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--validate-every", type=int, default=128)
    parser.add_argument("--encoder-batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.steps != 1024 or args.validate_every != 128:
        raise ValueError("Frozen pilot requires 1024 updates and validation every128")
    out = empty_directory(args.output)
    write(out / "protocol.json", protocol(args))
    from experiments.phase2.run_reader_scale_probe import configure_cuda
    hardware = configure_cuda()
    write(out / "hardware.json", hardware)
    start = time.monotonic()
    records, z, actions = generate_encode(args, out, "cuda:0")
    geometry = geometry_audit(records, z, out)
    if not geometry["feasibility_pass"]:
        write(out / "summary.json", {"decision": "stop_geometry_no_competition", "reader_fitted": False,
              "students_trained": False, "geometry": geometry["by_condition"]})
    else:
        basis = fit_basis(records, z, out, "cuda:0")
        model, samples = train_reader(args, records, z, actions, basis, out, "cuda:0")
        evaluate(model, samples, records, out)
    write(out / "manifest.json", {"status": "completed", "seconds": time.monotonic()-start,
          "files": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}})


if __name__ == "__main__":
    main()
