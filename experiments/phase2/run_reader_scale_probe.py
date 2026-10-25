"""Fixed-interface Gate reader scale pilot; export locally, run on one A100.

This is a diagnostic of the existing DynamicsPredictor, not a retention method
or a reproduction of the official Phase1 512-pair model. The 16/64/512 training
sets are nested, and all use the OLD 16-pair fitted feature transforms, the same
initialization, optimizer-update budget and new validation set. New evaluation
targets are opened only after every reader has been selected and saved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT, ROOT / "experiments/predictor/control/src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def empty_directory(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError("Require a new empty output directory: " + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_inputs(args):
    """Only this local packaging stage depends on the historical backup layout."""
    from experiments.phase2.portable_snapshot import PortableSnapshot
    from experiments.phase2.run_gate_followup_phase2 import FollowupGateEvaluator
    old = PortableSnapshot(args.snapshot)
    evaluator = FollowupGateEvaluator(old, args.followup)
    out = empty_directory(args.output)
    (out / "old_train").mkdir()
    records = []
    for r in old.records:
        if r["task"] != "gate" or r["split"] != "train":
            continue
        record = {k: r[k] for k in ("id", "pair", "sibling", "task", "split", "seed", "query", "frames")}
        record["feature_path"] = f"old_train/{r['id']}.npz"
        np.savez_compressed(out / record["feature_path"], z=old.features(r), actions=old.actions(r))
        records.append(record)
    if len(records) != 32 or len({r["pair"] for r in records}) != 16:
        raise ValueError("Expected the exact old 16 Gate training pairs")
    write(out / "old_train_records.json", records)
    np.savez_compressed(out / "normalization.npz", mean=old.norm["mean"], std=old.norm["std"])
    # Evaluator._basis intentionally exposes only input_mean/projection. The
    # freshly initialized reader also needs the matching frozen output basis.
    full_basis = args.followup / "gate/unit_columns/basis.npz"
    if sha(full_basis) != evaluator.binding["basis_sha256"]:
        raise ValueError("Full followed-up basis differs from verified binding")
    shutil.copy2(full_basis, out / "basis.npz")
    import torch
    torch.save({"model": evaluator._model.state_dict(), "role": "old_16_pair_epoch35_reference"}, out / "old_reader.pt")
    write(out / "model_config.json", old.config["model"])
    excluded = [{k: r[k] for k in ("id", "pair", "seed")} for r in old.all_records]
    for extra in args.exclude_records:
        excluded.extend({k: r[k] for k in ("id", "pair", "seed")} for r in read(extra))
    write(out / "excluded_identities.json", excluded)
    manifest = {"status": "verified_export", "old_train_pairs": 16,
                "old_reader_audit": evaluator.audit(),
                "representation": {"name": "facebook/ijepa_vith14_1k", "revision": "f157467ea509bc356ff9f61fd3c0d840eec5e04e",
                                   "pooling": "grid2", "dtype": "float32", "dimension": 5120},
                "fixed_transforms": "old train-only normalization and Gate unit-column basis; no refit",
                "files": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    write(out / "manifest.json", manifest)
    print(json.dumps({"exported": str(out), "pairs": 16, "files": len(manifest["files"])}), flush=True)


def verify_inputs(root):
    root = Path(root).resolve()
    m = read(root / "manifest.json")
    if m["status"] != "verified_export" or m["old_train_pairs"] != 16:
        raise ValueError("Wrong portable input export")
    for name, expected in m["files"].items():
        p = (root / name).resolve()
        if not p.is_relative_to(root) or sha(p) != expected:
            raise ValueError("Portable input checksum mismatch: " + name)
    return m


def configure_cuda():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from experiments.phase2.encode_fresh_gate_diagnostic import verify_a100
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return verify_a100()


def generate_and_encode(args, output, hardware):
    from experiments.phase2.encode_fresh_gate_diagnostic import (
        DATA_CONFIG, ENCODER_CONFIG, EXPECTED_ENCODER_METADATA_SHA256,
        verify_sources, digest,
    )
    from phase1.encoders import FrozenIJepa
    import torch
    envs, source_hashes = verify_sources(ROOT)
    data = output / "data"
    data.mkdir()
    (data / "features").mkdir()
    (data / "evaluation_only").mkdir()
    records = []
    for old in read(args.inputs / "old_train_records.json"):
        r = dict(old)
        r.update(legacy_training=True, condition="id", train_index=len(records) // 2)
        r["feature_path"] = f"features/{r['id']}.npz"
        shutil.copy2(args.inputs / old["feature_path"], data / r["feature_path"])
        records.append(r)
    # Stable sibling-preserving ordering makes 16 -> 64 -> 512 nested exactly.
    pairs = sorted({r["pair"] for r in records})
    for r in records:
        r["train_index"] = pairs.index(r["pair"])
    excluded = read(args.inputs / "excluded_identities.json")
    seen_seeds = {r["seed"] for r in excluded}
    seen_ids = {r["id"] for r in excluded}
    rgb, locations, payloads = [], {}, {}
    specs = [("train", "id", 20261025, 496), ("validation", "id", 20261026, 32)]
    specs += [("evaluation", cond, 20261027, 16) for cond in ("id", "long_gap", "distractor_shift")]
    for split, condition, data_seed, count in specs:
        legacy = {"id": "test_id", "long_gap": "test_long_gap", "distractor_shift": "test_distractor_shift"}[condition]
        if split in ("train", "validation"):
            legacy = split
        config = {"data": dict(DATA_CONFIG, seed=data_seed)}
        for index in range(count):
            scene = envs.scene(config, "gate", legacy, index)
            if scene["seed"] in seen_seeds:
                raise ValueError("New scene overlaps an old/new scene seed")
            seen_seeds.add(scene["seed"])
            pair = identity(["reader-scale-pilot-v1", data_seed, split, condition, index])
            ids = [identity([pair, j]) for j in (0, 1)]
            if set(ids) & seen_ids:
                raise ValueError("New episode ID collision")
            seen_ids.update(ids)
            episodes = [envs.episode(scene, lab) for lab in scene["label_order"]]
            q = scene["query"]
            if (not np.array_equal(episodes[0]["rgb"][q-3:q+1], episodes[1]["rgb"][q-3:q+1])
                    or not np.array_equal(episodes[0]["actions"], episodes[1]["actions"])
                    or np.array_equal(episodes[0]["rgb"][q+3], episodes[1]["rgb"][q+3])):
                raise ValueError("Sibling observation/action/future contract failed")
            for j, ep in enumerate(episodes):
                indices = []
                for frame in ep["rgb"]:
                    key = hashlib.sha256(frame.tobytes()).hexdigest()
                    if key not in locations:
                        locations[key] = len(rgb)
                        rgb.append(frame.copy())
                    indices.append(locations[key])
                r = {"id": ids[j], "pair": pair, "sibling": ids[1-j], "task": "gate", "split": split,
                     "seed": int(scene["seed"]), "data_seed": data_seed, "condition": condition,
                     "query": q, "frames": scene["length"], "legacy_training": False,
                     "train_index": index + 16 if split == "train" else None,
                     "rgb_indices": indices, "feature_path": f"features/{ids[j]}.npz",
                     "eval_path": f"evaluation_only/{ids[j]}.npz"}
                records.append(r)
                payloads[r["id"]] = ep["actions"]
                np.savez_compressed(data / r["eval_path"], evidence=ep["evidence"], label=ep["label"],
                                    states=ep["states"], classes=ep["classes"])
    write(data / "records.json", records)
    bank = np.stack(rgb)
    np.savez_compressed(data / "unique_rgb.npz", rgb=bank)
    config = dict(ENCODER_CONFIG, cache_dir=str(args.cache_dir.resolve()), batch_size=args.encoder_batch_size)
    write(data / "encoder_config.json", config)
    encoder = FrozenIJepa(config, torch.device("cuda:0"))
    if digest(encoder.metadata) != EXPECTED_ENCODER_METADATA_SHA256:
        raise ValueError("Pinned encoder/processor feature definition changed")
    if any(p.requires_grad for p in encoder.model.parameters()) or any(m.training for m in encoder.model.modules()):
        raise ValueError("Encoder must remain frozen/eval")
    encoded = []
    start = time.monotonic()
    for at in range(0, len(bank), args.encoder_batch_size):
        value = encoder.encode(bank[at:at + args.encoder_batch_size])
        if value.dtype != np.float32 or value.shape[1] != 5120 or not np.isfinite(value).all():
            raise ValueError("Bad official encoder output")
        encoded.append(value)
        if at % (args.encoder_batch_size * 50) == 0:
            print(json.dumps({"stage": "encode", "done": at + len(value), "total": len(bank)}), flush=True)
    raw = np.concatenate(encoded)
    with np.load(args.inputs / "normalization.npz", allow_pickle=False) as norm:
        normalized = ((raw - norm["mean"]) / norm["std"]).astype(np.float32)
    np.savez_compressed(data / "unique_features.npz", raw=raw, normalized=normalized)
    for r in records:
        if not r["legacy_training"]:
            np.savez_compressed(data / r["feature_path"], z=normalized[r["rgb_indices"]], actions=payloads[r["id"]])
    manifest = {"status": "encoded", "scope": "reader-scale-pilot", "hardware": hardware,
                "source_sha256": source_hashes, "encoder_metadata_sha256": digest(encoder.metadata),
                "encoder_metadata": encoder.metadata, "new_unique_frames": len(bank),
                "encoding_seconds": time.monotonic() - start,
                "pairs": {s: len({r["pair"] for r in records if r["split"] == s}) for s in ("train", "validation", "evaluation")},
                "files": {str(p.relative_to(data)): sha(p) for p in data.rglob("*") if p.is_file()}}
    write(data / "manifest.json", manifest)
    del encoder, encoded, raw, normalized, bank, rgb
    torch.cuda.empty_cache()
    return records


class Samples:
    """Only train/validation observations enter optimization/selection."""
    def __init__(self, data, records, basis):
        from experiments.phase2.training_samples import EpisodeIdentity, FeatureEpisode, SplitRegistry
        from phase1.model import project
        self.records = [r for r in records if r["split"] in ("train", "validation")]
        self.registry = SplitRegistry([EpisodeIdentity.from_record(r) for r in self.records])
        self.episodes, self.projected = {}, {}
        self.basis = basis
        for r in self.records:
            with np.load(data / r["feature_path"], allow_pickle=False) as f:
                z, actions = f["z"].copy(), f["actions"].copy()
            self.episodes[r["id"]] = FeatureEpisode(EpisodeIdentity.from_record(r), z, actions, "additive_controls", "fixed-old-ijepa")
            self.projected[r["id"]] = project(z, basis)
        self.validation = {}

    def batch(self, records, mode, draw, training):
        from experiments.phase2.training_samples import SamplingConfig, sample_predictor, phase1_arrays
        from phase1.inputs import collate
        import torch
        arrays, targets = [], []
        for r in records:
            s = sample_predictor(self.episodes[r["id"]], self.registry,
                                 SamplingConfig(memory_dropout=.15 if training else 0.),
                                 query=r["query"], base_seed=17, epoch=draw, expected_split=r["split"], history_mode=mode)
            a, y = phase1_arrays(s)
            times = [e.timestamp for e in s.causal.recent + s.causal.history]
            a["projected_z"] = self.projected[r["id"]][times]
            a["projected_current"] = self.projected[r["id"]][r["query"]]
            arrays.append(a)
            targets.append(y)
        return collate(arrays, "cuda:0"), torch.from_numpy(np.stack(targets)).cuda()

    def evaluate_validation(self, model):
        from experiments.phase2.training_samples import DEFAULT_MODES
        import torch
        model.eval()
        rows = [r for r in self.records if r["split"] == "validation"]
        values = {}
        with torch.inference_mode():
            for mode in DEFAULT_MODES:
                if mode not in self.validation:
                    self.validation[mode] = self.batch(rows, mode, 0, False)
                batch, target = self.validation[mode]
                values[mode] = float((model(batch) - target).square().mean())
        return {"equal_mode_mse": float(np.mean(list(values.values()))), "by_mode": values}


def train_readers(args, output, records):
    import torch
    from phase1.model import DynamicsPredictor
    from phase1.training import regression_loss
    from phase1.utils import seed_all
    from experiments.phase2.training_samples import DEFAULT_MODES
    with np.load(args.inputs / "basis.npz", allow_pickle=False) as f:
        basis = {k: f[k].copy() for k in f.files}
    config = read(args.inputs / "model_config.json")
    samples = Samples(output / "data", records, basis)
    all_pairs = {}
    for r in samples.records:
        if r["split"] == "train":
            all_pairs.setdefault(r["train_index"], []).append(r)
    if sorted(all_pairs) != list(range(512)) or any(len(pair) != 2 for pair in all_pairs.values()):
        raise ValueError("Non-nested or incomplete training pairs")
    results, initial_hash = [], None
    for count in (16, 64, 512):
        seed_all(args.model_seed, 4)
        model = DynamicsPredictor(5120, config, (3, 6), basis).cuda()
        h = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
        state_hash = h.hexdigest()
        if initial_hash is not None and state_hash != initial_hash:
            raise ValueError("Scale arms do not share exactly identical initialization")
        initial_hash = state_hash
        arm = output / f"pairs-{count}"
        arm.mkdir()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.steps // 3, 2 * args.steps // 3], gamma=.5)
        rng = np.random.default_rng(args.model_seed)
        best = float("inf")
        selected = None
        curve = []
        encountered = set()
        began = time.monotonic()
        for step in range(args.steps + 1):
            if step % args.validate_every == 0 or step == args.steps:
                validation = samples.evaluate_validation(model)
                curve.append({"step": step, **validation})
                if validation["equal_mode_mse"] < best:
                    best, selected = validation["equal_mode_mse"], step
                    torch.save({"model": model.state_dict(), "step": step, "train_pairs": count,
                                "validation": validation, "basis_sha256": sha(args.inputs / "basis.npz"),
                                "normalization_sha256": sha(args.inputs / "normalization.npz")}, arm / "selected.pt")
                write(arm / "learning_curve.json", curve)
                print(json.dumps({"stage": "train", "pairs": count, "step": step, "validation": validation["equal_mode_mse"]}), flush=True)
            if step == args.steps:
                break
            chosen = rng.choice(count, 16, replace=False)
            encountered.update(map(int, chosen))
            rows = [r for index in chosen for r in all_pairs[int(index)]]
            mode = DEFAULT_MODES[step % len(DEFAULT_MODES)]
            batch, target = samples.batch(rows, mode, step // len(DEFAULT_MODES), True)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = regression_loss(model(batch), target, 2., .5)[0]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); scheduler.step()
        result = {"train_pairs": count, "seed": args.model_seed, "steps": args.steps,
                  "selected_step": selected, "best_validation_mse": best,
                  "training_pair_draws": args.steps * 16,
                  "equivalent_dataset_passes": args.steps * 16 / count,
                  "unique_training_pairs_seen": len(encountered), "initial_state_sha256": initial_hash,
                  "selected_sha256": sha(arm / "selected.pt"), "elapsed_seconds": time.monotonic() - began}
        write(arm / "summary.json", result)
        results.append(result)
        del model, optimizer
        torch.cuda.empty_cache()
    write(output / "readers_locked.json", {"scope": "pilot", "arms": results,
          "evaluation_targets_opened_during_training": False, "basis_refitted": False,
          "normalization_refitted": False})
    return basis, config


def evaluate_readers(args, output, records, basis, config):
    import torch
    from phase1.model import DynamicsPredictor, project
    from phase1.inputs import collate
    from experiments.phase2.diagnose_gate_oracle_metric import paired_components
    if not (output / "readers_locked.json").exists():
        raise ValueError("Every scale reader must be selected before evaluation")
    data = output / "data"
    eval_records = [r for r in records if r["split"] == "evaluation"]
    by_id = {r["id"]: r for r in eval_records}
    episodes = {}
    for r in eval_records:
        with np.load(data / r["feature_path"], allow_pickle=False) as f:
            episodes[r["id"]] = (f["z"].copy(), f["actions"].copy())
    with np.load(args.inputs / "normalization.npz", allow_pickle=False) as f:
        std = f["std"].copy()
    rows, prediction_vectors = [], []
    model_paths = [(f"scale_{count}", output / f"pairs-{count}/selected.pt") for count in (16, 64, 512)]
    model_paths += [("old_16_reference", args.inputs / "old_reader.pt")]
    for name, checkpoint in model_paths:
        model = DynamicsPredictor(5120, config, (3, 6), basis).cuda()
        model.load_state_dict(torch.load(checkpoint, map_location="cuda:0", weights_only=False)["model"], strict=True)
        model.eval().requires_grad_(False)
        for r in eval_records:
            z, actions = episodes[r["id"]]
            sibling_z, sibling_actions = episodes[r["sibling"]]
            q = r["query"]
            if not np.array_equal(actions, sibling_actions):
                raise ValueError("Sibling actions differ")
            with np.load(data / r["eval_path"], allow_pickle=False) as f:
                cue = int(f["evidence"][0])
            if cue >= q - 4:
                raise ValueError("Cue/filler overlap")
            controls = {"recent": (), "no_cue_K2": (q-5, q-4), "cue_K2": (cue, q-4),
                        "wrong_cue_K2": (cue, q-4), "full_history": tuple(range(q-3))}
            for control, kept in controls.items():
                times = np.asarray(list(range(q-3, q+1)) + list(kept))
                values = z[times].copy()
                if control == "wrong_cue_K2":
                    values[4] = sibling_z[cue]
                prefix = np.vstack([np.zeros((1, 2), np.float32), np.cumsum(actions[:q], axis=0)])
                sample = {"z": values, "current": z[q], "age": (q-times).astype(np.float32),
                          "type": np.r_[np.zeros(4, np.int64), np.ones(len(kept), np.int64)],
                          "past_actions": prefix[q]-prefix[times], "actions": actions[q:q+6],
                          "projected_z": project(values, basis), "projected_current": project(z[q], basis)}
                with torch.inference_mode():
                    pred = model(collate([sample], "cuda:0"))[0].cpu().numpy()
                target, other = z[q+np.array([3, 6])], sibling_z[q+np.array([3, 6])]
                metrics = paired_components(pred, target, other)
                prediction_vectors.append(pred.copy())
                rows.append({"reader": name, "episode_id": r["id"], "pair": r["pair"], "condition": r["condition"],
                             "control": control, "query_first_action": float(actions[q, 0]),
                             "kept_times": list(kept), **metrics,
                             "raw_feature_mse": float(np.mean(((pred-target)*std)**2))})
        del model
    write(output / "evaluation_rows.json", rows)
    np.savez_compressed(output / "evaluation_predictions.npz", predictions=np.stack(prediction_vectors))
    summary = {"scope": "reader-scale single-seed pilot; not registered confirmation",
               "metric_note": "feature_branch_correct is sibling-contrast alpha>0, not decoded physical lane accuracy",
               "rows": len(rows), "pairs": len({r["pair"] for r in eval_records}), "arms": {}}
    for name, _ in model_paths:
        summary["arms"][name] = {}
        for control in ("recent", "no_cue_K2", "cue_K2", "wrong_cue_K2", "full_history"):
            selected = [r for r in rows if r["reader"] == name and r["control"] == control]
            summary["arms"][name][control] = {k: float(np.mean([r[k] for r in selected]))
                                             for k in ("mse", "raw_feature_mse", "feature_branch_correct", "contrast_mse")}
    write(output / "evaluation_summary.json", summary)
    print(json.dumps(summary), flush=True)


def run(args):
    args.inputs = args.inputs.resolve()
    input_manifest = verify_inputs(args.inputs)
    output = empty_directory(args.output)
    hardware = configure_cuda()
    protocol = {"version": "reader-scale-pilot-v1", "hardware": hardware, "train_pairs": [16, 64, 512],
                "train_expansion_seed": 20261025, "validation_seed": 20261026, "evaluation_seed": 20261027,
                "validation_pairs": 32, "evaluation_pairs": 48, "model_seed": args.model_seed,
                "steps_per_arm": args.steps, "pairs_per_update": 16, "validation_every": args.validate_every,
                "selection": "earliest minimum new-validation equal-six-mode feature MSE including step zero",
                "inputs_manifest_sha256": sha(args.inputs / "manifest.json"),
                "source_sha256": sha(__file__), "fixed_transforms": input_manifest["fixed_transforms"],
                "limitations": ["single training seed pilot", "same old limited transform subspace",
                                "sample-size and coverage improve together; not an architecture comparison",
                                "evaluations are diagnostic controls, not new online-policy results",
                                "full_history is over budget; wrong cue is evaluator-only"]}
    write(output / "protocol.json", protocol)
    records = generate_and_encode(args, output, hardware)
    basis, config = train_readers(args, output, records)
    evaluate_readers(args, output, records, basis, config)
    write(output / "completion.json", {"status": "completed", "hardware": hardware,
          "files": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("export")
    package.add_argument("--snapshot", type=Path, required=True)
    package.add_argument("--followup", type=Path, required=True)
    package.add_argument("--exclude-records", type=Path, nargs="*", default=[])
    package.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--inputs", type=Path, required=True)
    execute.add_argument("--cache-dir", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--steps", type=int, default=600)
    execute.add_argument("--validate-every", type=int, default=100)
    execute.add_argument("--model-seed", type=int, default=17)
    execute.add_argument("--encoder-batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.command == "export":
        export_inputs(args)
    else:
        if args.steps < 6 or args.validate_every < 1 or args.model_seed != 17 or not 1 <= args.encoder_batch_size <= 16:
            parser.error("Pilot requires positive update/evaluation counts, fixed seed17 and batch size1..16")
        run(args)


if __name__ == "__main__":
    main()
