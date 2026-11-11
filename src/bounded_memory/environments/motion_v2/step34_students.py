"""Frozen Motion student recipe adapted from the 10/26 terminal students.

This module performs no work on import.  The public fit entry point fixes the
registered full budget; it has no short-run, resume, or checkpoint-replace mode.
Only observed candidate items and common recent context enter the scorer.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


ITEM_DIM = 259
CONTEXT_DIM = 524
SEEDS = (17, 29, 43)
ARCHITECTURES = ("independent", "set")
RAW_LABELS = ("local_feature", "local_position", "use_time_feature",
              "use_time_position", "future_attention")
SHUFFLE_SOURCES = {"shuffle_" + name: name for name in RAW_LABELS[:-1]}
LABELS = RAW_LABELS + tuple(SHUFFLE_SOURCES)
STEPS = 1000
BATCH_SIZE = 128
VALIDATE_EVERY = 100
VALIDATION_BATCH_SIZE = 512
TRAIN_ROWS = 16384
VALIDATION_ROWS = 3072
SHUFFLE_SEED = 20261108
BATCH_SEED_OFFSET = 300000
INPUT_STD_FLOOR = 1e-4
LABEL_SCALE_FLOOR = 1e-8
TOTAL_UPDATES = len(ARCHITECTURES) * len(LABELS) * len(SEEDS) * STEPS
CHECKPOINT_RULE = "own_validation_label_cross_entropy_strict_min_earliest_tie_including_step0"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_record(value):
    a = np.ascontiguousarray(value)
    header = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(header + b"\0" + a.tobytes()).hexdigest()
    return {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": digest,
            "hash_format": "canonical_dtype_shape_json + NUL + contiguous_C_bytes"}


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _write_npz(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def validate_inputs(inputs, expected_rows=None):
    if not isinstance(inputs, dict) or set(inputs) != {"items", "context"}:
        raise ValueError("Only items and context are scorer inputs")
    items = np.asarray(inputs["items"])
    context = np.asarray(inputs["context"])
    if items.ndim != 3 or items.shape[1:] != (3, ITEM_DIM):
        raise ValueError("items must have shape [N,3,259]")
    if not len(items) or context.shape != (len(items), CONTEXT_DIM):
        raise ValueError("context must have shape [N,524] with N positive")
    if expected_rows is not None and len(items) != expected_rows:
        raise ValueError("Input row count differs from the frozen state manifest")
    result = {}
    for key, value in (("items", items), ("context", context)):
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError("Inputs must be finite numeric arrays")
        result[key] = np.asarray(value, dtype=np.float32)
        if not np.isfinite(result[key]).all():
            raise ValueError("Inputs exceed the FP32 range")
    return result


def validate_labels(labels, rows):
    if not isinstance(labels, dict) or set(labels) != set(RAW_LABELS):
        raise ValueError("Exactly five registered raw label arrays are required")
    result = {}
    for name in RAW_LABELS:
        value = np.asarray(labels[name])
        if value.shape != (rows, 3) or value.dtype.kind not in "fiu":
            raise ValueError("Each raw label must be numeric [N,3]")
        value = np.asarray(value, np.float64)
        if not np.isfinite(value).all() or (value < 0).any():
            raise ValueError("Raw losses and salience must be finite and nonnegative")
        result[name] = value
    attention = result["future_attention"]
    if (attention > 1).any() or not np.allclose(attention.sum(1), 1., rtol=0., atol=1e-6):
        raise ValueError("future_attention must contain three-candidate probabilities")
    return result


def _validate_ids(ids, rows):
    result = list(ids)
    if len(result) != rows or any(not isinstance(x, str) or not x for x in result):
        raise ValueError("Every row requires a nonempty stable parent record ID")
    if len(set(result)) != rows:
        raise ValueError("Parent record IDs must be unique; duplicate states keep distinct record IDs")
    return result


def fit_normalization(train_inputs):
    values = validate_inputs(train_inputs)
    result = {}
    for key, axes in (("items", (0, 1)), ("context", 0)):
        array = values[key].astype(np.float64)
        result[key + "_mean"] = array.mean(axis=axes).astype(np.float32)
        result[key + "_std"] = np.maximum(array.std(axis=axes), INPUT_STD_FLOOR).astype(np.float32)
    return result


def _validate_normalization(normalization):
    names = {kind + suffix for kind in ("items", "context") for suffix in ("_mean", "_std")}
    if set(normalization) != names:
        raise ValueError("Unexpected normalization fields")
    for kind, dim in (("items", ITEM_DIM), ("context", CONTEXT_DIM)):
        for suffix in ("_mean", "_std"):
            array = np.asarray(normalization[kind + suffix])
            if array.shape != (dim,) or not np.isfinite(array).all():
                raise ValueError("Invalid normalization shape or values")
            if suffix == "_std" and (array < np.float32(INPUT_STD_FLOOR)).any():
                raise ValueError("Normalization standard deviation below the fixed floor")


def normalize_inputs(inputs, normalization):
    """Apply the saved train-only transform to observed inputs, including online."""
    values = validate_inputs(inputs)
    _validate_normalization(normalization)
    result = {key: ((value - normalization[key + "_mean"]) /
                    normalization[key + "_std"]).astype(np.float32)
              for key, value in values.items()}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite normalized student inputs")
    return result


def parameter_formula(architecture, width):
    if architecture not in ARCHITECTURES:
        raise ValueError("Unknown architecture")
    return (ITEM_DIM + CONTEXT_DIM) * width + (1 if architecture == "independent" else 2) * width**2 + 3 * width + 1


def architecture_plan():
    independent = parameter_formula("independent", 64)
    width = min(range(1, 65), key=lambda w: (abs(parameter_formula("set", w) - independent), w))
    deepsets = parameter_formula("set", width)
    if (width, independent, deepsets) != (60, 54401, 54361):
        raise ValueError("Registered parameter matching changed")
    return {"independent": {"width": 64, "parameters": independent},
            "set": {"width": width, "parameters": deepsets}}


class TerminalScorer(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError("Unknown architecture")
        self.architecture = architecture
        width = architecture_plan()[architecture]["width"]
        self.phi = nn.Sequential(nn.Linear(ITEM_DIM + CONTEXT_DIM, width), nn.ReLU())
        head_width = width if architecture == "independent" else 2 * width
        self.head = nn.Sequential(nn.Linear(head_width, width), nn.ReLU(), nn.Linear(width, 1))
        if sum(p.numel() for p in self.parameters()) != parameter_formula(architecture, width):
            raise ValueError("Scorer parameter count differs from registration")

    def forward(self, items, context):
        if items.ndim != 3 or items.shape[1:] != (3, ITEM_DIM):
            raise ValueError("Scorer requires items [N,3,259]")
        if context.shape != (len(items), CONTEXT_DIM):
            raise ValueError("Scorer requires common context [N,524]")
        local = torch.cat((items, context[:, None].expand(-1, 3, -1)), dim=-1)
        hidden = self.phi(local)
        if self.architecture == "set":
            hidden = torch.cat((hidden, hidden.mean(1, keepdim=True).expand(-1, 3, -1)), dim=-1)
        return self.head(hidden).squeeze(-1)


def initialized_student(architecture, seed, device="cpu"):
    if seed not in SEEDS:
        raise ValueError("Unregistered training seed")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TerminalScorer(architecture)
    return model.to(device=device, dtype=torch.float32)


def training_label_scales(train_labels):
    first = np.asarray(train_labels[RAW_LABELS[0]])
    labels = validate_labels(train_labels, len(first))
    scales = {}
    for name, values in labels.items():
        centered = values - values.mean(1, keepdims=True)
        scale = max(float(np.sqrt(np.mean(centered**2))), LABEL_SCALE_FLOOR)
        if not math.isfinite(scale):
            raise ValueError("Nonfinite train-only label scale")
        scales[name] = scale
    scales.update({name: scales[source] for name, source in SHUFFLE_SOURCES.items()})
    return scales


def stable_parent_permutations(ids):
    result = []
    for parent in ids:
        digest = hashlib.sha256(json.dumps([SHUFFLE_SEED, parent], separators=(",", ":")).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        result.append(rng.permutation(3))
    return np.asarray(result, dtype=np.int64).reshape(-1, 3)


def soft_targets(raw_labels, ids, scales):
    labels = validate_labels(raw_labels, len(ids))
    _validate_ids(ids, len(ids))
    if set(scales) != set(LABELS) or any(not math.isfinite(float(x)) or x < LABEL_SCALE_FLOOR for x in scales.values()):
        raise ValueError("All nine fixed positive train label scales are required")
    if any(scales[name] != scales[source] for name, source in SHUFFLE_SOURCES.items()):
        raise ValueError("Shuffle must share the corresponding unshuffled train scale")
    permutations = stable_parent_permutations(ids)
    labels.update({name: np.take_along_axis(labels[source], permutations, axis=1)
                   for name, source in SHUFFLE_SOURCES.items()})
    targets = {}
    for name, values in labels.items():
        logits = -(values - values.mean(1, keepdims=True)) / scales[name]
        logits -= logits.max(1, keepdims=True)
        exp = np.exp(logits)
        targets[name] = (exp / exp.sum(1, keepdims=True)).astype(np.float32)
        if not np.isfinite(targets[name]).all():
            raise ValueError("Nonfinite soft supervision")
    return targets, permutations


def minibatch_indices(seed, rows):
    if seed not in SEEDS or not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
        raise ValueError("Registered seed and positive row count required")
    generator = torch.Generator(device="cpu").manual_seed(seed + BATCH_SEED_OFFSET)
    return torch.randint(rows, (STEPS, BATCH_SIZE), generator=generator).numpy()


def soft_cross_entropy(logits, targets):
    return -(targets * torch.log_softmax(logits, dim=-1)).sum(-1)


def _validation(model, inputs, targets):
    model.eval()
    total = 0.
    with torch.inference_mode():
        for at in range(0, len(targets), VALIDATION_BATCH_SIZE):
            end = min(at + VALIDATION_BATCH_SIZE, len(targets))
            logits = model(inputs["items"][at:end], inputs["context"][at:end])
            total += float(soft_cross_entropy(logits, targets[at:end]).double().sum().item())
    result = total / len(targets)
    if not math.isfinite(result):
        raise ValueError("Nonfinite validation own-label cross entropy")
    return result


def checkpoint_selection(best_ce, selected_step, candidate_ce, step):
    """Pure selection helper: ties never displace the earliest checkpoint."""
    if not math.isfinite(candidate_ce) or step < 0:
        raise ValueError("Finite validation CE and nonnegative step required")
    if candidate_ce < best_ce:
        return candidate_ce, step, True
    return best_ce, selected_step, False


def checkpoint_flags(selected_step):
    if not isinstance(selected_step, int) or isinstance(selected_step, bool) or selected_step not in range(0, STEPS + 1, VALIDATE_EVERY):
        raise ValueError("Selected step must be a registered evaluation checkpoint")
    return {"selected_step": selected_step, "is_trained": selected_step > 0,
            "training_updates_executed": STEPS, "step0_in_selection": True,
            "checkpoint_rule": CHECKPOINT_RULE}


def _state_record(state):
    records = {key: _array_record(value.detach().cpu().numpy()) for key, value in state.items()}
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "tensors": records}


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def fit_students(train_inputs, val_inputs, train_labels, val_labels, train_ids,
                 val_ids, output, device, budget):
    """Run exactly 54 registered fits; budget is a required keyword callable.

    budget(stage='students', completed=int, total=54000) may raise to stop.
    Its exception is never suppressed or converted into a shorter experiment.
    """
    if not callable(budget):
        raise ValueError("A live allocation budget callback is required")
    train = validate_inputs(train_inputs, TRAIN_ROWS)
    validation = validate_inputs(val_inputs, VALIDATION_ROWS)
    raw_train = validate_labels(train_labels, TRAIN_ROWS)
    raw_validation = validate_labels(val_labels, VALIDATION_ROWS)
    train_ids = _validate_ids(train_ids, TRAIN_ROWS)
    val_ids = _validate_ids(val_ids, VALIDATION_ROWS)
    if set(train_ids) & set(val_ids):
        raise ValueError("Train and validation parent IDs overlap")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("Student output must be a new directory")
    destination.mkdir(parents=True, exist_ok=False)
    completed, results = 0, {}
    started = time.perf_counter()
    try:
        budget(stage="students", completed=completed, total=TOTAL_UPDATES)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        source_sha = _sha(__file__)
        normalization = fit_normalization(train)
        _write_npz(destination / "normalization.npz", normalization)
        normalization_sha = _sha(destination / "normalization.npz")
        _write_json(destination / "source_ids.json", {"train": train_ids, "validation": val_ids,
                    "usage": "offline parent identity, fixed shuffle and provenance only; never scorer input"})
        scales = training_label_scales(raw_train)
        _write_json(destination / "label_scales.json", {"fit_split": "train", "floor": LABEL_SCALE_FLOOR,
                    "formula": "sqrt(mean_over_train_rows_and_candidates((value-parent_mean)^2))",
                    "shuffle_sources": SHUFFLE_SOURCES, "scales": scales})
        train_targets, train_permutations = soft_targets(raw_train, train_ids, scales)
        val_targets, val_permutations = soft_targets(raw_validation, val_ids, scales)
        supervision = {"train_permutations": train_permutations, "validation_permutations": val_permutations}
        for split, labels, targets in (("train", raw_train, train_targets), ("validation", raw_validation, val_targets)):
            supervision.update({split + "_raw_" + name: value for name, value in labels.items()})
            supervision.update({split + "_target_" + name: value for name, value in targets.items()})
        _write_npz(destination / "supervision_audit.npz", supervision)
        batches = {seed: minibatch_indices(seed, TRAIN_ROWS) for seed in SEEDS}
        _write_npz(destination / "minibatch_indices.npz", {"seed" + str(seed): value for seed, value in batches.items()})
        provenance = {"source_file": Path(__file__).name, "source_sha256": source_sha,
            "source_ids_sha256": _sha(destination / "source_ids.json"),
            "normalization_sha256": normalization_sha, "normalization_fit_split": "train",
            "normalization_std_floor": INPUT_STD_FLOOR,
            "label_scales_sha256": _sha(destination / "label_scales.json"),
            "supervision_audit_sha256": _sha(destination / "supervision_audit.npz"),
            "minibatch_indices_sha256": _sha(destination / "minibatch_indices.npz"),
            "raw_input_arrays": {split: {name: _array_record(value) for name, value in data.items()}
                                 for split, data in (("train", train), ("validation", validation))},
            "raw_label_arrays": {split: {name: _array_record(value) for name, value in data.items()}
                                 for split, data in (("train", raw_train), ("validation", raw_validation))},
            "architecture_plan": architecture_plan(), "labels": list(LABELS), "seeds": list(SEEDS),
            "train_rows": TRAIN_ROWS, "validation_rows": VALIDATION_ROWS,
            "same_architecture_seed_identical_initialization": True,
            "same_seed_identical_minibatch_indices_all_architectures_and_labels": True,
            "cross_architecture_identical_weights_claimed": False,
            "duplicate_states_keep_manifest_multiplicity": True,
            "optimizer": {"name": "Adam", "learning_rate": .001, "weight_decay": 0., "steps": STEPS,
                          "batch_size": BATCH_SIZE, "gradient_clip": 1., "scheduler": None, "dropout": 0.},
            "validation_batch_size": VALIDATION_BATCH_SIZE, "validate_every": VALIDATE_EVERY,
            "checkpoint_rule": CHECKPOINT_RULE, "device": str(torch.device(device)),
            "torch_version": str(torch.__version__), "numpy_version": np.__version__}
        _write_json(destination / "provenance.json", provenance)
        provenance_sha = _sha(destination / "provenance.json")
        tensor_inputs = {split: {name: torch.as_tensor(value, device=device, dtype=torch.float32)
                               for name, value in normalize_inputs(data, normalization).items()}
                         for split, data in (("train", train), ("validation", validation))}
        tensor_targets = {split: {name: torch.as_tensor(value, device=device, dtype=torch.float32)
                                for name, value in targets.items()}
                          for split, targets in (("train", train_targets), ("validation", val_targets))}
        tensor_batches = {seed: torch.as_tensor(value, device=device, dtype=torch.long) for seed, value in batches.items()}
        initial_records = {}
        for architecture in ARCHITECTURES:
            for label in LABELS:
                for seed in SEEDS:
                    budget(stage="students", completed=completed, total=TOTAL_UPDATES)
                    model = initialized_student(architecture, seed, device)
                    initial = _state_record(model.state_dict())
                    identity = (architecture, seed)
                    if identity in initial_records and initial_records[identity] != initial:
                        raise ValueError("Matched label arms do not share exact initialization")
                    initial_records[identity] = initial
                    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                    best_ce, selected_step, best_state = float("inf"), None, None
                    history, training_losses = [], []
                    _synchronize(device)
                    arm_started = time.perf_counter()
                    for step in range(STEPS + 1):
                        if step % VALIDATE_EVERY == 0:
                            ce = _validation(model, tensor_inputs["validation"], tensor_targets["validation"][label])
                            best_ce, selected_step, improved = checkpoint_selection(best_ce, selected_step, ce, step)
                            if improved:
                                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                            history.append({"step": step, "own_validation_label_cross_entropy": ce,
                                            "became_strict_best": improved, "is_trained": step > 0})
                            _synchronize(device)
                            budget(stage="students", completed=completed, total=TOTAL_UPDATES)
                        if step == STEPS:
                            break
                        model.train()
                        index = tensor_batches[seed][step]
                        logits = model(tensor_inputs["train"]["items"][index], tensor_inputs["train"]["context"][index])
                        loss = soft_cross_entropy(logits, tensor_targets["train"][label][index]).mean()
                        if not bool(torch.isfinite(loss).item()):
                            raise ValueError("Nonfinite student training loss")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        completed += 1
                        training_losses.append({"step": step + 1, "soft_cross_entropy": float(loss.detach().item()),
                                                "gradient_norm_before_clip": float(norm.item())})
                    _synchronize(device)
                    arm_seconds = time.perf_counter() - arm_started
                    name = f"{architecture}_{label}_seed{seed}"
                    metadata = {"architecture": architecture, "label": label, "seed": seed,
                        **checkpoint_flags(selected_step), "selected_validation_label_cross_entropy": best_ce,
                        "parameters": architecture_plan()[architecture]["parameters"],
                        "normalization_file": "normalization.npz", "normalization_sha256": normalization_sha,
                        "provenance_file": "provenance.json", "provenance_sha256": provenance_sha,
                        "source_sha256": source_sha, "target_scale": scales[label],
                        "initial_state": initial, "selected_state": _state_record(best_state),
                        "minibatch_indices": _array_record(batches[seed]), "training_seconds": arm_seconds,
                        "scorer_inputs": ["items", "context"], "targets_enter_forward": False}
                    with (destination / (name + ".pt")).open("xb") as stream:
                        torch.save({"model": best_state, "metadata": metadata}, stream)
                    _write_json(destination / (name + "_history.json"), {
                        "metadata": metadata, "validation": history, "training": training_losses})
                    results[name] = {**metadata, "checkpoint": name + ".pt",
                        "checkpoint_sha256": _sha(destination / (name + ".pt")),
                        "history_file": name + "_history.json", "history_sha256": _sha(destination / (name + "_history.json"))}
                    budget(stage="students", completed=completed, total=TOTAL_UPDATES)
        if completed != TOTAL_UPDATES or len(results) != 54:
            raise ValueError("Registered full student work was not completed")
        summary = {"status": "completed", "arms": results, "completed_updates": completed,
                   "total_updates": TOTAL_UPDATES, "arm_count": len(results),
                   "provenance_sha256": provenance_sha, "elapsed_seconds": time.perf_counter() - started,
                   "step0_selected_arms": [name for name, result in results.items() if not result["is_trained"]],
                   "test_or_confirmation_inputs_used": False, "resume_or_retry_supported": False}
        _write_json(destination / "summary.json", summary)
        return summary
    except Exception as error:
        _write_json(destination / "stopped.json", {"status": "stopped", "completed_updates": completed,
                    "completed_arms": list(results), "error_type": type(error).__name__, "error": str(error),
                    "elapsed_seconds": time.perf_counter() - started, "resume_authorized": False})
        raise


def _checkpoint_metadata_entries(summary):
    """Validate the full immutable fit summary without reading any files."""
    if not isinstance(summary, dict) or summary.get("status") != "completed":
        raise ValueError("Online evaluation requires a completed student summary")
    if summary.get("completed_updates") != TOTAL_UPDATES or summary.get("total_updates") != TOTAL_UPDATES or summary.get("arm_count") != 54:
        raise ValueError("Student summary does not describe all 54 full fits")
    provenance_sha = summary.get("provenance_sha256")
    if not isinstance(provenance_sha, str) or len(provenance_sha) != 64 or any(c not in "0123456789abcdef" for c in provenance_sha):
        raise ValueError("Missing canonical summary provenance SHA256")
    arms = summary.get("arms")
    expected = {f"{architecture}_{label}_seed{seed}"
                for architecture in ARCHITECTURES for label in LABELS for seed in SEEDS}
    if not isinstance(arms, dict) or set(arms) != expected:
        raise ValueError("Missing, extra, or duplicate-identity student arms")
    entries = []
    for architecture in ARCHITECTURES:
        for label in LABELS:
            for seed in SEEDS:
                name = f"{architecture}_{label}_seed{seed}"
                record = arms[name]
                if not isinstance(record, dict) or (record.get("architecture"), record.get("label"), record.get("seed")) != (architecture, label, seed):
                    raise ValueError("Student arm key and identity disagree")
                flags = checkpoint_flags(record.get("selected_step"))
                if any(record.get(key) != value for key, value in flags.items()):
                    raise ValueError("Student checkpoint selection metadata disagrees")
                if record.get("checkpoint") != name + ".pt":
                    raise ValueError("Checkpoint filename must match its arm without path components")
                if record.get("parameters") != architecture_plan()[architecture]["parameters"]:
                    raise ValueError("Student parameter count changed")
                digest = record.get("checkpoint_sha256")
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("Missing canonical checkpoint SHA256")
                if record.get("provenance_sha256") != summary.get("provenance_sha256"):
                    raise ValueError("Student arms do not share the summary provenance")
                entries.append({"architecture": architecture, "label": label, "seed": seed,
                                "checkpoint": record["checkpoint"], "checkpoint_sha256": digest,
                                "selected_step": record["selected_step"], "is_trained": record["is_trained"]})
    expected_step0 = {f"{entry['architecture']}_{entry['label']}_seed{entry['seed']}"
                      for entry in entries if not entry["is_trained"]}
    declared_step0 = summary.get("step0_selected_arms")
    if not isinstance(declared_step0, list) or len(set(declared_step0)) != len(declared_step0) or set(declared_step0) != expected_step0:
        raise ValueError("Step0-selected arm inventory differs from checkpoint metadata")
    return entries


def checkpoint_entries(summary, student_output):
    """Return all 54 entry dictionaries, ordered architecture/label/seed17,29,43.

    Each entry exposes architecture, label, seed, path, selected_step and
    is_trained. Paths must be regular checkpoint files directly inside the
    supplied student directory; the summary and checkpoint hashes are checked.
    """
    entries = _checkpoint_metadata_entries(summary)
    root = Path(student_output).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Student output must be a directory")
    if _sha(root / "provenance.json") != summary["provenance_sha256"]:
        raise ValueError("Student summary provenance digest changed")
    result = []
    for entry in entries:
        path = root / entry["checkpoint"]
        resolved = path.resolve(strict=True)
        if resolved.parent != root or path.is_symlink() or not resolved.is_file():
            raise ValueError("Checkpoint path escapes the student output or is not a regular file")
        if _sha(resolved) != entry["checkpoint_sha256"]:
            raise ValueError("Student checkpoint digest differs from completed summary")
        result.append({key: entry[key] for key in ("architecture", "label", "seed", "selected_step", "is_trained")})
        result[-1]["path"] = resolved
    return result


def load_student(path, device="cpu"):
    """Return (frozen model, train normalization, checkpoint metadata)."""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if set(checkpoint) != {"model", "metadata"}:
        raise ValueError("Unexpected student checkpoint schema")
    metadata = checkpoint["metadata"]
    if metadata["architecture"] not in ARCHITECTURES or metadata["label"] not in LABELS or metadata["seed"] not in SEEDS:
        raise ValueError("Unregistered student identity")
    flags = checkpoint_flags(metadata["selected_step"])
    if any(metadata.get(key) != value for key, value in flags.items()):
        raise ValueError("Checkpoint step0/training/selection metadata changed")
    if metadata["source_sha256"] != _sha(__file__):
        raise ValueError("Student implementation differs from fitted source")
    if metadata["normalization_file"] != "normalization.npz" or metadata["provenance_file"] != "provenance.json":
        raise ValueError("Unexpected artifact path")
    norm_path, provenance_path = path.parent / "normalization.npz", path.parent / "provenance.json"
    if _sha(norm_path) != metadata["normalization_sha256"] or _sha(provenance_path) != metadata["provenance_sha256"]:
        raise ValueError("Student normalization/provenance digest changed")
    provenance = json.loads(provenance_path.read_text())
    if provenance["source_sha256"] != metadata["source_sha256"] or provenance["normalization_fit_split"] != "train":
        raise ValueError("Invalid train-only normalization provenance")
    if _sha(path.parent / "source_ids.json") != provenance["source_ids_sha256"]:
        raise ValueError("Source parent IDs changed")
    if _state_record(checkpoint["model"]) != metadata["selected_state"]:
        raise ValueError("Checkpoint tensor digest changed")
    with np.load(norm_path, allow_pickle=False) as archive:
        normalization = {key: archive[key].copy() for key in archive.files}
    _validate_normalization(normalization)
    model = TerminalScorer(metadata["architecture"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    return model, normalization, metadata
