"""Fresh Motion v2 scenes for training the deletion-cost scorer: the registered generator, new scenes.

Why: the frozen readers were trained on the 1,024 registered training episodes and are far better on them
than on unseen scenes (the attention policy scores +3 position MSE 0.0044 on training scene pairs, 0.0091 on
validation). Deletion-cost labels computed on those episodes describe what the readers memorised, not which
frame matters (runs/memory_gaps_20261124/README.md, section 2). Labels and checkpoint selection on scenes the
readers never saw remove that bias without touching the readers, the validation sets or the protocol
constants (R = 4, B = 2, horizons +3/+6).

What is new data and what is not:

    scene / episode   envs_v2.scene and envs_v2.episode, copied statement for statement; the only change is
                      the split name (FRESH_SPLIT) in place of the registered train/validation check.
                      tests/motion_v2/test_fresh.py checks the copies reproduce the registered train and
                      validation scenes exactly, so fresh scenes come from the same distribution.
    seeds             stable_seed(20261030, "motion", FRESH_SPLIT, index), the registered recipe with a split
                      name of its own: disjoint from the train/validation scene seeds (checked) and never a
                      held-out split (names containing "test" or "confirm" are refused).
    features          the frozen I-JEPA recipe of the step-2 run (its resolved encoder config: official
                      revision, FP32, grid2 pooling, batches of 2 frames), normalised with the frozen
                      train-split mean/std and projected with the frozen basis, as FrozenAssets does.

Readers, decoder, normalisation and validation data are only read. Nothing here writes outside `root`.

    ranges / records / load   the config's split of the fresh scene pairs (training states, dev, holdout), the
                              written records, and FrozenAssets.episode-shaped episodes of chosen parts
    generate / encode         writing the scenes and their raw I-JEPA features (resumable)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import envs_v2
from .frozen import sha256

FRESH_SPLIT = "fresh_train_20261125"
EVAL_KEYS = ("states", "visible", "t_c", "end", "sufficient")


def _check_split(split):
    lowered = split.lower()
    if split in envs_v2.ALLOWED_SPLITS or "test" in lowered or "confirm" in lowered or "heldout" in lowered:
        raise ValueError(f"{split!r} is a registered or held-out split name; fresh scenes need a name of their own")


def draw(data, task, split, index):
    """The body of envs_v2.scene after its config/split checks, unchanged."""
    seed = envs_v2.stable_seed(data["seed"], task, split, index)
    rng = np.random.default_rng(seed)
    gap = int(rng.integers(5, 11))
    while True:
        t_on = int(rng.integers(0, 7))
        length_old = int(rng.integers(4, 11))
        length_new = int(rng.integers(3, 8))
        t_c = t_on + length_old
        end = t_c + length_new - 1
        query = end + gap + data["recent"]
        visible = np.array([
            (t_on <= t <= end) and rng.random() > 0.25
            for t in range(query + 1)
        ], dtype=bool)
        visible[end] = True
        if visible[t_c:end + 1].sum() >= 2 and visible[:t_c].sum() >= 2:
            break
    x_end = float(rng.uniform(0.20, 0.80))
    speed = float(rng.uniform(0.008, 0.012))
    x_pre = float(rng.uniform(0.20, 0.80))
    v0 = float(rng.choice([-1, 1]) * rng.uniform(0.005, 0.007))
    length = query + max(data["horizons"]) + 1
    full_visibility = np.zeros(length, dtype=bool)
    full_visibility[:query + 1] = visible
    full_visibility[query + 2:] = True
    actions = rng.uniform(-1, 1, (length - 1, 2)).astype(np.float32)
    label_order = rng.permutation([0, 1]).tolist()
    distractors = np.empty((length, 8, 3), dtype=np.float64)
    nuisance = rng.uniform(0, 1, (8, 3))
    for t in range(length):
        nuisance = np.clip(nuisance + rng.normal(0, 0.08, nuisance.shape), 0, 1)
        distractors[t] = nuisance
    return {
        "seed": seed, "task": task, "split": split, "query": query,
        "gap": gap, "length": length, "size": data["image_size"],
        "t_on": t_on, "t_c": t_c, "end": end,
        "L0": length_old, "L1": length_new,
        "x_end": x_end, "speed": speed, "x_pre": x_pre, "v0": v0,
        "visible": full_visibility, "actions": actions,
        "label_order": label_order, "distractors": distractors,
        "shift": False,
    }


def scene(index, split=FRESH_SPLIT):
    _check_split(split)
    if type(index) is not int or index < 0:
        raise ValueError("scene index must be a non-negative int")
    return draw(envs_v2.REGISTERED_DATA, "motion", split, index)


def render_episode(s, label):
    """The body of envs_v2.episode after its split check, unchanged (renderer and dynamics are envs_v2's)."""
    if type(label) is not int or label not in (0, 1):
        raise ValueError("A sibling label must be integer 0 or 1")
    length = s["length"]
    states = np.empty((length, 2), dtype=np.float64)
    velocity = np.empty(length, dtype=np.float64)
    x, v = s["x_pre"], s["v0"]
    for t in range(s["t_c"]):
        states[t], velocity[t] = (x, 0.40), v
        x, v = envs_v2.reflect(x + v, v)
    v1 = (2 * label - 1) * s["speed"]
    for t in range(s["t_c"], s["end"] + 1):
        states[t] = (s["x_end"] + v1 * (t - s["end"]), 0.62)
        velocity[t] = v1
    x, v = s["x_end"], v1
    for t in range(s["end"] + 1, length):
        x, v = envs_v2.motion_step(x, v, s["actions"][t - 1])
        states[t], velocity[t] = (x, 0.62), v
    visible = s["visible"].copy()
    sufficient = visible & (np.arange(length) >= s["t_c"]) & (np.arange(length) <= s["end"])
    identifying_times = np.flatnonzero(sufficient)
    if len(identifying_times) < 2 or identifying_times[-1] != s["end"]:
        raise ValueError("Scene violates its visible new-segment constraints")
    frames = np.stack([
        envs_v2.render(s, t, label, states[t, 0], states[t, 1], bool(visible[t]))
        for t in range(length)
    ])
    classes = np.where(visible, states[:, 0] > 0.5, -1).astype(np.int64)
    return {
        "observations": {
            "rgb": frames, "actions": s["actions"].copy(),
            "query": np.int64(s["query"]),
        },
        "evaluation_only": {
            "states": states.astype(np.float32),
            "velocity": velocity.astype(np.float32),
            "classes": classes, "visible": visible, "sufficient": sufficient,
            "label": np.int64(label),
            "evidence": np.asarray([identifying_times[0], identifying_times[-1]], dtype=np.int64),
            "t_on": np.int64(s["t_on"]), "t_c": np.int64(s["t_c"]),
            "end": np.int64(s["end"]), "L0": np.int64(s["L0"]), "L1": np.int64(s["L1"]),
            "v0": np.float64(s["v0"]), "v1": np.float64(v1),
            "speed": np.float64(s["speed"]), "x_end": np.float64(s["x_end"]),
            "x_pre": np.float64(s["x_pre"]),
        },
    }


def pair_ids(split, index):
    """generate_v2's pair and episode ids for a scene."""
    pair = envs_v2.digest([envs_v2.REGISTERED_DATA["seed"], "motion", split, index])[:24]
    return pair, [envs_v2.digest([pair, slot, "opaque-id"])[:24] for slot in range(2)]


def registered_scene_seeds():
    data = envs_v2.REGISTERED_DATA
    return {envs_v2.stable_seed(data["seed"], "motion", split, i)
            for split in envs_v2.ALLOWED_SPLITS for i in range(data["pairs"][split])}


def _save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def generate(root, pairs, split=FRESH_SPLIT, log=print):
    """Write `pairs` fresh scene pairs (two siblings each) under root/dataset; resumable; returns the records."""
    _check_split(split)
    root = Path(root)
    dataset = root / "dataset"
    registered = registered_scene_seeds()
    records = []
    for index in range(pairs):
        s = scene(index, split)
        if s["seed"] in registered:
            raise ValueError(f"fresh scene {index} reuses a registered train/validation scene seed")
        pair, ids = pair_ids(split, index)
        for slot, eid in enumerate(ids):
            obs = dataset / "observations" / split / f"{eid}.npz"
            meta = dataset / "evaluation_only" / split / f"{eid}.npz"
            if not (obs.exists() and meta.exists()):
                value = render_episode(s, s["label_order"][slot])
                _save_npz(meta, value["evaluation_only"])
                _save_npz(obs, value["observations"])
            records.append({
                "id": eid, "pair": pair, "sibling": ids[1 - slot], "split": split,
                "task": "motion", "seed": s["seed"], "scene_index": index,
                "query": s["query"], "gap": s["gap"], "frames": s["length"],
                "obs": str(obs.relative_to(dataset)), "eval": str(meta.relative_to(dataset)),
            })
        if index % 100 == 0:
            log(f"fresh scenes {index + 1}/{pairs}")
    if len({r["id"] for r in records}) != len(records) or len({r["seed"] for r in records}) != pairs:
        raise ValueError("fresh episode id or scene seed collision")
    manifest = json.dumps({"split": split, "pairs": pairs, "generator": envs_v2.GENERATOR_VERSION,
                           "data": envs_v2.REGISTERED_DATA, "records": records}, indent=1) + "\n"
    path = dataset / "manifest.json"
    if not path.exists() or path.read_text() != manifest:     # unchanged on a rerun, so readers never see it rewritten
        tmp = path.with_name("manifest.tmp.json")
        tmp.write_text(manifest)
        os.replace(tmp, path)
    return records


def encode(assets, root, records, device, log=print):
    """Raw I-JEPA features of every fresh episode under root/features/episodes/<id>.npz; resumable."""
    import torch
    folder = Path(root) / "features" / "episodes"
    todo = [r for r in records if not (folder / f"{r['id']}.npz").exists()]
    if todo:
        config = dict(assets.config["encoder"])
        encoder = assets.reader.load_reference("encoders").FrozenIJepa(config, torch.device(device))
        if encoder.metadata["resolved_revision"] != config["revision"]:
            raise RuntimeError("resolved I-JEPA revision differs from the frozen step-2 recipe")
        batch = int(config["batch_size"])
        for i, rec in enumerate(todo):
            with np.load(Path(root) / "dataset" / rec["obs"], allow_pickle=False) as f:
                rgb = f["rgb"]
            chunks = [encoder.encode(rgb[s:s + batch]) for s in range(0, len(rgb), batch)]
            values = np.concatenate(chunks).astype(np.float32)
            if values.shape != (rec["frames"], 5120) or not np.isfinite(values).all():
                raise RuntimeError(f"invalid features for fresh episode {rec['id']}")
            _save_npz(folder / f"{rec['id']}.npz", {"features": values})
            if i % 100 == 0 or i == len(todo) - 1:
                log(f"encoded {i + 1}/{len(todo)} new fresh episodes ({len(records) - len(todo)} cached)")
        del encoder
        torch.cuda.empty_cache()
    index = {r["id"]: {"path": f"episodes/{r['id']}.npz", "frames": r["frames"],
                       "sha256": sha256(folder / f"{r['id']}.npz")} for r in records}
    (Path(root) / "features" / "index.json").write_text(json.dumps({"episodes": index}, indent=1) + "\n")
    return index


def episode(assets, root, rec):
    """A FrozenAssets.episode-shaped dict for one fresh episode (frozen normalisation and basis)."""
    root = Path(root)
    with np.load(root / "features" / "episodes" / f"{rec['id']}.npz", allow_pickle=False) as f:
        raw = f["features"]
    z = ((raw - assets.store.norm["mean"]) / assets.store.norm["std"]).astype(np.float32)
    if z.shape != (rec["frames"], 5120) or not np.isfinite(z).all():
        raise ValueError(f"invalid fresh feature array {rec['id']}")
    p = assets.reader.load_reference("model").project(z, assets.basis)
    with np.load(root / "dataset" / rec["obs"], allow_pickle=False) as f:
        actions, query = f["actions"], int(f["query"])
    if query != rec["query"]:
        raise ValueError(f"fresh episode {rec['id']} query differs from its record")
    with np.load(root / "dataset" / rec["eval"], allow_pickle=False) as f:
        meta = {k: f[k] for k in EVAL_KEYS}
    prefix = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(actions, axis=0)])
    return dict(rec=rec, z=z, p=p, actions=actions, prefix=prefix, meta=meta)


def ranges(fresh):
    """Scene-pair index ranges of a config's `fresh` block: training states, dev selection, holdout check."""
    fit, dev, holdout = int(fresh["fit_pairs"]), int(fresh["dev_pairs"]), int(fresh["holdout_pairs"])
    return {"fit": range(0, fit), "dev": range(fit, fit + dev), "holdout": range(fit + dev, fit + dev + holdout)}


def records(root):
    """The records written by generate (root/dataset/manifest.json), read only."""
    path = Path(root) / "dataset" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/prepare_data/motion_v2_fresh.py first")
    return json.loads(path.read_text())["records"]


def load(assets, fresh, parts=("fit", "dev")):
    """{part: episodes} of the fresh scenes the config's `fresh` block names, from the written scenes and features."""
    wanted = ranges(fresh)
    written = records(fresh["root"])
    missing = [p for p in parts if max(wanted[p], default=-1) >= len(written) // 2]
    if missing:
        raise ValueError(f"fresh scenes for {missing} are not generated yet: run scripts/prepare_data/motion_v2_fresh.py")
    return {part: [episode(assets, fresh["root"], r) for r in written if r["scene_index"] in wanted[part]]
            for part in parts}
