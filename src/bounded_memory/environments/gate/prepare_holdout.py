"""Generate the registered Gate holdout and encode it with the frozen I-JEPA model."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from .generate import DESIGN_VERSION, _record_id
from .structured_design import ARROW_PANEL, STRUCTURED_LINE, _box, structured_pair_bundle
from .holdout_protocol import GENERATION_SEED, SPLITS, audit_identities, historical_records, original_records
from ...data.io import digest, file_hash, read_json, require_sha256, stable_seed, write_json, write_npz
from ...predictors.phase1.cache import encode
from ...predictors.phase1.data import dataset_signature


def pair_records(seed, split, i):
    pair = digest([seed, "gate", split, i, DESIGN_VERSION.lower()])[:24]
    ids = [_record_id(pair, label) for label in (0, 1)]
    scene_seed = stable_seed(seed, "gate", split, i, "position-noise-audit-v1")
    return [
        dict(
            task="gate",
            split=split,
            pair=pair,
            scene_index=i,
            seed=scene_seed,
            label=label,
            id=ids[label],
            sibling=ids[1 - label],
        )
        for label in (0, 1)
    ]


def render_pair(config, pair):
    split, i = pair[0]["split"], pair[0]["scene_index"]
    scene, design, noisy = structured_pair_bundle(
        config, split, i, include_vertical_lines=True, full_image_lines=True
    )
    clean_scene, _, clean = structured_pair_bundle(
        config, split, i, include_vertical_lines=False, full_image_lines=True
    )
    q = int(scene["query"])
    recent = config["data"]["recent"]
    if scene["seed"] != pair[0]["seed"] or clean_scene["seed"] != scene["seed"]:
        raise ValueError("Scene seeds differ")
    output = []
    for rec in pair:
        label = rec["label"]
        episode = noisy[label]["episode"]
        for key in ("actions", "states", "classes", "visible", "evidence"):
            if not np.array_equal(episode[key], clean[label]["episode"][key]):
                raise ValueError(f"Noisy and clean dynamics differ in {key}")
        frames = noisy[label]["frames"][: q + 1] + clean[label]["frames"][q + 1 :]
        counts = np.array([f.vertical_line_count for f in frames], np.int64)
        if np.any((counts[: q + 1] < 7) | (counts[: q + 1] > 14)) or counts[q + 1 :].any():
            raise ValueError("Incorrect noisy input or clean target rendering")
        evidence = np.asarray(episode["evidence"], np.int64)
        if np.any(evidence >= q - recent + 1):
            raise ValueError("Cue overlaps the recent buffer")
        obs = dict(rgb=np.stack([f.rgb for f in frames]), actions=episode["actions"], query=np.int64(q))
        meta = dict(
            states=episode["states"],
            classes=episode["classes"],
            visible=episode["visible"],
            label=np.int64(label),
            evidence=evidence,
            cue_mask=np.array([f.is_cue for f in frames]),
            vertical_line_counts=counts,
        )
        row = dict(
            rec,
            query=q,
            gap=design.post_cue_history_frames,
            history_depth=design.history_depth,
            frames=len(frames),
            cue_start=design.cue_start,
            cue_end=design.cue_end,
        )
        output.append((row, obs, meta))
    (_, left, lm), (_, right, rm) = output
    neutral = ~lm["cue_mask"][: q + 1]
    if (
        not np.array_equal(left["actions"], right["actions"])
        or not np.array_equal(lm["evidence"], rm["evidence"])
        or not np.array_equal(left["rgb"][: q + 1][neutral], right["rgb"][: q + 1][neutral])
        or any(np.array_equal(left["rgb"][q + h], right["rgb"][q + h]) for h in config["data"]["horizons"])
    ):
        raise ValueError("Sibling controls or targets failed")
    return output


def generate(models, trainval, prior_manifests, provenance, output):
    source = require_sha256(
        models / "resolved_config.json",
        "a602b7731d4af2825197160f57efa19201cbdc3c88d15a9eb93ea12435762c83",
        "team configuration",
    )
    config = read_json(source)
    config["output"] = str(output.resolve())
    config["data"]["seed"] = GENERATION_SEED
    config["data"]["pairs"].update({s: 64 for s in SPLITS})
    config["data"]["clean_future_protocol"].update(
        generated_splits=list(SPLITS), test_generated=True, test_accessed=True
    )
    old = original_records(trainval)
    prior = historical_records(old, prior_manifests)
    pairs = [pair_records(GENERATION_SEED, s, i) for s in SPLITS for i in range(64)]
    audit_identities([r for pair in pairs for r in pair], prior, GENERATION_SEED, provenance)
    output.mkdir(parents=True, exist_ok=False)
    root = output / "dataset"
    write_json(output / "resolved_config.json", config)
    records = []
    for pair in pairs:
        for rec, obs, meta in render_pair(config, pair):
            for kind, directory, arrays in (("obs", "observations", obs), ("eval", "evaluation_only", meta)):
                rel = Path(directory) / rec["split"] / (rec["id"] + ".npz")
                write_npz(root / rel, **arrays)
                rec[kind], rec[kind + "_sha256"] = str(rel), file_hash(root / rel)
            records.append(rec)
    signature = dataset_signature(config)
    write_json(
        root / "manifest.json", dict(signature=signature, design_version=DESIGN_VERSION, records=records)
    )
    write_json(
        root / "split_audit.json", dict(passed=True, signature=signature, splits=list(SPLITS), pairs=128)
    )


def encode_holdout(data, device):
    config = read_json(data / "resolved_config.json")
    if config["data"]["seed"] != GENERATION_SEED:
        raise ValueError("Expected a registered independent data directory")
    config.update(output=str(data.resolve()), device=device)
    index = encode(config)
    print("manifest_sha256", file_hash(data / "dataset/manifest.json"))
    print("index_sha256", file_hash(data / "features/index.json"))
    print("encoded_episodes", len(index["episodes"]))


def verify_videos(data, manifest_sha256):
    """Check paired cues, shared input noise and clean future targets against the renderer."""
    root = data / "dataset"
    config = read_json(data / "resolved_config.json")
    manifest = read_json(require_sha256(root / "manifest.json", manifest_sha256))
    pairs = defaultdict(list)
    for rec in manifest["records"]:
        pairs[rec["split"], rec["pair"]].append(rec)

    for (split, _), siblings in pairs.items():
        left, right = sorted(siblings, key=lambda rec: rec["label"])
        q, start, end = left["query"], left["cue_start"], left["cue_end"]
        _, _, clean = structured_pair_bundle(
            config, split, left["scene_index"], include_vertical_lines=False, full_image_lines=True
        )
        images, actions, masks = [], [], []
        for rec in (left, right):
            ref = np.stack([frame.rgb for frame in clean[rec["label"]]["frames"]])
            with np.load(require_sha256(root / rec["obs"], rec["obs_sha256"])) as obs:
                rgb = obs["rgb"]
                actions.append(obs["actions"])
            with np.load(require_sha256(root / rec["eval"], rec["eval_sha256"])) as meta:
                counts = meta["vertical_line_counts"]
                np.testing.assert_array_equal(np.flatnonzero(meta["cue_mask"]), np.arange(start, end + 1))
            mask = np.any(rgb != ref, axis=-1)
            if (
                np.any((counts[: q + 1] < 7) | (counts[: q + 1] > 14))
                or counts[q + 1 :].any()
                or not mask[: q + 1].any(axis=(1, 2)).all()
                or mask[q + 1 :].any()
                or not np.all(rgb[mask] == STRUCTURED_LINE)
            ):
                raise ValueError(f"Invalid input noise or future pixels in {rec['id']}")
            images.append(rgb)
            masks.append(mask)
        np.testing.assert_array_equal(actions[0], actions[1])
        np.testing.assert_array_equal(masks[0], masks[1])
        difference = np.any(images[0][: q + 1] != images[1][: q + 1], axis=-1)
        np.testing.assert_array_equal(np.flatnonzero(difference.any(axis=(1, 2))), np.arange(start, end + 1))
        x0, y0, x1, y1 = _box(ARROW_PANEL, 96)
        difference[:, y0 : y1 + 1, x0 : x1 + 1] = False
        if (
            end >= q - 3
            or difference.any()
            or any(np.array_equal(images[0][q + h], images[1][q + h]) for h in (3, 6))
        ):
            raise ValueError(f"Invalid sibling cue or route targets for {left['pair']}")
