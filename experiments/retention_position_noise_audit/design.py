"""Frozen Gate scene and noise generator for the retention fairness audit.

The scientific dataset is regenerated only through this module.  Cue position
is balanced across the complete legal history, including the two arrivals a
strict FIFO policy retains.  Additive noise is shared by sibling episodes and
does not alter evidence timestamps or task targets.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from phase1.envs import episode
from phase1.utils import stable_seed


BACKGROUND_COLORS = np.asarray(
    [
        (24, 29, 38),   # outside the arena
        (44, 49, 60),   # arena background
    ],
    dtype=np.uint8,
)
HISTORY_DEPTH_BY_SPLIT = {
    "test_long_gap": 40,
}


@dataclass(frozen=True)
class PairDesign:
    task: str
    split: str
    scene_index: int
    history_depth: int
    post_cue_history_frames: int
    cue_start: int
    cue_end: int


def fifo_overlap_count(design: PairDesign) -> int:
    final_two = {
        design.history_depth - 1,
        design.history_depth,
    }
    cue = set(range(design.cue_start, design.cue_end + 1))
    return len(final_two & cue)


def _integer(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    return int(rng.integers(int(bounds[0]), int(bounds[1]) + 1))


def _position_offset(duration: int, history_depth: int, rng: np.random.Generator) -> int:
    """Sample ordinary legal-history frames after the cue uniformly."""
    maximum = history_depth - duration + 1
    if maximum < 0:
        raise ValueError("Cue duration does not fit the history")
    return int(rng.integers(0, maximum + 1))


def random_position_scene(
    source_config: dict,
    task: str,
    split: str,
    pair_index: int,
    *,
    history_depth: int | None = None,
) -> tuple[dict, PairDesign]:
    """Create a deterministic Gate scene with a uniformly random cue location."""
    if task != "gate":
        raise ValueError("This audit isolates the Gate cue-position confound")
    c = copy.deepcopy(source_config)
    d = c["data"]
    history_depth = (
        int(history_depth)
        if history_depth is not None
        else HISTORY_DEPTH_BY_SPLIT.get(split, 24)
    )
    seed = stable_seed(d["seed"], task, split, pair_index, "position-noise-audit-v1")
    rng = np.random.default_rng(seed)
    duration = _integer(rng, d["cue_duration"])
    offset = _position_offset(duration, history_depth, rng)
    end = history_depth - offset
    start = end - duration + 1
    query = history_depth + int(d["recent"])
    length = query + max(int(v) for v in d["horizons"]) + 1
    actions = rng.uniform(-1, 1, (length - 1, 2)).astype(np.float32)
    label_order = rng.permutation([0, 1]).tolist()
    distractors = np.repeat(rng.uniform(0, 1, (1, 8, 3)), length, axis=0)
    scene = {
        "seed": seed,
        "task": task,
        "split": split,
        "query": query,
        "start": start,
        "end": end,
        "gap": offset,
        "duration": duration,
        "length": length,
        "actions": actions,
        "label_order": label_order,
        "xref": float(rng.uniform(0.38, 0.62)),
        "speed": float(rng.uniform(0.005, 0.007)),
        "y": float(rng.uniform(0.43, 0.57)),
        "distractors": distractors,
        "shift": split == "test_distractor_shift",
        "size": d["image_size"],
        "history_depth": history_depth,
    }
    design = PairDesign(
        task=task,
        split=split,
        scene_index=pair_index,
        history_depth=history_depth,
        post_cue_history_frames=offset,
        cue_start=start,
        cue_end=end,
    )
    return scene, design


def _background_mask(rgb: np.ndarray) -> np.ndarray:
    background = np.zeros(rgb.shape[:2], dtype=bool)
    for color in BACKGROUND_COLORS:
        background |= np.all(rgb == color, axis=2)
    return background


def shared_background_noise(
    frames: list[np.ndarray],
    *,
    scene_seed: int,
    frame_index: int,
    sigma: float,
) -> np.ndarray:
    """Construct one noise tensor from the common sibling background."""
    if sigma < 0:
        raise ValueError("Noise sigma must be nonnegative")
    cleaned = [np.asarray(rgb, dtype=np.uint8) for rgb in frames]
    if not cleaned or any(rgb.shape != cleaned[0].shape for rgb in cleaned):
        raise ValueError("Sibling frames must share one RGB shape")
    if sigma == 0:
        return np.zeros_like(cleaned[0], dtype=np.int16)
    rng = np.random.default_rng(
        stable_seed(scene_seed, frame_index, "shared-background-noise")
    )
    noise = np.rint(rng.normal(0.0, sigma, cleaned[0].shape)).astype(np.int16)
    noise = np.clip(noise, -int(round(3 * sigma)), int(round(3 * sigma)))
    common_background = np.ones(cleaned[0].shape[:2], dtype=bool)
    for rgb in cleaned:
        common_background &= _background_mask(rgb)
    noise[~common_background] = 0
    return noise


def paired_noisy_episodes(
    source_config: dict,
    split: str,
    pair_index: int,
    *,
    sigma: float = 12.0,
    history_depth: int | None = None,
) -> tuple[dict, PairDesign, list[dict]]:
    """Render both sibling labels with an identical per-frame noise tensor."""
    scene, design = random_position_scene(
        source_config,
        "gate",
        split,
        pair_index,
        history_depth=history_depth,
    )
    clean_by_label = {}
    for label in scene["label_order"]:
        clean_by_label[int(label)] = episode(scene, int(label))
    by_label = {}
    for label in scene["label_order"]:
        clean = clean_by_label[int(label)]
        noisy_frames = []
        noises = []
        for frame_index, rgb in enumerate(clean["rgb"]):
            noise = shared_background_noise(
                [item["rgb"][frame_index] for item in clean_by_label.values()],
                scene_seed=scene["seed"],
                frame_index=frame_index,
                sigma=sigma,
            )
            noisy = np.clip(
                rgb.astype(np.int16) + noise,
                0,
                255,
            ).astype(np.uint8)
            noisy_frames.append(noisy)
            noises.append(noise)
        by_label[int(label)] = {
            "rgb": np.stack(noisy_frames),
            "noise": np.stack(noises),
            "clean_rgb": clean["rgb"],
            "actions": clean["actions"],
            "states": clean["states"],
            "classes": clean["classes"],
            "visible": clean["visible"],
            "label": int(label),
            "evidence": clean["evidence"],
            "scene": scene,
            "design": design,
            "noise_sigma": float(sigma),
        }
    return scene, design, [by_label[int(label)] for label in scene["label_order"]]


