"""Independent Motion v2.1 renderer; hidden fields are evaluator-only.

The scene API is compatible with the Phase 1 task/split/index convention, but
deliberately refuses all held-out splits. Nothing in this module writes data.
"""

import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw


GENERATOR_VERSION = "motion_v2.1"
ALLOWED_SPLITS = ("train", "validation")
REGISTERED_DATA = {
    "seed": 20261030,
    "tasks": ["motion"],
    "image_size": 96,
    "recent": 4,
    "budget": 2,
    "horizons": [3, 6],
    "gap": [5, 10],
    "long_gap": [20, 32],
    "pairs": {"train": 512, "validation": 96},
}


def digest(obj):
    """The original Phase 1 canonical JSON hash, without importing torch."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def stable_seed(*parts):
    return int(digest(parts)[:16], 16) % (2**32)


def validate_config(config):
    if config.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("Expected generator_version='motion_v2.1'")
    data = config.get("data")
    if not isinstance(data, dict) or data != REGISTERED_DATA:
        raise ValueError("Data settings must exactly match the registered train/validation configuration")
    # In particular, adding a held-out split cannot silently generate it.
    if set(data["pairs"]) != set(ALLOWED_SPLITS):
        raise ValueError("Only train and validation are authorized")


def reflect(x, v):
    """Phase 1 elastic reflection, including its boundary convention."""
    while x < 0.08 or x > 0.92:
        if x < 0.08:
            x, v = 0.16 - x, -v
        if x > 0.92:
            x, v = 1.84 - x, -v
    return x, v


def motion_step(x, v, action):
    """action[t] affects t -> t+1; action displacement does not accelerate v."""
    return reflect(x + v + 0.0015 * float(action[0]) + 0.001 * float(action[1]), v)


def scene(config, task, split, index):
    """Return shared scene factors for one pair, never a test scene.

    Hidden factors in this in-memory object must not be passed to a model or
    copied to observation files/the public manifest. generate_v2 enforces the
    explicit storage allowlists.
    """
    validate_config(config)
    if task != "motion" or split not in ALLOWED_SPLITS:
        raise ValueError("Only Motion train/validation scenes are authorized")
    if type(index) is not int or not 0 <= index < config["data"]["pairs"][split]:
        raise ValueError("scene_index is outside the registered split")
    data = config["data"]
    seed = stable_seed(data["seed"], task, split, index)
    rng = np.random.default_rng(seed)
    gap = int(rng.integers(5, 11))
    # Match the amendment's rejection sampling: a failed mask resamples timing
    # and mask together. End is forced visible before the count checks.
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
    # Preserve Phase 1's shutter through query+1; q+3 and q+6 are visible.
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


def render(s, t, label, x, y, visible):
    """Phase 1 Motion drawing style, with a real shutter for every hidden frame.

    The original shift palette/shape/count branch is retained for future work;
    scene() never constructs a test or shifted scene in this execution.
    label is unused by the renderer, as in Phase 1 Motion.
    """
    n = s["size"]
    im = Image.new("RGB", (n, n), (24, 29, 38))
    draw = ImageDraw.Draw(im)
    box = lambda b: tuple(int(v * n) for v in b)
    draw.rectangle(box((0.05, 0.25, 0.95, 0.78)), fill=(44, 49, 60),
                   outline=(110, 118, 133), width=2)
    if visible:
        r = 0.055
        draw.ellipse(box((x-r, y-r, x+r, y+r)), fill=(237, 175, 70),
                     outline=(255, 235, 174), width=2)
    else:
        # Never draw a hidden body, including the hidden pre-roll and blips.
        draw.rectangle(box((0.055, 0.255, 0.945, 0.775)), fill=(80, 89, 102))
        for xx in np.linspace(0.1, 0.9, 7):
            draw.line(box((xx, 0.28, xx, 0.74)), fill=(99, 108, 120), width=1)
    count = 7 if s["shift"] else 3
    for xx, yy, cc in s["distractors"][t, :count]:
        xx = 0.08 + 0.84 * xx
        yy = 0.84 + 0.10 * yy
        color = (int(80 + 150 * cc), 100, 210) if s["shift"] else (85, int(80 + 130 * cc), 115)
        bounds = box((xx-0.025, yy-0.025, xx+0.025, yy+0.025))
        (draw.rectangle if s["shift"] else draw.ellipse)(bounds, fill=color)
    return np.asarray(im, dtype=np.uint8).copy()


def episode(s, label):
    """Construct observations and explicitly separate evaluator-only arrays."""
    if s["task"] != "motion" or s["split"] not in ALLOWED_SPLITS:
        raise ValueError("Only Motion train/validation episodes are authorized")
    if type(label) is not int or label not in (0, 1):
        raise ValueError("A sibling label must be integer 0 or 1")
    length = s["length"]
    states = np.empty((length, 2), dtype=np.float64)
    velocity = np.empty(length, dtype=np.float64)
    x, v = s["x_pre"], s["v0"]
    for t in range(s["t_c"]):
        states[t], velocity[t] = (x, 0.40), v
        x, v = reflect(x + v, v)
    v1 = (2 * label - 1) * s["speed"]
    # The authorized kick at t_c is an exogenous reset into the lower lane.
    # Actions, like Phase 1, do not act before/at the matched endpoint.
    for t in range(s["t_c"], s["end"] + 1):
        states[t] = (s["x_end"] + v1 * (t - s["end"]), 0.62)
        velocity[t] = v1
    x, v = s["x_end"], v1
    for t in range(s["end"] + 1, length):
        x, v = motion_step(x, v, s["actions"][t - 1])
        states[t], velocity[t] = (x, 0.62), v
    visible = s["visible"].copy()
    sufficient = visible & (np.arange(length) >= s["t_c"]) & (np.arange(length) <= s["end"])
    identifying_times = np.flatnonzero(sufficient)
    if len(identifying_times) < 2 or identifying_times[-1] != s["end"]:
        raise ValueError("Scene violates its visible new-segment constraints")
    frames = np.stack([
        render(s, t, label, states[t, 0], states[t, 1], bool(visible[t]))
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
