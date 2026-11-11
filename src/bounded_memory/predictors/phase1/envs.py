"""Pure, deterministic RGB simulators. All hidden state is confined here/evaluation metadata."""
import numpy as np
from PIL import Image, ImageDraw

from .utils import stable_seed


def reflect(x, v):
    # Elastic 1D reflections; works even for large externally configured gaps.
    while x < 0.08 or x > 0.92:
        if x < 0.08:
            x, v = 0.16 - x, -v
        if x > 0.92:
            x, v = 1.84 - x, -v
    return x, v


def motion_step(x, v, action):
    # a_t applies only to transition t -> t+1. Exogenous displacement changes position.
    return reflect(x + v + 0.0015 * float(action[0]) + 0.001 * float(action[1]), v)


def scene(c, task, split, index):
    d = c["data"]
    seed = stable_seed(d["seed"], task, split, index)
    rng = np.random.default_rng(seed)
    integer = lambda key: int(rng.integers(d[key][0], d[key][1] + 1))
    start = integer("cue_start")
    duration = integer("cue_duration") if task == "gate" else integer("motion_dt")
    gap = integer("long_gap" if split == "test_long_gap" else "gap")
    end = start + duration - 1 if task == "gate" else start + duration
    q = end + gap + d["recent"]
    length = q + max(d["horizons"]) + 1
    actions = rng.uniform(-1, 1, (length - 1, 2)).astype(np.float32)
    # Balanced paired labels are assigned to opaque IDs independently of public scene factors.
    order = rng.permutation([0, 1]).tolist()
    distractors = np.repeat(rng.uniform(0, 1, (1, 8, 3)), length, axis=0)
    return {"seed": seed, "task": task, "split": split, "query": q, "start": start, "end": end,
            "gap": gap, "duration": duration, "length": length, "actions": actions,
            "label_order": order, "xref": float(rng.uniform(0.38, 0.62)),
            "speed": float(rng.uniform(0.005, 0.007)), "y": float(rng.uniform(0.43, 0.57)),
            "distractors": distractors, "shift": split == "test_distractor_shift", "size": d["image_size"]}


def render(s, t, label, x, y, visible):
    n = s["size"]
    im = Image.new("RGB", (n, n), (24, 29, 38))
    draw = ImageDraw.Draw(im)
    box = lambda b: tuple(int(v * n) for v in b)
    draw.rectangle(box((0.05, 0.25, 0.95, 0.78)), fill=(44, 49, 60), outline=(110, 118, 133), width=2)
    if s["task"] == "gate":
        draw.line(box((0.10, 0.36, 0.90, 0.36)), fill=(105, 109, 120), width=2)
        draw.line(box((0.10, 0.68, 0.90, 0.68)), fill=(105, 109, 120), width=2)
        if s["start"] <= t <= s["end"]:
            # Physical two-way gate diagram, geometric cue, no text/colour label.
            draw.rectangle(box((0.31, 0.04, 0.69, 0.22)), fill=(70, 79, 92))
            draw.line(box((0.34, 0.18 if label == 0 else 0.07, 0.66, 0.07 if label == 0 else 0.18)), fill=(238, 232, 195), width=max(3, n // 20))
    if visible:
        r = 0.055
        draw.ellipse(box((x-r, y-r, x+r, y+r)), fill=(237, 175, 70), outline=(255, 235, 174), width=2)
    if s["task"] == "motion" and s["end"] < t <= s["query"] + 1:
        # RGB shutter covers the whole arena BEFORE encoding. Hidden body is never drawn.
        draw.rectangle(box((0.055, 0.255, 0.945, 0.775)), fill=(80, 89, 102))
        for xx in np.linspace(0.1, 0.9, 7):
            draw.line(box((xx, 0.28, xx, 0.74)), fill=(99, 108, 120), width=1)
    # Shared exogenous distractors outside the arena; shift changes shapes/count/palette.
    count = 7 if s["shift"] else 3
    for xx, yy, cc in s["distractors"][t, :count]:
        xx = 0.08 + 0.84 * xx
        yy = 0.84 + 0.10 * yy
        color = (int(80 + 150 * cc), 100, 210) if s["shift"] else (85, int(80 + 130 * cc), 115)
        b = box((xx-0.025, yy-0.025, xx+0.025, yy+0.025))
        (draw.rectangle if s["shift"] else draw.ellipse)(b, fill=color)
    return np.asarray(im, dtype=np.uint8).copy()


def episode(s, label):
    frames, states, classes, visibility = [], [], [], []
    sign = 2 * label - 1
    x = s["xref"] - sign * s["speed"] * (s["end"] - s["start"])
    v = sign * s["speed"]
    if s["task"] == "gate":
        x = 0.20
    for t in range(s["length"]):
        if s["task"] == "gate":
            lane = label ^ int(s["actions"][s["query"], 0] > 0)
            y = 0.36 if lane == 0 else 0.68
            if t <= s["query"]:
                x, y = 0.20, 0.52
            else:
                x = min(0.86, x + 0.035 + 0.015 * float(s["actions"][t-1, 1]))
            visible = True
            cls = lane if t > s["query"] else -1
        else:
            if t <= s["end"]:
                # Constant visible velocity before occlusion; end position matched exactly.
                x = s["xref"] + sign * s["speed"] * (t-s["end"])
            elif t > s["end"]:
                x, v = motion_step(x, v, s["actions"][t-1])
            y = s["y"]
            visible = not (s["end"] < t <= s["query"] + 1)
            cls = int(x > 0.5) if visible else -1
        frames.append(render(s, t, label, x, y, visible))
        states.append([x, y])
        classes.append(cls)
        visibility.append(visible)
    evidence = [s["start"]] if s["task"] == "gate" else [s["start"], s["end"]]
    return {"rgb": np.stack(frames), "actions": s["actions"].copy(),
            "states": np.asarray(states, np.float32), "classes": np.asarray(classes, np.int64),
            "visible": np.asarray(visibility, bool), "evidence": evidence, "label": label}
