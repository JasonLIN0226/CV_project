"""Raw visual terminal tasks: redundancy, decoy and relational XOR synergy.

Only RGB and executed actions are learner observations. Records, roles, bits,
tags, class IDs, states and semantic deletion sets are evaluator metadata. All
families use a legal R=4, K=2 FIFO parent at q with cues q-6,q-5,q-4; admission
stops after that single terminal deletion. This module never encodes or fits.

The runner owns the frozen split seeds/counts. A base family contains R12,
D12 and S48 episodes. S uses all six nonconstant binary-tag triples and all
eight slot-payload triples. At fixed family/slot/tag/payload its cue pixels are
identical regardless of the other candidates or its evaluator-assigned role.
"""
from __future__ import annotations

from collections import Counter
from itertools import permutations, product
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw

VERSION = "terminal-structures-v1"
IMAGE_SIZE = 96
RECENT_LENGTH = 4
MEMORY_CAPACITY = 2
HORIZONS = (3, 6)
TASK_FAMILIES = ("R", "D", "S")
BITS = tuple(product((0, 1), repeat=2))
SLOT_BITS = tuple(product((0, 1), repeat=3))
TAG_PATTERNS = tuple(t for t in SLOT_BITS if len(set(t)) == 2)
R_STYLES = ("neutral", "salient_redundant", "salient_unique")
D_STYLES = ("neutral", "salient_decoy", "salient_useful")
R_ORDERS = tuple(permutations(("A", "Aprime", "B")))
D_ORDERS = tuple(permutations(("A", "B", "D")))
AXES = ("a", "b")
PUBLIC_OBSERVATION_KEYS = ("rgb", "actions")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _integer(value, name):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 0:
        raise ValueError(name+" must be a nonnegative integer")
    return int(value)


def family(seed, index, split):
    """Base nuisance scene shared by all 72 episodes; seeds are runner-owned."""
    seed, index = _integer(seed, "seed"), _integer(index, "index")
    if split not in ("train", "validation", "geometry", "development", "confirmation"):
        raise ValueError("Unknown split")
    family_id = digest([VERSION, seed, index, split])[:24]
    rng = np.random.default_rng(int(digest([VERSION, "public", seed, index, split])[:16], 16))
    q = int(rng.integers(12, 19))
    length = q+max(HORIZONS)+1
    controls = np.zeros((length-1, 2), np.float32)
    controls[q:] = rng.uniform(.75, 1.05, (6, 2)).astype(np.float32)
    return {"version": VERSION, "family_id": family_id, "seed": seed, "index": index, "split": split,
            "image_size": IMAGE_SIZE, "query": q, "retention_stop": q, "length": length,
            "recent_length": RECENT_LENGTH, "memory_capacity": MEMORY_CAPACITY,
            "cue_times": [q-6, q-5, q-4], "recent_times": list(range(q-3, q+1)),
            "horizons": list(HORIZONS), "target_times": [q+h for h in HORIZONS],
            "origin": [float(rng.uniform(.47, .53)), float(rng.uniform(.61, .65))],
            "motion_scale": .03, "actions": controls.tolist(),
            "decor": [[int(v) for v in row] for row in rng.integers(75, 170, (3, 3))],
            "duplicate_axis": AXES[index % 2], "order_index": (index//2) % 6,
            "salient_redundant_role": ("A", "Aprime")[(index//12) % 2],
            "salient_useful_role": ("A", "B")[(index//12) % 2],
            "decoy_axis": AXES[int(rng.integers(2))], "decoy_bit": int(rng.integers(2)),
            # Fixed by position within a family, never by the eventual S role.
            "synergy_salient_slot": None if index % 4 == 0 else index % 4-1}


def target_class_count(task_family):
    if task_family not in TASK_FAMILIES:
        raise ValueError("Unknown task family")
    return 2 if task_family == "S" else 4


def _record(base, task, condition, role_order, axes, payload, tags, bits, outcome, correct, salient):
    group = digest([base["family_id"], task, condition, tags if task == "S" else None])[:24]
    rid = digest([VERSION, group, payload])[:24]
    return {"id": rid, "episode_id": rid, "family": base["family_id"], "family_id": base["family_id"],
            "split": base["split"], "task_family": task, "condition": condition,
            "query": base["query"], "retention_stop": base["query"],
            "evidence": list(base["cue_times"]), "target_times": list(base["target_times"]),
            "counterfactual_group": group, "quartet": group, "roles": list(role_order),
            "semantic_axes": list(axes), "slot_payload": list(payload), "tags": tags,
            "bits": list(bits), "factor_bits": list(payload) if task == "S" else list(bits),
            "outcome_class": int(outcome), "target_class_count": target_class_count(task),
            "semantic_correct_delete_set": list(correct), "salient_slot": salient, "scene": base}


def records_for_family(base):
    """Return JSON metadata; never pass this dict directly as a model input."""
    if base.get("version") != VERSION:
        raise ValueError("Require this generator's base family")
    out = []
    r_order = R_ORDERS[base["order_index"]]
    duplicate = base["duplicate_axis"]
    other = AXES[1-AXES.index(duplicate)]
    r_axes = [other if role == "B" else duplicate for role in r_order]
    for style in R_STYLES:
        salient_role = None if style == "neutral" else (base["salient_redundant_role"] if style == "salient_redundant" else "B")
        salient = None if salient_role is None else r_order.index(salient_role)
        for bits in BITS:
            payload = [bits[AXES.index(axis)] for axis in r_axes]
            out.append(_record(base, "R", style, r_order, r_axes, payload, None, bits, 2*bits[0]+bits[1],
                               [j for j, role in enumerate(r_order) if role != "B"], salient))
    d_order = D_ORDERS[base["order_index"]]
    d_axes = [{"A": "a", "B": "b", "D": base["decoy_axis"]}[role] for role in d_order]
    for style in D_STYLES:
        salient_role = None if style == "neutral" else ("D" if style == "salient_decoy" else base["salient_useful_role"])
        salient = None if salient_role is None else d_order.index(salient_role)
        for bits in BITS:
            payload = [{"A": bits[0], "B": bits[1], "D": base["decoy_bit"]}[role] for role in d_order]
            out.append(_record(base, "D", style, d_order, d_axes, payload, None, bits, 2*bits[0]+bits[1],
                               [d_order.index("D")], salient))
    salient = base["synergy_salient_slot"]
    condition = "neutral" if salient is None else "salient_slot_"+str(salient)
    for tags in TAG_PATTERNS:
        singleton = next(j for j, tag in enumerate(tags) if tags.count(tag) == 1)
        pair = [j for j in range(3) if j != singleton]
        roles = ["D" if j == singleton else ("A" if j == pair[0] else "B") for j in range(3)]
        for payload in SLOT_BITS:
            bits = [payload[j] for j in pair]
            out.append(_record(base, "S", condition, roles, ["parity"]*3, payload, list(tags), bits,
                               bits[0] ^ bits[1], [singleton], salient))
    if len(out) != 72 or len({r["id"] for r in out}) != 72:
        raise AssertionError("Family must have exactly72 distinct records")
    return out


def content_oracle_delete_set(record):
    """Evaluator-only semantic action set, not a learned-reader loss oracle."""
    return tuple(record["semantic_correct_delete_set"])


def _arrow(draw, axis, bit, center, lengths, color, width):
    cx, cy = center
    dx, dy = (lengths[0], 0) if axis == "a" else (0, lengths[1])
    sign = 2*int(bit)-1
    start, end = (cx-sign*dx/2, cy-sign*dy/2), (cx+sign*dx/2, cy+sign*dy/2)
    draw.line((start, end), fill=color, width=width)
    wings = [(end[0]-sign*7, end[1]-6), (end[0]-sign*7, end[1]+6)] if axis == "a" else [(end[0]-6, end[1]-sign*7), (end[0]+6, end[1]-sign*7)]
    draw.polygon([end]+wings, fill=color)


def _render_frame(record, t, state):
    base, task = record["scene"], record["task_family"]
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (24, 29, 38))
    draw = ImageDraw.Draw(image)
    draw.rectangle((7, 35, 89, 91), fill=(44, 49, 60), outline=(112, 121, 136), width=2)
    # Visible fixed grammar marker: identifies only task rule, never cue roles.
    mode_color = {"R": (104, 164, 211), "D": (106, 191, 139), "S": (195, 134, 213)}[task]
    draw.rectangle((10, 38, 22, 50), fill=mode_color)
    draw.text((13, 38), {"R": "R", "D": "D", "S": "X"}[task], fill=(20, 25, 33))
    for j, color in enumerate(base["decor"]):
        draw.rectangle((13+28*j, 92, 23+28*j, 95), fill=tuple(color))
    x, y = float(state[0])*IMAGE_SIZE, float(state[1])*IMAGE_SIZE
    draw.ellipse((x-4, y-4, x+4, y+4), fill=(237, 175, 70), outline=(255, 235, 174), width=1)
    if t not in record["evidence"]:
        return np.asarray(image, dtype=np.uint8).copy()
    slot = record["evidence"].index(t)
    salient = slot == record["salient_slot"]
    if salient:
        draw.rectangle((1, 1, 94, 33), fill=(213, 179, 70), outline=(230, 97, 181), width=3)
        for yy in range(5, 30, 8):
            draw.rectangle((5, yy, 23, yy+3), fill=(111, 52, 150))
        center, lengths, color, width = (61, 17), (38, 23), (31, 39, 57), 5
    else:
        draw.rectangle((15, 3, 81, 31), fill=(47, 67, 83), outline=(144, 164, 181), width=2)
        center, lengths, color, width = (48, 17), (29, 19), (240, 234, 202), 4
    bit = record["slot_payload"][slot]
    axis = "a" if task == "S" else record["semantic_axes"][slot]
    if task == "S":
        # Only local tag and local payload affect this frame. No A/B/D marker.
        center = (60, 17)
        lengths = (26, 19)
    _arrow(draw, axis, bit, center, lengths, color, width)
    if task == "S":
        tag = record["tags"][slot]
        box = (25, 9, 39, 24)
        if tag == 0:
            draw.ellipse(box, fill=(69, 183, 224), outline=(240, 244, 247), width=2)
        else:
            draw.rectangle(box, fill=(230, 94, 108), outline=(240, 244, 247), width=2)
    elif task == "D" and record["roles"][slot] == "D":
        # Observable stable non-task channel badge; D is not secretly assigned.
        draw.ellipse((19, 10, 30, 23), outline=(244, 244, 244), width=2)
        draw.line((19, 23, 30, 10), fill=(244, 244, 244), width=2)
    return np.asarray(image, dtype=np.uint8).copy()


def _state_actions(record, future_actions=None):
    base = record["scene"]
    if base.get("version") != VERSION or record["task_family"] not in TASK_FAMILIES:
        raise ValueError("Invalid terminal-structures record")
    q = int(record["query"])
    actions = np.asarray(base["actions"], np.float32).copy()
    if future_actions is not None:
        plan = np.asarray(future_actions, np.float32)
        if plan.shape != (6, 2) or not np.isfinite(plan).all() or np.any(plan <= 0):
            raise ValueError("Future plan must be six finite positive two-axis controls")
        actions[q:q+6] = plan
    if actions.shape != (base["length"]-1, 2) or np.any(actions[:q]) or np.any(actions[q:] <= 0):
        raise ValueError("Invalid zero-prefix/positive-query controls")
    outcome = record["outcome_class"]
    bits = (outcome, 1) if record["task_family"] == "S" else (outcome//2, outcome % 2)
    signs = np.asarray([2*b-1 for b in bits], np.float32)
    states = np.repeat(np.asarray(base["origin"], np.float32)[None], base["length"], axis=0)
    for t in range(q+1, base["length"]):
        states[t] = states[t-1]+signs*np.float32(base["motion_scale"])*actions[t-1]
    if not np.isfinite(states).all() or not np.all((states > .08) & (states < .90)):
        raise ValueError("Controls moved state outside the visible arena")
    return states, actions


def reference_plan(record):
    """Original exogenous six-step plan; evaluator/offline TRAIN-bank helper."""
    q = record["query"]
    return np.asarray(record["scene"]["actions"], np.float32)[q:q+6].copy()


def target_frames(record, plan=None, horizons=HORIZONS):
    """Render only future targets; equal to episode(...)[q+h] pixel for pixel.

    At fixed family/task/outcome/plan these pixels are independent of cue style,
    tag pattern and payload factorization, enabling offline target caching.
    This function belongs only on supervision/evaluation paths.
    """
    horizons = tuple(horizons)
    if not horizons or any(not isinstance(h, (int, np.integer)) or not 1 <= h <= 6 for h in horizons):
        raise ValueError("Target horizons must be integers in1..6")
    states, _ = _state_actions(record, plan)
    q = record["query"]
    return np.stack([_render_frame(record, q+h, states[q+h]) for h in horizons])


def episode(record, future_actions=None):
    """Render observations plus separately named evaluator metadata.

    An optional six-step positive plan changes only q→q+6 transitions and
    targets. It never changes current/recent/candidate observations or prefix
    actions. Raw encoder caching should key RGB bytes, not metadata IDs.
    """
    states, actions = _state_actions(record, future_actions)
    q = int(record["query"])
    rgb = np.stack([_render_frame(record, t, state) for t, state in enumerate(states)])
    return {"id": record["id"], "family": record["family"], "task_family": record["task_family"],
            "rgb": rgb, "actions": actions, "states": states, "query": q, "retention_stop": q,
            "evidence": np.asarray(record["evidence"], np.int64),
            "target_times": np.asarray(record["target_times"], np.int64),
            "outcome_class": record["outcome_class"], "semantic_correct_delete_set": tuple(record["semantic_correct_delete_set"]),
            "observation_keys": PUBLIC_OBSERVATION_KEYS}


render = episode


def validate_family(base):
    """Raw structural invariants only: no encoder, reader or policy outcomes."""
    records = records_for_family(base)
    groups, prefixes, local_s = {}, {}, {}
    for r in records:
        ep = episode(r)
        q = r["query"]
        groups.setdefault(r["counterfactual_group"], []).append((r, ep))
        mask = [t for t in range(q+1) if t not in r["evidence"]]
        signature = ep["rgb"][mask].tobytes()
        if r["task_family"] in prefixes and prefixes[r["task_family"]] != signature:
            raise AssertionError("Information leaked outside cues before query")
        prefixes[r["task_family"]] = signature
        if r["task_family"] == "S":
            for j, t in enumerate(r["evidence"]):
                key = (j, r["tags"][j], r["slot_payload"][j])
                raw = ep["rgb"][t].tobytes()
                if key in local_s and local_s[key] != raw:
                    raise AssertionError("S cue pixels depend on another candidate or role")
                local_s[key] = raw
    for members in groups.values():
        task = members[0][0]["task_family"]
        expected = target_class_count(task)
        for h in HORIZONS:
            targets = {ep["rgb"][r["query"]+h].tobytes() for r, ep in members}
            if len(targets) != expected:
                raise AssertionError("Unexpected distinct target count")
        if len(members) != (8 if task == "S" else 4):
            raise AssertionError("Incomplete factorial group")
    return {"status": "passed_raw_structure", "family": base["family_id"],
            "counts": dict(Counter(r["task_family"] for r in records)),
            "synergy_local_pixel_keys": len(local_s), "counterfactual_groups": len(groups),
            "encoding_calls": 0, "model_calls": 0}
