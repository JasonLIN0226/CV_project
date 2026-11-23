"""Hidden-CoG PushT with a visible CoG marker (design v5, "marker").

The PushT analogue of Gate: the hidden CoG class is shown as a small red dot at
the CoG's position on the block for MARKER_FRAMES consecutive frames, starting at
a random early frame. There is no probe push: the block is untouched until the
test push, and every action is planned from the shared start pose, so siblings
share all actions and are pixel-identical everywhere except the marker frames and
after the first test contact. The future still depends on the class through the
test push, so a predictor must turn "where the dot was" into block dynamics.

Episode: reset -> pusher wanders on an orbit (marker shown early) -> transit to
the test side -> hold -> gap -> query -> top-face test push -> target = query + 12.
The dot is drawn on the 96x96 observation after rendering (not part of the
physics); its position comes from the block's actual pose and CoG.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

import pusht_cog_coherent as v2
import pusht_cog_cue as v1
from pusht_cog_cue import (COG_CLASSES, HORIZON, TEST_DIR, TEST_FACE_Y, TEST_OFFSET_MAX, first_index,
                           keypoints, make_env, place, rotate)

RECENT_LENGTH = 4
START_RANGE = (195.0, 317.0)
ORBIT_RADIUS = 140.0
WANDER_STEPS = 20
MARKER_FRAMES = 3
MARKER_START_RANGE = (2, 10)          # inclusive; random so a scorer cannot just keep a fixed frame
MARKER_RADIUS_PX = 2                  # at 96x96
MARKER_COLOR = (220, 20, 20)          # distinct from block (119,136,153), pusher (65,105,225), goal (144,238,144)
# v5c distractors (Nov 23): on v5b, visual diversity kept a marker frame 99% of the time because the red dot
# was the most salient thing in the history. Distractor dots (marker size, palette including the marker red)
# on empty background in every history frame make "visually distinct" stop meaning "cue"; the cue is still
# unambiguous: a red dot ON the block. Seeded per family and frame, so siblings share them. 0 = v5/v5b exactly.
DISTRACTOR_DOTS = 0                   # dots per history frame: uniform in 1..DISTRACTOR_DOTS
DISTRACTOR_PALETTE = ((220, 20, 20), (240, 140, 20), (150, 60, 200), (30, 30, 30))
BACKGROUND = (255, 255, 255)


@dataclass(frozen=True)
class FamilySpec:
    seed: int
    start_center: tuple[float, float]
    start_angle: float
    pusher_angle: float
    wander_rate: float     # rad per step along the orbit
    marker_start: int
    test_offset: float
    gap_steps: int
    gap_seed: int


def sample_family(seed: int, *, gap_range=(8, 14)) -> FamilySpec:
    rng = np.random.default_rng(seed)
    return FamilySpec(
        seed=seed,
        start_center=tuple(rng.uniform(*START_RANGE, size=2).round(2)),
        start_angle=float(rng.uniform(-np.pi, np.pi)),
        pusher_angle=float(rng.uniform(-np.pi, np.pi)),
        wander_rate=float(rng.choice([-1, 1]) * rng.uniform(0.04, 0.12)),
        marker_start=int(rng.integers(MARKER_START_RANGE[0], MARKER_START_RANGE[1] + 1)),
        test_offset=float(rng.uniform(-TEST_OFFSET_MAX, TEST_OFFSET_MAX)),
        gap_steps=int(rng.integers(gap_range[0], gap_range[1] + 1)),
        gap_seed=int(rng.integers(2**31)),
    )


def plan(spec: FamilySpec):
    """Open-loop pusher targets from the shared start pose; identical for every class."""
    c, a = np.asarray(spec.start_center), spec.start_angle
    p0 = c + ORBIT_RADIUS * np.array([np.cos(spec.pusher_angle), np.sin(spec.pusher_angle)])
    wander = [c + ORBIT_RADIUS * np.array([np.cos(spec.pusher_angle + spec.wander_rate * k),
                                           np.sin(spec.pusher_angle + spec.wander_rate * k)])
              for k in range(1, WANDER_STEPS + 1)]
    test_start, test = v1.push_waypoints(c, a, (spec.test_offset, TEST_FACE_Y), TEST_DIR,
                                         standoff=v2.STANDOFF, depth=v2.DEPTH, speed=v2.SPEED)
    d = rotate(TEST_DIR, a)
    perp = np.array([-d[1], d[0]])
    rendezvous = test_start - d * 45
    transit = v2.transit_path(wander[-1], rendezvous, c, v2.TRANSIT_STEPS)
    hold = [rendezvous] * v2.HOLD_STEPS
    rng = np.random.default_rng(spec.gap_seed)
    gap = [rendezvous + perp * rng.uniform(-25, 25) - d * rng.uniform(0, 30)
           for _ in range(spec.gap_steps - 1)] + [test_start]
    steps = wander + transit + hold + gap + test
    phase = (["reset"] + ["wander"] * len(wander) + ["transit"] * len(transit) + ["hold"] * len(hold)
             + ["gap"] * len(gap) + ["test"] * len(test))
    return p0, np.array(steps), phase


def cog_pixel(env, size=96):
    """CoG world position mapped to observation pixels (gym-pusht renders world x->col, y->row)."""
    world = np.asarray(env.block.local_to_world(env.block.center_of_gravity))
    return world / 512.0 * size


def draw_marker(rgb, xy):
    import cv2
    out = rgb.copy()
    cv2.circle(out, (int(round(xy[0])), int(round(xy[1]))), MARKER_RADIUS_PX, MARKER_COLOR, -1, lineType=cv2.LINE_8)
    return out


def draw_distractors(rgb, seed: int, t: int):
    """1..DISTRACTOR_DOTS dots on empty background (every pixel within radius + 1 is background).

    Sampled on the frame before the marker is drawn, from a generator keyed by (family seed, frame), so
    every sibling gets the same dots. Returns the new frame and the dots as (x, y, rgb)."""
    import cv2
    rng = np.random.default_rng([seed, 7_000_003, t])
    out, dots, r = rgb.copy(), [], MARKER_RADIUS_PX
    size = rgb.shape[0]
    yy, xx = np.mgrid[:size, :size]
    for _ in range(int(rng.integers(1, DISTRACTOR_DOTS + 1))):
        for _attempt in range(200):
            x, y = (int(v) for v in rng.integers(r + 2, size - r - 2, size=2))
            disk = (xx - x) ** 2 + (yy - y) ** 2 <= (r + 1) ** 2
            if (out[disk] == BACKGROUND).all():
                color = DISTRACTOR_PALETTE[int(rng.integers(len(DISTRACTOR_PALETTE)))]
                cv2.circle(out, (x, y), r, color, -1, lineType=cv2.LINE_8)
                dots.append((x, y, list(color)))
                break
    return out, dots


def run_episode(spec: FamilySpec, cog_class: str):
    p0, actions, phase = plan(spec)
    marker = set(range(spec.marker_start, spec.marker_start + MARKER_FRAMES))
    history_end = phase.index("test") - 1 - RECENT_LENGTH   # last frame that can enter memory
    env = make_env(COG_CLASSES[cog_class])
    env.reset(seed=spec.seed)
    place(env, p0, spec.start_center, spec.start_angle)
    out = {"rgb": [], "pose": [], "keypoints": [], "agent": [], "contacts": [0], "marker_px": [], "distractors": []}

    def observe(rgb):
        t = len(out["rgb"])
        xy = cog_pixel(env)
        dots = []
        if DISTRACTOR_DOTS and t <= history_end:
            rgb, dots = draw_distractors(rgb, spec.seed, t)
        out["rgb"].append(draw_marker(rgb, xy) if t in marker else rgb)
        out["marker_px"].append(xy.tolist() if t in marker else None)
        out["distractors"].append(dots)
        out["pose"].append([*env.block.position, env.block.angle])
        out["keypoints"].append(keypoints(env))
        out["agent"].append(np.array(env.agent.position))

    observe(env.get_obs())
    for action in actions:
        obs, _, _, _, _ = env.step(action)
        out["contacts"].append(env.n_contact_points)
        observe(obs)
    env.close()
    run = {k: np.stack(v) for k, v in out.items() if k in ("pose", "keypoints", "agent")}
    run.update(rgb=np.stack(out["rgb"]).astype(np.uint8), contacts=np.array(out["contacts"]),
               marker_px=out["marker_px"], distractors=out["distractors"])
    return run, actions, phase


def check_family(spec: FamilySpec, *, classes=tuple(COG_CLASSES), min_target_px=12.0):
    episodes = {c: run_episode(spec, c) for c in classes}
    runs = {c: e[0] for c, e in episodes.items()}
    actions, phase = episodes[classes[0]][1], episodes[classes[0]][2]
    marker = list(range(spec.marker_start, spec.marker_start + MARKER_FRAMES))
    test0 = phase.index("test")
    query, target = test0 - 1, test0 - 1 + HORIZON
    first_test_contact = min((first_index(r["contacts"][test0:] > 0) or 10**9) + test0 for r in runs.values())
    pre_test_contacts = int(sum(r["contacts"][1:test0].sum() for r in runs.values()))
    ref = runs[classes[0]]["rgb"]
    same = np.array([all(np.array_equal(ref[t], r["rgb"][t]) for r in runs.values()) for t in range(len(phase))])
    differ_before_test = [int(t) for t in np.flatnonzero(~same[:first_test_contact])]
    in_bounds = all(((r["keypoints"] > 20) & (r["keypoints"] < 492)).all() for r in runs.values())
    actions_in_arena = bool(((actions > 20) & (actions < 492)).all())
    marker_visible = all(not np.array_equal(r["rgb"][t], runs[c]["rgb"][t])
                         for t in marker for c in classes for r in [runs[o] for o in classes if o != c])
    has_target = target < len(phase)

    def pair_px(t):
        kp = {c: r["keypoints"][t] for c, r in runs.items()}
        return {f"{a}|{b}": float(np.linalg.norm(kp[a] - kp[b], axis=1).mean())
                for i, a in enumerate(classes) for b in classes[i + 1:]}

    target_px = pair_px(target) if has_target else {}
    result = {
        "spec": asdict(spec), "steps": len(actions), "marker_frames": marker,
        "query_frame": query, "target_frame": target if has_target else None,
        "test_start_frame": test0, "first_test_contact_frame": first_test_contact,
        "pre_test_contacts": pre_test_contacts, "frames_differing_before_test_contact": differ_before_test,
        "marker_frames_distinct_across_classes": marker_visible,
        "block_in_bounds": in_bounds, "actions_in_arena": actions_in_arena,
        "target_pair_keypoint_px": target_px,
        "target_min_pair_px": min(target_px.values()) if has_target else None,
    }
    result["valid"] = bool(pre_test_contacts == 0 and differ_before_test == marker and marker_visible
                           and in_bounds and actions_in_arena and has_target and first_test_contact > query
                           and marker[-1] < query - RECENT_LENGTH + 1)
    result["informative"] = bool(result["valid"] and result["target_min_pair_px"] >= min_target_px)
    return result, runs, actions, phase
