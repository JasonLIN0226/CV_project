"""PushT with a hidden, discrete center of gravity (CoG) as a dynamics cue.

The T block looks identical under every CoG class; only its response to a push
differs. Each episode has two phases separated by a scene cut:

  probe   the pusher drives the stem tip along the stem axis. The block turns
          one way, the other way, or barely turns, depending on the CoG class.
  cut     the block and pusher are placed at a fixed test pose. From here on
          frames are pixel-identical across classes until the test contact.
  gap     the pusher moves around without touching the block.
  test    a different push (crossbar face or underside). Its outcome depends on
          the CoG, which is only visible in the probe frames.

A family is one sampled spec (poses, pushes, gap) run under every CoG class with
identical actions. Siblings in a family share actions and pre-contact pixels but
differ in the hidden CoG and the future. Actions are computed from the spec only,
never from the class or from simulator outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

# Body-local coordinates of the gym-pusht T (scale 30): crossbar x in [-60, 60],
# y in [0, 30]; stem x in [-15, 15], y in [30, 120]. Default CoG is (0, 45).
COG_CLASSES = {"left": (-40.0, 20.0), "center": (0.0, 45.0), "right": (40.0, 20.0)}
PUSHER_RADIUS = 15.0
# The query is the last gap frame; the prediction target is HORIZON frames later.
# Test contact comes 4 frames after the query, so shorter horizons (+3, +6) barely
# depend on the CoG (Oct 30 measurement: 0 and ~1 px apart at 96x96).
HORIZON = 12
BLOCK_CENTER = np.array([0.0, 60.0])  # local point used to place the block
# Pushes are (local contact point, inward direction). The probe drives the stem
# tip toward the crossbar; the test drives the crossbar's top face toward the
# stem at a lateral offset. Both act along the local y axis, so the lateral CoG
# offset sets the torque sign and size. A dev sweep (Oct 30) found top-face
# offsets within +-20 separate all three classes; underside and side pushes
# leave one pair nearly identical because two CoGs share a side of the line.
PROBE_PUSH = ((0.0, 120.0), (0.0, -1.0))
TEST_FACE_Y, TEST_DIR, TEST_OFFSET_MAX = 0.0, (0.0, 1.0), 20.0


def rotate(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    v = np.asarray(v, dtype=float)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


@dataclass(frozen=True)
class FamilySpec:
    seed: int
    probe_center: tuple[float, float]
    probe_angle: float
    probe_offset: float  # lateral offset of the probe contact on the stem tip
    test_center: tuple[float, float]
    test_angle: float
    test_offset: float  # lateral contact offset on the crossbar's top face
    gap_steps: int
    gap_seed: int


def sample_family(seed: int, *, gap_range=(12, 20)) -> FamilySpec:
    rng = np.random.default_rng(seed)
    return FamilySpec(
        seed=seed,
        probe_center=tuple(rng.uniform(210, 300, size=2).round(2)),
        probe_angle=float(rng.uniform(-np.pi, np.pi)),
        probe_offset=float(rng.uniform(-4, 4)),
        test_center=tuple(rng.uniform(210, 300, size=2).round(2)),
        test_angle=float(rng.uniform(-np.pi, np.pi)),
        test_offset=float(rng.uniform(-TEST_OFFSET_MAX, TEST_OFFSET_MAX)),
        gap_steps=int(rng.integers(gap_range[0], gap_range[1] + 1)),
        gap_seed=int(rng.integers(2**31)),
    )


def block_origin(center, angle):
    return np.asarray(center, dtype=float) - rotate(BLOCK_CENTER, angle)


def to_world(center, angle, local):
    return block_origin(center, angle) + rotate(local, angle)


def push_waypoints(center, angle, contact, inward, *, standoff, depth, speed):
    """Pusher targets from `standoff` px before contact to `depth` px past it."""
    p = to_world(center, angle, contact)
    d = rotate(inward, angle)
    start = p - d * (PUSHER_RADIUS + standoff)
    n = int(np.ceil((standoff + depth) / speed))
    return start, [start + d * speed * (k + 1) for k in range(n)]


@dataclass
class Plan:
    actions: np.ndarray       # (T, 2) absolute pusher targets, one per step
    phase: list[str]          # phase label of each step
    cut_step: int             # step index at which the scene cut happens
    probe_start: np.ndarray   # pusher position at reset
    test_start: np.ndarray    # pusher position right after the cut


def build_plan(spec: FamilySpec, *, speed=6.0, depth=100.0) -> Plan:
    contact = (PROBE_PUSH[0][0] + spec.probe_offset, PROBE_PUSH[0][1])
    probe_start, probe = push_waypoints(spec.probe_center, spec.probe_angle, contact,
                                        PROBE_PUSH[1], standoff=12, depth=depth, speed=speed)
    probe_dir = rotate(PROBE_PUSH[1], spec.probe_angle)
    retreat = [probe[-1] - probe_dir * 12 * (k + 1) for k in range(3)]

    test_start, test = push_waypoints(spec.test_center, spec.test_angle,
                                      (spec.test_offset, TEST_FACE_Y), TEST_DIR,
                                      standoff=12, depth=depth, speed=speed)
    d = rotate(TEST_DIR, spec.test_angle)
    perp = np.array([-d[1], d[0]])
    # Gap: wander in a box behind the test start (away from the block), end at start.
    rng = np.random.default_rng(spec.gap_seed)
    hold = test_start - d * 45
    gap = [hold + perp * rng.uniform(-25, 25) - d * rng.uniform(0, 30)
           for _ in range(spec.gap_steps - 1)] + [test_start]

    steps = probe + retreat + [test_start - d * 45] + gap + test
    phase = (["probe"] * len(probe) + ["retreat"] * len(retreat) + ["cut"]
             + ["gap"] * len(gap) + ["test"] * len(test))
    return Plan(np.array(steps), phase, len(probe) + len(retreat), probe_start,
                test_start - d * 45)


def make_env(cog):
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (registers the environment)
    return gym.make("gym_pusht/PushT-v0", obs_type="pixels", render_mode="rgb_array",
                    block_cog=tuple(cog)).unwrapped


def place(env, agent_xy, center, angle):
    """Teleport block and pusher. Angle first, then position: pymunk rotates
    about the CoG, so the reverse order would shift the block per CoG class.

    Chipmunk stores an overlap-correction (bias) velocity from the last solve
    and applies it on the next position update; it is not exposed, and it
    would nudge the block with no visible contact. One physics step from the
    new poses spends it, then the poses are set again. The new poses must not
    overlap anything, so that step stores no new bias."""
    block, agent = env.block, env.agent
    for flush in (True, False):
        block.velocity, block.angular_velocity = (0.0, 0.0), 0.0
        block.angle = float(angle)
        block.position = tuple(block_origin(center, angle))
        agent.velocity = (0.0, 0.0)
        agent.position = tuple(np.asarray(agent_xy, dtype=float))
        if flush:
            env.space.step(env.dt)
    env.space.reindex_shapes_for_body(block)
    env.space.reindex_shapes_for_body(agent)


def keypoints(env):
    return env.get_keypoints(env._block_shapes)


def run_episode(spec: FamilySpec, cog_class: str, plan: Plan | None = None):
    plan = plan or build_plan(spec)
    env = make_env(COG_CLASSES[cog_class])
    env.reset(seed=spec.seed)
    place(env, plan.probe_start, spec.probe_center, spec.probe_angle)
    frames, poses, kps, contacts = [env.get_obs()], [], [keypoints(env)], [0]
    poses.append([*env.block.position, env.block.angle])
    for t, action in enumerate(plan.actions):
        if t == plan.cut_step:
            place(env, plan.test_start, spec.test_center, spec.test_angle)
            env._last_action = action
            obs, n = env.get_obs(), 0
        else:
            obs, _, _, _, _ = env.step(action)
            n = env.n_contact_points
        frames.append(obs)
        poses.append([*env.block.position, env.block.angle])
        kps.append(keypoints(env))
        contacts.append(n)
    env.close()
    return {"rgb": np.stack(frames).astype(np.uint8), "pose": np.array(poses),
            "keypoints": np.stack(kps), "contacts": np.array(contacts)}


def first_index(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def check_family(spec: FamilySpec, *, classes=tuple(COG_CLASSES), min_target_px=12.0):
    """Run every class and report identity, cue strength and future divergence.

    Frame index i is the observation after action i-1 (index 0 is the reset).
    """
    plan = build_plan(spec)
    runs = {c: run_episode(spec, c, plan) for c in classes}
    phase = ["reset"] + plan.phase
    cut = plan.cut_step + 1
    test_idx = [i for i, p in enumerate(phase) if p == "test"]
    first_probe_contact = min(first_index(r["contacts"][1:cut] > 0) or 10**9 for r in runs.values()) + 1
    first_test_contact = min(
        (first_index(r["contacts"][test_idx[0]:] > 0) or 10**9) + test_idx[0] for r in runs.values())

    ref = runs[classes[0]]
    pre_probe_same = all(np.array_equal(ref["rgb"][:first_probe_contact], r["rgb"][:first_probe_contact])
                         for r in runs.values())
    pre_test_same = all(np.array_equal(ref["rgb"][cut:first_test_contact], r["rgb"][cut:first_test_contact])
                        for r in runs.values())
    gap_contacts = int(sum(r["contacts"][cut:test_idx[0]].sum() for r in runs.values()))
    in_bounds = all(((r["keypoints"] > 20) & (r["keypoints"] < 492)).all() for r in runs.values())
    actions_in_arena = bool(((plan.actions > 20) & (plan.actions < 492)).all())

    query = test_idx[0] - 1
    target = query + HORIZON
    has_target = target < len(phase)

    def pair_px_at(t):
        kp = {c: r["keypoints"][t] for c, r in runs.items()}
        return {f"{a}|{b}": float(np.linalg.norm(kp[a] - kp[b], axis=1).mean())
                for i, a in enumerate(classes) for b in classes[i + 1:]}

    target_px = pair_px_at(target) if has_target else {}
    probe_turn = {c: float(np.degrees(wrap(r["pose"][cut - 1, 2] - spec.probe_angle))) for c, r in runs.items()}
    final_kp = {c: r["keypoints"][-1] for c, r in runs.items()}
    pair_px = {f"{a}|{b}": float(np.linalg.norm(final_kp[a] - final_kp[b], axis=1).mean())
               for i, a in enumerate(classes) for b in classes[i + 1:]}
    probe_kp = {c: r["keypoints"][cut - 1] for c, r in runs.items()}
    probe_px = {f"{a}|{b}": float(np.linalg.norm(probe_kp[a] - probe_kp[b], axis=1).mean())
                for i, a in enumerate(classes) for b in classes[i + 1:]}
    result = {
        "spec": asdict(spec),
        "steps": len(plan.actions), "cut_frame": cut, "query_frame": query,
        "horizon": HORIZON, "target_frame": target if has_target else None,
        "first_probe_contact_frame": first_probe_contact,
        "first_test_contact_frame": first_test_contact,
        "pre_probe_pixels_identical": pre_probe_same,
        "pre_test_pixels_identical": pre_test_same,
        "gap_contacts": gap_contacts,
        "block_in_bounds": in_bounds, "actions_in_arena": actions_in_arena,
        "probe_turn_deg": probe_turn,
        "probe_pair_keypoint_px": probe_px,
        "target_pair_keypoint_px": target_px,
        "target_min_pair_px": min(target_px.values()) if has_target else None,
        "final_pair_keypoint_px": pair_px,
        "final_min_pair_px": min(pair_px.values()),
        "test_final_turn_deg": {c: float(np.degrees(wrap(r["pose"][-1, 2] - spec.test_angle)))
                                for c, r in runs.items()},
    }
    # Contact during the retreat is harmless: the scene cut follows it.
    # The recent window must end before any test contact, or it shows the class.
    result["valid"] = bool(pre_probe_same and pre_test_same and gap_contacts == 0
                           and in_bounds and actions_in_arena and has_target
                           and first_test_contact > query)
    result["informative"] = bool(result["valid"] and result["target_min_pair_px"] >= min_target_px)
    return result, runs, plan


def contact_sheet(runs, plan, path, *, n_cols=12):
    from PIL import Image, ImageDraw
    phase = ["reset"] + plan.phase
    T = len(phase)
    cols = sorted(set(np.linspace(0, T - 1, n_cols).round().astype(int))
                  | {plan.cut_step, plan.cut_step + 1})
    tile, pad, label_w = 96, 2, 60
    rows = list(runs)
    W = label_w + len(cols) * (tile + pad)
    H = 14 + len(rows) * (tile + pad)
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    for j, t in enumerate(cols):
        draw.text((label_w + j * (tile + pad), 1), f"{t}:{phase[t][:5]}", fill="black")
    for i, c in enumerate(rows):
        draw.text((2, 14 + i * (tile + pad) + 40), c, fill="black")
        for j, t in enumerate(cols):
            img.paste(Image.fromarray(runs[c]["rgb"][t]), (label_w + j * (tile + pad), 14 + i * (tile + pad)))
    img.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=2026110000)
    parser.add_argument("--sheets", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)

    results = []
    for k in range(args.families):
        spec = sample_family(args.base_seed + k)
        result, runs, plan = check_family(spec)
        results.append(result)
        if k < args.sheets:
            contact_sheet(runs, plan, args.output / f"family_{spec.seed}.png")
    valid = [r for r in results if r["valid"]]
    by_offset = {}
    for r in valid:
        by_offset.setdefault(f"|offset|<{5 * (abs(r['spec']['test_offset']) // 5 + 1):.0f}",
                             []).append(r["target_min_pair_px"])
    summary = {
        "cog_classes": COG_CLASSES, "families": len(results), "valid": len(valid),
        "informative": sum(r["informative"] for r in results),
        "probe_turn_deg_median": {c: float(np.median([r["probe_turn_deg"][c] for r in valid]))
                                  for c in COG_CLASSES} if valid else None,
        "probe_min_pair_px_median": float(np.median([min(r["probe_pair_keypoint_px"].values())
                                                     for r in valid])) if valid else None,
        "horizon": HORIZON,
        "target_min_pair_px_median_by_offset": {k: float(np.median(v)) for k, v in sorted(by_offset.items())},
        "target_min_pair_px_quantiles": (np.quantile([r["target_min_pair_px"] for r in valid],
                                                   [0.1, 0.25, 0.5, 0.75, 0.9]).tolist() if valid else None),
        "invalid_reasons": {
            "pre_probe_pixels_differ": sum(not r["pre_probe_pixels_identical"] for r in results),
            "pre_test_pixels_differ": sum(not r["pre_test_pixels_identical"] for r in results),
            "gap_contact": sum(r["gap_contacts"] > 0 for r in results),
            "block_out_of_bounds": sum(not r["block_in_bounds"] for r in results),
            "actions_out_of_arena": sum(not r["actions_in_arena"] for r in results),
        },
    }
    (args.output / "families.json").write_text(json.dumps(results, indent=1) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
