"""Hidden-CoG PushT as one continuous physical trajectory (no scene cut).

v1 (pusht_cog_cue.py) teleported the block after the probe so siblings could
share the present. Here siblings instead start from *different* poses chosen
so the probe push leaves every class at the same block pose:

  1. Sample the post-probe pose Q shared by the family.
  2. Calibrate: run the probe (+ retreat), written in the block's own frame,
     from a reference pose for each class and record the relative motion T_c.
     Gravity-free, damping-free PushT is invariant to rigid transforms of the
     whole scene, so starting from Q * T_c^-1 ends at Q.
  3. The pushers then travel, in the same number of steps, from their
     class-dependent end points to a shared rendezvous R, hold there, wander
     without contact, and run the same test push as v1.

Siblings differ in history (start pose, probe motion, transit) and share the
present: the class is only readable by comparing an early frame with now.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

import pusht_cog_cue as v1
from pusht_cog_cue import (COG_CLASSES, HORIZON, PROBE_PUSH, PUSHER_RADIUS, TEST_DIR,
                           TEST_FACE_Y, TEST_OFFSET_MAX, first_index, keypoints, make_env,
                           place, rotate, to_world, wrap)

REFERENCE_POSE = ((256.0, 256.0), 0.0)
TRANSIT_RADIUS = 135.0   # orbit around the block center; the T reaches ~85 px
TRANSIT_STEPS = 16
HOLD_STEPS = 6
SPEED, DEPTH, STANDOFF = 6.0, 100.0, 12.0
RECENT_LENGTH = 4


@dataclass(frozen=True)
class FamilySpec:
    seed: int
    shared_center: tuple[float, float]   # block pose after the probe, shared by siblings
    shared_angle: float
    probe_offset: float
    test_offset: float
    gap_steps: int
    gap_seed: int


def sample_family(seed: int, *, gap_range=(8, 14)) -> FamilySpec:
    rng = np.random.default_rng(seed)
    return FamilySpec(
        seed=seed,
        shared_center=tuple(rng.uniform(220, 292, size=2).round(2)),
        shared_angle=float(rng.uniform(-np.pi, np.pi)),
        probe_offset=float(rng.uniform(-4, 4)),
        test_offset=float(rng.uniform(-TEST_OFFSET_MAX, TEST_OFFSET_MAX)),
        gap_steps=int(rng.integers(gap_range[0], gap_range[1] + 1)),
        gap_seed=int(rng.integers(2**31)),
    )


def probe_local(spec: FamilySpec):
    """Probe + retreat pusher targets in the block frame (BLOCK_CENTER-relative pose 0)."""
    center, angle = (0.0, 0.0), 0.0
    contact = (PROBE_PUSH[0][0] + spec.probe_offset, PROBE_PUSH[0][1])
    start, push = v1.push_waypoints(center, angle, contact, PROBE_PUSH[1],
                                    standoff=STANDOFF, depth=DEPTH, speed=SPEED)
    d = np.asarray(PROBE_PUSH[1], dtype=float)
    retreat = [push[-1] - d * 12 * (k + 1) for k in range(3)]
    return start, push + retreat


def pose_to_world(center, angle, local_pts):
    """Map points given relative to a block posed at ((0,0), 0) onto pose (center, angle)."""
    return [np.asarray(center, dtype=float) + rotate(p, angle) for p in local_pts]


def block_center_pose(env):
    """Current (center, angle) of the block, using the v1 BLOCK_CENTER convention."""
    a = float(env.block.angle)
    return np.asarray(env.block.position) + rotate(v1.BLOCK_CENTER, a), a


def calibrate(spec: FamilySpec, cog_class: str):
    """Relative motion (dp in the start frame, dtheta) of probe + retreat."""
    c0, a0 = REFERENCE_POSE
    start, steps = probe_local(spec)
    env = make_env(COG_CLASSES[cog_class])
    env.reset(seed=spec.seed)
    place(env, pose_to_world(c0, a0, [start])[0], c0, a0)
    for action in pose_to_world(c0, a0, steps):
        env.step(action)
    c1, a1 = block_center_pose(env)
    env.close()
    return rotate(c1 - np.asarray(c0), -a0), float(a1 - a0)


def start_pose(spec: FamilySpec, motion):
    dp, dtheta = motion
    angle = spec.shared_angle - dtheta
    return np.asarray(spec.shared_center) - rotate(dp, angle), angle


def transit_path(p_from, p_to, center, steps):
    """`steps` targets from p_from to p_to, via an orbit of TRANSIT_RADIUS around center."""
    center = np.asarray(center, dtype=float)
    r0, r1 = p_from - center, p_to - center
    a0, a1 = np.arctan2(r0[1], r0[0]), np.arctan2(r1[1], r1[0])
    da = wrap(a1 - a0)
    out = int(steps * 0.25)
    orbit = steps - 2 * out
    pts = []
    # out: radially to the orbit; orbit: around; in: radially to the target.
    for k in range(1, out + 1):
        rad = np.linalg.norm(r0) + (TRANSIT_RADIUS - np.linalg.norm(r0)) * k / out
        pts.append(center + rad * np.array([np.cos(a0), np.sin(a0)]))
    for k in range(1, orbit + 1):
        a = a0 + da * k / orbit
        pts.append(center + TRANSIT_RADIUS * np.array([np.cos(a), np.sin(a)]))
    for k in range(1, out + 1):
        rad = TRANSIT_RADIUS + (np.linalg.norm(r1) - TRANSIT_RADIUS) * k / out
        pts.append(center + rad * np.array([np.cos(a1), np.sin(a1)]))
    return pts


@dataclass
class Plan:
    actions: np.ndarray        # (T, 2) world targets for this sibling
    phase: list[str]
    start_center: np.ndarray
    start_angle: float
    pusher_start: np.ndarray
    shared_from_step: int      # actions are identical across siblings from here on


def build_plan(spec: FamilySpec, motion) -> Plan:
    center, angle = start_pose(spec, motion)
    start_l, probe_l = probe_local(spec)
    pusher_start = pose_to_world(center, angle, [start_l])[0]
    probe = pose_to_world(center, angle, probe_l)

    q_center, q_angle = np.asarray(spec.shared_center), spec.shared_angle
    test_start, test = v1.push_waypoints(q_center, q_angle, (spec.test_offset, TEST_FACE_Y), TEST_DIR,
                                         standoff=STANDOFF, depth=DEPTH, speed=SPEED)
    d = rotate(TEST_DIR, q_angle)
    perp = np.array([-d[1], d[0]])
    rendezvous = test_start - d * 45
    transit = transit_path(probe[-1], rendezvous, q_center, TRANSIT_STEPS)
    hold = [rendezvous] * HOLD_STEPS
    rng = np.random.default_rng(spec.gap_seed)
    gap = [rendezvous + perp * rng.uniform(-25, 25) - d * rng.uniform(0, 30)
           for _ in range(spec.gap_steps - 1)] + [test_start]
    steps = probe + transit + hold + gap + test
    n_probe = len(probe_l)
    phase = (["probe"] * (n_probe - 3) + ["retreat"] * 3 + ["transit"] * len(transit)
             + ["hold"] * len(hold) + ["gap"] * len(gap) + ["test"] * len(test))
    return Plan(np.array(steps), phase, center, angle, pusher_start, n_probe + len(transit))


def run_episode(spec: FamilySpec, cog_class: str, plan: Plan):
    env = make_env(COG_CLASSES[cog_class])
    env.reset(seed=spec.seed)
    place(env, plan.pusher_start, plan.start_center, plan.start_angle)
    frames, poses, kps, contacts, agent = [env.get_obs()], [], [keypoints(env)], [0], [np.array(env.agent.position)]
    poses.append([*env.block.position, env.block.angle])
    for action in plan.actions:
        obs, _, _, _, _ = env.step(action)
        frames.append(obs)
        poses.append([*env.block.position, env.block.angle])
        kps.append(keypoints(env))
        contacts.append(env.n_contact_points)
        agent.append(np.array(env.agent.position))
    env.close()
    return {"rgb": np.stack(frames).astype(np.uint8), "pose": np.array(poses),
            "keypoints": np.stack(kps), "contacts": np.array(contacts), "agent": np.stack(agent)}


def check_family(spec: FamilySpec, *, classes=tuple(COG_CLASSES), min_target_px=12.0):
    """Frame i follows action i-1. `converge_frame`: first frame from which all
    siblings are pixel-identical up to the first test contact."""
    motions = {c: calibrate(spec, c) for c in classes}
    plans = {c: build_plan(spec, motions[c]) for c in classes}
    runs = {c: run_episode(spec, c, plans[c]) for c in classes}
    ref_plan = plans[classes[0]]
    phase = ["reset"] + ref_plan.phase
    shared = ref_plan.shared_from_step
    actions_shared = all(np.array_equal(p.actions[shared:], ref_plan.actions[shared:]) for p in plans.values())
    test0 = phase.index("test")
    query, target = test0 - 1, test0 - 1 + HORIZON
    first_test_contact = min((first_index(r["contacts"][test0:] > 0) or 10**9) + test0 for r in runs.values())
    retreat_end = phase.index("transit")  # first transit frame = obs after first transit action

    ref = runs[classes[0]]["rgb"]
    same = np.array([all(np.array_equal(ref[t], r["rgb"][t]) for r in runs.values())
                     for t in range(len(phase))])
    diff_before = np.flatnonzero(~same[:first_test_contact])
    converge = int(diff_before[-1] + 1) if diff_before.size else 0
    transit_contacts = int(sum(r["contacts"][retreat_end:test0].sum() for r in runs.values()))
    in_bounds = all(((r["keypoints"] > 20) & (r["keypoints"] < 492)).all() for r in runs.values())
    actions_in_arena = all(((p.actions > 20) & (p.actions < 492)).all() for p in plans.values())
    probe_done = retreat_end - 1
    pose_mismatch = max(float(np.abs(r["keypoints"][probe_done] - ref_kp).max())
                        for ref_kp in [runs[classes[0]]["keypoints"][probe_done]] for r in runs.values())
    agent_mismatch_at_query = max(float(np.abs(r["agent"][query] - runs[classes[0]]["agent"][query]).max())
                                  for r in runs.values())

    def pair_px(t):
        kp = {c: r["keypoints"][t] for c, r in runs.items()}
        return {f"{a}|{b}": float(np.linalg.norm(kp[a] - kp[b], axis=1).mean())
                for i, a in enumerate(classes) for b in classes[i + 1:]}

    has_target = target < len(phase)
    target_px = pair_px(target) if has_target else {}
    result = {
        "spec": asdict(spec), "steps": len(ref_plan.actions),
        "probe_turn_deg": {c: float(np.degrees(m[1])) for c, m in motions.items()},
        "start_angle_deg": {c: float(np.degrees(wrap(p.start_angle))) for c, p in plans.items()},
        "probe_end_frame": probe_done, "shared_actions_from_frame": shared + 1, "actions_shared_after": actions_shared,
        "converge_frame": converge, "query_frame": query, "target_frame": target if has_target else None,
        "first_test_contact_frame": first_test_contact,
        "post_probe_keypoint_mismatch_px": pose_mismatch,
        "pusher_mismatch_at_query_px": agent_mismatch_at_query,
        "transit_gap_contacts": transit_contacts,
        "block_in_bounds": in_bounds, "actions_in_arena": actions_in_arena,
        "target_pair_keypoint_px": target_px,
        "target_min_pair_px": min(target_px.values()) if has_target else None,
    }
    result["valid"] = bool(actions_shared and transit_contacts == 0 and in_bounds and actions_in_arena
                           and has_target and first_test_contact > query
                           and converge <= query - RECENT_LENGTH + 1)
    result["informative"] = bool(result["valid"] and result["target_min_pair_px"] >= min_target_px)
    return result, runs, plans
