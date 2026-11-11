"""Hidden-CoG PushT as natural continuous episodes (design v4).

No teleport, no matching. A family shares the start pose, the pusher start and
one probe push (drawn from the v3 probe library) across the three CoG classes,
so the first frame and the probe contact are class-neutral and only the block's
response differs. After the probe, everything follows the block's actual pose:
the pusher transits around it, holds, wanders without contact and runs a
top-face test push. Recent frames may differ between classes (the turned block
stays turned); the design aims for the present to say little about the class
while the probe frames say a lot, so retaining them is what helps.

Per family the check records:
  * simulator-state features at the reset, the end of the probe and the query,
    for leak / history-gain classifiers (dev driver);
  * counterfactual divergence: from each episode's exact query state, the test
    push is re-run under every CoG; the spread at +HORIZON is how much the hidden
    cause changes the future given the same present.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

import pusht_cog_coherent as v2
import pusht_cog_cue as v1
import pusht_cog_matched as v3
from pusht_cog_cue import (COG_CLASSES, HORIZON, TEST_DIR, TEST_FACE_Y, TEST_OFFSET_MAX, first_index,
                           keypoints, make_env, place, rotate, wrap)
from pusht_cog_matched import FACES, Probe

RECENT_LENGTH = 4
START_RANGE = (195.0, 317.0)        # wider than v1/v2 (weakens the arena cue); [180, 332] lost 59/192 families out of bounds
PROBE_TURN_SPREAD_MIN = 20.0        # degrees: library probes whose class turns differ at least this much
PROBE_SHIFT_MAX = 70.0              # px: keeps the post-probe block and orbits inside the arena


@dataclass(frozen=True)
class FamilySpec:
    seed: int
    start_center: tuple[float, float]
    start_angle: float
    pusher_angle: float
    probe: dict            # asdict(Probe)
    test_offset: float
    gap_steps: int
    gap_seed: int


def probe_pool(library):
    """Library probes with no pre-push contact, bounded shift and a real turn difference."""
    by_probe = {}
    for r in library:
        by_probe.setdefault(json_key(r["probe"]), []).append(r)
    pool = []
    for rows in by_probe.values():
        if len(rows) != len(COG_CLASSES) or any(r["early_contacts"] for r in rows):
            continue
        turns = [np.degrees(r["motion"][2]) for r in rows]
        shifts = [np.hypot(*r["motion"][:2]) for r in rows]
        if max(turns) - min(turns) >= PROBE_TURN_SPREAD_MIN and max(shifts) <= PROBE_SHIFT_MAX:
            pool.append(rows[0]["probe"])
    return pool


def json_key(probe: dict):
    return (probe["face"], round(probe["offset"], 6), round(probe["tilt"], 6), round(probe["depth"], 6))


def sample_family(seed: int, pool, *, gap_range=(8, 14)) -> FamilySpec:
    rng = np.random.default_rng(seed)
    base = Probe(**pool[rng.integers(len(pool))])
    lo, hi = FACES[base.face][2]
    jitter = rng.uniform(-1, 1, 3) * [(hi - lo) / 10, np.radians(5), 8.0]
    probe = Probe(base.face, base.offset + jitter[0], base.tilt + jitter[1], base.depth + jitter[2]).clipped()
    return FamilySpec(
        seed=seed,
        start_center=tuple(rng.uniform(*START_RANGE, size=2).round(2)),
        start_angle=float(rng.uniform(-np.pi, np.pi)),
        pusher_angle=float(rng.uniform(-np.pi, np.pi)),
        probe=asdict(probe),
        test_offset=float(rng.uniform(-TEST_OFFSET_MAX, TEST_OFFSET_MAX)),
        gap_steps=int(rng.integers(gap_range[0], gap_range[1] + 1)),
        gap_seed=int(rng.integers(2**31)),
    )


def after_probe_actions(spec: FamilySpec, last_target, q_center, q_angle):
    """Transit, hold, gap and test targets planned from the block's actual post-probe pose."""
    q_center = np.asarray(q_center)
    test_start, test = v1.push_waypoints(q_center, q_angle, (spec.test_offset, TEST_FACE_Y), TEST_DIR,
                                         standoff=v2.STANDOFF, depth=v2.DEPTH, speed=v2.SPEED)
    d = rotate(TEST_DIR, q_angle)
    perp = np.array([-d[1], d[0]])
    rendezvous = test_start - d * 45
    transit = v2.transit_path(np.asarray(last_target), rendezvous, q_center, v2.TRANSIT_STEPS)
    hold = [rendezvous] * v2.HOLD_STEPS
    rng = np.random.default_rng(spec.gap_seed)
    gap = [rendezvous + perp * rng.uniform(-25, 25) - d * rng.uniform(0, 30)
           for _ in range(spec.gap_steps - 1)] + [test_start]
    steps = transit + hold + gap + test
    phase = ["transit"] * len(transit) + ["hold"] * len(hold) + ["gap"] * len(gap) + ["test"] * len(test)
    return steps, phase


def _record(env, out):
    out["pose"].append([*env.block.position, env.block.angle])
    out["keypoints"].append(keypoints(env))
    out["agent"].append(np.array(env.agent.position))


def run_episode(spec: FamilySpec, cog_class: str):
    probe = Probe(**spec.probe)
    p0, probe_steps, probe_phase = v3.probe_actions(spec, probe)
    env = make_env(COG_CLASSES[cog_class])
    env.reset(seed=spec.seed)
    place(env, p0, spec.start_center, spec.start_angle)
    out = {"rgb": [env.get_obs()], "pose": [], "keypoints": [], "agent": [], "contacts": [0],
           "agent_velocity": [np.array(env.agent.velocity)]}
    _record(env, out)

    def step(action):
        obs, _, _, _, _ = env.step(action)
        out["rgb"].append(obs)
        out["contacts"].append(env.n_contact_points)
        out["agent_velocity"].append(np.array(env.agent.velocity))
        _record(env, out)

    for action in probe_steps:
        step(action)
    q_center, q_angle = v2.block_center_pose(env)
    rest, rest_phase = after_probe_actions(spec, probe_steps[-1], q_center, q_angle)
    for action in rest:
        step(action)
    env.close()
    run = {k: np.stack(v) if k != "rgb" else np.stack(v).astype(np.uint8) for k, v in out.items()}
    run["contacts"] = np.array(out["contacts"])
    return run, np.array(probe_steps + rest), ["reset"] + probe_phase + rest_phase


def counterfactual_targets(run, actions, query, target):
    """Keypoints at `target` when the actions after `query` replay from the exact query state
    under each CoG. The pusher's velocity is restored; the block is at rest (zero damping)."""
    x, y, a = run["pose"][query]
    center = np.array([x, y]) + rotate(v1.BLOCK_CENTER, a)
    out = {}
    for cog in COG_CLASSES:
        env = make_env(COG_CLASSES[cog])
        env.reset(seed=0)
        place(env, run["agent"][query], center, a)
        env.agent.velocity = tuple(run["agent_velocity"][query])
        for action in actions[query:target]:
            env.step(action)
        out[cog] = keypoints(env)
        env.close()
    return out


def check_family(spec: FamilySpec, *, classes=tuple(COG_CLASSES)):
    episodes = {c: run_episode(spec, c) for c in classes}
    phase = episodes[classes[0]][2]
    probe0, transit0, test0 = phase.index("probe"), phase.index("transit"), phase.index("test")
    query, target = test0 - 1, test0 - 1 + HORIZON
    runs = {c: e[0] for c, e in episodes.items()}
    actions = {c: e[1] for c, e in episodes.items()}
    ref = runs[classes[0]]["rgb"]
    reset_same = all(np.array_equal(ref[0], r["rgb"][0]) for r in runs.values())
    pre_contact_same = all(np.array_equal(ref[:probe0], r["rgb"][:probe0]) for r in runs.values())
    early_contacts = int(sum(r["contacts"][1:probe0].sum() for r in runs.values()))
    transit_contacts = int(sum(r["contacts"][transit0:test0].sum() for r in runs.values()))
    first_test_contact = min((first_index(r["contacts"][test0:] > 0) or 10**9) + test0 for r in runs.values())
    in_bounds = all(((r["keypoints"] > 20) & (r["keypoints"] < 492)).all() for r in runs.values())
    actions_in_arena = all(((a > 20) & (a < 492)).all() for a in actions.values())
    has_target = target < len(phase)

    def state(r, t):
        x, y, a = r["pose"][t]
        return {"center": (np.array([x, y]) + rotate(v1.BLOCK_CENTER, a)).tolist(), "angle": float(a),
                "pusher": r["agent"][t].tolist()}

    per_class = {}
    for c, r in runs.items():
        entry = {"reset": state(r, 0), "probe_end": state(r, transit0 - 1), "query": state(r, query),
                 "probe_turn_deg": float(np.degrees(wrap(r["pose"][transit0 - 1][2] - r["pose"][0][2])))}
        if has_target:
            cf = counterfactual_targets(r, actions[c], query, target)
            names = list(cf)
            entry["counterfactual_min_pair_px"] = min(
                float(np.linalg.norm(cf[a] - cf[b], axis=1).mean()) for i, a in enumerate(names) for b in names[i + 1:])
            entry["counterfactual_true_matches_run_px"] = float(np.abs(cf[c] - r["keypoints"][target]).max())
        per_class[c] = entry
    result = {
        "spec": asdict(spec), "steps": len(phase) - 1, "probe_start_frame": probe0,
        "probe_end_frame": transit0 - 1, "query_frame": query, "target_frame": target if has_target else None,
        "first_test_contact_frame": first_test_contact,
        "reset_frames_identical": reset_same, "pre_contact_frames_identical": pre_contact_same,
        "approach_contacts": early_contacts, "transit_gap_contacts": transit_contacts,
        "block_in_bounds": in_bounds, "actions_in_arena": actions_in_arena, "per_class": per_class,
    }
    result["valid"] = bool(reset_same and early_contacts == 0 and transit_contacts == 0 and in_bounds
                           and actions_in_arena and has_target and first_test_contact > query)
    return result, runs, actions, phase
