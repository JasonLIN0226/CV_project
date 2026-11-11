"""Hidden-CoG PushT with matched probes (design v3): shared start, shared post-probe pose.

v2 (pusht_cog_coherent.py) reached a shared post-probe pose by starting each
class from a different pose, which let the reset frame leak the class. Here
every sibling starts from the same block pose and pusher position. Each class
gets its own probe push (contact face, offset, tilt, depth), solved so that the
block ends at the same post-probe pose for every class:

  1. Anchor: one class, drawn per family, gets a random probe from a library of
     pushes; its simulated outcome defines the shared post-probe pose Q.
  2. Match: for each other class, start from the library push whose block-frame
     motion is nearest, then solve (offset, tilt, depth) by least squares on the
     real episode prefix until the block lands on Q.
  3. As in v2, pushers then transit to a shared rendezvous, hold, wander without
     contact, and run the same test push.

All phases have the same step counts for every class. Siblings are identical at
reset, differ only in the probe (where/how the block is pushed, and the pusher
route), and are identical again from `converge_frame`. The class is visible only
by relating a push to the block motion it produced.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np

import pusht_cog_coherent as v2
import pusht_cog_cue as v1
from pusht_cog_cue import (COG_CLASSES, HORIZON, PUSHER_RADIUS, TEST_DIR, TEST_FACE_Y,
                           TEST_OFFSET_MAX, first_index, keypoints, make_env, place, rotate, wrap)

RECENT_LENGTH = 4
STANDOFF = 12.0
ORBIT_RADIUS = 140.0       # approach orbit around the block centre; the T reaches ~85 px
ORBIT_STEPS, INBOUND_STEPS, PAUSE_STEPS = 9, 5, 3
PUSH_STEPS, RETREAT_STEPS = 16, 3
TILT_MAX, DEPTH_RANGE = np.radians(35.0), (30.0, 150.0)
ANCHOR_TURN_MAX, ANCHOR_SHIFT_MAX = 45.0, 80.0   # degrees, px
Q_CENTER_RANGE = (160.0, 352.0)                  # keeps transit orbits inside the arena
ANGLE_SCALE = 100.0        # residual weight: 1 rad of turn ~ 100 px
MATCH_TOL = 1e-6           # max residual (px, or rad * ANGLE_SCALE)

# Faces in block-centre coordinates (v1 local minus BLOCK_CENTER): contact(s), inward normal, s range.
FACES = {
    "top":    (lambda s: (s, -60.0), (0.0, 1.0), (-55.0, 55.0)),
    "stem":   (lambda s: (s, 60.0), (0.0, -1.0), (-12.0, 12.0)),
    "underL": (lambda s: (s, -30.0), (0.0, -1.0), (-57.0, -32.0)),
    "underR": (lambda s: (s, -30.0), (0.0, -1.0), (32.0, 57.0)),
    "sideL":  (lambda s: (-60.0, s - 60.0), (1.0, 0.0), (3.0, 27.0)),
    "sideR":  (lambda s: (60.0, s - 60.0), (-1.0, 0.0), (3.0, 27.0)),
    "stemL":  (lambda s: (-15.0, s - 60.0), (1.0, 0.0), (35.0, 115.0)),
    "stemR":  (lambda s: (15.0, s - 60.0), (-1.0, 0.0), (35.0, 115.0)),
}


@dataclass(frozen=True)
class Probe:
    face: str
    offset: float
    tilt: float     # radians, rotation of the push direction from the face normal
    depth: float    # px past contact

    def clipped(self):
        lo, hi = FACES[self.face][2]
        return Probe(self.face, float(np.clip(self.offset, lo, hi)),
                     float(np.clip(self.tilt, -TILT_MAX, TILT_MAX)),
                     float(np.clip(self.depth, *DEPTH_RANGE)))


@dataclass(frozen=True)
class FamilySpec:
    seed: int
    start_center: tuple[float, float]
    start_angle: float
    pusher_angle: float      # where the pusher starts on the approach orbit
    test_offset: float
    gap_steps: int
    gap_seed: int
    anchor_seed: int


def sample_family(seed: int, *, gap_range=(8, 14)) -> FamilySpec:
    rng = np.random.default_rng(seed)
    return FamilySpec(
        seed=seed,
        start_center=tuple(rng.uniform(215, 297, size=2).round(2)),
        start_angle=float(rng.uniform(-np.pi, np.pi)),
        pusher_angle=float(rng.uniform(-np.pi, np.pi)),
        test_offset=float(rng.uniform(-TEST_OFFSET_MAX, TEST_OFFSET_MAX)),
        gap_steps=int(rng.integers(gap_range[0], gap_range[1] + 1)),
        gap_seed=int(rng.integers(2**31)),
        anchor_seed=int(rng.integers(2**31)),
    )


def probe_geometry(probe: Probe):
    """Contact point and push direction in block-centre coordinates."""
    contact_fn, normal, _ = FACES[probe.face]
    return np.array(contact_fn(probe.offset)), rotate(normal, probe.tilt)


def to_world(center, angle, local):
    return np.asarray(center, dtype=float) + rotate(local, angle)


def probe_actions(spec: FamilySpec, probe: Probe):
    """Approach (orbit, inbound), pause, push and retreat targets; fixed step counts."""
    c, a = np.asarray(spec.start_center), spec.start_angle
    contact, n = probe_geometry(probe)
    start = to_world(c, a, contact - n * (PUSHER_RADIUS + STANDOFF))
    n_w = rotate(n, a)
    # Entry point: where the outward push line meets the orbit.
    rel = start - c
    b = rel @ (-n_w)
    t = -b + np.sqrt(max(b * b - (rel @ rel - ORBIT_RADIUS**2), 0.0))
    entry = start - n_w * t
    p0 = c + ORBIT_RADIUS * np.array([np.cos(spec.pusher_angle), np.sin(spec.pusher_angle)])
    a0 = spec.pusher_angle
    a1 = np.arctan2(*(entry - c)[::-1])
    da = wrap(a1 - a0)
    orbit = [c + ORBIT_RADIUS * np.array([np.cos(a0 + da * k / ORBIT_STEPS), np.sin(a0 + da * k / ORBIT_STEPS)])
             for k in range(1, ORBIT_STEPS + 1)]
    inbound = [entry + (start - entry) * k / INBOUND_STEPS for k in range(1, INBOUND_STEPS + 1)]
    pause = [start] * PAUSE_STEPS
    travel = STANDOFF + probe.depth
    push = [start + n_w * travel * k / PUSH_STEPS for k in range(1, PUSH_STEPS + 1)]
    retreat = [push[-1] - n_w * 12 * (k + 1) for k in range(RETREAT_STEPS)]
    steps = orbit + inbound + pause + push + retreat
    phase = (["approach"] * (ORBIT_STEPS + INBOUND_STEPS) + ["pause"] * PAUSE_STEPS
             + ["probe"] * PUSH_STEPS + ["retreat"] * RETREAT_STEPS)
    return p0, steps, phase


def simulate_prefix(spec: FamilySpec, cog_class: str, probe: Probe):
    """Run reset + probe prefix; return final block-centre pose and contact count before the push."""
    p0, steps, phase = probe_actions(spec, probe)
    env = make_env(COG_CLASSES[cog_class])
    env.reset(seed=spec.seed)
    place(env, p0, spec.start_center, spec.start_angle)
    early_contacts = 0
    for action, ph in zip(steps, phase):
        env.step(action)
        if ph in ("approach", "pause"):
            early_contacts += env.n_contact_points
    c, a = v2.block_center_pose(env)
    env.close()
    return np.asarray(c), float(a), early_contacts


# ---- motion library (block-frame motion of each probe, pose-invariant up to approach lag) ----

def library_grid():
    for face, (_, _, (lo, hi)) in FACES.items():
        n_off = 5 if hi - lo > 30 else 3
        for s in np.linspace(lo, hi, n_off):
            for tilt in np.radians([-25.0, 0.0, 25.0]):
                for depth in (40.0, 70.0, 100.0, 130.0):
                    yield Probe(face, float(s), float(tilt), depth)


LIBRARY_REFERENCE = FamilySpec(seed=1, start_center=(256.0, 256.0), start_angle=0.0, pusher_angle=0.0,
                               test_offset=0.0, gap_steps=8, gap_seed=0, anchor_seed=0)


def _library_row(job):
    probe, cog = job
    c, a, early = simulate_prefix(LIBRARY_REFERENCE, cog, probe)
    return {"probe": asdict(probe), "cog": cog, "early_contacts": early,
            "motion": [float(c[0] - 256.0), float(c[1] - 256.0), float(a)]}


def build_library(path: Path | None = None, processes: int = 1):
    """Block-frame motion (dx, dy, dtheta) of every grid probe for every class.

    Cached at `path`; written atomically so an interrupted build leaves no file."""
    if path is not None and path.exists():
        return json.loads(path.read_text())
    jobs = [(probe, cog) for probe in library_grid() for cog in COG_CLASSES]
    if processes > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(processes) as pool:
            rows = pool.map(_library_row, jobs, chunksize=8)
    else:
        rows = [_library_row(job) for job in jobs]
    if path is not None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows))
        tmp.replace(path)
    return rows


def motion_vector(motion):
    return np.array([motion[0], motion[1], motion[2] * ANGLE_SCALE])


# ---- matching ----

def residual(spec, cog, probe, q_center, q_angle):
    c, a, early = simulate_prefix(spec, cog, probe.clipped())
    r = np.array([c[0] - q_center[0], c[1] - q_center[1], wrap(a - q_angle) * ANGLE_SCALE])
    return r, early


def solve_probe(spec, cog, q_center, q_angle, library, *, tries=4):
    """Least squares over (offset, tilt, depth) from the nearest library probes of this class."""
    from scipy.optimize import least_squares
    want = motion_vector([*rotate(np.asarray(q_center) - spec.start_center, -spec.start_angle),
                          wrap(q_angle - spec.start_angle)])
    cands = [r for r in library if r["cog"] == cog and r["early_contacts"] == 0]
    cands.sort(key=lambda r: np.linalg.norm(motion_vector(r["motion"]) - want))
    seen, best = set(), None
    for row in cands:
        key = (row["probe"]["face"], row["probe"]["offset"])
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > tries:
            break
        face = row["probe"]["face"]
        lo, hi = FACES[face][2]
        x0 = np.array([row["probe"]["offset"], row["probe"]["tilt"], row["probe"]["depth"]])
        bounds = ([lo, -TILT_MAX, DEPTH_RANGE[0]], [hi, TILT_MAX, DEPTH_RANGE[1]])
        fun = lambda x: residual(spec, cog, Probe(face, *map(float, x)), q_center, q_angle)[0]
        fit = least_squares(fun, x0, bounds=bounds, x_scale=[10.0, 0.2, 20.0], diff_step=1e-6,
                            xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=200)
        probe = Probe(face, *map(float, fit.x)).clipped()
        r, early = residual(spec, cog, probe, q_center, q_angle)
        err = float(np.abs(r).max())
        if early == 0 and (best is None or err < best[1]):
            best = (probe, err)
        if best is not None and best[1] < MATCH_TOL:
            break
    return best


ANCHOR_NEIGHBOR_MAX = 6.0   # anchor motion must have a library neighbour this close in every other class


def anchor_pool(library):
    """Library probes whose motion every other class can approximately reproduce.

    The v3 dev run (job 22139069) matched only 6/32 families when anchors were drawn
    from the whole library: most motions lie outside the other classes' reach."""
    from scipy.spatial import cKDTree
    names = list(COG_CLASSES)
    usable = [r for r in library if r["early_contacts"] == 0]
    trees = {c: cKDTree([motion_vector(r["motion"]) for r in usable if r["cog"] == c]) for c in names}
    pool = []
    for r in usable:
        if abs(np.degrees(r["motion"][2])) > ANCHOR_TURN_MAX or np.hypot(*r["motion"][:2]) > ANCHOR_SHIFT_MAX:
            continue
        v = motion_vector(r["motion"])
        if all(trees[c].query(v)[0] <= ANCHOR_NEIGHBOR_MAX for c in names if c != r["cog"]):
            pool.append(r)
    return pool


def design_family(spec: FamilySpec, library, *, max_anchor_draws=12, pool=None, log=None):
    """Pick an anchor probe and match every class to it. Returns None if no draw matches.

    Draws are rejected only on matchability and arena bounds, never on test outcomes.
    `log`, if a list, receives one record per draw.
    """
    rng = np.random.default_rng(spec.anchor_seed)
    names = list(COG_CLASSES)
    pool = anchor_pool(library) if pool is None else pool
    for draw in range(max_anchor_draws):
        anchor_cog = names[rng.integers(len(names))]
        options = [r for r in pool if r["cog"] == anchor_cog]
        if not options:
            continue
        row = options[rng.integers(len(options))]
        base = Probe(**row["probe"])
        lo, hi = FACES[base.face][2]
        jitter = rng.uniform(-1, 1, 3) * [(hi - lo) / 10, np.radians(5), 8.0]
        anchor = Probe(base.face, base.offset + jitter[0], base.tilt + jitter[1], base.depth + jitter[2]).clipped()
        q_center, q_angle, early = simulate_prefix(spec, anchor_cog, anchor)
        record = {"draw": draw, "anchor_class": anchor_cog, "anchor": asdict(anchor),
                  "turn_deg": float(np.degrees(wrap(q_angle - spec.start_angle)))}
        if log is not None:
            log.append(record)
        if early or not all(Q_CENTER_RANGE[0] <= v <= Q_CENTER_RANGE[1] for v in q_center):
            record["rejected"] = "anchor_contact" if early else "q_out_of_range"
            continue
        probes, errors = {anchor_cog: anchor}, {anchor_cog: 0.0}
        for cog in names:
            if cog == anchor_cog:
                continue
            found = solve_probe(spec, cog, q_center, q_angle, library)
            record[f"{cog}_best_error"] = None if found is None else found[1]
            if found is None or found[1] >= MATCH_TOL:
                record["rejected"] = f"no_match_{cog}"
                break
            probes[cog], errors[cog] = found
        else:
            return {"anchor_class": anchor_cog, "anchor_draw": draw, "probes": probes,
                    "match_error": errors, "q_center": q_center.tolist(), "q_angle": q_angle}
    return None


# ---- episodes ----

@dataclass
class Plan:
    actions: np.ndarray
    phase: list[str]
    pusher_start: np.ndarray
    shared_from_step: int


def build_plan(spec: FamilySpec, probe: Probe, q_center, q_angle) -> Plan:
    p0, steps, phase = probe_actions(spec, probe)
    q_center = np.asarray(q_center)
    test_start, test = v1.push_waypoints(q_center, q_angle, (spec.test_offset, TEST_FACE_Y), TEST_DIR,
                                         standoff=v2.STANDOFF, depth=v2.DEPTH, speed=v2.SPEED)
    d = rotate(TEST_DIR, q_angle)
    perp = np.array([-d[1], d[0]])
    rendezvous = test_start - d * 45
    transit = v2.transit_path(steps[-1], rendezvous, q_center, v2.TRANSIT_STEPS)
    hold = [rendezvous] * v2.HOLD_STEPS
    rng = np.random.default_rng(spec.gap_seed)
    gap = [rendezvous + perp * rng.uniform(-25, 25) - d * rng.uniform(0, 30)
           for _ in range(spec.gap_steps - 1)] + [test_start]
    all_steps = steps + transit + hold + gap + test
    phase = phase + (["transit"] * len(transit) + ["hold"] * len(hold) + ["gap"] * len(gap)
                     + ["test"] * len(test))
    return Plan(np.array(all_steps), phase, p0, len(steps) + len(transit))


def run_episode(spec: FamilySpec, cog_class: str, plan: Plan):
    env = make_env(COG_CLASSES[cog_class])
    env.reset(seed=spec.seed)
    place(env, plan.pusher_start, spec.start_center, spec.start_angle)
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


def check_family(spec: FamilySpec, design: dict, *, classes=tuple(COG_CLASSES), min_target_px=12.0):
    """Frame i follows action i-1."""
    q_center, q_angle = design["q_center"], design["q_angle"]
    plans = {c: build_plan(spec, design["probes"][c], q_center, q_angle) for c in classes}
    runs = {c: run_episode(spec, c, plans[c]) for c in classes}
    ref_plan = plans[classes[0]]
    phase = ["reset"] + ref_plan.phase
    shared = ref_plan.shared_from_step
    actions_shared = all(np.array_equal(p.actions[shared:], ref_plan.actions[shared:]) for p in plans.values())
    test0 = phase.index("test")
    query, target = test0 - 1, test0 - 1 + HORIZON
    first_test_contact = min((first_index(r["contacts"][test0:] > 0) or 10**9) + test0 for r in runs.values())
    probe0, transit0 = phase.index("probe"), phase.index("transit")

    ref = runs[classes[0]]["rgb"]
    same = np.array([all(np.array_equal(ref[t], r["rgb"][t]) for r in runs.values()) for t in range(len(phase))])
    diff_before = np.flatnonzero(~same[:first_test_contact])
    converge = int(diff_before[-1] + 1) if diff_before.size else 0
    first_diverge = int(diff_before[0]) if diff_before.size else None
    early_contacts = int(sum(r["contacts"][1:probe0].sum() for r in runs.values()))
    transit_contacts = int(sum(r["contacts"][transit0:test0].sum() for r in runs.values()))
    in_bounds = all(((r["keypoints"] > 20) & (r["keypoints"] < 492)).all() for r in runs.values())
    actions_in_arena = all(((p.actions > 20) & (p.actions < 492)).all() for p in plans.values())
    probe_end = transit0 - 1
    ref_kp = runs[classes[0]]["keypoints"][probe_end]
    pose_mismatch = max(float(np.abs(r["keypoints"][probe_end] - ref_kp).max()) for r in runs.values())

    def pair_px(t):
        kp = {c: r["keypoints"][t] for c, r in runs.items()}
        return {f"{a}|{b}": float(np.linalg.norm(kp[a] - kp[b], axis=1).mean())
                for i, a in enumerate(classes) for b in classes[i + 1:]}

    has_target = target < len(phase)
    target_px = pair_px(target) if has_target else {}
    turn = {c: float(np.degrees(wrap(q_angle - spec.start_angle))) for c in classes}
    result = {
        "spec": asdict(spec), "steps": len(ref_plan.actions),
        "anchor_draw": design["anchor_draw"],
        "probes": {c: asdict(p) for c, p in design["probes"].items()},
        "match_error": design["match_error"], "shared_turn_deg": turn[classes[0]],
        "first_sibling_difference_frame": first_diverge, "probe_start_frame": probe0,
        "probe_end_frame": probe_end, "shared_actions_from_frame": shared + 1,
        "actions_shared_after": actions_shared, "converge_frame": converge,
        "query_frame": query, "target_frame": target if has_target else None,
        "first_test_contact_frame": first_test_contact,
        "post_probe_keypoint_mismatch_px": pose_mismatch,
        "approach_contacts": early_contacts, "transit_gap_contacts": transit_contacts,
        "block_in_bounds": in_bounds, "actions_in_arena": actions_in_arena,
        "reset_frames_identical": bool(same[0]),
        "target_pair_keypoint_px": target_px,
        "target_min_pair_px": min(target_px.values()) if has_target else None,
    }
    result["valid"] = bool(actions_shared and early_contacts == 0 and transit_contacts == 0 and in_bounds
                           and actions_in_arena and has_target and same[0]
                           and first_test_contact > query and converge <= query - RECENT_LENGTH + 1)
    result["informative"] = bool(result["valid"] and result["target_min_pair_px"] >= min_target_px)
    return result, runs, plans
