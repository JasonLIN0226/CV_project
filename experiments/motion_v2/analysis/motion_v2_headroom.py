"""Model-free online B=2 retention headroom: Motion v1 (Phase 1) vs a proposed Motion v2.

Success = at query time the two retained history frames are jointly sufficient to
recover the final motion direction (both visible, and for v2 both after the last
velocity change). No encoder or predictor is used; diversity/surprise use raw pixels
as a cheap proxy for their feature-space versions.
"""
import json, sys
import numpy as np
from PIL import Image, ImageDraw

N = 96
RECENT = 4


def render(x, visible, distractors):
    im = Image.new("RGB", (N, N), (24, 29, 38)); d = ImageDraw.Draw(im)
    b = lambda r: tuple(int(v * N) for v in r)
    d.rectangle(b((0.05, 0.25, 0.95, 0.78)), fill=(44, 49, 60), outline=(110, 118, 133), width=2)
    if visible:
        r = 0.055; d.ellipse(b((x - r, 0.5 - r, x + r, 0.5 + r)), fill=(237, 175, 70))
    else:
        d.rectangle(b((0.055, 0.255, 0.945, 0.775)), fill=(80, 89, 102))
    for xx, cc in distractors:
        xx = 0.08 + 0.84 * xx
        d.ellipse(b((xx - 0.025, 0.86, xx + 0.025, 0.91)), fill=(85, int(80 + 130 * cc), 115))
    return np.asarray(im, np.float32) / 255.0


def reflect(x, v):
    while x < 0.08 or x > 0.92:
        x, v = (0.16 - x, -v) if x < 0.08 else (1.84 - x, -v)
    return x, v


def episode(version, long_gap, rng, label):
    gap = int(rng.integers(20, 33) if long_gap else rng.integers(5, 11))
    if version == "v1":  # Phase 1: visible constant velocity from t=0 to end, then shutter
        start = int(rng.integers(2, 8)); end = start + int(rng.integers(3, 7))
        q = end + gap + RECENT
        vis = np.array([t <= end for t in range(q + 1)])
        sufficient = vis.copy()  # any two visible frames give velocity
        sign = 2 * label - 1; speed = rng.uniform(0.005, 0.007); xref = rng.uniform(0.38, 0.62)
        xs = [xref + sign * speed * (t - end) for t in range(q + 1)]
        moving_d = False
    else:  # v2: hidden pre-roll, shared stale segment, velocity change, random blips
        while True:
            t_on = int(rng.integers(0, 7)); L0 = int(rng.integers(4, 11)); L1 = int(rng.integers(3, 8))
            t_c = t_on + L0; end = t_c + L1 - 1; q = end + gap + RECENT
            vis = np.array([(t_on <= t <= end) and rng.random() > 0.25 for t in range(q + 1)])
            vis[end] = True
            post = vis & (np.arange(q + 1) >= t_c)
            if post.sum() >= 2 and vis[:t_c].sum() >= 2:
                break
        sufficient = post
        v0 = rng.choice([-1, 1]) * rng.uniform(0.005, 0.007)  # identical for both siblings
        v1 = (2 * label - 1) * rng.uniform(0.008, 0.012)      # sibling-specific final velocity
        xs = [0.0] * (q + 1); x, v = 0.5, v0
        for t in range(t_c, -1, -1):
            xs[t] = x; x, v = reflect(x - v, v)
        x, v = xs[t_c], v1
        for t in range(t_c + 1, q + 1):
            x, v = reflect(x + v, v); xs[t] = x
        moving_d = True
    dist = rng.uniform(0, 1, (3, 2))
    frames = []
    for t in range(q + 1):
        if moving_d:
            dist = np.clip(dist + rng.normal(0, 0.08, dist.shape), 0, 1)
        frames.append(render(xs[t], vis[t], dist))
    return dict(q=q, vis=vis, sufficient=sufficient, frames=frames)


def run_policy(name, ep, rng):
    stop = ep["q"] - RECENT + 1  # history frames 0..stop-1 (recent window excluded)
    F = ep["frames"]
    surprise = [0.0] + [float(np.abs(F[t] - F[t - 1]).mean()) for t in range(1, stop)]
    mem = [0, 1]
    for t in range(2, stop):
        cand = mem + [t]
        if name == "fifo":
            mem = cand[1:]
        elif name == "keep_first":
            pass
        elif name == "reservoir":
            if rng.random() < 2 / (t + 1):
                mem[rng.integers(2)] = t
        elif name == "temporal_buckets":
            b = {}
            for c in cand:
                b[min(1, 2 * c // (t + 1))] = c  # newest in each half of [0, t]
            keep = sorted(b.values())
            if len(keep) < 2:
                keep = sorted(cand)[-2:]
            mem = keep
        elif name == "pixel_diversity":
            pairs = [(a, c) for i, a in enumerate(cand) for c in cand[i + 1:]]
            mem = list(max(pairs, key=lambda p: float(np.square(F[p[0]] - F[p[1]]).sum())))
        elif name == "pixel_surprise":
            mem = sorted(sorted(cand, key=lambda c: surprise[c])[-2:])
        elif name == "latest_visible_reference":  # structure-aware hand rule (uses visibility)
            keep = [c for c in cand if ep["vis"][c]]
            if len(keep) >= 2:
                keep = keep[-2:]
            else:  # pad with the newest occluded candidates, never dropping a visible one
                keep = keep + [c for c in cand if c not in keep][-(2 - len(keep)):]
            mem = sorted(keep)
        else:
            raise ValueError(name)
    return bool(all(ep["sufficient"][m] for m in mem))


POLICIES = ["fifo", "keep_first", "reservoir", "temporal_buckets", "pixel_diversity",
            "pixel_surprise", "latest_visible_reference"]

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    out = {}
    for version in ("v1", "v2"):
        for long_gap in (False, True):
            rng = np.random.default_rng(20261030); res = {p: [] for p in POLICIES}; feasible = []
            for i in range(n):
                ep = episode(version, long_gap, rng, i % 2)
                feasible.append(int(ep["sufficient"][: ep["q"] - RECENT + 1].sum() >= 2))
                for p in POLICIES:
                    res[p].append(run_policy(p, ep, np.random.default_rng(i)))
            key = f"{version}_{'long_gap' if long_gap else 'ordinary'}"
            out[key] = {"episodes": n, "hindsight_oracle": float(np.mean(feasible)),
                        **{p: round(float(np.mean(v)), 3) for p, v in res.items()}}
    print(json.dumps(out, indent=1))
