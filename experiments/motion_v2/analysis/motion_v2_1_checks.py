"""Model-free checks for Motion v1 (Phase 1) and the revised Motion v2 (v2.1).

Episodes are generated as true sibling pairs. Three checks:
1. pair_audit: siblings must agree on everything except the intended evidence.
2. leakage: a classifier trained across scenes tries to recover the sibling label
   from chosen frame subsets (relative time, x, y, visibility). Single frames and
   (stale frame + end frame) must stay near chance; two post-change frames must not.
3. online B=2 policies: success = the two retained history frames form a label-
   identifying pair (v1: two visible frames; v2.1: two visible frames after t_c).
Diversity/surprise use raw pixels, a crude proxy for their feature versions.
"""
import json, sys
import numpy as np
from PIL import Image, ImageDraw

N, RECENT = 96, 4
Y_PRE, Y_POST = 0.40, 0.62


def render(x, y, visible, distractors):
    im = Image.new("RGB", (N, N), (24, 29, 38)); d = ImageDraw.Draw(im)
    b = lambda r: tuple(int(v * N) for v in r)
    d.rectangle(b((0.05, 0.25, 0.95, 0.78)), fill=(44, 49, 60), outline=(110, 118, 133), width=2)
    if visible:
        r = 0.055; d.ellipse(b((x - r, y - r, x + r, y + r)), fill=(237, 175, 70))
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


def scene(version, long_gap, rng):
    """Shared scene factors; the sibling label only flips the final velocity sign."""
    gap = int(rng.integers(20, 33) if long_gap else rng.integers(5, 11))
    if version == "v1":
        start = int(rng.integers(2, 8)); end = start + int(rng.integers(3, 7))
        q = end + gap + RECENT
        return dict(version=version, q=q, end=end, t_c=0, vis=np.array([t <= end for t in range(q + 1)]),
                    x_end=rng.uniform(0.38, 0.62), speed=rng.uniform(0.005, 0.007), gap=gap)
    while True:
        t_on = int(rng.integers(0, 7)); L0 = int(rng.integers(4, 11)); L1 = int(rng.integers(3, 8))
        t_c = t_on + L0; end = t_c + L1 - 1; q = end + gap + RECENT
        vis = np.array([(t_on <= t <= end) and rng.random() > 0.25 for t in range(q + 1)])
        vis[end] = True
        if (vis & (np.arange(q + 1) >= t_c)).sum() >= 2 and vis[:t_c].sum() >= 2:
            break
    return dict(version=version, q=q, end=end, t_c=t_c, vis=vis, gap=gap,
                x_end=rng.uniform(0.20, 0.80),          # shared end position (wide range limits single-frame leakage)
                speed=rng.uniform(0.008, 0.012),        # shared |v1|
                x_pre=rng.uniform(0.2, 0.8), v0=rng.choice([-1, 1]) * rng.uniform(0.005, 0.007))


def states(s, label):
    """Hidden positions for one sibling. v2.1: the stale pre-change segment runs in a
    separate lane and is identical for both siblings; at t_c the ball is kicked into
    the lower lane; post-change positions are anchored at the shared end position."""
    sign = 2 * label - 1; q = s["q"]
    xs, ys = np.zeros(q + 1), np.zeros(q + 1)
    if s["version"] == "v1":
        for t in range(q + 1):
            xs[t] = s["x_end"] + sign * s["speed"] * (t - s["end"]); ys[t] = 0.5
        return xs, ys
    x, v = s["x_pre"], s["v0"]
    for t in range(s["t_c"]):
        xs[t], ys[t] = x, Y_PRE; x, v = reflect(x + v, v)
    for t in range(s["t_c"], s["end"] + 1):
        xs[t], ys[t] = s["x_end"] + sign * s["speed"] * (t - s["end"]), Y_POST
    x, v = s["x_end"], sign * s["speed"]
    for t in range(s["end"] + 1, q + 1):
        x, v = reflect(x + v, v); xs[t], ys[t] = x, Y_POST
    return xs, ys


def identifying(s):
    """Frames that can belong to a label-identifying pair."""
    return s["vis"] & (np.arange(s["q"] + 1) >= s["t_c"])


def pair_audit(s, frames_a, frames_b, xa, xb):
    stop = s["q"] - RECENT + 1
    differ = [t for t in range(stop) if not np.array_equal(frames_a[t], frames_b[t])]
    ok_recent = all(np.array_equal(frames_a[t], frames_b[t]) for t in range(stop, s["q"] + 1))
    ok_end = abs(xa[s["end"]] - xb[s["end"]]) < 1e-12
    ok_pre = all(np.array_equal(frames_a[t], frames_b[t]) for t in range(s["t_c"]))
    return ok_recent and ok_end and ok_pre and set(differ) <= set(np.flatnonzero(identifying(s)))


def run_policy(name, s, F, rng):
    stop = s["q"] - RECENT + 1
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
                b[min(1, 2 * c // (t + 1))] = c
            keep = sorted(b.values()); mem = keep if len(keep) == 2 else sorted(cand)[-2:]
        elif name == "pixel_diversity":
            pairs = [(a, c) for i, a in enumerate(cand) for c in cand[i + 1:]]
            mem = list(max(pairs, key=lambda p: float(np.square(F[p[0]] - F[p[1]]).sum())))
        elif name == "pixel_surprise":
            mem = sorted(sorted(cand, key=lambda c: surprise[c])[-2:])
        elif name == "latest_visible_reference":
            keep = [c for c in cand if s["vis"][c]]
            keep = keep[-2:] if len(keep) >= 2 else keep + [c for c in cand if c not in keep][-(2 - len(keep)):]
            mem = sorted(keep)
        else:
            raise ValueError(name)
    ident = identifying(s)
    return bool(ident[mem[0]] and ident[mem[1]])


POLICIES = ["fifo", "keep_first", "reservoir", "temporal_buckets", "pixel_diversity",
            "pixel_surprise", "latest_visible_reference"]


def leakage(version, n_scenes, seed):
    """Cross-scene label decoding from frame subsets, grouped 5-fold CV."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold, cross_val_score
    rng = np.random.default_rng(seed)
    subsets = {"end_only": [], "one_post_non_end": [], "stale_plus_end": [], "two_post": []}
    groups = []  # unused
    for i in range(n_scenes):
        s = scene(version, True, rng)
        ident = np.flatnonzero(identifying(s)); stale = np.flatnonzero(s["vis"][: s["t_c"]])
        non_end = [t for t in ident if t != s["end"]]
        pick_post = int(rng.choice(non_end)); pick_two = sorted(rng.choice(ident, 2, replace=False))
        pick_stale = int(rng.choice(stale)) if len(stale) else None
        for label in (0, 1):
            xs, ys = states(s, label)
            f = lambda t: [t - s["q"], xs[t], ys[t], float(s["vis"][t])]
            subsets["end_only"].append(f(s["end"]) + [label])
            subsets["one_post_non_end"].append(f(pick_post) + [label])
            if pick_stale is not None:
                subsets["stale_plus_end"].append(f(pick_stale) + f(s["end"]) + [label])
            subsets["two_post"].append(f(pick_two[0]) + f(pick_two[1]) + [label])
            groups.append(i)
    out = {}
    for k, rows in subsets.items():
        if not rows:
            out[k] = None  # v1 has no stale segment
            continue
        a = np.array(rows); X, y = a[:, :-1], a[:, -1]
        g = np.repeat(np.arange(len(a) // 2), 2)  # siblings stay in the same fold
        acc = cross_val_score(HistGradientBoostingClassifier(max_iter=200), X, y, groups=g, cv=GroupKFold(5))
        out[k] = round(float(acc.mean()), 3)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    out = {"online_policies": {}, "pair_audit": {}, "leakage_long_gap": {}}
    for version in ("v1", "v2"):
        for long_gap in (False, True):
            rng = np.random.default_rng(20261030); res = {p: [] for p in POLICIES}; audit = []; feas = []
            for i in range(n // 2):
                s = scene(version, long_gap, rng)
                dist_seq = []; d = rng.uniform(0, 1, (3, 2))
                for t in range(s["q"] + 1):
                    if version == "v2":
                        d = np.clip(d + rng.normal(0, 0.08, d.shape), 0, 1)
                    dist_seq.append(d.copy())
                sib = []
                for label in (0, 1):
                    xs, ys = states(s, label)
                    F = [render(xs[t], ys[t], s["vis"][t], dist_seq[t]) for t in range(s["q"] + 1)]
                    sib.append((F, xs))
                    feas.append(int(identifying(s)[: s["q"] - RECENT + 1].sum() >= 2))
                    for p in POLICIES:
                        res[p].append(run_policy(p, s, F, np.random.default_rng(i)))
                audit.append(pair_audit(s, sib[0][0], sib[1][0], sib[0][1], sib[1][1]))
            key = f"{version}_{'long_gap' if long_gap else 'ordinary'}"
            out["online_policies"][key] = {"episodes": len(feas), "hindsight_oracle": float(np.mean(feas)),
                                           **{p: round(float(np.mean(v)), 3) for p, v in res.items()}}
            out["pair_audit"][key] = f"{sum(audit)}/{len(audit)}"
        out["leakage_long_gap"][version] = leakage(version, 2000, 7)
    print(json.dumps(out, indent=1))
