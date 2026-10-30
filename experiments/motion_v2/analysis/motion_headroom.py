"""Model-free information headroom of the Phase 1 Motion deletion pools.
Rebuilds the fresh seed-9137 test scenes and the exact clock-only pools, then
counts how many visible (pre-shutter) frames each deletion policy keeps."""
import sys, json, hashlib, types, importlib.util
from collections import Counter, defaultdict
import numpy as np

src = sys.argv[1]  # .../original_src/phase1
cfg = json.load(open(sys.argv[2]))

def digest(o): return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def stable_seed(*p): return int(digest(p)[:16], 16) % (2**32)
pkg = types.ModuleType("phase1"); pkg.__path__ = []
ut = types.ModuleType("phase1.utils"); ut.stable_seed = stable_seed; ut.digest = digest
sys.modules["phase1"] = pkg; sys.modules["phase1.utils"] = ut
spec = importlib.util.spec_from_file_location("phase1.envs", f"{src}/envs.py")
envs = importlib.util.module_from_spec(spec); spec.loader.exec_module(envs)

def pools(query, seed, recent=4, count=6):  # verbatim logic of clock_only_static_pools
    stop = query-recent+1
    rng = np.random.default_rng(seed); chosen = set()
    while len(chosen) < count:
        chosen.add(tuple(sorted(int(x) for x in rng.choice(stop, 3, replace=False))))
    return sorted(chosen)

out = {}
for split in ("test_id", "test_long_gap", "test_distractor_shift"):
    nvis = Counter(); pol = defaultdict(list); best_is = defaultdict(int); n = 0
    gap_from_end = []
    for i in range(cfg["data"]["pairs"][split]):
        s = envs.scene(cfg, "motion", split, i)
        pair = digest([cfg["data"]["seed"], "motion", split, i])[:24]
        q, end = s["query"], s["end"]
        for parent in pools(q, stable_seed(20261017, "clock_only", pair)):
            n += 1
            vis = [t <= end for t in parent]  # frame visible iff t <= end (shutter after end)
            nv = sum(vis); nvis[nv] += 1
            keep = {d: sum(v for t, v in zip(parent, vis) if t != d) for d in parent}
            best = max(keep.values())
            for name, d in (("fifo_delete_oldest", parent[0]), ("reject_newest", parent[-1]), ("delete_middle", parent[1])):
                pol[name].append(keep[d]); best_is[name] += keep[d] == best
            pol["uniform"].append(np.mean(list(keep.values())))
            pol["best_possible"].append(best)
    out[split] = {
        "pools": n,
        "visible_frames_in_pool": {k: round(v/n, 4) for k, v in sorted(nvis.items())},
        "mean_visible_kept": {k: round(float(np.mean(v)), 4) for k, v in pol.items()},
        "share_pools_policy_matches_best": {k: round(v/n, 4) for k, v in best_is.items()},
        "share_pools_with_velocity_recoverable_best": round(float(np.mean(np.array(pol["best_possible"]) >= 2)), 4),
    }
print(json.dumps(out, indent=1))
