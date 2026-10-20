import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import SPLITS
from .envs import scene, episode
from .utils import code_hash, digest, file_hash, read_json, write_json, write_npz


def dataset_signature(c):
    return digest({"data": c["data"], "implementation": code_hash("envs.py", "data.py", "utils.py", "config.py")})


def generate(c):
    root = Path(c["output"]) / "dataset"
    p = root / "manifest.json"
    sig = dataset_signature(c)
    if p.exists():
        m = read_json(p)
        if m["signature"] != sig:
            raise ValueError("Dataset configuration/code mismatch; use new output")
        validate_data(c)
        return m
    began = time.perf_counter()
    records = []
    for task in c["data"]["tasks"]:
        for split in SPLITS:
            for i in range(c["data"]["pairs"][split]):
                s = scene(c, task, split, i)
                pair_id = digest([c["data"]["seed"], task, split, i])[:24]
                ids = [digest([pair_id, j, "opaque-id"])[:24] for j in range(2)]
                for j, eid in enumerate(ids):
                    e = episode(s, s["label_order"][j])
                    obs = root / "observations" / split / f"{eid}.npz"
                    meta = root / "evaluation_only" / split / f"{eid}.npz"
                    write_npz(obs, rgb=e["rgb"], actions=e["actions"], query=np.int64(s["query"]))
                    write_npz(meta, states=e["states"], classes=e["classes"], visible=e["visible"],
                              label=np.int64(e["label"]), evidence=np.asarray(e["evidence"], np.int64))
                    records.append({"id": eid, "pair": pair_id, "sibling": ids[1-j], "split": split,
                                    "task": task, "seed": s["seed"], "scene_index": i, "query": s["query"],
                                    "gap": s["gap"], "frames": s["length"], "obs": str(obs.relative_to(root)),
                                    "eval": str(meta.relative_to(root)), "obs_sha256": file_hash(obs),
                                    "eval_sha256": file_hash(meta)})
    manifest = {"signature": sig, "config_hash": digest(c["data"]), "code_hash": code_hash("envs.py", "data.py"),
                "mode": c["mode"], "generation_seconds": time.perf_counter()-began, "records": records}
    write_json(p, manifest)
    return manifest


def manifest(c):
    m = read_json(Path(c["output"]) / "dataset/manifest.json")
    if m["signature"] != dataset_signature(c):
        raise ValueError("Stale/mismatched dataset manifest")
    return m


def load_episode(c, rec, metadata=False, keys=None):
    p = Path(c["output"]) / "dataset" / rec["eval" if metadata else "obs"]
    with np.load(p, allow_pickle=False) as f:
        return {k: f[k] for k in (f.files if keys is None else keys)}


def contact_sheet(c, pair, name):
    rows = []
    for rec in pair:
        o, m = load_episode(c, rec), load_episode(c, rec, True)
        q = rec["query"]
        times = list(m["evidence"]) + [q-c["data"]["recent"]+1, q] + [q+h for h in c["data"]["horizons"]]
        panels = []
        for t in times:
            im = Image.fromarray(o["rgb"][t]).resize((160, 160))
            tile = Image.new("RGB", (160, 187), "white")
            tile.paste(im, (0, 27))
            label = "evidence" if t in m["evidence"] else "recent" if t <= q else "TRUE future"
            ImageDraw.Draw(tile).text((3, 6), f"t={t} {label}", fill="black")
            panels.append(tile)
        rows.append(panels)
    canvas = Image.new("RGB", (160*len(rows[0]), 187*2), "white")
    for r, panels in enumerate(rows):
        for col, panel in enumerate(panels):
            canvas.paste(panel, (col*160, r*187))
    path = Path(c["output"]) / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def validate_data(c):
    m = manifest(c)
    groups, seen_seeds, summary = {}, {}, []
    for rec in m["records"]:
        groups.setdefault(rec["pair"], []).append(rec)
        old = seen_seeds.setdefault(rec["seed"], rec["pair"])
        assert old == rec["pair"], "Seed overlap across base scenes"
        for kind in ["obs", "eval"]:
            assert file_hash(Path(c["output"])/"dataset"/rec[kind]) == rec[kind+"_sha256"], "Data checksum mismatch"
    for pid, pair in groups.items():
        assert len(pair) == 2 and len({r["split"] for r in pair}) == 1, "Pair split leakage"
        a, b = pair
        assert a["sibling"] == b["id"] and b["sibling"] == a["id"]
        oa, ob = load_episode(c, a), load_episode(c, b)
        ma, mb = load_episode(c, a, True), load_episode(c, b, True)
        q, R = a["query"], c["data"]["recent"]
        assert q == b["query"] and q == int(oa["query"])
        assert np.array_equal(oa["rgb"][q-R+1:q+1], ob["rgb"][q-R+1:q+1]), "Recent RGB differs"
        assert np.array_equal(oa["actions"], ob["actions"]), "Actions leak sibling label"
        assert {int(ma["label"]), int(mb["label"])} == {0, 1}
        assert np.array_equal(ma["evidence"], mb["evidence"])
        assert np.all(ma["evidence"] < q-R+1), "Evidence overlaps recent"
        assert not np.array_equal(oa["rgb"][ma["evidence"]], ob["rgb"][mb["evidence"]]), "No visual evidence"
        for h in c["data"]["horizons"]:
            assert q+h < len(oa["rgb"]) and len(oa["actions"][q:q+h]) == h
            assert ma["visible"][q+h] and mb["visible"][q+h], "Invisible target"
            assert not np.array_equal(oa["rgb"][q+h], ob["rgb"][q+h]), "Future RGB identical"
            assert not np.array_equal(ma["states"][q+h], mb["states"][q+h]), "Future state identical"
        if a["task"] == "motion":
            assert np.array_equal(oa["rgb"][ma["evidence"][1]], ob["rgb"][mb["evidence"][1]]), "Motion end position leaks direction"
        # Recompute BOTH siblings; verifies determinism, label-independent nuisance/timing/actions.
        s = scene(c, a["task"], a["split"], a["scene_index"])
        for rec, o, em in [(a, oa, ma), (b, ob, mb)]:
            regen = episode(s, int(em["label"]))
            assert np.array_equal(regen["rgb"], o["rgb"]) and np.array_equal(regen["actions"], o["actions"])
        if a["scene_index"] == 0:
            contact_sheet(c, pair, f"pair_{a['task']}_{a['split']}.png")
    for task in c["data"]["tasks"]:
        for split in SPLITS:
            rows = [r for r in m["records"] if r["task"] == task and r["split"] == split]
            labels = [int(load_episode(c, r, True)["label"]) for r in rows]
            summary.append({"task": task, "split": split, "pairs": len(rows)//2, "episodes": len(rows),
                            "frames": sum(r["frames"] for r in rows), "labels": {"0": labels.count(0), "1": labels.count(1)},
                            "gap_min": min(r["gap"] for r in rows), "gap_max": max(r["gap"] for r in rows),
                            "query_min": min(r["query"] for r in rows), "query_max": max(r["query"] for r in rows)})
    audit = {"passed": True, "signature": m["signature"], "checks": ["exact_recent_rgb", "exact_all_actions",
             "historical_evidence_differs", "visible_targets_differ", "historical_timestamps", "target_action_indices",
             "pair_split_disjoint", "seed_disjoint", "label_balance", "deterministic_generation",
             "motion_second_frame_matched", "pairwise_timing_action_nuisance_independence"], "splits": summary}
    write_json(Path(c["output"])/"dataset/split_audit.json", audit)
    return audit


def require_audit(c):
    m = manifest(c)
    audit = read_json(Path(c["output"])/"dataset/split_audit.json")
    if not audit["passed"] or audit["signature"] != m["signature"]:
        raise ValueError("Structural audit absent or stale; run validate-data")
    # The audit is invalid after any source data/metadata mutation, even if config is unchanged.
    for rec in m["records"]:
        for kind in ["obs", "eval"]:
            if file_hash(Path(c["output"])/"dataset"/rec[kind]) != rec[kind+"_sha256"]:
                raise ValueError("Source checksum changed after structural audit; do not reuse the cache")
    return m
