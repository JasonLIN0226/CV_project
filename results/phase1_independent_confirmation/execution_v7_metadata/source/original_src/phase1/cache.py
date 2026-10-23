import time
from pathlib import Path

import numpy as np

from .data import require_audit, load_episode
from .encoders import build_encoder
from .utils import code_hash, digest, file_hash, read_json, write_json, write_npz, device_of, environment


def cache_signature(c, m):
    return digest({"dataset": m["signature"], "encoder": c["encoder"], "code": code_hash("encoders.py", "cache.py")})


def load_index(c, complete=True):
    m = require_audit(c)
    ix = read_json(Path(c["output"])/"features/index.json")
    if ix["signature"] != cache_signature(c, m):
        raise ValueError("Feature cache config/code mismatch")
    if complete and set(ix["episodes"]) != {r["id"] for r in m["records"]}:
        raise ValueError("Incomplete cache; resume encode")
    if complete:
        audit = read_json(Path(c["output"])/"features/pair_feature_audit.json")
        if not audit["passed"] or audit["signature"] != ix["signature"] or audit["index_sha256"] != file_hash(Path(c["output"])/"features/index.json"):
            raise ValueError("Feature-level pairing audit absent/stale")
    return ix


def load_feature(c, rec, ix=None):
    ix = load_index(c) if ix is None else ix
    info = ix["episodes"][rec["id"]]
    p = Path(c["output"])/"features"/info["path"]
    if file_hash(p) != info["sha256"]:
        raise ValueError("Corrupted feature checksum")
    with np.load(p, allow_pickle=False) as f:
        z = f["z"]
        if str(f["signature"]) != ix["signature"] or str(f["encoder_hash"]) != ix["encoder_hash"]:
            raise ValueError("Mismatched feature provenance")
    if list(z.shape) != info["shape"] or z.dtype != np.float32 or not np.isfinite(z).all():
        raise ValueError("Invalid feature array")
    return z


def encode(c, max_episodes=None):
    m = require_audit(c)
    root = Path(c["output"])/"features"
    p = root/"index.json"
    sig = cache_signature(c, m)
    if p.exists():
        ix = load_index(c, complete=False)
        for rec in m["records"]:
            if rec["id"] in ix["episodes"]:
                load_feature(c, rec, ix)
    else:
        ix = {"signature": sig, "dataset_signature": m["signature"], "mode": c["mode"], "episodes": {}, "seconds": 0.0}
    pending = [r for r in m["records"] if r["id"] not in ix["episodes"]]
    if not pending:
        audit_features(c, m, ix)
        return ix
    began = time.perf_counter()
    encoder = build_encoder(c, device_of(c))
    encoder_hash = digest(encoder.metadata)
    if "encoder_hash" in ix and ix["encoder_hash"] != encoder_hash:
        raise ValueError("Resolved checkpoint/processor differs from partial cache. Pin revision or start a new run.")
    ix.update(encoder=encoder.metadata, encoder_hash=encoder_hash, extraction_environment=environment(c["device"]))
    write_json(p, ix)
    processed = 0
    for rec in pending:
        rgb = load_episode(c, rec)["rgb"]
        bs = c["encoder"]["batch_size"]
        z = np.concatenate([encoder.encode(rgb[i:i+bs]) for i in range(0, len(rgb), bs)]).astype(np.float32)
        if not np.isfinite(z).all():
            raise ValueError("Non-finite encoder features")
        path = root/rec["split"]/(rec["id"]+".npz")
        write_npz(path, z=z, signature=np.array(sig), encoder_hash=np.array(encoder_hash))
        ix["episodes"][rec["id"]] = {"path": str(path.relative_to(root)), "sha256": file_hash(path),
            "shape": list(z.shape), "dtype": "float32", "feature_bytes": z.nbytes,
            "entry_bytes": z.shape[1]*z.dtype.itemsize, "split": rec["split"], "pair": rec["pair"],
            "seed": rec["seed"], "source_sha256": rec["obs_sha256"]}
        # Index commit follows atomic episode commit; an interrupted unindexed file is safely recomputed.
        ix["seconds"] += time.perf_counter()-began
        began = time.perf_counter()
        write_json(p, ix)
        processed += 1
        if processed % 32 == 0:
            print(f"encoded {len(ix['episodes'])}/{len(m['records'])} episodes", flush=True)
        if max_episodes is not None and processed >= max_episodes:
            break
    if set(ix["episodes"]) == {r["id"] for r in m["records"]}:
        audit_features(c, m, ix)
    return ix


def audit_features(c, m, ix):
    # Exact equal RGB must remain equal features; stochastic/batch-coupled encoders would invalidate chance.
    records = {r["id"]: r for r in m["records"]}
    checked = set()
    for rec in m["records"]:
        if rec["pair"] in checked:
            continue
        sibling = records[rec["sibling"]]
        a, b = load_feature(c, rec, ix), load_feature(c, sibling, ix)
        q, R = rec["query"], c["data"]["recent"]
        if not np.array_equal(a[q-R+1:q+1], b[q-R+1:q+1]):
            raise ValueError("Equal recent RGB became unequal features; inspect deterministic frame encoder before training")
        checked.add(rec["pair"])
    write_json(Path(c["output"])/"features/pair_feature_audit.json", {"passed": True, "pairs": len(checked),
        "signature": ix["signature"], "exact_recent_feature_equality": True,
        "index_sha256": file_hash(Path(c["output"])/"features/index.json")})


def fit_normalization(c):
    m, ix = require_audit(c), load_index(c)
    p = Path(c["output"])/"features/normalization.npz"
    if p.exists():
        return load_normalization(c)
    total, sq, n = None, None, 0
    ids = []
    for rec in m["records"]:
        if rec["split"] != "train":
            continue
        z = load_feature(c, rec, ix).astype(np.float64)
        total = z.sum(0) if total is None else total+z.sum(0)
        sq = (z*z).sum(0) if sq is None else sq+(z*z).sum(0)
        n += len(z)
        ids.append(rec["id"])
    mean = total/n
    std = np.maximum(np.sqrt(np.maximum(sq/n-mean*mean, 0)), c["normalization"]["std_floor"])
    write_npz(p, mean=mean.astype(np.float32), std=std.astype(np.float32), train_ids=np.array(ids),
              count=np.int64(n), signature=np.array(ix["signature"]), floor=np.float32(c["normalization"]["std_floor"]))
    write_json(p.with_suffix(".json"), {"fit_split": "train", "frames": n, "episodes": len(ids),
        "std_floor": c["normalization"]["std_floor"], "cache_signature": ix["signature"], "sha256": file_hash(p)})
    return load_normalization(c)


def load_normalization(c):
    ix = load_index(c)
    p = Path(c["output"])/"features/normalization.npz"
    meta = read_json(p.with_suffix(".json"))
    if file_hash(p) != meta["sha256"]:
        raise ValueError("Normalization checksum mismatch")
    with np.load(p, allow_pickle=False) as f:
        norm = {k: f[k] for k in f.files}
    if str(norm["signature"]) != ix["signature"] or not np.isclose(norm["floor"], c["normalization"]["std_floor"]):
        raise ValueError("Normalization provenance mismatch")
    train_ids = {r["id"] for r in require_audit(c)["records"] if r["split"] == "train"}
    if set(norm["train_ids"].tolist()) != train_ids:
        raise ValueError("Normalization must fit train only")
    return norm


def normalize(z, norm):
    return ((z-norm["mean"])/norm["std"]).astype(np.float32)
