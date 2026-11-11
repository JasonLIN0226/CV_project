"""Read a Gate asset tree as plain arrays.

An asset tree is what the clean-future pipeline writes and what the team's frozen bundle holds:

    <root>/resolved_config.json
    <root>/dataset/manifest.json, split_audit.json, observations/, evaluation_only/
    <root>/features/index.json, pair_feature_audit.json, normalization.{npz,json}, train/, validation/
    <root>/predictor/gate/basis.{npz,json}, seed_<s>/best.pt
    <root>/probes/metrics.json, gate_state_decoder.npz

The historical readers (phase1/data.py, cache.py, model.py, probes.py, training.py) accepted these
files only when signatures built from their own source hashes matched. Those checks cannot follow
code that has moved, so every file is verified here against the checksum the tree records for it,
and predictor checkpoints against checksums supplied by the caller (the team bundle's are pinned
in `bundle.py`). The arithmetic is unchanged: `normalize` is phase1/cache.py's, `project` is
phase1/model.py's, and the evidence rule is phase1/inputs.py's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ...data.io import file_hash, read_json, require_sha256
from ...evaluation.metrics.gate import LinearReadout
from ...predictors.checkpoints import load_basis, load_dynamics_predictor
from ...predictors.dynamics import project
from ...predictors.inputs import selected_evidence
from .protocol import B, HORIZONS, R


def normalize(z, norm):
    return ((z-norm["mean"])/norm["std"]).astype(np.float32)


@dataclass
class Episode:
    id: str
    pair: str
    sibling: str
    split: str
    q: int
    cue: tuple            # inclusive (start, end) of the cue interval
    evidence: np.ndarray  # privileged timestamps, used only by the oracle reference
    z: np.ndarray         # [T, D] train-normalised I-JEPA features; frames after q are clean targets
    p: np.ndarray         # [T, d] projected features (the predictor's token space)
    actions: np.ndarray   # [T-1, 2]
    classes: np.ndarray   # [T] route label per frame (evaluation only)
    prefix: np.ndarray    # [T, 2] cumulative executed actions

    @property
    def candidates(self):
        return self.q - R + 1  # frames 0 .. q-R are eligible history at the query


@dataclass
class GateAssets:
    root: Path
    c: dict
    records: list
    episodes: list
    basis: dict
    decoder: object
    device: torch.device
    predictor_sha256: dict = field(default_factory=dict)

    def split(self, name):
        return [e for e in self.episodes if e.split == name]

    def predictor(self, seed):
        if seed not in self.predictor_sha256:
            raise ValueError(f"no recorded checksum for the seed-{seed} predictor; refusing to deserialise it")
        return load_dynamics_predictor(self.root / f"predictor/gate/seed_{seed}/best.pt", self.basis,
                                       sha256=self.predictor_sha256[seed], seed=seed, task="gate",
                                       horizons=HORIZONS, device=self.device)


def load_normalization(root, c, index, records):
    p = root / "features/normalization.npz"
    require_sha256(p, read_json(p.with_suffix(".json"))["sha256"], "feature normalisation")
    with np.load(p, allow_pickle=False) as f:
        norm = {k: f[k] for k in f.files}
    if str(norm["signature"]) != index["signature"] or not np.isclose(norm["floor"], c["normalization"]["std_floor"]):
        raise ValueError("Normalization provenance mismatch")
    if set(norm["train_ids"].tolist()) != {r["id"] for r in records if r["split"] == "train"}:
        raise ValueError("Normalization must fit train only")
    return norm


def load_feature(root, rec, index):
    info = index["episodes"][rec["id"]]
    p = require_sha256(root / "features" / info["path"], info["sha256"], "feature cache")
    with np.load(p, allow_pickle=False) as f:
        z = f["z"]
        if str(f["signature"]) != index["signature"] or str(f["encoder_hash"]) != index["encoder_hash"]:
            raise ValueError("Mismatched feature provenance")
    if list(z.shape) != info["shape"] or z.dtype != np.float32 or not np.isfinite(z).all():
        raise ValueError("Invalid feature array")
    return z


def load_episode_arrays(root, rec, kind, keys):
    p = require_sha256(root / "dataset" / rec[kind], rec[kind + "_sha256"], "dataset episode")
    with np.load(p, allow_pickle=False) as f:
        return {k: f[k] for k in keys}


def load_gate(root, device="cpu", *, predictor_sha256=None, basis_sha256=None):
    """Load every Gate train/validation episode, the bases and the state decoder from `root`."""
    root = Path(root).resolve()
    c = read_json(root / "resolved_config.json")
    if (c["data"]["recent"], c["data"]["budget"], tuple(c["data"]["horizons"])) != (R, B, HORIZONS):
        raise ValueError("expected R=4, B=2, horizons (3, 6)")
    manifest = read_json(root / "dataset/manifest.json")
    audit = read_json(root / "dataset/split_audit.json")
    if not audit["passed"] or audit["signature"] != manifest["signature"]:
        raise ValueError("Structural audit absent or stale")
    records = manifest["records"]
    index = read_json(root / "features/index.json")
    if set(index["episodes"]) != {r["id"] for r in records}:
        raise ValueError("Incomplete feature cache")
    pairing = read_json(root / "features/pair_feature_audit.json")
    if not pairing["passed"] or pairing["signature"] != index["signature"] \
            or pairing["index_sha256"] != file_hash(root / "features/index.json"):
        raise ValueError("Feature-level pairing audit absent/stale")
    norm = load_normalization(root, c, index, records)

    basis_path = root / "predictor/gate/basis.npz"
    recorded = read_json(basis_path.with_suffix(".json"))["sha256"]
    if basis_sha256 is not None and basis_sha256 != recorded:
        raise ValueError("basis differs from the pinned checksum")
    basis = load_basis(basis_path, recorded)

    probes = read_json(root / "probes/metrics.json")
    decoder_path = root / "probes/gate_state_decoder.npz"
    require_sha256(decoder_path, probes["checkpoints"][decoder_path.name], "state decoder")
    if probes["cache_signature"] != index["signature"]:
        raise ValueError("State decoder provenance mismatch")

    episodes = []
    for rec in records:
        if rec["task"] != "gate" or rec["split"] not in ("train", "validation"):
            raise ValueError("Gate train/validation records only")
        meta = load_episode_arrays(root, rec, "eval", ("classes", "evidence"))
        actions = load_episode_arrays(root, rec, "obs", ("actions",))["actions"].astype(np.float32)
        z = normalize(load_feature(root, rec, index), norm)
        q = int(rec["query"])
        if len(z) < q + max(HORIZONS) + 1 or len(actions) < q + max(HORIZONS):
            raise ValueError("episode is shorter than query + horizon")
        episodes.append(Episode(
            id=rec["id"], pair=rec["pair"], sibling=rec["sibling"], split=rec["split"], q=q,
            cue=(int(rec["cue_start"]), int(rec["cue_end"])),
            evidence=selected_evidence(c, rec, meta["evidence"]),
            z=z, p=project(z, basis), actions=actions, classes=np.asarray(meta["classes"]),
            prefix=np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(actions, axis=0)]).astype(np.float32)))
    return GateAssets(root=root, c=c, records=records, episodes=episodes, basis=basis,
                      decoder=LinearReadout.load(decoder_path), device=torch.device(device),
                      predictor_sha256=dict(predictor_sha256 or {}))
