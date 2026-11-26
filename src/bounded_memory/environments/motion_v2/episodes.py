"""Motion v2 episodes for the literature baselines and for scoring kept frames on the frozen study readers.

PaFu-KV, Compress to Remember and FrameMorrow are the unchanged Gate implementations
(methods/{pafu_kv,pacc,framemorrow}/gate.py). Motion v2 uses the same protocol constants (R = 4 recent
frames, B = 2 memory items, horizons +3 and +6) and the same predictor class (the Phase 1
DynamicsPredictor), so the Motion-specific parts are here and in methods/pacc/motion_v2.py:

    MotionEpisode   the fields those implementations read (id, pair, q, z, p, actions, prefix),
                    built from FrozenAssets.episode; evaluation-only metadata is not copied in
    score_frames    kept frame pairs scored by FrozenAssets.losses, the path the published Motion v2
                    table was computed with: three readers x 96 scene pairs x 2 siblings x 2 horizons
    batch_equivalence   checks that Gate's batch builder and Motion's give the same predictions for the
                    same kept frames, which methods/pacc/motion_v2.score_tokens relies on
    summarize       pooled and matched-reader means of scored seeds
    MEMORY_CLASS    what each literature policy may store (bounded, full history, privileged)
    registered_parts    the registered training episodes split into training states and a dev set

Each baseline is trained against one reader (its seed) and scored under all three readers, like the
students; `matched` in the summaries is the trained-on reader alone.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from ..gate.protocol import item_batch
from .step34_core import SEEDS

HORIZONS = np.array([3, 6])
MEMORY_CLASS = {"pafu_kv": "bounded", "pafu_teacher_top2": "privileged (reads the future)", "pacc_full": "full history",
                "pacc_recurrent": "bounded", "fm_ijepa_published": "full history",
                "fm_ijepa_teacher_top2": "privileged (reads the future)", "fm_ijepa_online": "bounded"}


@dataclass
class MotionEpisode:
    id: str
    pair: str
    q: int
    z: np.ndarray        # [T, 5120] train-normalised I-JEPA features
    p: np.ndarray        # [T, d] projected features (the readers' token space)
    actions: np.ndarray  # [T-1, 2]
    prefix: np.ndarray   # [T, 2] cumulative executed actions

    @property
    def candidates(self):
        return self.q - 3     # frames 0 .. q-4 are legal history at the query (R = 4)


def adapt(raw):
    """FrozenAssets.episode dict -> MotionEpisode (no evaluation-only fields)."""
    rec = raw["rec"]
    return MotionEpisode(rec["id"], rec["pair"], int(rec["query"]), raw["z"], raw["p"], raw["actions"], raw["prefix"])


def split_episodes(assets, records, split, external=None):
    """Episodes of one split in evaluate_online's order (scene pairs, then siblings)."""
    from .step34_core import groups
    raw = [assets.episode(r, external) for pair in groups(records, split) for r in pair]
    return raw, [adapt(r) for r in raw]


def by_scene_pair(values, pairs):
    """[3 readers, episodes, 2 horizons] arrays -> [3 readers, scene pairs, 2 siblings, 2 horizons]."""
    return {k: v.reshape(3, pairs, 2, 2) for k, v in values.items()}


def score_frames(assets, raw, kept):
    """Losses of kept frame pairs under the three readers, plus whether both cue frames were kept."""
    for r, k in zip(raw, kept):
        if len(k) != 2 or len(set(k)) != 2 or min(k) < 0 or max(k) > r["rec"]["query"] - 4:
            raise ValueError(f"illegal memory {k} for episode {r['rec']['id']}")
    out = by_scene_pair(assets.losses(list(zip(raw, [list(map(int, k)) for k in kept]))), len(raw) // 2)
    out["success"] = np.array([bool(np.all(r["meta"]["sufficient"][list(k)])) for r, k in zip(raw, kept)]).reshape(-1, 2)
    return out


@torch.inference_mode()
def batch_equivalence(assets, raw, episodes, kept, reader=0):
    """Largest absolute difference between the reader's predictions from Motion's and Gate's batch builders."""
    model = assets.models[reader]
    motion = model(assets.collate([assets.sample(r, r["rec"]["query"], k) for r, k in zip(raw, kept)], assets.device))
    gate = model(item_batch(episodes, [e.q for e in episodes], kept, assets.device))
    return float((motion - gate).abs().max())


def summarize(scores, seeds, matched=True):
    """Pooled means over seeds and readers. scores: seed -> {position, feature [3 readers, pairs, 2, 2], success}.

    matched=True also reports the mean under the reader each seed was trained against (seed = reader seed, as
    for the literature baselines); leave it off for a method trained against all three readers at once."""
    position = np.stack([scores[s]["position"] for s in seeds])     # [seed, reader, pair, sibling, horizon]
    feature = np.stack([scores[s]["feature"] for s in seeds])
    success = np.stack([scores[s]["success"] for s in seeds]).astype(np.float64)
    out = {"seeds": list(seeds)}
    for j, h in enumerate(HORIZONS):
        out[f"position_mse_+{h}"] = float(position[..., j].mean())
        out[f"feature_mse_+{h}"] = float(feature[..., j].mean())
        if matched:
            same = np.stack([position[i, SEEDS.index(s)] for i, s in enumerate(seeds)])
            out[f"position_mse_+{h}_matched_reader"] = float(same[..., j].mean())
        out[f"position_mse_+{h}_per_seed"] = [float(position[i, ..., j].mean()) for i in range(len(seeds))]
    out["success_rate"] = None if np.isnan(success).all() else float(np.nanmean(success))
    return out


def registered_dev_pairs(records, modulus=10):
    """Registered training scene pairs held out as a dev set (sha256 of the pair id, 1 in `modulus`)."""
    return {r["pair"] for r in records if r["split"] == "train"
            and int(hashlib.sha256(r["pair"].encode()).hexdigest(), 16) % modulus == 0}


def registered_parts(assets):
    """{fit, dev}: the registered training episodes, which the frozen readers were trained on, split by scene pair."""
    records = assets.store.manifest["records"]
    dev = registered_dev_pairs(records)
    train = [assets.episode(r) for r in records if r["split"] == "train"]
    return {"fit": [ep for ep in train if ep["rec"]["pair"] not in dev],
            "dev": [ep for ep in train if ep["rec"]["pair"] in dev]}
