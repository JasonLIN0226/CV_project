"""Compress to Remember on Motion v2: the Gate compressor's memory tokens in the frozen Motion v2 readers.

Training is the unchanged Gate implementation (methods/pacc/gate.py) on Motion episodes
(environments/motion_v2/episodes.py). Compress to Remember emits memory tokens, not frames, so its losses
are computed from the readers' predictions with those tokens in the two memory slots; Gate's batch builder
gives the same predictions as Motion's for the same kept frames (episodes.batch_equivalence).
"""
from __future__ import annotations

import numpy as np
import torch

from ...environments.gate.protocol import item_batch, with_memory_tokens
from ...environments.motion_v2.episodes import HORIZONS, by_scene_pair
from .gate import _history, _recurrent_memory


@torch.inference_mode()
def score_tokens(assets, compressor, raw, episodes, variant, chunk=64):
    """Compress to Remember: readers' losses with the compressor's two memory tokens in the memory slots."""
    device = assets.device
    feature = np.empty((3, len(raw), 2), np.float64)
    position = np.empty_like(feature)
    for at in range(0, len(raw), chunk):
        eps, rows = episodes[at:at + chunk], raw[at:at + chunk]
        clocks = [e.q for e in eps]
        memory = (compressor.compress(*_history(eps, clocks, device)) if variant == "full"
                  else _recurrent_memory(compressor, eps, clocks, device))
        batch = with_memory_tokens(item_batch(eps, clocks, [[] for _ in eps], device), *compressor.items(memory))
        target = np.stack([r["z"][r["rec"]["query"] + HORIZONS] for r in rows])
        truth = np.stack([r["meta"]["states"][r["rec"]["query"] + HORIZONS] for r in rows])
        for mi, model in enumerate(assets.models):
            predicted = model(batch).cpu().numpy()
            decoded = assets.decoder.predict(predicted.reshape(-1, predicted.shape[-1])).reshape(-1, 2, 2)
            feature[mi, at:at + len(rows)] = np.square(predicted.astype(np.float64) - target).mean(-1)
            position[mi, at:at + len(rows)] = np.square(decoded - truth.astype(np.float64)).mean(-1)
    if not (np.isfinite(feature).all() and np.isfinite(position).all()):
        raise ValueError("nonfinite Compress to Remember loss")
    out = by_scene_pair({"feature": feature, "position": position}, len(raw) // 2)
    out["success"] = np.full((len(raw) // 2, 2), np.nan)   # no frames are kept
    return out
