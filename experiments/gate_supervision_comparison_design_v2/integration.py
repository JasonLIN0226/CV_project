"""Explicit adapters for the existing Gate FeatureStore; import has no side effects.

These functions are intended for a later authorized train/validation run. They
never open test records, fit predictors, encode data, or persist outputs. The
caller must first verify the frozen source/model/cache hashes and freeze its
state manifest. Synthetic adapter tests do not verify real HPC assets.
"""
from __future__ import annotations

import numpy as np
from .core import OnlineState
from .teachers import OfflineTargets, teacher_pair


def labels_for_record(store, rec, state: OnlineState, *, load_episode, model, collate, device):
    """Create both offline labels for one pre-frozen query-time state.

    FeatureStore.features returns already-normalized latent features; its
    positions q+3/q+6 are the clean future targets of the frozen dataset.
    The selector's state retains only q-prefix features and past actions.
    The sole observation that the action continuation comes from the original
    episode is not proof of data provenance: caller must verify the source lock.
    """
    split = rec.get("split")
    if split not in {"train", "validation"} or rec.get("task") != "gate":
        raise ValueError("only frozen Gate train/validation records allowed")
    q = int(rec["query"])
    if q != state.clock:
        raise ValueError("state is not the record's final query")
    if tuple(store.c["data"]["horizons"]) != (3, 6):
        raise ValueError("expected frozen horizons 3 and 6")
    z = np.asarray(store.features(rec))
    p = np.asarray(store.projected_features(rec))
    actions = np.asarray(load_episode(store.c, rec, keys=("actions",))["actions"])
    if z.ndim != 2 or p.ndim != 2 or z.shape[0] <= q + 6 or p.shape[0] <= q:
        raise ValueError("incomplete frozen feature episode")
    if actions.ndim != 2 or actions.shape[1] != 2 or actions.shape[0] < q + 6:
        raise ValueError("incomplete recorded actions")
    if not np.array_equal(p[:q + 1], state.projected_prefix):
        raise ValueError("state and record projected prefixes differ")
    if not np.array_equal(actions[:q], state.actions_prefix):
        raise ValueError("state and record action prefixes differ")
    target = OfflineTargets(split, z[[q + 3, q + 6]].copy(), actions[q:q + 6].copy())
    return teacher_pair(state, target, z[:q + 1].copy(), model, collate, device)
