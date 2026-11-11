"""The ONLY predictor input boundary. No labels/states/IDs in returned model tensors."""
from collections import OrderedDict

import numpy as np
import torch

from .cache import load_index, load_feature, load_normalization, normalize
from .data import require_audit, load_episode
from .utils import stable_seed


def selected_evidence(c, rec, raw_evidence):
    """Fixed privileged timestamps; reads no hidden state, label, or future image.

    The wide motion policy uses the already visible frame at zero and the same
    pre-occlusion end frame.  It changes neither the legal history boundary nor
    the random/full/recent memory rules.
    """
    evidence = np.asarray(raw_evidence)
    if evidence.ndim != 1 or not np.issubdtype(evidence.dtype, np.integer):
        raise ValueError("Evidence timestamps must be a one-dimensional integer array")
    evidence = evidence.astype(np.int64, copy=True)
    stop = rec["query"]-c["data"]["recent"]+1
    if not len(evidence) or not np.all((evidence >= 0) & (evidence < stop)):
        raise ValueError("Evidence must precede recent buffer")
    if np.any(np.diff(evidence) <= 0):
        raise ValueError("Evidence timestamps must be distinct and increasing")
    experiment = c.get("experiment", {})
    wide = experiment.get("arm") in {"motion_wide_span", "confirm_candidate"}
    if rec["task"] == "motion":
        if len(evidence) != 2:
            raise ValueError("Motion evidence requires exactly two timestamps")
        if wide:
            evidence = np.array([0, evidence[-1]], np.int64)
            if evidence[-1] <= 0:
                raise ValueError("Wide motion evidence requires a positive end timestamp")
    return evidence


def memory_indices(q, recent, budget, evidence, condition, rng):
    stop = q-recent+1
    valid = np.arange(stop, dtype=np.int64)
    evidence = np.asarray(evidence, np.int64)
    if not np.all((evidence >= 0) & (evidence < stop)):
        raise ValueError("Evidence must precede recent buffer")
    if condition == "recent_only":
        return np.empty(0, np.int64)
    if condition == "full_history":
        return valid
    if len(valid) < budget:
        raise ValueError("Insufficient history: no duplication or crossing recent boundary")
    if condition in {"privileged_small_budget", "wrong_evidence"}:
        if len(evidence) > budget:
            raise ValueError("Evidence does not fit budget")
        filler = [int(i) for i in valid[::-1] if i not in evidence][:budget-len(evidence)]
        return np.sort(np.concatenate([evidence, filler])).astype(np.int64)
    if condition == "random_small_budget":
        return np.sort(rng.choice(valid, size=budget, replace=False))
    if condition == "random_subset":
        return np.sort(rng.choice(valid, size=int(rng.integers(0, len(valid)+1)), replace=False))
    raise ValueError(f"Unknown condition {condition}")


class FeatureStore:
    def __init__(self, c, max_cached=16):
        self.c, self.manifest, self.ix = c, require_audit(c), load_index(c)
        self.norm = load_normalization(c)
        self.records = {r["id"]: r for r in self.manifest["records"]}
        self.lru, self.max_cached = OrderedDict(), max_cached
        self.bases, self.projected = {}, OrderedDict()

    def projected_features(self, rec):
        from .model import project
        if rec['id'] not in self.projected:
            self.projected[rec['id']] = project(self.features(rec), self.bases[rec['task']])
            if len(self.projected) > self.max_cached:
                self.projected.popitem(last=False)
        self.projected.move_to_end(rec['id'])
        return self.projected[rec['id']]

    def features(self, rec):
        key = rec["id"]
        if key not in self.lru:
            self.lru[key] = normalize(load_feature(self.c, rec, self.ix), self.norm)
            if len(self.lru) > self.max_cached:
                self.lru.popitem(last=False)
        self.lru.move_to_end(key)
        return self.lru[key]

    def sample(self, rec, condition, rng=None, dropout=0):
        c, q = self.c, rec["query"]
        R, B = c["data"]["recent"], c["data"]["budget"]
        # Whitelist metadata: only evidence timestamps, never label/state.
        evidence = selected_evidence(c, rec, load_episode(c, rec, True, keys=("evidence",))["evidence"])
        if rng is None:
            # Same pair -> same random subset, so recent-only/random pair comparisons stay exact.
            rng = np.random.default_rng(stable_seed(c["evaluation"]["random_seed"], rec["pair"], condition))
        idx = memory_indices(q, R, B, evidence, condition, rng)
        if dropout and rng.random() < dropout:
            idx = idx[:0]
        z = self.features(rec)
        memory = z[idx].copy()
        if condition == "wrong_evidence":
            sibling = self.records[rec["sibling"]]
            sibling_z = self.features(sibling)
            for j, ti in enumerate(idx):
                if ti in evidence:
                    memory[j] = sibling_z[ti]
        times = np.concatenate([np.arange(q-R+1, q+1), idx])
        # Predictor training reads actions only; never decompresses RGB or accesses future images.
        o = load_episode(c, rec, keys=("actions",))
        # All controls executed since each observation are known at query time.
        # This is a legal summary of additive controls, not simulator state.
        prefix = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(o["actions"][:q], axis=0)])
        prev_actions = prefix[q]-prefix[times]
        sample = {"z": np.concatenate([z[q-R+1:q+1], memory]), "age": (q-times).astype(np.float32),
                  "type": np.concatenate([np.zeros(R, np.int64), np.ones(len(idx), np.int64)]),
                  "past_actions": prev_actions, "actions": o["actions"][q:q+max(c["data"]["horizons"])].copy(),
                  "current": z[q].copy()}
        if rec['task'] in self.bases:
            p = self.projected_features(rec)
            sample['projected_z'] = p[times].copy()
            if condition == 'wrong_evidence':
                sibling_p = self.projected_features(self.records[rec['sibling']])
                for j, ti in enumerate(idx):
                    if ti in evidence:
                        sample['projected_z'][R+j] = sibling_p[ti]
            sample['projected_current'] = p[q].copy()
        target = z[q+np.asarray(c["data"]["horizons"])].copy()
        info = {"indices": idx, "history_count": len(idx), "evidence_hit": bool(set(evidence.tolist()).issubset(idx.tolist()))}
        return sample, target, info


def collate(samples, device):
    # Padding is sanitized before linear layers; both encoder and cross attention mask it.
    n = max(len(s["z"]) for s in samples)
    batch = {}
    for key in ["z", "age", "type", "past_actions"]+(['projected_z'] if 'projected_z' in samples[0] else []):
        a = samples[0][key]
        out = np.zeros((len(samples), n)+a.shape[1:], dtype=a.dtype)
        for i, s in enumerate(samples):
            out[i, :len(s[key])] = s[key]
        batch[key] = torch.from_numpy(out).to(device)
    batch["mask"] = torch.tensor([[j < len(s["z"]) for j in range(n)] for s in samples], dtype=torch.bool, device=device)
    for key in ["actions", "current"]+(['projected_current'] if 'projected_current' in samples[0] else []):
        batch[key] = torch.from_numpy(np.stack([s[key] for s in samples])).to(device)
    return batch
