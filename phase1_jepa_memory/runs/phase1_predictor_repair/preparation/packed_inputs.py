"""Execution-only gather cache; copies the original legal sample byte for byte.

The model, loss, optimizer and numerical training operations are unchanged.
Only already audited train rows are resident; validation/test rows are rejected.
"""
import numpy as np
import torch
from phase1.inputs import collate, memory_indices, selected_evidence


class PackedTrainInputs:
    sequence_keys = ('z', 'age', 'type', 'past_actions', 'projected_z')
    fixed_keys = ('actions', 'current', 'projected_current')

    def __init__(self, store, pairs, device):
        self.store = store
        self.rows = [r for pair in pairs for r in pair]
        if any(r['split'] != 'train' for r in self.rows):
            raise ValueError('Execution cache only permits the fixed training rows')
        self.ids = {r['id']: i for i, r in enumerate(self.rows)}
        if len(self.ids) != len(self.rows):
            raise ValueError('Duplicate training row')
        self.device = device
        self.recent, self.budget = store.c['data']['recent'], store.c['data']['budget']
        full = [store.sample(r, 'full_history', np.random.default_rng(0)) for r in self.rows]
        self.base = collate([s[0] for s in full], device)
        self.target = torch.from_numpy(np.stack([s[1] for s in full])).to(device)
        self.evidence = {}
        for r in self.rows:
            raw = store.read_episode(store.c, r, True, keys=('evidence',))['evidence']
            self.evidence[r['id']] = selected_evidence(store.c, r, raw)
        self.versions = {k: v._version for k, v in self.base.items()}
        self.target_version = self.target._version

    def get(self, selected, condition, sample_seeds):
        if condition not in ('recent_only', 'full_history', 'privileged_small_budget', 'random_small_budget'):
            raise ValueError('Original four training conditions only')
        if len(selected) != len(sample_seeds):
            raise ValueError('Exactly one original RNG seed per sibling pair')
        rows = [r for pair in selected for r in pair]
        row_indices, positions, infos = [], [], []
        for pair, sample_seed in zip(selected, sample_seeds):
            for r in pair:
                if r['id'] not in self.ids or self.rows[self.ids[r['id']]] != r:
                    raise ValueError('Unknown or changed original training row')
                evidence = self.evidence[r['id']]
                idx = memory_indices(r['query'], self.recent, self.budget, evidence,
                                     condition, np.random.default_rng(int(sample_seed)))
                pos = np.concatenate((np.arange(self.recent), self.recent + idx))
                row_indices.append(self.ids[r['id']]); positions.append(pos)
                infos.append({'indices':idx, 'history_count':len(idx),
                              'evidence_hit':bool(set(evidence.tolist()).issubset(idx.tolist()))})
        n = max(map(len, positions))
        gather = np.zeros((len(rows), n), dtype=np.int64)
        mask = np.zeros((len(rows), n), dtype=bool)
        for i, pos in enumerate(positions):
            gather[i, :len(pos)] = pos
            mask[i, :len(pos)] = True
        ri = torch.as_tensor(row_indices, dtype=torch.long, device=self.device)
        gi = torch.from_numpy(gather).to(self.device)
        valid = torch.from_numpy(mask).to(self.device)
        result = {'mask':valid}
        for key in self.sequence_keys:
            if key not in self.base: continue
            value = self.base[key][ri[:,None], gi]
            padding = ~valid
            while padding.ndim < value.ndim: padding = padding[...,None]
            result[key] = value.masked_fill(padding, 0)
        for key in self.fixed_keys:
            if key in self.base: result[key] = self.base[key].index_select(0, ri)
        return result, self.target.index_select(0, ri), infos

    def verify_unchanged(self):
        if any(v._version != self.versions[k] for k, v in self.base.items()):
            raise ValueError('Model or caller changed the exact execution cache')
        if self.target._version != self.target_version:
            raise ValueError('Model or caller changed the target cache')


def compare_original(store, packed, selected, condition, seeds, device):
    original = [store.sample(r, condition, np.random.default_rng(int(seed)))
                for pair, seed in zip(selected, seeds) for r in pair]
    expected = collate([s[0] for s in original], device)
    target = torch.from_numpy(np.stack([s[1] for s in original])).to(device)
    actual, actual_target, info = packed.get(selected, condition, seeds)
    if set(expected) != set(actual) or any(not torch.equal(v, actual[k]) for k, v in expected.items()):
        raise ValueError('Packed tensor differs from original sampling/collation')
    if not torch.equal(target, actual_target):
        raise ValueError('Packed target differs')
    for left, right in zip([s[2] for s in original], info):
        if set(left) != set(right) or any(not np.array_equal(v, right[k]) for k,v in left.items()):
            raise ValueError('Packed legal indices / metadata differ')
    packed.verify_unchanged()
    return len(original)
