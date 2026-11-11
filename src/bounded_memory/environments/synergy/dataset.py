"""Synergy (terminal R/D/S) dataset: frame generation, I-JEPA encoding and the reader's input adapter.

Extracted unchanged from experiments/phase2/run_terminal_structures.py (only the import lines inside make_banks, generate and encode
changed). generate() renders every episode with the frozen generator (generate.py) and stores each
distinct frame once; encode() runs the pinned I-JEPA and normalises with TRAIN statistics (or a
given normalisation file); Data, causal_capsule and reader_sample build what a reader sees: the
four recent frames plus the kept ones of the three candidates, never targets or labels.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

H = (3, 6)
TASKS = ('R', 'D', 'S')


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def emit(stage, **kw):
    print(json.dumps({'stage': stage, **kw}), flush=True)


def make_banks(c):
    from . import generate as env
    tc = c['splits']['train']
    families = [env.family(tc['seed'], i, 'train') for i in range(tc['families'])]
    selected = np.random.default_rng(c['reference_plan_seed']).choice(len(families), c['reference_plans'], replace=False)
    refs, sources = [], []
    for i in selected:
        r = env.records_for_family(families[int(i)])[0]
        refs.append(env.reference_plan(r)); sources.append(r['family'])
    evaluation = np.random.default_rng(c['evaluation_plan_seed']).uniform(c['action_prior_uniform_low'],
        c['action_prior_uniform_high'],(c['evaluation_plans'],6,2)).astype(np.float32)
    return {'reference': np.asarray(refs, np.float32), 'evaluation': np.asarray(evaluation, np.float32)}, sources


class RGBRegistry:
    """Stream unique RGB to disk; keep only hashes/indices in RAM."""
    def __init__(self, path):
        self.path, self.lookup, self.frames = path, {}, 0
        self.file = path.open('wb')
        self.shape = None

    def add(self, frame):
        x = np.asarray(frame)
        if x.dtype != np.uint8 or x.shape != (96, 96, 3):
            raise ValueError('Expected uint8 96x96 RGB')
        key = hashlib.sha256(x.tobytes()).hexdigest()
        if key not in self.lookup:
            self.lookup[key] = self.frames
            self.file.write(x.tobytes()); self.frames += 1
        return self.lookup[key]

    def close(self):
        self.file.close()


def fifo_parent_trace(query):
    memory=[];trace=[]
    for t in range(query+1):
        incoming=t-4
        if incoming<0:continue
        before=list(memory);memory.append(incoming);dropped=None
        if t<query and len(memory)>2:dropped=memory.pop(0)
        trace.append({'time':t,'before':before,'incoming':incoming,'prefix_deleted':dropped,'after':list(memory)})
    if memory!=[query-6,query-5,query-4]:raise ValueError('Actual FIFO prefix does not form terminal parent')
    return trace


def generate(c, data, splits, profile=False, existing_banks=None):
    from . import generate as env
    if data.exists() and any(data.iterdir()):
        raise ValueError('Do not overwrite generated dataset')
    data.mkdir(parents=True, exist_ok=True)
    banks, sources = make_banks(c) if existing_banks is None else (existing_banks, [])
    np.savez(data/'action_banks.npz', **banks)
    write(data/'action_bank_sources.json', {'reference_train_families': sources,
          'reference_seed': c['reference_plan_seed'], 'evaluation_seed': c['evaluation_plan_seed'],
          'same_reference_bank_all_parents': True})
    rgb = RGBRegistry(data/'unique_rgb.bin')
    records, actions, futures = [], {}, {}
    counts = {};prefix_traces={}
    try:
        for split in splits:
            n = 1 if profile else c['splits'][split]['families']
            before = rgb.frames
            for index in range(n):
                family = env.family(c['splits'][split]['seed'], index, split)
                prefix_traces[family['family_id']]=fifo_parent_trace(family['query'])
                for r in env.records_for_family(family):
                    ep = env.episode(r)
                    q = int(r['query'])
                    if ep['actions'].shape != (len(ep['rgb'])-1, 2) or np.any(ep['actions'][:q]):
                        raise ValueError('Executed actions/prefix contract')
                    if list(r['evidence']) != [q-6, q-5, q-4]:
                        raise ValueError('Terminal candidate times differ')
                    r['rgb_indices'] = [rgb.add(f) for f in ep['rgb']]
                    actions[r['id']] = ep['actions']
                    use = ('reference',) if split == 'train' else (('evaluation',) if split == 'validation' else ('reference', 'evaluation'))
                    r['plan_targets'] = {}
                    for name in use:
                        key = (r['family'], r['task_family'], r['outcome_class'], name)
                        if key not in futures:
                            futures[key] = [[rgb.add(x) for x in env.target_frames(r, plan)] for plan in banks[name]]
                        r['plan_targets'][name] = futures[key]
                    records.append(r)
                emit('generating', split=split, family=index+1, total=n, unique_rgb=rgb.frames)
            counts[split] = {'families_generated': n, 'records': sum(r['split'] == split for r in records),
                             'additional_unique_rgb': rgb.frames-before}
    finally:
        rgb.close()
    write(data/'records.json', records)
    write(data/'prefix_traces.json',prefix_traces)
    np.savez_compressed(data/'actions.npz', **actions)
    write(data/'rgb_index.json', rgb.lookup)
    info = {'unique_rgb': rgb.frames, 'rgb_shape': [rgb.frames, 96, 96, 3],
            'rgb_bytes': rgb.frames*96*96*3, 'one_feature_array_bytes': rgb.frames*5120*4,
            'estimated_raw_plus_normalized_feature_bytes': rgb.frames*5120*8,
            'profile_only': profile, 'counts': counts,
            'records_sha256': sha(data/'records.json'), 'action_banks_sha256': sha(data/'action_banks.npz')}
    if profile:
        estimate = sum(counts[s]['additional_unique_rgb']*c['splits'][s]['families'] for s in splits)
        info['linear_profile_unique_rgb_estimate'] = estimate
        info['linear_profile_disk_bytes_rgb_plus_two_features'] = estimate*(96*96*3+5120*8)
        info['estimate_limit'] = 'one family per split; different appearance/geometry may change deduplication'
    write(data/'generation.json', info)
    emit('prepared', **info)


def encode(c, data, cache, device, normalization_source=None):
    from ...predictors.phase1.encoders import FrozenIJepa
    from ...encoders.ijepa import ENCODER_CONFIG, EXPECTED_ENCODER_METADATA_SHA256, digest
    import torch
    if (data/'features.npy').exists():
        raise ValueError('Do not overwrite encoded features')
    info = read(data/'generation.json')
    if info['profile_only']:
        raise ValueError('Profile data cannot be encoded as experiment')
    records = read(data/'records.json')
    bank = np.memmap(data/'unique_rgb.bin', dtype=np.uint8, mode='r', shape=tuple(info['rgb_shape']))
    raw = np.lib.format.open_memmap(data/'features_raw.npy', mode='w+', dtype=np.float32, shape=(len(bank), 5120))
    enc = FrozenIJepa(dict(ENCODER_CONFIG, cache_dir=str(cache), batch_size=c['encoder_batch_size']), torch.device(device))
    if digest(enc.metadata) != EXPECTED_ENCODER_METADATA_SHA256:
        raise ValueError('Pinned encoder metadata mismatch')
    began = time.perf_counter()
    for at in range(0, len(bank), c['encoder_batch_size']):
        z = enc.encode(np.array(bank[at:at+c['encoder_batch_size']]))
        if z.shape[1:] != (5120,) or not np.isfinite(z).all():
            raise ValueError('Invalid encoded features')
        raw[at:at+len(z)] = z
        if at % (c['encoder_batch_size']*25) == 0:
            emit('encoding', done=at+len(z), total=len(bank))
    raw.flush()
    if normalization_source is None:
        counts = np.zeros(len(bank), np.int64)
        for r in records:
            if r['split'] == 'train':
                np.add.at(counts, r['rgb_indices'][:r['query']+1], 1)
        if counts.sum() == 0:
            raise ValueError('TRAIN prefixes absent for normalization')
        mean, second = np.zeros(5120, np.float64), np.zeros(5120, np.float64)
        for at in range(0, len(raw), 1024):
            x = np.asarray(raw[at:at+1024], np.float64); w = counts[at:at+len(x), None]
            mean += (w*x).sum(0); second += (w*x*x).sum(0)
        mean /= counts.sum(); second /= counts.sum()
        std = np.maximum(np.sqrt(np.maximum(second-mean*mean, 0)), c['normalization_std_floor'])
        np.savez(data/'normalization.npz', mean=mean.astype(np.float32), std=std.astype(np.float32), train_frame_counts=counts)
    else:
        a = np.load(normalization_source, allow_pickle=False)
        mean, std = a['mean'], a['std']
        np.savez(data/'normalization.npz', mean=mean, std=std)
    a = np.load(data/'normalization.npz', allow_pickle=False); mean, std = a['mean'], a['std']
    normalized = np.lib.format.open_memmap(data/'features.npy', mode='w+', dtype=np.float32, shape=raw.shape)
    for at in range(0, len(raw), 1024):
        normalized[at:at+1024] = (raw[at:at+1024]-mean)/std
        if not np.isfinite(normalized[at:at+1024]).all():
            raise ValueError('Nonfinite normalized features')
    normalized.flush()
    write(data/'encoding.json', {'seconds': time.perf_counter()-began, 'metadata': enc.metadata,
          'metadata_sha256': digest(enc.metadata), 'features_sha256': sha(data/'features.npy'),
          'normalization_sha256': sha(data/'normalization.npz'), 'normalization_source': str(normalization_source)})


def grouped(records, split):
    groups = defaultdict(list)
    for r in records:
        if r['split'] == split:
            groups[(r['task_family'], r['counterfactual_group'])].append(r)
    return {task: [sorted(v, key=lambda r:r['id']) for (t, _), v in sorted(groups.items()) if t == task] for task in TASKS}


class Data:
    def __init__(self, path, basis):
        self.path, self.basis = path, basis
        self.records = read(path/'records.json')
        self.z = np.load(path/'features.npy', mmap_mode='r', allow_pickle=False)
        self.actions = np.load(path/'actions.npz', allow_pickle=False)
        self.banks = np.load(path/'action_banks.npz', allow_pickle=False)

    def project(self, z):
        return ((z-self.basis['input_mean'])@self.basis['projection']).astype(np.float32)

    def capsule(self, r):
        q = r['query']; z = self.z[r['rgb_indices'][:q+1]]; ac = self.actions[r['id']][:q]
        return causal_capsule(z, ac, q)

    def actual_plan(self, r):
        return self.actions[r['id']][r['query']:r['query']+6].copy()

    def target(self, r):
        return self.z[np.asarray(r['rgb_indices'])[r['query']+np.asarray(H)]].copy()


def causal_capsule(observed_features, executed_actions, clock):
    """Only already observed prefix and executed controls cross this boundary."""
    if observed_features.shape[0] != clock+1 or executed_actions.shape != (clock, 2):
        raise ValueError('Expected causal prefixes only')
    times = np.r_[np.arange(clock-3, clock+1), np.arange(clock-6, clock-3)]
    prefix = np.concatenate([np.zeros((1,2), np.float32), executed_actions.cumsum(0)])
    return {'z': np.asarray(observed_features[times], np.float32), 'age': (clock-times).astype(np.float32),
            'past_actions': (prefix[clock]-prefix[times]).astype(np.float32),
            'current': np.asarray(observed_features[clock], np.float32)}


def reader_sample(capsule, keep, plan, basis, history_override=None, ages_override=None, projection_cache=None):
    """Pure tensor adapter; accepts no role, split, target, id or future observation."""
    if set(capsule) != {'z','age','past_actions','current'}:
        raise ValueError('Causal capsule field whitelist')
    ix = [0,1,2,3]+[4+i for i in keep]
    z, age, past = capsule['z'][ix].copy(), capsule['age'][ix].copy(), capsule['past_actions'][ix].copy()
    if history_override is not None:
        z[4:] = history_override
    if ages_override is not None:
        age[4:] = ages_override
    if projection_cache is not None and history_override is not None:
        raise ValueError('Changed content cannot reuse native projection')
    projected=((z-basis['input_mean'])@basis['projection']).astype(np.float32) if projection_cache is None else projection_cache[0][ix]
    current=((capsule['current']-basis['input_mean'])@basis['projection']).astype(np.float32) if projection_cache is None else projection_cache[1]
    return {'z': z, 'age': age, 'past_actions': past, 'current': capsule['current'],
            'type': np.asarray([0]*4+[1]*len(keep), np.int64), 'actions': np.asarray(plan, np.float32),
            'projected_z': projected,'projected_current': current}


def group_loss(p, y):
    """Proper conditional-mean target for every history size, including XOR.

    No pair or ranking term is used. Complete balanced factorial groups ensure
    identical insufficient inputs receive both future outcomes without a loss
    that rewards pushing them away from their conditional mean.
    """
    return (p-y).square().mean()
