"""Portable, inference-only adapter for the two explicitly separated releases.

Inputs are normalized I-JEPA feature vectors, not pixels. This adapter neither
encodes images nor trains/fits anything. Checkpoint hashes are checked before
deserializing the trusted project checkpoints. No historical /scratch access.
"""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'results/phase1_independent_confirmation/execution_v10_checkpoint_resume/source'
sys.path.insert(0, str(REPO / 'experiments/predictor/control/src'))
from phase1.model import DynamicsPredictor
from phase1.centered_transport import CenteredTransportPredictor
from phase1.inputs import collate
from phase1.probes import LinearReadout

try:
    from .verify_handoff import sha
except ImportError:
    from verify_handoff import sha


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _frozen_imports():
    # The frozen scripts have generic absolute import names. Isolate those names
    # while loading; functions retain references to their own original modules.
    names = ('core', 'geometry', 'guard', 'kernel', 'gate_intervention')
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            sys.modules[name] = _module('_handoff_' + name, SOURCE / (name + '.py'))
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


with _frozen_imports():
    LOCKED = _module('_handoff_locked_inference', SOURCE / 'locked_inference.py')


def npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def pack(features, actions, query, retained):
    """Recent four plus chronologically ordered, actually retained past frames."""
    z = np.asarray(features, np.float32)
    actions = np.asarray(actions, np.float32)
    retained = tuple(retained)
    if (query < 3 or tuple(sorted(set(retained))) != retained
            or any(t < 0 or t >= query - 3 for t in retained)
            or len(actions) < query + 6 or len(z) <= query):
        raise ValueError('Invalid observed timestamps or incomplete known actions')
    times = np.asarray(list(range(query - 3, query + 1)) + list(retained), np.int64)
    prefix = np.vstack([np.zeros((1, 2), np.float32), np.cumsum(actions[:query], axis=0)])
    return dict(z=z[times].copy(), current=z[query].copy(),
                age=(query-times).astype(np.float32),
                type=np.r_[np.zeros(4, np.int64), np.ones(len(retained), np.int64)],
                past_actions=prefix[query]-prefix[times], actions=actions[query:query+6].copy())


class Predictor:
    """One family/task/seed with its matching reference and train-only assets."""
    def __init__(self, intake, family, task, seed, manifest=None):
        if family not in ('development', 'independent_confirmation') or task not in ('gate', 'motion') or seed not in (17,29,43):
            raise ValueError('Choose an explicit published family, task and seed')
        self.family, self.task, self.seed = family, task, seed
        self.root = Path(intake).expanduser().resolve()
        self.manifest = json.loads(Path(manifest or REPO/'handoff/asset_manifest.json').expanduser().read_text())
        records = [r for r in self.manifest['files'] if r['variant'] == family]

        def checked(record):
            p = (self.root/record['relative_path']).resolve()
            if not p.is_relative_to(self.root) or p.stat().st_size != record['bytes'] or sha(p) != record['sha256']:
                raise ValueError('Asset hash/size mismatch: '+record['relative_path'])
            return p

        def ending(suffix):
            choices = [r for r in records if r['relative_path'].endswith(suffix)]
            if len(choices) != 1:
                raise ValueError('Ambiguous or missing asset: '+suffix)
            return checked(choices[0])

        def checkpoint(role):
            choices = [r for r in records if r.get('role') == role and r.get('task') == task
                       and r.get('seed') == seed and r['relative_path'].endswith('/best.pt')]
            if len(choices) != 1:
                raise ValueError('Missing unique matching checkpoint')
            p = checked(choices[0])
            ck = torch.load(p, map_location='cpu', weights_only=False)
            if not all(torch.isfinite(t).all() for t in ck['model'].values()):
                raise ValueError('Nonfinite model state')
            return ck

        base = f'phase1_v3_dev/{task}/reference/' if family == 'development' else 'fresh/'
        config_path = ending(base+'resolved_config.json') if family == 'development' else ending('execution_v10_checkpoint_resume/config.json')
        self.config = json.loads(config_path.read_text())
        self.basis = npz(ending(base+f'predictor/{task}/basis.npz'))
        self.normalization = npz(ending(base+'features/normalization.npz'))
        reference_ck, candidate_ck = checkpoint('reference'), checkpoint('candidate')
        self.reference = DynamicsPredictor(len(self.basis['input_mean']), self.config['model'], (3,6), self.basis)
        self.reference.load_state_dict(reference_ck['model'], strict=True)
        self.candidate = CenteredTransportPredictor(self.reference, self.basis)
        self.candidate.load_state_dict(candidate_ck['model'], strict=True)
        # Loading the candidate must never silently substitute a different teacher.
        for key, tensor in reference_ck['model'].items():
            if not torch.equal(self.candidate.reference.state_dict()[key], tensor):
                raise ValueError('Candidate contains a different frozen reference: '+key)
        for name in ('input_mean', 'projection', 'output_basis'):
            expected = torch.as_tensor(self.basis[name])
            if (not torch.equal(getattr(self.candidate, name), expected)
                    or not torch.equal(getattr(self.reference, name), expected)):
                raise ValueError('Checkpoint contains a different train-only basis: '+name)
        for model in (self.reference, self.candidate):
            model.eval().requires_grad_(False)
        if task == 'motion':
            decoder = LinearReadout.load(ending(base+'probes/motion_state_decoder.npz'))
            prefix = 'phase1_motion_observation_map/gpu_output/' if family == 'development' else 'fresh/observation_maps/fitted/'
            self.motion = LOCKED.MotionInference(self.basis, decoder,
                        {h: npz(ending(prefix+f'head_{h}.npz')) for h in (3,6)})

    def normalize(self, encoded_features):
        """Apply the matching train-only normalization once to raw encoder output."""
        return ((np.asarray(encoded_features)-self.normalization['mean'])/self.normalization['std']).astype(np.float32)

    def predict(self, sample, *, query=None, retained_rgb=None, current_rgb=None, own_actions=None):
        """Return normalized future features at h=3,6, shape (2,5120).

        Motion's fixed K=2 repair requires supplied observed RGB and known
        actions; all other history counts retain the original neural output.
        No test labels, future RGB, sibling IDs or hidden simulator state enter.
        """
        batch = collate([sample], 'cpu')
        if self.task == 'gate':
            return LOCKED.gate_transport_off(self.candidate, batch)[0].numpy()
        with torch.inference_mode():
            base = self.candidate(batch)[0].numpy()
            history = self.reference(batch)[0].numpy()
        if int(np.sum(sample['type'] == 1)) != 2:
            return base
        if any(x is None for x in (query, retained_rgb, current_rgb, own_actions)):
            raise ValueError('Motion K=2 requires its actual observed context')
        times = tuple(int(query - age) for age in sample['age'][4:])
        enabled, geometry = self.motion.context(sample, times, retained_rgb, current_rgb, own_actions, query)
        return self.motion.correct(base, history, sample, enabled, geometry)


def smoke(intake):
    """Load all 24 best checkpoints and four production heads; no dataset access."""
    torch.set_num_threads(2)
    rows = []
    for family in ('development', 'independent_confirmation'):
        for task in ('gate', 'motion'):
            for seed in (17,29,43):
                model = Predictor(intake, family, task, seed)
                rng = np.random.default_rng(1729)
                features = model.basis['input_mean'] + rng.normal(0,.001,(40,5120)).astype(np.float32)
                actions = np.zeros((40,2), np.float32)
                sample = pack(features, actions, 24, (2,10))
                kwargs = dict(query=24, retained_rgb=np.zeros((2,96,96,3),np.uint8),
                              current_rgb=np.zeros((96,96,3),np.uint8), own_actions=actions)
                a = model.predict(sample, **kwargs)
                kwargs['retained_rgb'] = np.full((2,96,96,3),255,np.uint8)
                b = model.predict(sample, **kwargs)
                assert a.shape == b.shape == (2,5120) and np.isfinite(a).all() and np.isfinite(b).all()
                # Empty-history identity is a nontrivial invariant of the repair.
                empty = pack(features, actions, 24, ())
                pred = model.predict(empty)
                with torch.inference_mode():
                    ref = model.reference(collate([empty], 'cpu'))[0].numpy()
                assert np.array_equal(pred, ref)
                rows.append(dict(family=family,task=task,seed=seed,strict_load=True,
                                 finite_forward=True,empty_history_identity=True))
    return dict(status='PASS', best_checkpoints_loaded=24, production_maps_loaded=4,
                pairs=rows, scope='CPU loading and synthetic legal-input inference only; no scientific rerun')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--intake', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(smoke(args.intake), indent=2))
