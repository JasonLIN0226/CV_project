from pathlib import Path

import numpy as np
import pytest
import torch

from .core import (BEHAVIORS, OnlineState, behavior_update,
                   choose, paired_initialization, rollout)
from .fit import fit_pair, minibatch_indices
from .integration import labels_for_record
from .pack import VerifiedManifest, manifest_digest, train_inputs
from .artifacts import ASSET_LOCK, read_state_pack, write_state_pack, read_teacher_pack, write_teacher_pack
from .asset_audit import verify_entries
from . import teacher_stage
from .validation import BASELINES, NEW_POLICIES, aggregate_validation, evaluate_episode, verify_trace
from .fresh_inference import verify_fresh_manifest, paired_route_inference
from .checkpoint_fit import fit_pair_checkpointed
from .states import deterministic_seed, load_frozen_behavior, validate_manifest
from .teachers import (OfflineTargets, attention_energy, deletion_distribution,
                       predictor_samples, teacher_pair)


def fixture_state():
    rng = np.random.default_rng(7)
    return OnlineState(10, (0, 3, 6), rng.normal(size=(11, 128)).astype(np.float32),
                       rng.normal(size=(10, 2)).astype(np.float32))


def test_matched_initialization_and_equivariance():
    a, d = paired_initialization(17)
    assert sum(p.numel() for p in a.parameters()) == 45633
    for left, right in zip(a.parameters(), d.parameters()):
        assert torch.equal(left, right) and left.data_ptr() != right.data_ptr()
    optim_a = torch.optim.AdamW(a.parameters(), lr=.001, weight_decay=.0001)
    optim_d = torch.optim.AdamW(d.parameters(), lr=.001, weight_decay=.0001)
    assert optim_a.defaults == optim_d.defaults and optim_a.state_dict()['state'] == optim_d.state_dict()['state']
    x = torch.from_numpy(fixture_state().features()).unsqueeze(0)
    permutation = [2, 0, 1]
    torch.testing.assert_close(a(x[:, permutation]), a(x)[:, permutation])
    assert np.array_equal(minibatch_indices(17, 17), minibatch_indices(17, 17))


def test_feature_boundary_and_past_actions():
    s = fixture_state()
    x = s.features()
    assert x.shape == (3, 517)
    np.testing.assert_allclose(x[:, :128], s.projected_prefix[list(s.candidates)])
    for row, timestamp in enumerate(s.candidates):
        np.testing.assert_allclose(x[row, 257:259], s.actions_prefix[timestamp:10].sum(axis=0), atol=1e-6)
    with pytest.raises(ValueError):
        OnlineState(10, s.candidates, np.vstack((s.projected_prefix, np.ones((1, 128)))), s.actions_prefix)
    with pytest.raises(ValueError):
        OnlineState(10, (0, 3, 7), s.projected_prefix, s.actions_prefix)
    with pytest.raises(ValueError):
        OfflineTargets('test', np.zeros((2, 128)), np.zeros((6, 2)))


def test_deletion_sign_zero_norm_ties_and_permutation():
    p = np.zeros((3, 128)); f = np.zeros((2, 128))
    np.testing.assert_allclose(attention_energy(p, f), np.full(3, 1/3))
    np.testing.assert_allclose(deletion_distribution([1, 1, 1]), np.full(3, 1/3))
    p[0, 0] = 1; p[1, 0] = -1; f[:, 0] = 1
    e = attention_energy(p, f)
    t = deletion_distribution(e)
    assert e[0] > e[2] > e[1] and t[0] < t[2] < t[1]
    perm = [2, 0, 1]
    np.testing.assert_allclose(attention_energy(p[perm], f), e[perm])
    assert choose([1, 1, 0], (6, 0, 3)) == [3, 6]


def test_real_legacy_behavior_code_and_no_resurrection():
    path = Path(__file__).resolve().parents[1] / 'gate_retention_comparison_v1' / 'online.py'
    update_impl = load_frozen_behavior(path)
    rng = np.random.default_rng(8)
    p = rng.normal(size=(29, 128)).astype(np.float32)
    actions = rng.normal(size=(28, 2)).astype(np.float32)
    for policy in BEHAVIORS:
        decision = behavior_update(policy, np.random.default_rng(4), update_impl)
        memory, trace = rollout(p, actions, 28, decision)
        previous = []
        for row in trace:
            assert row['previous'] == previous
            assert set(row['selected']) <= set(previous + [row['arrival']])
            previous = row['selected']
        assert memory == previous and len(memory) == 2
        state, _ = rollout(p, actions, 28, behavior_update(policy, np.random.default_rng(4), update_impl), capture_final=True)
        assert state.clock == 28 and len(state.candidates) == 3
    with pytest.raises(ValueError, match='resurrection'):
        rollout(p, actions, 28, lambda state: [900, 901])


def test_equal_b_predictor_teacher_schema_and_evaluation():
    s = fixture_state(); z = np.zeros((11, 128), np.float32)
    future_actions = np.ones((6, 2), np.float32)
    samples = predictor_samples(s, z, future_actions)
    for i, sample in enumerate(samples):
        assert sample['type'].tolist() == [0] * 4 + [1] * 2
        expected = sorted(t for t in s.candidates if t != s.candidates[i])
        np.testing.assert_array_equal(sample['projected_z'][-2:], s.projected_prefix[expected])
        np.testing.assert_array_equal(sample['actions'], future_actions)

    class FakePredictor(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.horizons = (3, 6)
            self.register_buffer('input_mean', torch.zeros(128))
            self.register_buffer('projection', torch.eye(128))
        def forward(self, batch):
            values = batch['projected_z'][:, -2:, 0].mean(1)
            return values[:, None, None].expand(-1, 2, 128)
    def collate(items, device):
        return {'projected_z': torch.from_numpy(np.stack([s['projected_z'] for s in items])).to(device)}
    target = OfflineTargets('train', np.zeros((2, 128), np.float32), future_actions)
    result = teacher_pair(s, target, z, FakePredictor(), collate, 'cpu')
    for name in ('attention_target', 'deletion_target'):
        assert result[name].shape == (3,) and result[name].sum() == pytest.approx(1)
    expected = []
    for sample in samples:
        value = float(sample['projected_z'][-2:, 0].mean())
        expected.append(value * value)
    np.testing.assert_allclose(result['deletion_energy'], expected, rtol=1e-6)


def test_train_only_and_manifest_firewall():
    with pytest.raises(ValueError, match='only training'):
        fit_pair(np.zeros((1, 3, 517)), np.ones((1, 3))/3, np.ones((1, 3))/3, seed=17, split='validation')
    with pytest.raises(ValueError, match='only original'):
        validate_manifest({'records': [{'split': 'test', 'task': 'gate'}]})
    with pytest.raises(ValueError, match='complete 4096'):
        fit_pair(np.zeros((1, 3, 517)), np.ones((1, 3))/3, np.ones((1, 3))/3, seed=17, split='train')
    assert deterministic_seed(17, 'a', 'fifo') == deterministic_seed(17, 'a', 'fifo')
    assert deterministic_seed(17, 'a', 'fifo') != deterministic_seed(17, 'a', 'reservoir')


def test_reject_tampered_behavior_source(tmp_path):
    altered = tmp_path / 'online.py'
    altered.write_text('def update(*args): return []\n')
    with pytest.raises(ValueError, match='frozen audit'):
        load_frozen_behavior(altered)


def test_frozen_manifest_index_rejects_mutations():
    rows = [{'episode': 'x', 'split': 'train', 'clock': 10, 'candidates': [0, 3, 6]}]
    digest = manifest_digest(rows)
    frozen = VerifiedManifest(rows, digest)
    assert frozen.contains(rows[0])
    rows[0]['candidates'][0] = 1
    assert not frozen.contains(rows[0])
    with pytest.raises(ValueError, match='frozen digest'):
        VerifiedManifest(rows, digest)


def test_training_pack_excludes_validation_and_requires_four_behaviors():
    rows = [{'episode': f'e{i}', 'behavior': p, 'split': 'train', 'seed': 17}
            for i in range(1024) for p in BEHAVIORS]
    rows += [{'episode': f'v{i}', 'behavior': p, 'split': 'validation', 'seed': 17}
             for i in range(192) for p in BEHAVIORS]
    frozen = VerifiedManifest(rows, manifest_digest(rows))
    features = np.broadcast_to(np.zeros((1, 3, 517), np.float32), (len(rows), 3, 517))
    labels = np.broadcast_to(np.ones((1, 3), np.float32)/3, (len(rows), 3))
    x, a, d = train_inputs(rows, features, labels, labels, frozen, 17)
    assert x.shape == (4096, 3, 517) and a.shape == d.shape == (4096, 3)
    swapped = rows.copy(); swapped[0], swapped[-1] = swapped[-1], swapped[0]
    with pytest.raises(ValueError, match='metadata order'):
        train_inputs(swapped, features, labels, labels, frozen, 17)


def test_sealed_artifacts_detect_tampering_and_bad_teacher(tmp_path):
    rows = []
    for split, n in [('train', 1024), ('validation', 192)]:
        for i in range(n):
            for policy in BEHAVIORS:
                rows.append({'split': split, 'episode': f'{split}{i}', 'pair': f'{split}{i//2}',
                             'seed': 17, 'behavior': policy, 'rng_seed': 7,
                             'clock': 10, 'candidates': [0, 3, 6]})
    x = np.broadcast_to(np.zeros((1, 3, 517), np.float32), (len(rows), 3, 517))
    pack = {'metadata': rows, 'features': x, 'manifest_sha256': manifest_digest(rows)}
    state = tmp_path / 'state'
    write_state_pack(state, pack, seed=17, asset_lock_sha256=ASSET_LOCK,
                     protocol_sha256='a'*64, implementation_sha256='b'*64)
    loaded_rows, loaded_features, _ = read_state_pack(state)
    assert loaded_rows == rows and loaded_features.shape == (4864, 3, 517)
    with pytest.raises(FileExistsError):
        write_state_pack(state, pack, seed=17, asset_lock_sha256=ASSET_LOCK,
                         protocol_sha256='a'*64, implementation_sha256='b'*64)
    energy = np.zeros((len(rows), 3), np.float64)
    uniform = np.full((len(rows), 3), 1/3, np.float32)
    with pytest.raises(ValueError, match='disagrees'):
        write_teacher_pack(tmp_path/'bad', state, energy, energy, uniform*0, uniform,
                           timing={'attention_seconds': 1, 'deletion_seconds': 2})
    teacher = tmp_path / 'teacher'
    write_teacher_pack(teacher, state, energy, energy, uniform, uniform,
                       timing={'attention_seconds': 1, 'deletion_seconds': 2})
    assert read_teacher_pack(teacher, state)[3]['label_costs_equal'] is False
    with (state/'features.npz').open('ab') as out:
        out.write(b'x')
    with pytest.raises(ValueError, match='hash mismatch'):
        read_state_pack(state)
    with pytest.raises(ValueError, match='hash mismatch'):
        read_teacher_pack(teacher, state)


def test_teacher_stage_requires_frozen_order_before_calling_labeler(monkeypatch):
    state = fixture_state()
    rows = [{'episode': 'synthetic', 'split': 'train', 'seed': 17,
             'clock': 10, 'candidates': list(state.candidates)}]
    features = np.stack([state.features()])
    monkeypatch.setattr(teacher_stage, 'read_state_pack',
                        lambda root: (rows, features, {'seed': 17, 'manifest_sha256': manifest_digest(rows)}))
    class Store:
        manifest = {'records': [{'id': 'synthetic'}]}
    calls = []
    def labeler(*args, **kwargs):
        calls.append(1)
        return {'attention_energy': np.ones(3), 'deletion_energy': np.ones(3),
                'attention_target': np.ones(3)/3, 'deletion_target': np.ones(3)/3,
                'attention_seconds': 1., 'deletion_seconds': 2.}
    monkeypatch.setattr(teacher_stage, 'label_one_frozen_state', labeler)
    monkeypatch.setattr(teacher_stage, 'iter_states', lambda *args: iter([(rows[0], state)]))
    arrays, timing = teacher_stage.make_teacher_arrays('unused', Store(), None, None, None, None, 'cpu')
    assert arrays['targets_a'].shape == (1, 3) and timing['deletion_seconds'] == 2 and len(calls) == 1
    altered = {**rows[0], 'episode': 'other'}
    monkeypatch.setattr(teacher_stage, 'iter_states', lambda *args: iter([(altered, state)]))
    with pytest.raises(ValueError, match='replayed state'):
        teacher_stage.make_teacher_arrays('unused', Store(), None, None, None, None, 'cpu')
    assert len(calls) == 1


def test_teacher_checkpoint_resume_skips_committed_rows(monkeypatch, tmp_path):
    state = fixture_state()
    rows = [{'episode': f'e{i}', 'split': 'train', 'seed': 17, 'clock': 10,
             'candidates': list(state.candidates)} for i in range(2)]
    state_root = tmp_path/'state'; state_root.mkdir(); (state_root/'seal.json').write_text('fixture')
    monkeypatch.setattr(teacher_stage, 'read_state_pack', lambda root:
        (rows, np.stack([state.features()] * 2), {'seed': 17, 'manifest_sha256': manifest_digest(rows)}))
    class Store:
        manifest = {'records': [{'id': 'e0'}, {'id': 'e1'}]}
    monkeypatch.setattr(teacher_stage, 'iter_states', lambda *args: iter((r, state) for r in rows))
    calls = []
    def labeler(store, rec, *args, **kwargs):
        calls.append(rec['id'])
        return {'attention_energy': np.ones(3), 'deletion_energy': np.ones(3),
                'attention_target': np.ones(3)/3, 'deletion_target': np.ones(3)/3,
                'attention_seconds': 1., 'deletion_seconds': 2.}
    monkeypatch.setattr(teacher_stage, 'label_one_frozen_state', labeler)
    path = tmp_path/'teacher-checkpoint.npz'
    for _ in range(2):
        with pytest.raises(InterruptedError):
            teacher_stage.make_teacher_arrays_checkpointed(state_root, Store(), None, None, None, None, 'cpu',
                    checkpoint=path, stop_requested=lambda: True)
    with np.load(path, allow_pickle=False) as ck:
        assert int(ck['completed']) == 2 and float(ck['deletion_seconds']) == 4
    assert calls == ['e0', 'e1']


def test_frozen_asset_audit_detects_byte_changes_and_path_escape(tmp_path):
    import hashlib
    permitted = tmp_path/'permitted'; permitted.mkdir()
    asset = permitted/'input'; asset.write_bytes(b'original')
    entries = {str(asset): {'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest()}}
    assert verify_entries(entries, [permitted]) == {str(asset): entries[str(asset)]['sha256']}
    asset.write_bytes(b'changed!')
    with pytest.raises(ValueError, match='locked asset differs'):
        verify_entries(entries, [permitted])
    outside = tmp_path/'outside'; outside.write_bytes(b'original')
    with pytest.raises(ValueError, match='outside authorized roots'):
        verify_entries({str(outside): entries[str(asset)]}, [permitted])


def test_validation_episode_has_prefix_firewall_and_audited_online_traces():
    q = 10
    rng = np.random.default_rng(9)
    features = rng.normal(size=(17, 128)).astype(np.float32)
    actions = rng.normal(size=(16, 2)).astype(np.float32)
    rec = {'id': 'synthetic', 'split': 'validation', 'task': 'gate', 'query': q,
           'cue_start': 0, 'cue_end': 1}
    class Store:
        c = {'data': {'recent': 4, 'budget': 2, 'horizons': [3, 6]}}
        def features(self, r): return features
        def projected_features(self, r): return features
    class Predictor(torch.nn.Module):
        def forward(self, batch):
            return torch.zeros((batch['size'], 2, 128))
    baseline_called = []
    def baseline(z, p, a, query, *args):
        assert z.shape == p.shape == (q+1, 128) and a.shape == (q, 2)
        chosen, trace = rollout(p, a, query, lambda s: sorted(s.candidates)[-2:])
        baseline_called.append(True)
        return ({name: chosen for name in BASELINES},
                [{**row, 'policy': name} for name in BASELINES for row in trace], {}, 0)
    def loader(c, r, eval_only=False, keys=None):
        if eval_only:
            assert baseline_called and keys == ('evidence',)
            return {'evidence': np.array([0])}
        assert keys == ('actions',)
        return {'actions': actions}
    def sample(store, r, idx, *unused):
        return {}, np.zeros((2,128), np.float32), {}
    def collate(samples, device): return {'size': len(samples)}
    def append(rows, **kwargs): rows.append({'condition': kwargs['policy']})
    a, d = paired_initialization(17)
    result = evaluate_episode(Store(), rec, Predictor(), dict(zip(NEW_POLICIES, (a, d))), None,
        seed=17, device='cpu', collate=collate, load_episode=loader,
        selected_evidence=lambda c, r, x: x, baseline_replay=baseline,
        sample_with_indices=sample, append_metric_rows=append,
        pair_diagnostic=None, reservoir_seed=1)
    assert len(result['rows']) == 10 and len(result['traces']) == 8
    assert result['selected']['manual_keyframe'] == [0, 6]
    assert not next(r for r in result['rows'] if r['condition'] == 'manual_keyframe')['online_irreversible']
    broken = [dict(row) for row in result['traces']['fifo']]
    broken[-1] = {**broken[-1], 'selected': [0, 6]}
    with pytest.raises(ValueError, match='resurrection'):
        verify_trace(broken, q, broken[-1]['selected'])
    with pytest.raises(ValueError, match='validation'):
        evaluate_episode(None, {**rec, 'split': 'test'}, None, {}, None, seed=17, device='cpu',
            collate=None, load_episode=None, selected_evidence=None, baseline_replay=None,
            sample_with_indices=None, append_metric_rows=None, pair_diagnostic=None, reservoir_seed=1)


def test_validation_aggregation_requires_complete_matched_rows():
    records = [{'id': f'e{i}', 'pair': f'p{i//2}', 'split': 'validation', 'task': 'gate'} for i in range(192)]
    policies = list(BASELINES) + list(NEW_POLICIES) + ['recent_only', 'manual_keyframe']
    rows = [{'seed': seed, 'episode': rec['id'], 'pair': rec['pair'], 'condition': policy,
             'horizon': h, 'split': 'validation', 'state_accuracy': 0.6 if policy in NEW_POLICIES else 0.5,
             'normalized_mse': 0.1, 'state_position_mse': 0.2, 'pair_accuracy': 0.5,
             'event_hit': 1 if int(rec['id'][1:]) % 2 else 0, 'evidence_hit': 0}
            for seed in (17,29,43) for rec in records for policy in policies for h in (3,6)]
    result = aggregate_validation(rows, records)
    assert result['rows'] == 11520
    assert all(result['students_exceed_recent_only_on_pooled_validation'].values())
    assert len(result['group_metrics']) == 60
    with pytest.raises(ValueError, match='incomplete'):
        aggregate_validation(rows[:-1], records)
    with pytest.raises(ValueError, match='duplicate'):
        aggregate_validation(rows + rows[:1], records)
    poisoned = [{**r, 'split': 'test'} if i == 0 else r for i, r in enumerate(rows)]
    with pytest.raises(ValueError, match='identity or split'):
        aggregate_validation(poisoned, records)


def test_fresh_inference_scene_bootstrap_and_exposure_firewall():
    records = []
    for split in ('test_id', 'test_long_gap'):
        for pair in range(64):
            group = f'{split}-p{pair}'
            for sibling in (0, 1):
                records.append({'id': f'{group}-{sibling}', 'pair': group,
                                'sibling': f'{group}-{1-sibling}', 'seed': pair + (0 if split == 'test_id' else 64),
                                'task': 'gate', 'split': split})
    audit = verify_fresh_manifest(records, generation_seed=2026102701, prior_ids={'old'},
             prior_pairs={'old-pair'}, prior_scene_seeds={999},
             model_frozen_before_generation=True, previously_unopened=True)
    assert audit['episodes'] == 256
    with pytest.raises(ValueError, match='previously exposed'):
        verify_fresh_manifest(records, generation_seed=2026102701, prior_ids={records[0]['id']},
            prior_pairs=set(), prior_scene_seeds=set(), model_frozen_before_generation=True, previously_unopened=True)
    with pytest.raises(ValueError, match='non-exposure'):
        verify_fresh_manifest(records, generation_seed=2026102701, prior_ids=set(), prior_pairs=set(),
            prior_scene_seeds=set(), model_frozen_before_generation=True, previously_unopened=False)
    rows = [{'seed': seed, 'episode': rec['id'], 'pair': rec['pair'], 'split': rec['split'],
             'condition': policy, 'horizon': h, 'state_accuracy': int(policy == 'deletion_cost_student')}
            for seed in (17,29,43) for rec in records for policy in NEW_POLICIES for h in (3,6)]
    result = paired_route_inference(rows, records)
    assert result['positive_on_both_conditions'] and result['conditions']['test_id']['ci_97_5_percent'] == [1., 1.]
    with pytest.raises(ValueError, match='incomplete'):
        paired_route_inference(rows[:-1], records)


def test_paired_checkpoint_resume_keeps_equal_step_and_rejects_other_input(tmp_path):
    rng = np.random.default_rng(3)
    one = rng.normal(size=(1,3,517)).astype(np.float32)
    x = np.broadcast_to(one, (4096,3,517)).copy()
    y = np.full((4096,3), 1/3, np.float32)
    path = tmp_path/'checkpoint.pt'
    with pytest.raises(InterruptedError):
        fit_pair_checkpointed(x, y, y, seed=17, signature='a'*64, checkpoint=path,
                              device='cpu', stop_requested=lambda: True)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['step'] == 1
    assert all(int(v['step']) == 1 for opt in ck['optimizers'] for v in opt['state'].values())
    with pytest.raises(ValueError, match='signature'):
        fit_pair_checkpointed(x, y, y, seed=17, signature='b'*64, checkpoint=path, device='cpu')
    with pytest.raises(InterruptedError):
        fit_pair_checkpointed(x, y, y, seed=17, signature='a'*64, checkpoint=path,
                              device='cpu', stop_requested=lambda: True)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['step'] == 2
    assert all(int(v['step']) == 2 for opt in ck['optimizers'] for v in opt['state'].values())


def test_store_adapter_rejects_test_before_reading_and_preserves_prefix():
    state = fixture_state()
    class Store:
        c = {'data': {'horizons': [3, 6]}}
        def features(self, rec):
            return np.zeros((17, 128), np.float32)
        def projected_features(self, rec):
            return np.concatenate([state.projected_prefix, np.zeros((6, 128), np.float32)])
    def loader(c, rec, keys):
        assert keys == ('actions',)
        return {'actions': np.concatenate([state.actions_prefix, np.zeros((6, 2), np.float32)])}
    class Model(torch.nn.Module):
        horizons = (3, 6)
        def __init__(self):
            super().__init__()
            self.register_buffer('input_mean', torch.zeros(128))
            self.register_buffer('projection', torch.eye(128))
        def forward(self, batch):
            return torch.zeros((3, 2, 128))
    def collate(samples, device):
        assert len(samples) == 3 and all(len(s['z']) == 6 for s in samples)
        return {}
    rec = {'task': 'gate', 'split': 'train', 'query': 10}
    labels = labels_for_record(Store(), rec, state, load_episode=loader, model=Model(), collate=collate, device='cpu')
    assert labels['attention_target'].sum() == pytest.approx(1)
    np.testing.assert_allclose(labels['deletion_target'], np.full(3, 1/3))
    with pytest.raises(ValueError, match='train/validation'):
        labels_for_record(None, {'task': 'gate', 'split': 'test', 'query': 10}, state,
                          load_episode=None, model=None, collate=None, device='cpu')
    with pytest.raises(ValueError, match='final query'):
        labels_for_record(Store(), {**rec, 'query': 9}, state,
                          load_episode=loader, model=Model(), collate=collate, device='cpu')
