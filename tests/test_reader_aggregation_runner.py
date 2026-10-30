"""Small integration guards; synthetic tensors are not experimental evidence."""
from pathlib import Path
import copy
import json
import zipfile

import numpy as np
import pytest
import torch

from experiments.phase2 import run_reader_aggregation as runner
from experiments.phase2.reader_aggregation import VARIANTS, build_reader


def tiny_basis():
    rng = np.random.default_rng(19)
    return {'input_mean': rng.normal(size=12).astype('float32'),
            'projection': rng.normal(size=(12, 8)).astype('float32') / 4,
            'target_mean': rng.normal(size=12).astype('float32'),
            'output_basis': rng.normal(size=(12, 6)).astype('float32') / 4}


@pytest.fixture
def parent_cache(tmp_path, monkeypatch):
    source = tmp_path/'run'; source.mkdir()
    names = ['basis.npz', 'protocol.json', 'reader_selected.pt', 'data/records.json',
             'data/features.npy', 'data/features_raw.npy', 'data/normalization.npz',
             'data/actions.npz', 'data/action_banks.npz', 'data/rgb_index.json']
    files = {}
    for name in names:
        p = source/name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(('synthetic bound artifact: '+name).encode())
        files['run/'+name] = {'bytes': p.stat().st_size, 'sha256': runner.sha(p)}
    archive = tmp_path/'terminal-structures-results.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('package_manifest.json', json.dumps({'files': files,
                        'derived_data_preserved_on_scratch': {}}))
    scientific_source = {'unchanged.py': 'sha-placeholder'}
    monkeypatch.setattr(runner.legacy, 'source_files', lambda: scientific_source)
    config = {'source_run_protocol_sha256': runner.sha(source/'protocol.json'),
              'parent_source_files_sha256': scientific_source,
              'parent_archive_sha256': runner.sha(archive)}
    return config, source


def test_parent_cache_all_bytes_bound_and_confirmation_stays_closed(parent_cache):
    c, source = parent_cache
    verified = runner.verify_parent(c, source)
    assert len(verified['verified_files']) == 10
    (source/'confirmation_opened.json').write_text('{}')
    with pytest.raises(ValueError, match='confirmation'):
        runner.verify_parent(c, source)


@pytest.mark.parametrize('name', ['data/features_raw.npy', 'data/features.npy',
                                'basis.npz', 'data/records.json', 'data/action_banks.npz'])
def test_parent_cache_tamper_rejected_even_if_same_byte_count(parent_cache, name):
    c, source = parent_cache
    p = source/name; value = bytearray(p.read_bytes()); value[-1] ^= 1; p.write_bytes(value)
    with pytest.raises(ValueError, match='cache binding'):
        runner.verify_parent(c, source)


@pytest.mark.parametrize('variant', VARIANTS)
def test_paired_initialization_copies_every_shape_compatible_tensor(variant):
    b = tiny_basis(); base_c = {'reader': {'hidden': 16, 'dropout': 0.}}
    torch.manual_seed(17)
    common = build_reader(12, base_c['reader'], (3, 6), b, 'original').state_dict()
    preserved = {k: v.clone() for k, v in common.items()}
    m, keys = runner.initialize_model({'seed': 17, 'heads': 4}, base_c, b, variant, common)
    m2, keys2 = runner.initialize_model({'seed': 17, 'heads': 4}, base_c, b, variant, common)
    assert keys == keys2 and 'action.0.weight' in keys
    expected = {k for k, v in m.state_dict().items() if k in common and v.shape == common[k].shape}
    assert set(keys) == expected
    assert all(torch.equal(m.state_dict()[k], common[k]) for k in keys)
    assert all(torch.equal(m.state_dict()[k], v) for k, v in m2.state_dict().items())
    assert all(torch.equal(common[k], v) for k, v in preserved.items())


@pytest.mark.parametrize('variant', VARIANTS+('uniform_intervention',))
def test_legacy_forecast_capture_reconstructs_every_aggregation_variant(tmp_path, variant):
    b = tiny_basis(); source = tmp_path/'source'; source.mkdir(); out = tmp_path/'new'
    np.savez(source/'basis.npz', **b)
    checkpoint_variant = 'original' if variant == 'uniform_intervention' else variant
    dest = out/'readers'/checkpoint_variant; dest.mkdir(parents=True)
    base_c = {'reader': {'hidden': 16, 'dropout': 0.}, 'predict_batch_size': 2}
    torch.manual_seed(17)
    m = build_reader(12, base_c['reader'], (3, 6), b, checkpoint_variant)
    torch.save({'model': m.state_dict()}, dest/'reader_selected.pt')
    runner.write(dest/'reader_locked.json', {'checkpoint_sha256': runner.sha(dest/'reader_selected.pt'),
                                           'basis_sha256': runner.sha(source/'basis.npz')})
    predictor = runner.Predictor({'heads': 4}, base_c, source, out, variant, 'cpu')
    rng = np.random.default_rng(51)
    cap = {'z': rng.normal(size=(7, 12)).astype('float32'),
           'current': rng.normal(size=12).astype('float32'),
           'age': np.array([3, 2, 1, 0, 6, 5, 4], dtype='float32'),
           'past_actions': np.zeros((7, 2), dtype='float32')}
    plan = np.full((6, 2), .9, dtype='float32')
    samples = [runner.legacy.reader_sample(cap, keep, plan, b) for keep in ([], [0], [0, 2], [0, 1, 2])]
    prediction, coeff = predictor.predict(samples)
    assert prediction.shape == (4, 2, 12) and coeff.shape == (4, 2, 6)
    expected = cap['current'][None, None]+b['target_mean']+coeff@b['output_basis'].T
    np.testing.assert_allclose(prediction, expected, atol=2e-6, rtol=2e-6)
    assert predictor.calls == 2 and predictor.examples == 4
    assert predictor.reconstruction_max < 2e-6


@pytest.fixture
def model_lock(tmp_path, monkeypatch):
    source, out = tmp_path/'source', tmp_path/'out'; source.mkdir(); out.mkdir()
    (source/'basis.npz').write_bytes(b'frozen basis')
    (out/'probes').mkdir(); (out/'probes/model.npz').write_bytes(b'frozen probe')
    readers = {}
    for name in VARIANTS:
        path = out/f'{name}.pt'; path.write_bytes(name.encode())
        readers[name] = {'checkpoint': path.name, 'checkpoint_sha256': runner.sha(path)}
    source_hashes = {'scientific.py': 'stable'}
    monkeypatch.setattr(runner, 'source_hashes', lambda: source_hashes)
    c = {'variants': list(VARIANTS)}
    config_path = tmp_path/'config.json'; runner.write(config_path, c)
    runner.write(out/'parent_binding.json', {'fixture': 'bound'})
    runner.write(out/'protocol.json', {'config': c, 'config_file_sha256': runner.sha(config_path),
                                      'parent_binding_sha256': runner.sha(out/'parent_binding.json')})
    (out/'shared_initial_state.pt').write_bytes(b'bound initialization')
    lock = {'source_sha256': source_hashes, 'readers': readers,
            'probe_files': {'model.npz': runner.sha(out/'probes/model.npz')},
            'basis_sha256': runner.sha(source/'basis.npz'),
            'protocol_sha256': runner.sha(out/'protocol.json'),
            'config_file_path': str(config_path), 'config_file_sha256': runner.sha(config_path),
            'shared_initial_state_sha256': runner.sha(out/'shared_initial_state.pt')}
    runner.write(out/'all_models_locked.json', lock)
    return c, source, out, lock


def test_all_models_lock_requires_each_preregistered_arm(model_lock):
    c, source, out, lock = model_lock
    runner.verify_prefit_lock(c, source, out)
    lock = copy.deepcopy(lock); del lock['readers']['item_attention']
    runner.write(out/'all_models_locked.json', lock)
    with pytest.raises(ValueError, match='Missing preregistered reader'):
        runner.verify_prefit_lock(c, source, out)


@pytest.mark.parametrize('name', ['original.pt', 'probes/model.npz'])
def test_all_models_lock_rejects_checkpoint_or_probe_tamper(model_lock, name):
    c, source, out, _ = model_lock
    (out/name).write_bytes(b'mutated')
    with pytest.raises(ValueError, match='changed'):
        runner.verify_prefit_lock(c, source, out)


def test_all_models_lock_rejects_scientific_source_change(model_lock, monkeypatch):
    c, source, out, _ = model_lock
    monkeypatch.setattr(runner, 'source_hashes', lambda: {'scientific.py': 'changed'})
    with pytest.raises(ValueError, match='Source changed'):
        runner.verify_prefit_lock(c, source, out)


@pytest.mark.parametrize('name', ['shared_initial_state.pt', 'protocol.json', 'parent_binding.json'])
def test_all_models_lock_rejects_frozen_artifact_tamper(model_lock, name):
    c, source, out, _ = model_lock
    (out/name).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='changed'):
        runner.verify_prefit_lock(c, source, out)


def test_all_models_lock_rejects_config_file_tamper(model_lock):
    c, source, out, lock = model_lock
    Path(lock['config_file_path']).write_text('{}')
    with pytest.raises(ValueError, match='Config file changed'):
        runner.verify_prefit_lock(c, source, out)


def test_all_models_lock_rejects_runtime_config_drift_but_allows_device(model_lock):
    c, source, out, _ = model_lock
    runner.verify_prefit_lock({**c, 'runtime_device': 'cpu'}, source, out)
    with pytest.raises(ValueError, match='Runtime config changed'):
        runner.verify_prefit_lock({**c, 'seed': 999}, source, out)


@pytest.mark.parametrize('split', ['train', 'validation', 'development', 'confirmation'])
def test_fresh_seed_rejects_every_parent_split_even_if_generator_role_differs(split):
    base = {'splits': {name: {'seed': i} for i, name in enumerate(
        ['train', 'validation', 'development', 'confirmation'])}}
    with pytest.raises(ValueError, match='overlaps parent splits: '+split):
        runner.validate_diagnostic_seed({'diagnostic': {'seed': base['splits'][split]['seed']}}, base)
    runner.validate_diagnostic_seed({'diagnostic': {'seed': 99}}, base)


@pytest.mark.parametrize('change', [
    {'students_enabled': True}, {'original_confirm_open_enabled': True},
    {'variants': ['original']}, {'status': 'DRAFT-not-frozen-not-executed'},
    {'must_resolve_before_freeze': ['missing source binding']},
])
def test_scope_or_unresolved_protocol_rejected_before_parent_access(tmp_path, monkeypatch, change):
    c = {'status': 'ready-for-source-freeze-not-yet-executed', 'must_resolve_before_freeze': [],
         'variants': list(VARIANTS), 'students_enabled': False, 'original_confirm_open_enabled': False}
    c.update(change); path = tmp_path/'config.json'; runner.write(path, c)
    def forbidden(*args):
        raise AssertionError('Invalid scope reached parent artifacts')
    monkeypatch.setattr(runner, 'verify_parent', forbidden)
    with pytest.raises(ValueError, match='scope guard|not ready'):
        runner.run(path, tmp_path/'absent-source', tmp_path/'output', 'cpu', tmp_path/'cache')
    assert not (tmp_path/'output').exists()


def test_monolithic_tiny_fixture_finishes_all_arms_before_new_diagnostic(tmp_path, monkeypatch):
    """Real stages on tiny synthetic features; no scientific output/encoder run."""
    root = Path(__file__).resolve().parents[1]
    base = runner.read(root/'experiments/phase2/configs/terminal_structures_20261025.json')
    base.update(steps=4, validate_every=2, groups_per_task=1, predict_batch_size=128)
    base['reader'] = {'hidden': 16, 'heads': 4, 'layers': 2, 'dropout': 0., 'rank': 8}
    base['splits'] = {'train': {'seed': 11, 'families': 2},
                      'validation': {'seed': 12, 'families': 1},
                      'development': {'seed': 13, 'families': 1},
                      'confirmation': {'seed': 14, 'families': 1}}
    base['reference_plans'] = 1; base['evaluation_plans'] = 1
    banks = {name: np.full((1, 6, 2), .9, dtype='float32') for name in ('reference', 'evaluation')}
    source = tmp_path/'parent/run'; source.mkdir(parents=True)
    runner.legacy.generate(base, source/'data', ['train', 'validation', 'development'], existing_banks=banks)
    b = tiny_basis(); np.savez(source/'basis.npz', **b)

    def synthetic_encode(config, data, cache, device, normalization_source=None):
        frames = len(runner.read(data/'rgb_index.json'))
        values = np.random.default_rng(120).normal(size=(frames, 12)).astype('float32')
        np.save(data/'features_raw.npy', values); np.save(data/'features.npy', values)
        np.savez(data/'normalization.npz', mean=np.zeros(12, dtype='float32'), std=np.ones(12, dtype='float32'))
    synthetic_encode(base, source/'data', None, 'cpu')
    runner.write(source/'protocol.json', {'config': base})
    torch.save({'model': build_reader(12, base['reader'], (3, 6), b).state_dict(),
                'basis_sha256': runner.sha(source/'basis.npz')}, source/'reader_selected.pt')
    names = ['basis.npz', 'protocol.json', 'reader_selected.pt', 'data/records.json',
             'data/features.npy', 'data/features_raw.npy', 'data/normalization.npz',
             'data/actions.npz', 'data/action_banks.npz', 'data/rgb_index.json']
    entries = {'run/'+n: {'bytes': (source/n).stat().st_size, 'sha256': runner.sha(source/n)} for n in names}
    archive = source.parent/'terminal-structures-results.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('package_manifest.json', json.dumps({'files': entries, 'derived_data_preserved_on_scratch': {}}))
    c = runner.read(root/'experiments/phase2/configs/reader_aggregation_20261025.json')
    c.update(status='ready-for-source-freeze-not-yet-executed', must_resolve_before_freeze=[],
             source_run_protocol_sha256=runner.sha(source/'protocol.json'),
             parent_archive_sha256=runner.sha(archive), parent_source_files_sha256={'fixture': 'bound'},
             steps=4, validate_every=2, groups_per_task=1, reader=base['reader'],
             scheduler={'milestones': [1, 2], 'gamma': .5},
             diagnostic={'seed': 15, 'families': 1, 'split': 'diagnostic'})
    c['bootstrap']['resamples'] = 10
    c['probe']['lambdas'] = [.01, 1.]
    config = tmp_path/'tiny_config.json'; runner.write(config, c)
    monkeypatch.setattr(runner.legacy, 'source_files', lambda: {'fixture': 'bound'})
    monkeypatch.setattr(runner, 'source_hashes', lambda: {'fixture': 'bound'})
    monkeypatch.setattr(runner.legacy, 'encode', synthetic_encode)
    out = tmp_path/'new_run'
    runner.run(config, source, out, 'cpu', tmp_path/'unused-cache')
    status = runner.read(out/'study_status.json')
    assert status['status'] == 'completed_locked_reader_diagnostic_no_students_no_original_confirmation'
    names = [x['stage'] for x in status['stages']]
    assert max(names.index('train_'+v) for v in VARIANTS) < names.index('generate_fresh_diagnostic')
    assert names.index('qualify_existing_development') < names.index('reverify_parent_before_diagnostic') < names.index('generate_fresh_diagnostic')
    assert not status['original_confirmation_opened'] and not status['retention_students_trained']
    assert not (source/'confirmation_opened.json').exists()
    locked = runner.read(out/'all_models_locked.json')
    assert len({v['training_schedule_sha256'] for v in locked['readers'].values()}) == 1
    summary = runner.read(out/'summary.json')
    for split in ('existing_development', 'fresh_diagnostic'):
        assert set(summary['splits'][split]['readers']) == set(VARIANTS)|{'uniform_intervention'}
    assert set(runner.read(out/'probes/manifest.json')['label_arms']) == {'real', 'shuffle', 'blank'}
    # The separate intervention uses the already bound source checkpoint,
    # not any newly trained checkpoint, and has no new-data stage.
    from experiments.phase2 import run_archived_reader_intervention as archived
    archive_out = tmp_path/'archived_intervention'
    result = archived.run(config, source, archive_out)
    assert result['status'] == 'completed_archived_checkpoint_weight_intervention_development_only'
    assert not result['new_reader_trained'] and not result['data_generated']
    assert set(result['qualification']) == {'original', 'uniform_intervention'}
    assert result['execution']['original']['same_checkpoint_sha256'] == result['execution']['uniform_intervention']['same_checkpoint_sha256']
