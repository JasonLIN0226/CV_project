import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

d = module('signal_diagnostics', 'tools/diagnose_signal.py')
p = module('counterbalanced_pools', 'experiments/counterbalanced_pools.py')


def test_signs_keep_zero_and_negative_cases():
    out = d.sign_summary([1., 0., -1., 0.])
    assert (out['positive'], out['tied'], out['negative']) == (1, 2, 1)
    assert out['positive_fraction'] == .25
    assert d.wilson(0, 128)[0] == 0
    assert np.isclose(d.wilson(128, 128)[1], 1)


def test_rank_ties_and_undefined_correlation():
    assert np.array_equal(d.ranks([3, 1, 1]), [2, .5, .5])
    assert d.spearman([1, 1], [2, 3]) is None
    assert np.isclose(d.spearman([1, 2, 3], [3, 2, 1]), -1)


def test_duplicate_pair_axis_rejected():
    r = dict(task='gate', split='test_id', horizon=3, role='cue', seed=17,
             pair='x', kind='critical_vs_nuisance', critical_feature=1,
             nuisance_feature=0, critical_task=1, nuisance_task=0)
    with pytest.raises(ValueError, match='Duplicate'):
        d.pair_cube([r, r])


def test_feature_oracle_can_be_worse_on_task_metric():
    records = []
    for name in ('uniform_delete', 'FIFO_delete_oldest', 'reject_newest', 'future_feature_oracle'):
        records.append(dict(task='gate', split='test_id', horizon=3, seed=17, baseline=name,
            metrics={'feature_mse': {'mean': .5 if name=='future_feature_oracle' else 1.},
                     'task_error': {'mean': 2. if name=='future_feature_oracle' else 1.}}))
    rows = d.headroom_rows(records)
    assert rows[0]['relative_gap'] == .5
    assert rows[1]['relative_gap'] == -1.  # Never clamp away the counterexample.


def test_equal_capacity_and_clock_boundary():
    pools = p.gate_age_counterbalance(4, 6, 20)
    for expected_rank, pool in enumerate(pools.values()):
        assert len(set(pool)) == 3 and list(pool) == sorted(pool)
        assert pool.index(4) == expected_rank and max(pool) < 17
        assert not {5, 6}.intersection(pool)
    a = p.clock_only_static_pools(20, 41)
    assert a == p.clock_only_static_pools(20, 41)
    assert len(set(a)) == 6 and all(len(set(t))==3 and max(t)<17 for t in a)
    with pytest.raises(ValueError):p.gate_age_counterbalance(1, 2, 20)
    with pytest.raises(ValueError):p.clock_only_static_pools(4, 41, count=2)


def test_gate_counterbalance_uses_real_noncue_rgb():
    import json
    from phase1.envs import scene, episode
    c = json.loads((ROOT/'results/reference/gate/resolved_config.json').read_text())
    for split in d.SPLITS:
        for index in (0, 1, 2):
            s = scene(c, 'gate', split, index)
            a, b = episode(s, 0)['rgb'], episode(s, 1)['rgb']
            pools = p.gate_age_counterbalance(s['start'], s['end'], s['query'])
            for pool in pools.values():
                assert not np.array_equal(a[s['start']], b[s['start']])
                assert all(np.array_equal(a[t], b[t]) for t in pool if t!=s['start'])


def test_counterbalanced_forks_stay_inside_original_model_input_boundary():
    import ast
    tree = ast.parse((ROOT/'experiments/deletion/run_assay.py').read_text())
    sample = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='sample')
    ns = {'np': np}
    exec(compile(ast.Module(body=[sample], type_ignores=[]), '<original_sample>', 'exec'), ns)
    z = np.arange(32*8, dtype=np.float32).reshape(32, 8)
    actions = np.ones((31, 2), dtype=np.float32)
    changed = z.copy();changed[21:] = -1e6
    for pool in p.gate_age_counterbalance(4, 6, 20).values():
        for removed in pool:
            kept = tuple(t for t in pool if t!=removed)
            a = ns['sample'](z, z[:, :3], actions, 20, kept)
            b = ns['sample'](changed, changed[:, :3], actions, 20, kept)
            assert a['z'].shape == (6, 8)  # Four recent plus two real history entries.
            assert all(np.array_equal(a[k], b[k]) for k in a)
