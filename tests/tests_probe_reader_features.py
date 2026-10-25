"""Focused data-boundary and weighted-ridge contracts for diagnostic probes."""
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.phase2 import probe_reader_features as probe


def make_source(path):
    data = path/'data'; data.mkdir(parents=True)
    features, lookup, records = [], {}, []
    def add(key, x):
        if key not in lookup:
            lookup[key] = len(features); features.append(x)
        return lookup[key]
    for split in ('train', 'validation', 'development'):
        for f in range(4):
            family = split+'-'+str(f)
            blank = add(family+'blank', np.zeros(8))
            for task in ('R', 'D', 'S'):
                for bit in (0, 1):
                    slots, payload, tags = [], [], []
                    for slot in range(3):
                        axis = 'a' if slot < 2 else 'b'
                        target = bit if slot < 2 else 1-bit
                        tag = (slot+bit) % 2
                        x = np.zeros(8)
                        if task == 'S':
                            x[4], x[5] = target*2-1, tag*2-1
                        else:
                            x[(0 if task == 'R' else 2)+(axis == 'b')] = target*2-1
                        x[6] = f/10; x[7] = f % 2
                        key = family+task+axis+str(target)+(str(tag) if task == 'S' else '')
                        slots.append(add(key, x)); payload.append(target); tags.append(tag)
                    records.append({'id': family+task+str(bit), 'family': family, 'split': split,
                        'task_family': task, 'roles': ['A', 'Aprime', 'B'], 'semantic_axes': ['a', 'a', 'b'] if task != 'S' else ['parity']*3,
                        'slot_payload': payload, 'tags': tags if task == 'S' else None,
                        'salient_slot': None, 'condition': 'neutral', 'evidence': [0, 1, 2],
                        'query': 3, 'rgb_indices': slots+[blank]})
    probe.write(data/'records.json', records); probe.write(data/'rgb_index.json', lookup)
    np.save(data/'features_raw.npy', features); np.save(data/'features.npy', features)
    np.savez(path/'basis.npz', input_mean=np.zeros(8), projection=np.eye(8)[:, :6])
    return data


def test_ridge_objective_matches_closed_form_and_free_intercept():
    rng = np.random.default_rng(2)
    for n, d in [(20, 3), (5, 9)]:
        x = rng.normal(size=(n, d))+8
        y = rng.normal(size=(n, 2))+3
        w = rng.uniform(.1, 1, n); w /= w.sum()
        coef, intercept = probe.ridge_path(x, y, w, [.1])[.1]
        xc, yc = x-w@x, y-w@y
        expected = np.linalg.solve(xc.T@(w[:, None]*xc)+.1*np.eye(d), xc.T@(w[:, None]*yc))
        assert np.allclose(coef, expected, atol=1e-10)
        assert np.allclose(intercept, w@y-(w@x)@expected, atol=1e-10)
        diagnostics = probe.normal_equation_diagnostics(x, y[:, 0], w, coef[:, 0], intercept[0], .1)
        assert diagnostics['all_finite']
        assert diagnostics['residual_l2_divided_by_max_one_rhs_l2'] < 1e-10
        with pytest.raises(ValueError, match='residual exceeds'):
            probe.normal_equation_diagnostics(x, y[:, 0], w, coef[:, 0]+.1, intercept[0], .1)


def test_nonfinite_solver_inputs_fail_closed():
    with pytest.raises(ValueError, match='Nonfinite ridge inputs'):
        probe.ridge_path(np.array([[1.], [np.inf]]), np.array([0., 1.]), np.array([.5, .5]), [.1])


def test_bootstrap_resamples_families_and_reports_class_recall_deterministically():
    # Family a is always correct, family b always wrong, each contains both labels.
    rows = [{'family': 'a'}, {'family': 'a'}, {'family': 'b'}, {'family': 'b'}]
    y, scores = np.array([0, 1, 0, 1]), np.array([-1., 1., 1., -1.])
    config = {'seed': 20261122, 'resamples': 2000}
    first = probe.metric(y, scores, rows, config)
    assert first['class_recall'] == {'0': .5, '1': .5}
    assert first['bootstrap'] == probe.metric(y, scores, rows, config)['bootstrap']
    assert first['bootstrap']['family_equal_balanced_accuracy'] == {'valid_resamples': 2000, 'ci95': [0., 1.]}
    # Repeating every row of one family must not give it extra family-level mass.
    indices = [0, 1, 0, 1, 0, 1, 2, 3]
    repeated = probe.metric(y[indices], scores[indices], [rows[i] for i in indices], config)
    assert first['class_recall'] == repeated['class_recall']
    assert first['bootstrap']['family_equal_accuracy'] == repeated['bootstrap']['family_equal_accuracy']
    single = probe.metric(y[:2], scores[:2], rows[:2], config)
    assert single['bootstrap']['status'] == 'omitted_single_family'


def test_normalization_train_only_floor_and_family_weighting():
    rows = [{'family': 'a'}, {'family': 'a'}, {'family': 'b'}]
    x = np.array([[0, 4], [2, 4], [5, 4]], float)
    mean, std = probe.normalize_fit(x, rows, 1e-4)
    assert np.allclose(mean, [3, 4]); assert std[1] == 1e-4
    frozen = mean.copy(), std.copy()
    future = (np.full((3, 2), 999.)-mean)/std
    assert np.max(future) > 1000
    assert np.array_equal(frozen[0], mean) and np.array_equal(frozen[1], std)


def test_shuffle_once_preserves_within_head_salience_counts():
    rows = [{'head': head, 'salient': salient, 'target': i % 2} for head in ('R_a', 'S_tag') for salient in (False, True) for i in range(20)]
    y, order = probe.shuffled_targets(rows, 20261121)
    assert np.array_equal(order, probe.shuffled_targets(rows, 20261121)[1])
    assert not np.array_equal(order, np.arange(len(rows)))
    for head in ('R_a', 'S_tag'):
        for salient in (False, True):
            indices = [i for i,r in enumerate(rows) if r['head'] == head and r['salient'] == salient]
            assert sum(y[indices]) == sum(rows[i]['target'] for i in indices)
            assert set(order[indices]) == set(indices)


def test_dedup_and_family_overlap_fail_closed(tmp_path):
    data = make_source(tmp_path/'source')
    rows = probe.cue_rows(data, 'train')
    assert len({(r['family'],r['rgb_sha256'],r['head']) for r in rows}) == len(rows)
    assert len(rows) < 4*3*2*3+4*2*3
    with pytest.raises(ValueError, match='Family overlap'):
        probe.assert_disjoint(rows, rows)


def test_full_fit_has_no_dev_normalization_or_selection_and_evaluates_frozen(tmp_path):
    data = make_source(tmp_path/'source')
    before = probe.fit(data.parent, tmp_path/'fit', {'lambdas': [.01, .1]})
    assert len(before['models']) == 36
    assert all(m['validation_balanced_accuracy'] == .5 for m in before['models'] if m['label_arm'] == 'blank')
    assert set(before['fit_families']) == {'train', 'validation'}
    assert all(m['validation_balanced_accuracy'] == 1 for m in before['models'] if m['label_arm'] == 'real')
    dev = probe.cue_rows(data, 'development')
    feature = np.load(data/'features_raw.npy')
    feature[[r['frame_index'] for r in dev]] += 999
    np.save(data/'features_raw.npy', feature)
    after = probe.fit(data.parent, tmp_path/'fit2', {'lambdas': [.01, .1]})
    assert before['normalization'] == after['normalization']
    assert before['inputs_hashes'] == after['inputs_hashes']
    assert [(m['selected_lambda'],m['checkpoint_sha256']) for m in before['models']] == [(m['selected_lambda'],m['checkpoint_sha256']) for m in after['models']]
    report = probe.evaluate(tmp_path/'fit', data, 'development', tmp_path/'eval')
    assert len(report['models']) == 36
    assert all(v['all']['balanced_accuracy'] == 1 for k,v in report['models'].items() if k.startswith('projected') and k.endswith('real'))
    with pytest.raises(ValueError, match='overlaps'):
        probe.evaluate(tmp_path/'fit', data, 'train', tmp_path/'bad_eval')
    norm = tmp_path/'fit'/'normalization.npz'
    norm.write_bytes(norm.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        probe.evaluate(tmp_path/'fit', data, 'development', tmp_path/'tampered_eval')
