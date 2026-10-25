"""Synthetic arithmetic/report checks; no learned reader or student fitting."""
import json

import numpy as np

from experiments.phase2.summarize_terminal_students import (
    build_report, family_means, gram_scores, metrics_for_drops, summarize_values,
)


def test_family_means_do_not_treat_siblings_as_independent_families():
    values = np.asarray([[0., 1., 0.], [1., 3., 2.], [1., 3., 2.], [1., 3., 2.]])
    keys, means = family_means(values, ['a', 'b', 'b', 'b'])
    assert keys == ['a', 'b']
    assert np.array_equal(means, [[0., 1., 0.], [1., 3., 2.]])
    summary = summarize_values(values, ['a', 'b', 'b', 'b'])
    assert summary['metrics']['necessary_evidence_loss_rate']['equal_family_mean'] == .5
    assert summary['metrics']['necessary_evidence_loss_rate']['raw_parent_mean'] == .75
    paired_zero = summarize_values(values-values, ['a', 'b', 'b', 'b'])
    assert all(x['family_percentile95'] == [0., 0.] for x in paired_zero['metrics'].values())


def test_coefficient_gram_matches_nonorthogonal_full_feature_distances():
    rng = np.random.default_rng(17)
    basis = rng.normal(size=(9, 3))  # deliberately not orthonormal
    coefficients = rng.normal(size=(4, 4, 2, 3))
    common = rng.normal(size=(2, 9))
    predictions = common[None, None]+coefficients@basis.T
    computed = gram_scores(coefficients, basis)
    distances = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            distances[a, b] = ((predictions[:, a]-predictions[:, b])**2).mean()
    full = ((predictions[:, :3]-predictions[:, 3:4])**2).mean((0, 2, 3))
    assert np.allclose(computed['medoid'], distances.sum(1), atol=1e-13)
    assert np.allclose(computed['full_sensitivity'], full, atol=1e-13)
    # Any common current/horizon offset cancels, even if large.
    shifted = predictions+100.
    assert np.allclose(((shifted[:, :3]-shifted[:, 3:4])**2).mean((0, 2, 3)), full, atol=1e-12)


def test_regret_is_relative_to_minimum_of_expected_losses():
    row = {'expected_losses': [5., 5., 8.], 'semantic_correct_delete_set': [0, 1]}
    values = metrics_for_drops([row], np.asarray([1]))
    assert np.array_equal(values, [[0., 5., 0.]])


def test_failed_gate_baseline_only_report_and_audit_need_no_students(tmp_path):
    rng = np.random.default_rng(29)
    dest = tmp_path/'development'
    export = tmp_path/'exports/development'
    dest.mkdir(); export.mkdir(parents=True)
    basis = rng.normal(size=(7, 3)).astype(np.float32)
    np.savez(tmp_path/'basis.npz', output_basis=basis)
    coefficients = rng.normal(size=(6, 2, 4, 2, 3)).astype(np.float32)
    np.save(dest/'reference_coefficients.npy', coefficients)
    rows = []
    for i in range(6):
        c = coefficients[i].astype(np.float64)
        p = c@basis.astype(np.float64).T
        losses = (p*p).mean((2, 3))
        expected = losses[:, :3].mean(0)
        scores = gram_scores(c, basis)
        choices = {'fifo': 0, 'terminal_uniform': 1, 'visual_diversity': 2, 'reader_attention': 0,
                   'medoid': int(scores['medoid'].argmin()), 'full_sensitivity': int(scores['full_sensitivity'].argmin())}
        rows.append({'id': f'parent-{i}', 'family': f'family-{i//3}', 'task_family': ['R', 'D', 'S'][i%3],
            'condition': 'neutral', 'semantic_correct_delete_set': [0], 'label_bank': 'evaluation',
            'expected_losses': expected.tolist(), 'per_plan_losses': {'evaluation': losses.tolist()},
            'selection_details': {k: {'drop': choices[k], 'scores': scores[k].tolist(),
                **({'pairwise_mse': scores['pairwise_mse'].tolist()} if k == 'medoid' else {})}
                for k in ('medoid', 'full_sensitivity')},
            'policies': {k: {'drop': d, 'mse': float(expected[d]), 'regret': float(expected[d]-expected.min()),
                            'semantic_correct': d == 0} for k, d in choices.items()}})
    def write(path, value):
        path.write_text(json.dumps(value))
    write(dest/'baseline_decisions.json', rows)
    write(dest/'summary.json', {'coefficient_reconstruction_max_abs_difference': 0.})
    write(dest/'reader_qualification.json', {'all_task_k2_qualified': False, 'all_task_k3_qualified': False})
    write(export/'ids.json', [r['id'] for r in rows])
    write(export/'metadata.json', {'records': rows})
    np.savez(export/'labels.npz', cost=np.asarray([r['expected_losses'] for r in rows]), attention=np.ones((6, 3))/3)
    result = build_report(tmp_path, 'development', audit=True, baseline_only=True)
    assert result['status'] == 'completed_analysis_of_frozen_baselines'
    assert result['reader_qualification']['all_task_k2_qualified'] is False
    assert result['coefficient_audit']['exact_choice_matches'] == {'medoid': 6, 'full_sensitivity': 6}
    assert result['hashes']['student_manifest'] is None
    assert not (tmp_path/'students').exists()
