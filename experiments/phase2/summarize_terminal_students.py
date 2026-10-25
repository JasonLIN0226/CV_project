"""Join frozen terminal decisions with completed student predictions.

Analysis only: no reader, encoder, policy fitting, checkpoint selection or new
seed selection. The primary summaries separate R/D/S and conditions, average
the three fixed training seeds within each parent, then bootstrap base families.
An optional independent coefficient/Gram audit recomputes target-free forecast
distances without rerunning the reader or changing the original saved choices.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


SEEDS = (17, 29, 43)
BOOTSTRAP_SEED = 20261107
BOOTSTRAP_DRAWS = 2000
SCORE_ATOL = 1e-5
SCORE_RTOL = 1e-5
METRICS = ('necessary_evidence_loss_rate', 'mse', 'regret')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def family_means(values, families):
    values = np.asarray(values, np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) != len(families):
        raise ValueError('Require aligned [parent,3 metrics] and family IDs')
    index = defaultdict(list)
    for i, family in enumerate(families):
        index[family].append(i)
    keys = sorted(index)
    return keys, np.stack([values[index[key]].mean(0) for key in keys])


def summarize_values(values, families):
    keys, means = family_means(values, families)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = means[rng.integers(0, len(means), (BOOTSTRAP_DRAWS, len(means)))].mean(1)
    return {'parents': len(values), 'families': len(keys),
            'metrics': {name: {'equal_family_mean': float(means[:, k].mean()),
                               'raw_parent_mean': float(np.asarray(values)[:, k].mean()),
                               'family_percentile95': np.quantile(draws[:, k], [.025, .975]).tolist()}
                        for k, name in enumerate(METRICS)}}


def metrics_for_drops(rows, drops):
    drops = np.asarray(drops)
    if drops.shape != (len(rows),) or not np.issubdtype(drops.dtype, np.integer) or ((drops < 0) | (drops > 2)).any():
        raise ValueError('Require one integer deletion0..2 per parent')
    result = []
    for row, drop in zip(rows, drops):
        losses = np.asarray(row['expected_losses'], np.float64)
        if losses.shape != (3,) or not np.isfinite(losses).all():
            raise ValueError('Native expected losses must be finite length3')
        result.append([float(int(drop) not in row['semantic_correct_delete_set']), losses[drop], losses[drop]-losses.min()])
    return np.asarray(result, np.float64)


def gram_scores(coefficients, output_basis):
    """Independent distance algebra; shared current/target-mean terms cancel."""
    c, basis = np.asarray(coefficients, np.float64), np.asarray(output_basis, np.float64)
    if c.ndim != 4 or c.shape[1:3] != (4, 2) or c.shape[-1] != basis.shape[1]:
        raise ValueError('Require coefficients[plans,drop0/drop1/drop2/full,2,rank]')
    gram = basis.T@basis/basis.shape[0]
    return _scores_with_gram(c, gram)


def _scores_with_gram(c, gram):
    pairwise = np.zeros((3, 3), np.float64)
    for i in range(3):
        for j in range(i+1, 3):
            delta = c[:, i]-c[:, j]
            pairwise[i, j] = pairwise[j, i] = np.einsum('lhr,rs,lhs->lh', delta, gram, delta, optimize=True).mean()
    sensitivity = []
    for i in range(3):
        delta = c[:, i]-c[:, 3]
        sensitivity.append(np.einsum('lhr,rs,lhs->lh', delta, gram, delta, optimize=True).mean())
    return {'medoid': pairwise.sum(1), 'full_sensitivity': np.asarray(sensitivity), 'pairwise_mse': pairwise}


def audit_coefficients(run, split, rows):
    coefficient_path = run/split/'reference_coefficients.npy'
    basis_path = run/'basis.npz'
    coefficients = np.load(coefficient_path, mmap_mode='r', allow_pickle=False)
    with np.load(basis_path, allow_pickle=False) as stored:
        basis = stored['output_basis'].astype(np.float64)
    if len(coefficients) != len(rows):
        raise ValueError('Reference coefficient row count mismatch')
    gram = basis.T@basis/basis.shape[0]
    max_error = {'medoid': 0., 'full_sensitivity': 0., 'pairwise_mse': 0.}
    action_matches = {'medoid': 0, 'full_sensitivity': 0}
    disagreements = []
    for index, row in enumerate(rows):
        c = np.asarray(coefficients[index], np.float64)
        if c.ndim != 4 or c.shape[1:3] != (4, 2) or c.shape[-1] != basis.shape[1] or not np.isfinite(c).all():
            raise ValueError('Invalid reference coefficient tensor')
        computed = _scores_with_gram(c, gram)
        details = row['selection_details']
        original_pairwise = np.asarray(details['medoid']['pairwise_mse'], np.float64)
        err = float(np.max(abs(computed['pairwise_mse']-original_pairwise)))
        max_error['pairwise_mse'] = max(max_error['pairwise_mse'], err)
        if not np.allclose(computed['pairwise_mse'], original_pairwise, atol=SCORE_ATOL, rtol=SCORE_RTOL):
            raise ValueError('Coefficient pair distances disagree beyond frozen numerical tolerance')
        for name in ('medoid', 'full_sensitivity'):
            original = np.asarray(details[name]['scores'], np.float64)
            error = float(np.max(abs(computed[name]-original)))
            max_error[name] = max(max_error[name], error)
            if not np.allclose(computed[name], original, atol=SCORE_ATOL, rtol=SCORE_RTOL):
                raise ValueError(f'Coefficient {name} score differs beyond frozen numerical tolerance')
            saved_drop = int(details[name]['drop'])
            if saved_drop != int(original.argmin()) or saved_drop != row['policies'][name]['drop']:
                raise ValueError('Saved policy did not use exact earliest-index argmin of its saved scores')
            recovered = int(computed[name].argmin())
            if recovered == saved_drop:
                action_matches[name] += 1
            else:
                gap = float(original[recovered]-original[saved_drop])
                if gap > 2*error+1e-12:
                    raise ValueError('Action disagreement is not explained by measured score perturbation')
                disagreements.append({'id': row['id'], 'method': name, 'saved_drop': saved_drop,
                                      'coefficient_argmin': recovered, 'original_score_gap': gap,
                                      'maximum_score_difference': error})
    source_summary = read(run/split/'summary.json')
    result = {'status': 'passed_independent_coefficient_distance_audit', 'parents': len(rows),
              'formula': 'G=output_basis.T@output_basis / feature_dimension; mean over plans/horizons of delta_coeff.T G delta_coeff',
              'cancellation': 'current feature and common horizon target_mean are identical for all four subsets, so cancel from deletion-vs-deletion and deletion-vs-full differences',
              'maximum_score_difference': max_error, 'exact_choice_matches': action_matches,
              'floating_order_disagreements': disagreements,
              'numerical_comparison': {'absolute_tolerance': SCORE_ATOL, 'relative_tolerance': SCORE_RTOL,
                                       'choice_rule_not_changed': 'Original saved exact-argmin choices remain evaluation actions'},
              'reader_recorded_reconstruction_max_abs_difference': source_summary['coefficient_reconstruction_max_abs_difference'],
              'reader_reconstruction_field_is_source_evidence_not_independent_inference': True,
              'reader_calls': 0, 'target_losses_recomputed_from_features': False,
              'hashes': {'coefficients': sha(coefficient_path), 'basis': sha(basis_path)}}
    write(run/split/'student_coefficient_audit.json', result)
    return result


def build_report(run, split, audit=False, baseline_only=False):
    dest, export = run/split, run/'exports'/split
    rows = read(dest/'baseline_decisions.json')
    ids = read(export/'ids.json')
    ids = [x['id'] if isinstance(x, dict) else x for x in ids]
    if ids != [r['id'] for r in rows] or len(ids) != len(set(ids)):
        raise ValueError('Decision/export parent order must match exactly without duplicate IDs')
    if set(r['task_family'] for r in rows) != {'R', 'D', 'S'}:
        raise ValueError('Report requires all three task families')
    metadata = read(export/'metadata.json')
    if [r['id'] for r in metadata['records']] != ids:
        raise ValueError('Export metadata row order mismatch')
    predictions_path = run/'student_predictions'/(split+'.npz')
    manifest_path = run/'students/manifest.json'
    manifest = None
    if not baseline_only:
        receipt = read(predictions_path.with_suffix('.json'))
        manifest = read(manifest_path)
        if sha(predictions_path) != receipt['predictions_sha256'] or sha(export/'inputs.npz') != receipt['inputs_npz_sha256']:
            raise ValueError('Student predictions/input hashes differ from prediction receipt')
        if sha(manifest_path) != receipt['student_manifest_sha256']:
            raise ValueError('Student checkpoint selection manifest changed after prediction')
        if receipt['labels_npz_opened'] or receipt['ids_json_opened']:
            raise ValueError('Prediction receipt violates deployed input boundary')
    with np.load(export/'labels.npz', allow_pickle=False) as f:
        expected, attention = np.asarray(f['cost'], np.float64), np.asarray(f['attention'], np.float64)
    if expected.shape != (len(rows), 3) or attention.shape != expected.shape or not np.isfinite(expected).all() or not np.isfinite(attention).all():
        raise ValueError('Export label rows mismatch')
    methods, descriptions = {}, {}
    baseline_names = sorted(rows[0]['policies'])
    for row, label in zip(rows, expected):
        if set(row['policies']) != set(baseline_names) or row['label_bank'] != 'evaluation':
            raise ValueError('DEV/CONF must share baseline arms and use independent evaluation plans')
        if not np.array_equal(np.asarray(row['expected_losses']), label):
            raise ValueError('Expected native losses differ from exported evaluator cost')
        per_plan = np.asarray(row['per_plan_losses']['evaluation'], np.float64)
        if per_plan.ndim != 2 or per_plan.shape[1] != 4 or not np.isfinite(per_plan).all() or not np.allclose(per_plan[:, :3].mean(0), label, atol=1e-12, rtol=1e-12):
            raise ValueError('Oracle must minimize mean plan loss, not average per-plan minima')
    for name in baseline_names:
        drops = np.asarray([r['policies'][name]['drop'] for r in rows], np.int64)
        metrics = metrics_for_drops(rows, drops)
        for row, values in zip(rows, metrics):
            old = row['policies'][name]
            if not np.allclose([old['mse'], old['regret']], values[1:], atol=1e-12, rtol=1e-12) or bool(old['semantic_correct']) != (values[0] == 0):
                raise ValueError('Saved baseline metrics do not match one shared deletion loss')
        key = 'baseline/'+name
        methods[key] = metrics
        descriptions[key] = {'type': 'fixed_baseline', 'seed_aggregation': 'not applicable'}
    for key, drops, scope in (
            ('diagnostic/expected_loss_oracle', expected.argmin(1), 'Evaluator-only minimum after averaging the fixed independent action bank'),
            ('diagnostic/future_attention_label_oracle', attention.argmin(1), 'Privileged future-query attention label argmin; not a deployable causal baseline')):
        methods[key] = metrics_for_drops(rows, drops)
        descriptions[key] = {'type': 'privileged_diagnostic', 'scope': scope}
    seed_groups = defaultdict(dict)
    if not baseline_only:
        with np.load(predictions_path, allow_pickle=False) as predictions:
            expected_keys = {name+'__'+kind for name in manifest['arms'] for kind in ('logits', 'drop')}
            if set(predictions.files) != expected_keys:
                raise ValueError('Student prediction arms differ from frozen fit manifest')
            for name, arm in manifest['arms'].items():
                logits, drops = predictions[name+'__logits'], predictions[name+'__drop']
                if logits.shape != (len(rows), 3) or not np.isfinite(logits).all() or not np.array_equal(logits.argmax(1), drops):
                    raise ValueError('Student must select exact earliest-index max deletion logit')
                key = 'student/'+name
                methods[key] = metrics_for_drops(rows, drops)
                descriptions[key] = {'type': 'individual_training_seed', 'architecture': arm['architecture'],
                                     'label': arm['label'], 'seed': arm['seed']}
                seed_groups[(arm['architecture'], arm['label'])][int(arm['seed'])] = key
        for architecture in ('independent', 'set'):
            for label in ('cost', 'future_attention', 'shuffle'):
                if set(seed_groups[(architecture, label)]) != set(SEEDS):
                    raise ValueError('All six primary arms require exactly the three frozen training seeds')
    for (architecture, label), members in seed_groups.items():
        if set(members) != set(SEEDS):
            raise ValueError('Do not average an incomplete or posthoc selected seed subset')
        key = f'mean3/{architecture}/{label}'
        methods[key] = np.stack([methods[members[seed]] for seed in SEEDS]).mean(0)
        descriptions[key] = {'type': 'mean_performance_over_fixed_training_seeds', 'seeds': list(SEEDS),
                             'not_an_ensemble_policy': True,
                             'order': 'average metric over seeds per parent, then parents per base family; no best seed selection'}
    comparison_keys = {
        'set_cost_minus_independent_cost': ('mean3/set/cost', 'mean3/independent/cost'),
        'set_cost_minus_set_future_attention': ('mean3/set/cost', 'mean3/set/future_attention'),
        'set_cost_minus_set_shuffle': ('mean3/set/cost', 'mean3/set/shuffle'),
        'independent_cost_minus_independent_future_attention': ('mean3/independent/cost', 'mean3/independent/future_attention'),
        'set_cost_minus_medoid': ('mean3/set/cost', 'baseline/medoid'),
        'set_cost_minus_full_sensitivity': ('mean3/set/cost', 'baseline/full_sensitivity'),
        'set_cost_minus_reader_attention': ('mean3/set/cost', 'baseline/reader_attention'),
    }
    if baseline_only:
        comparison_keys = {'medoid_minus_full_sensitivity': ('baseline/medoid', 'baseline/full_sensitivity'),
                           'medoid_minus_reader_attention': ('baseline/medoid', 'baseline/reader_attention')}
    subgroup_keys = [(task, None) for task in ('R', 'D', 'S')]
    subgroup_keys += [(task, condition) for task in ('R', 'D', 'S') for condition in sorted({r['condition'] for r in rows if r['task_family'] == task})]
    groups = {}
    family_records = []
    for task, condition in subgroup_keys:
        indices = [i for i, r in enumerate(rows) if r['task_family'] == task and (condition is None or r['condition'] == condition)]
        families = [rows[i]['family'] for i in indices]
        group_name = task+'/'+('all_conditions' if condition is None else condition)
        summaries = {name: summarize_values(value[indices], families) for name, value in methods.items()}
        for name, values in methods.items():
            keys, means = family_means(values[indices], families)
            for family, mean in zip(keys, means):
                family_records.append({'group': group_name, 'method': name, 'family': family, **dict(zip(METRICS, mean.tolist()))})
        comparisons = {name: {'left': pair[0], 'right': pair[1], 'negative_favors_left': True,
                              **summarize_values(methods[pair[0]][indices]-methods[pair[1]][indices], families)}
                       for name, pair in comparison_keys.items()}
        groups[group_name] = {'task_family': task, 'condition': condition, 'methods': summaries, 'paired_comparisons': comparisons}
    qualification_path = dest/'reader_qualification.json'
    qualification = read(qualification_path) if qualification_path.exists() else None
    coefficient_audit = audit_coefficients(run, split, rows) if audit else None
    report = {'status': 'completed_analysis_of_frozen_baselines' if baseline_only else 'completed_analysis_of_frozen_predictions', 'split': split,
              'scope': 'Development analysis; not confirmation' if split == 'development' else 'Locked confirmation results; interpretation depends on predeclared reader/method gates',
              'parents': len(rows), 'base_families': len({r['family'] for r in rows}),
              'no_new_models_or_training': True, 'no_checkpoint_or_seed_selection': True,
              'baseline_only': baseline_only,
              'endpoints': ['necessary evidence loss', 'expected feature MSE', 'regret = mean-plan selected loss minus minimum mean-plan deletion loss'],
              'statistical_protocol': {'training_seeds': list(SEEDS), 'bootstrap_seed': BOOTSTRAP_SEED,
                  'bootstrap_draws': BOOTSTRAP_DRAWS, 'unit': 'base family',
                  'aggregation': 'three fixed model seed performances averaged within parent before family means',
                  'conditions': 'R,D,S and each condition reported separately; no pooled primary result',
                  'interval_scope': 'percentile95 paired family bootstrap, conditional on fitted seeds',
                  'multiplicity': 'Intervals are unadjusted descriptive intervals; this script makes no multiplicity-adjusted superiority declaration'},
              'reader_qualification': qualification, 'full_sensitivity_requires_qualified_K3_reader': True,
              'retention_method_interpretation_requires_passed_K2_reader_gates': True,
              'method_definitions': descriptions, 'groups': groups,
              'coefficient_audit': None if coefficient_audit is None else {k: coefficient_audit[k] for k in ('status', 'parents', 'maximum_score_difference', 'exact_choice_matches', 'floating_order_disagreements')},
              'hashes': {'source': sha(__file__), 'decisions': sha(dest/'baseline_decisions.json'), 'student_manifest': None if baseline_only else sha(manifest_path),
                         'student_predictions': None if baseline_only else sha(predictions_path), 'prediction_receipt': None if baseline_only else sha(predictions_path.with_suffix('.json')),
                         'export_labels': sha(export/'labels.npz'), 'export_ids': sha(export/'ids.json')},
              'limits': ['An oracle with low expected loss is not a deployable strategy.', 'Full-reference sensitivity is not qualified if K3 content gates fail.',
                         'Seed-average performance is not a model ensemble and does not treat seeds or siblings as independent samples.',
                         'No empirical bound on independent scorers is inferred without the separately checked generator conditions.']}
    write(dest/'student_comparison_family_metrics.json', family_records)
    write(dest/'student_comparison_summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--split', choices=('development', 'confirmation'), required=True)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--baseline-only', action='store_true', help='Archive failed-reader baseline evidence without requiring any student checkpoint')
    args = parser.parse_args()
    result = build_report(args.run.resolve(), args.split, args.audit, args.baseline_only)
    print(json.dumps({'status': result['status'], 'split': args.split, 'parents': result['parents'],
                      'families': result['base_families'], 'groups': len(result['groups']),
                      'coefficient_audit': result['coefficient_audit']}))


if __name__ == '__main__':
    main()
