"""Recompute locked causal-consensus choices from saved forecasts; no inference.

Reports all development parents and family-cluster uncertainty. This audit does
not choose a policy, alter its locked predictions, or make confirmation claims.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def interval(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['family']].append(float(field(row)))
    means = np.asarray([np.mean(grouped[k]) for k in sorted(grouped)])
    rng = np.random.default_rng(20261025)
    draws = means[rng.integers(0, len(means), (2000, len(means)))].mean(1)
    return {'mean': float(means.mean()), 'family_percentile95': np.quantile(draws, [.025, .975]).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--causal', type=Path, required=True)
    p.add_argument('--forecasts', type=Path, required=True)
    p.add_argument('--full-vector', type=Path, required=True)
    a = p.parse_args()
    lock = read(a.causal/'choices_locked.json')
    assert sha(a.forecasts) == lock['reference_predictions_sha256']
    assert sha(a.causal/'choices.json') == lock['choices_sha256']
    predictions = np.load(a.forecasts, mmap_mode='r', allow_pickle=False)
    rows = read(a.causal/'evaluated_rows.json')
    choices = read(a.causal/'choices.json')
    native = {r['id']: r for r in read(a.run/'terminal_decisions.json')}
    records = {r['id']: r for r in read(a.run/'data/records.json')}
    full = {r['id']: r for r in read(a.full_vector/'episode_results.json')}
    assert len(rows) == len(choices) == len(predictions) == 192
    max_delta = 0.
    for i, row in enumerate(rows):
        assert row['id'] == choices[i]['id']
        pred = predictions[i].astype(np.float64)
        assert pred.shape[:3] == (16, 3, 2) and np.isfinite(pred).all()
        pairwise = np.zeros((3, 3))
        for j in range(3):
            for k in range(j+1, 3):
                pairwise[j, k] = pairwise[k, j] = ((pred[:, j]-pred[:, k])**2).mean((1, 2)).mean()
        scores = pairwise.sum(1)
        assert int(scores.argmin()) == row['drop'] == choices[i]['drop']
        max_delta = max(max_delta, float(np.max(abs(scores-np.asarray(row['sum_distances'])))))
        assert np.allclose(scores, row['sum_distances'], atol=1e-12, rtol=1e-12)
        n = native[row['id']]
        assert row['mse'] == n['losses'][row['drop']]
        assert row['regret'] == row['mse']-min(n['losses'])
        assert row['semantic_correct'] == (row['drop'] in n['semantic_correct_delete_set'])
        row['duplicate_axis'] = records[row['id']]['scene']['duplicate_axis']
        row['full_vector'] = full[row['id']]['policies']['full_prediction_medoid']
    result = {'scope': 'posthoc already-inspected DEV, not confirmation', 'checks': {
        'forecast_sha256_matches_lock': True, 'choices_sha256_matches_lock': True,
        'recomputed_parents': len(rows), 'all_choices_match': True,
        'maximum_score_difference': max_delta, 'all_archived_losses_and_semantic_labels_match': True,
        'new_reader_calls': 0}, 'bootstrap': {'unit': 'family', 'samples': 2000, 'seed': 20261025},
        'same_choice_as_actual_plan_full_vector': sum(r['drop'] == r['full_vector']['drop'] for r in rows),
        'subgroups': {}, 'semantic_failures': []}
    styles = ['benign_duplicate', 'distant_style_duplicate']
    for key in ['all']+styles+['axis/a', 'axis/b']+[s+'/'+axis for s in styles for axis in ['a', 'b']]:
        rr = [r for r in rows if key == 'all' or key == r['condition'] or key == 'axis/'+r['duplicate_axis'] or key == r['condition']+'/'+r['duplicate_axis']]
        result['subgroups'][key] = {'episodes': len(rr), 'families': len({r['family'] for r in rr}),
            'unique_evidence_loss_count': sum(not r['semantic_correct'] for r in rr),
            'loss_rate': interval(rr, lambda r: not r['semantic_correct']),
            'regret': interval(rr, lambda r: r['regret']),
            'paired_mse_differences': {name: interval(rr, lambda r, name=name: r['mse']-r['baseline'][name]['mse'])
                for name in ['fifo', 'terminal_uniform', 'visual_diversity', 'reader_attention']},
            'minus_actual_plan_full_vector': interval(rr, lambda r: r['mse']-r['full_vector']['mse'])}
    for r in rows:
        if not r['semantic_correct']:
            result['semantic_failures'].append({k: r[k] for k in ['id', 'family', 'condition', 'duplicate_axis', 'drop', 'regret', 'feature_class_correct', 'sum_distances']})
    result['source_and_inputs_sha256'] = {str(x): sha(x) for x in [Path(__file__), a.forecasts, a.causal/'choices.json', a.causal/'evaluated_rows.json', a.run/'terminal_decisions.json', a.full_vector/'episode_results.json']}
    (a.causal/'independent_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'checks': result['checks'], 'all': result['subgroups']['all'], 'same_choice_as_full': result['same_choice_as_actual_plan_full_vector']}))


if __name__ == '__main__':
    main()
