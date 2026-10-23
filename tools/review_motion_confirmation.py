"""Summarize retained Motion evidence; no model calls, fitting or fresh data access."""
import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-root', type=Path, default=repo.parent / 'phase1_jepa_memory/runs/phase1_predictor_repair_v3/native_diagnosis7580')
    args = parser.parse_args()
    out = repo / 'results/motion_confirmation_review'
    out.mkdir(parents=True, exist_ok=True)
    ledger = {}

    def read(path):
        raw = path.read_bytes()
        ledger[str(path.relative_to(repo.parent))] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        return json.loads(raw)

    base = repo / 'results/predictor/evaluation/job_7575'
    comp = read(base / 'motion/development_comparison.json')
    metrics = read(base / 'motion/evaluation/metrics.json')
    audit = read(base / 'independent_scalar_audit.json')
    config = read(repo / 'results/reference/motion/resolved_config.json')['decision']
    assert metrics['raw_sha256'] == comp['candidate_raw_sha256'] == audit['tasks']['motion']['raw_sha256']
    assert audit['tasks']['motion']['all_axes_finite_pairs_pass']
    assert 'PASS' in audit['status']

    def counts(rows):
        total = Counter(r['kind'] for r in rows)
        passed = Counter(r['kind'] for r in rows if r['passed'])
        return {k: {'passed': passed[k], 'total': v} for k, v in total.items()}

    rows = []
    for c in metrics['comparisons']:
        if c['split'] not in ('test_id', 'test_long_gap', 'test_distractor_shift') or c['alternative'] != 'privileged_small_budget':
            continue
        r = {k: c[k] for k in ('split', 'horizon', 'seed')}
        for metric, label in [('normalized_mse', 'feature'), ('state_position_mse', 'position')]:
            x = c['improvement'][metric]
            threshold = config['latent_relative_improvement' if label == 'feature' else 'state_relative_improvement'] if x['scale'].startswith('relative') else config['absolute_latent_gain' if label == 'feature' else 'absolute_position_mse_gain']
            r.update({label + '_' + k: x[k] for k in ('mean', 'low', 'high')})
            r[label + '_passed'] = x['mean'] >= threshold and x['low'] > 0
        name = f"motion/{c['split']}/h{c['horizon']}/seed{c['seed']}"
        original = next(x for x in comp['original_checks'] if x['kind'] == 'privileged_benefit' and x['slice'] == name)
        assert original['passed'] == (r['feature_passed'] and r['position_passed'])
        rows.append(r)
    assert len(rows) == 24
    with (out / 'original_benefit_components.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    transfer = read(args.diagnostic_root / 'transfer_integrity.json')
    diag_path = args.diagnostic_root / 'diagnostic_results_v2.json'
    diag = read(diag_path)
    assert transfer['native_before_local_native_after_all_match']
    assert hashlib.sha256(diag_path.read_bytes()).hexdigest() == transfer['files'][diag_path.name]['sha256']
    assert diag['no_training_or_optimizer'] and diag['GPU'] == 0 and not diag['fresh9137_opened']
    subset = []
    for split in ('train', 'validation', 'test_id', 'test_long_gap', 'test_distractor_shift'):
        for horizon in (3, 6):
            group = [s for s in diag['slices'] if s['task'] == 'motion' and s['condition'] == 'full_history' and s['split'] == split and s['horizon'] == horizon]
            assert len(group) == 3 and {s['seed'] for s in group} == {17, 29, 43} and all(s['pairs'] == 64 for s in group)
            r = {'split': split, 'horizon': horizon, 'pairs_per_seed': 64, 'condition': 'full_history'}
            for key in group[0]:
                if key not in ('task', 'seed', 'split', 'condition', 'pairs', 'horizon'):
                    r[key] = statistics.mean(s[key] for s in group)
            assert abs(r['common_MSE'] + r['quarter_pair_difference_MSE'] - r['pred_MSE']) < 1e-7
            r['common_error_fraction'] = r['common_MSE'] / r['pred_MSE']
            subset.append(r)
    with (out / 'historical_full_history_diagnosis.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(subset[0])); w.writeheader(); w.writerows(subset)

    settled = read(repo / 'results/age_control/native_completed/output/job_8275/summary.json')
    gpu_ledger = settled['GPU_native_ledger']
    spent = sum(gpu_ledger.values())
    assert spent == 12723
    probe = read(repo / 'results/predictor/training/job_7574/native_resource_probe.json')
    times = {}
    for task in ('gate', 'motion'):
        logs = [e for seed in (17, 29, 43) for e in read(repo / f'results/predictor/training/job_7574/predictor/{task}/seed_{seed}/training.json')['logs']]
        assert len(logs) == 360
        times[task] = {'mean': statistics.mean(e['epoch_seconds'] for e in logs), 'maximum': max(e['epoch_seconds'] for e in logs)}
    encoding = probe['confirmation_full_frame_bound'] / probe['encoder_frames_per_second']
    reference = 360 * sum(probe['reference_epoch_seconds'].values())
    candidate = 360 * sum(v['maximum'] for v in times.values())
    forecast = 1.15 * (encoding + reference + candidate) + 900 + 2 * probe['encoder_setup_and_actual_weight_proof_seconds']
    result = {
        'scope': 'Local reanalysis of revealed6301 results and historical64-pair full-history diagnosis; no new inference',
        'original': counts(comp['original_checks']), 'fixed_reference': counts(comp['fixed_reference_checks']),
        'split_components': {s: {k: sum(r[k + '_passed'] for r in rows if r['split'] == s) for k in ('feature', 'position')} for s in ('test_id', 'test_long_gap', 'test_distractor_shift')},
        'position_component_passed': sum(r['position_passed'] for r in rows),
        'feature_component_passed': sum(r['feature_passed'] for r in rows),
        'critical_seed_mean_failures': [x for x in comp['fixed_reference_checks'] if x['kind'] == 'critical_absolute_mse_reduction' and 'all_seeds_mean' in x['slice']],
        'diagnostic_scope': 'First64 pairs per split, three seeds, full_history only; not full-test acceptance and no fresh evidence',
        'GPU_budget': {'cap_seconds': 28800, 'native_settled_by_job': gpu_ledger, 'settled_seconds': spent, 'remaining_seconds': 28800 - spent, 'new_seconds_this_review': 0},
        'provisional_confirmation_forecast': {'encoding_seconds': encoding, 'reference_training_seconds': reference, 'candidate_training_seconds': candidate, 'margin': 1.15, 'fixed_allowance_seconds': 900, 'encoder_setup_twice_seconds': 2 * probe['encoder_setup_and_actual_weight_proof_seconds'], 'total_seconds': forecast, 'residual_after_reservation_seconds': 28800 - spent - forecast, 'qualification': 'Unvalidated projection using current architecture; excludes queue and CPU fitting/evaluation wall. Remeasure for any new Motion candidate. Not an executed confirmation or funding guarantee.'},
        'confirmation_eligible': False, 'fresh9137_opened_this_review': False, 'Phase1_GO': False,
        'integrity': 'Retained raw-digest identity matches independent audit; historical diagnostic SHA matches native triple-SHA receipt. No raw model inference replay this turn.'
    }
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'input_hashes.json').write_text(json.dumps(ledger, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout='constrained')
    labels = ['Ordinary', 'Long gap', 'Distractor shift']
    splits = ['test_id', 'test_long_gap', 'test_distractor_shift']
    ax = axes[0]
    for offset, key, color, label in [(-.18, 'feature', '.2', 'Full visual-feature benefit'), (.18, 'position', '.8', 'Position benefit')]:
        values = [result['split_components'][s][key] for s in splits]
        bars = ax.bar([i + offset for i in range(3)], values, .34, color=color, edgecolor='black', label=label)
        ax.bar_label(bars, labels=[f'{v}/8' for v in values], padding=3)
    ax.set(xticks=range(3), xticklabels=labels, ylim=(0, 9.5), yticks=[0, 2, 4, 6, 8], ylabel='Original component checks met', title='A  What still fails? (all 128 pairs)')
    ax.legend(fontsize=8, loc='upper right')
    ax = axes[1]
    selected = [next(x for x in subset if x['split'] == s and x['horizon'] == 6) for s in ['validation', 'test_long_gap', 'test_distractor_shift']]
    for offset, key, color, label in [(-.18, 'pred_MSE', '.2', 'Current Motion'), (.18, 'frame_only_MSE', '.8', 'Transport off')]:
        values = [r[key] for r in selected]
        bars = ax.bar([i + offset for i in range(3)], values, .34, color=color, edgecolor='black', label=label)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
    ax.set(xticks=range(3), xticklabels=['Validation', 'Long gap', 'Distractor shift'], ylim=(0, .37), ylabel='Visual-feature MSE (lower is better)', title='B  Gate-style switch is not a general fix')
    ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('Motion: useful historical information, incomplete shifted-condition prediction', fontsize=12)
    fig.savefig(out / 'motion_gap.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'feature': result['feature_component_passed'], 'position': result['position_component_passed'], 'remaining_GPU_seconds': 28800 - spent, 'confirmation_forecast_seconds': forecast, 'output': str(out)}))


if __name__ == '__main__':
    main()
