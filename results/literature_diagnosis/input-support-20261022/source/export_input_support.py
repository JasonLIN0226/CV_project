"""Audit and publish saved input-support diagnostics; no new model calls."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments/predictor/control/src')]
import numpy as np
from experiments.phase2.portable_snapshot import PortableSnapshot
from experiments.phase2.run_fresh_literature_diagnostic import FreshGateDataset

RAW = ROOT / 'build/literature-diagnosis-20261022/input-support-v2'
OUT = ROOT / 'results/literature_diagnosis/input-support-20261022'


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read(RAW / 'episode_rows.json')
    with np.load(RAW / 'predictions.npz', allow_pickle=False) as a:
        pred, names = a['predictions'].copy(), a['case_names'].tolist()
    old = PortableSnapshot(ROOT / 'build/hpc-20261020/development-assets')
    fresh = FreshGateDataset(ROOT / 'build/literature-baselines-20261022/fresh-data')
    fresh.verify_old_normalization(old.norm)
    gm = ROOT / 'build/literature-diagnosis-20261022/metric-reader-v2'
    lm = ROOT / 'build/literature-diagnosis-20261022/reader-local-v2'
    with np.load(gm / 'predictions.npz', allow_pickle=False) as a:
        old_global = dict(zip(a['case_names'].tolist(), a['predictions']))
    with np.load(lm / 'predictions.npz', allow_pickle=False) as a:
        old_local = {(r['episode_id'], r['arm'], r['keeps_cue']): p for r, p in
                     zip(read(lm / 'episode_rows.json'), a['predictions'])}
    baseline = {r['episode_id']: r for r in read(ROOT / 'build/literature-baselines-20261022/fresh-eval/episode_results.json')
                if r['policy'] == 'attention'}
    scalar_error = 0.; original_vectors = 0; attention_replays = 0
    assert len(rows) == len(pred) == len(names) == 1440
    for r, p, name in zip(rows, pred, names):
        assert name == f"{r['episode_id']}:{r['family']}:{r['arm']}"
        rec = fresh.by_id[r['episode_id']]; z, _ = fresh.episode(rec)
        sibling, _ = fresh.episode(fresh.by_id[rec['sibling']])
        times = rec['query'] + np.array([3, 6])
        target = z[times].astype(np.float64); sibling = sibling[times].astype(np.float64)
        p = p.astype(np.float64)
        mse = float(np.mean((p-target)**2))
        midpoint = (target+sibling)/2
        half = (target-sibling)/2
        unit = half / np.linalg.norm(half, axis=-1, keepdims=True)
        component = np.sum((p-target)*unit, axis=-1, keepdims=True)*unit
        along = float(np.mean(component**2))
        remainder = float(np.mean((p-target-component)**2))
        raw_mse = float(np.mean(((p-target)*old.norm['std'])**2))
        for measured, saved in [(mse, r['mse']), (along, r['contrast_mse']),
                                 (remainder, r['orthogonal_mse']), (raw_mse, r['raw_feature_mse'])]:
            scalar_error = max(scalar_error, abs(measured-saved))
            assert abs(measured-saved) < 1e-12
        if r['arm'] == 'original':
            if r['family'].startswith('global_'):
                key = f"{r['episode_id']}:original:{int(r['family']=='global_cue')}"
                assert np.array_equal(p, old_global[key]); original_vectors += 1
            elif r['family'].startswith('local_'):
                assert np.array_equal(p, old_local[r['episode_id'], 'original', r['family']=='local_cue'])
                original_vectors += 1
            elif r['family'] == 'attention':
                assert r['kept_times'] == baseline[r['episode_id']]['kept_times']
                assert abs(mse-baseline[r['episode_id']]['mean_mse']) < 1e-12
                attention_replays += 1
    prior_probe = {(r['id'], r['arm']): r['score'] for r in read(lm / 'cue_probe.json')}
    probe = read(RAW / 'cue_probe.json')
    assert all(r['score'] == prior_probe[r['id'], r['probe']] for r in probe if r['arm'] == 'original')
    support = read(RAW / 'training_support.json')
    with np.load(RAW / 'training_support.npz', allow_pickle=False) as a:
        assert np.array_equal(a['unsupported'], a['second_moment'] < 1e-4)
        assert np.flatnonzero(a['unsupported']).tolist() == support['unsupported_indices']
    source_hash = sha(ROOT / 'experiments/phase2/diagnose_gate_input_support.py')
    summary = read(RAW / 'summary.json')
    assert source_hash == summary['source_sha256']
    gaps = read(RAW / 'cue_gaps.json')
    cohort_stats = {}
    for scope in ('global', 'local'):
        original = {r['episode_id']: r for r in gaps if r['scope'] == scope and r['arm'] == 'original'}
        masked = {r['episode_id']: r for r in gaps if r['scope'] == scope and r['arm'] == 'train_support_mask'}
        bad = [k for k, r in original.items() if r['mse'] > 1e-8]
        cohort_stats[scope] = {'episodes': len(bad),
            'episode_weighted_original_mean': float(np.mean([original[k]['mse'] for k in bad])),
            'episode_weighted_masked_mean': float(np.mean([masked[k]['mse'] for k in bad])),
            'rescued_ids': [k for k in bad if masked[k]['mse'] <= 1e-8],
            'mean_convention': 'These are episode-weighted descriptive cohort means. Main summary CIs give equal weight to represented scene pairs; bad subsets can have one or two siblings per scene.'}
    audit = {'status': 'passed_saved_vector_and_arithmetic_audit', 'new_model_calls': 0,
             'prediction_rows': len(rows), 'original_global_local_vectors_bitwise_equal': original_vectors,
             'original_attention_sets_and_losses_verified': attention_replays,
             'prior_cue_probe_scores_exact': len(prior_probe), 'max_metric_discrepancy': scalar_error,
             'support_mask_matches_fixed_second_moment_rule': True, 'source_sha256': source_hash,
             'bad_cohort_episode_weighted_descriptions': cohort_stats}
    save(RAW / 'independent_audit.json', audit)
    for name in ['protocol.json', 'summary.json', 'training_support.json', 'training_support.npz',
                 'episode_rows.json', 'input_diagnostics.json', 'mask_minus_original.json',
                 'cue_gaps.json', 'cue_probe.json', 'cue_readout.npz', 'cue_minus_current_readout.npz',
                 'independent_audit.json']:
        shutil.copy2(RAW / name, OUT / name)
    shutil.copytree(RAW / 'source', OUT / 'source', dirs_exist_ok=True)
    shutil.copy2(__file__, OUT / 'source/export_input_support.py')
    failure = OUT / 'failed-first-runtime'
    failure.mkdir(exist_ok=True)
    for name in ('failure_receipt.json', 'protocol.json'):
        shutil.copy2(RAW.parent / 'input-support-v1' / name, failure / name)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4), constrained_layout=True)
    colors = ['#35618d', '#d07a30']
    conditions = ['id', 'long_gap', 'distractor_shift']; labels = ['ID', 'Long gap', 'Distractor shift']
    for i, condition in enumerate(conditions):
        v = summary['mask_minus_original']['attention']['mse'][condition]
        axes[0].errorbar(1000*v['mean'], i, xerr=[[1000*(v['mean']-v['ci95'][0])], [1000*(v['ci95'][1]-v['mean'])]],
                         fmt='o', capsize=4, color=colors[0])
    axes[0].axvline(0, color='0.6', lw=1)
    axes[0].set_yticks(range(3), labels); axes[0].invert_yaxis()
    axes[0].set_xlabel('Masked − original feature MSE (×10⁻³)\nScene-pair bootstrap 95% CI')
    axes[0].set_title('A  Frozen predictor: small MSE changes', loc='left', fontsize=11)
    for j, arm in enumerate(('original', 'train_support_mask')):
        v = [summary['cue_probe']['cue_minus_current'][arm][c]['accuracy']['mean']*100 for c in conditions]
        axes[1].bar(np.arange(3)+(j-.5)*.34, v, .34, color=colors[j],
                    label='Original probe' if j == 0 else 'Unsupported coefficients zeroed')
    axes[1].set_xticks(range(3), labels); axes[1].set_ylim(0, 109)
    axes[1].set_ylabel('Cue label accuracy (%)'); axes[1].set_title('B  Frozen cue−current probe: instability isolated', loc='left', fontsize=11)
    axes[1].legend(loc='lower right', fontsize=8)
    for ax in axes:
        ax.spines[['right', 'top']].set_visible(False)
    (OUT / 'figures').mkdir(exist_ok=True)
    fig.savefig(OUT / 'figures/input_support.png', dpi=180)
    fig.savefig(OUT / 'figures/input_support.pdf')
    plt.close(fig)
    manifest = {
        'status': 'completed_fixed_support_diagnostic_export', 'runtime': read(RAW / 'protocol.json')['runtime'],
        'no_new_predictions_during_export': True, 'excluded_large_artifacts': {
            'predictions.npz': {'path': str((RAW / 'predictions.npz').relative_to(ROOT)),
                                'sha256': sha(RAW / 'predictions.npz'), 'bytes': (RAW / 'predictions.npz').stat().st_size}},
        'files': {str(p.relative_to(OUT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                  for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json'}}
    save(OUT / 'manifest.json', manifest)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
