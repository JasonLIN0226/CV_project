"""Validate/export existing factorial predictions without new model calls."""
from itertools import permutations, product
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT), str(ROOT/'experiments/predictor/control/src')]
import numpy as np
from experiments.phase2.portable_snapshot import PortableSnapshot
from experiments.phase2.run_fresh_literature_diagnostic import FreshGateDataset
from experiments.phase2.diagnose_gate_cue_factors import factor_effects, grouped

RAW = ROOT/'build/literature-diagnosis-20261022/cue-factors-v1'
OUT = ROOT/'results/literature_diagnosis/cue-factors-20261022'


def read(p): return json.loads(Path(p).read_text())
def save(p, value): Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read(RAW/'episode_rows.json'); summary = read(RAW/'summary.json')
    effects = read(RAW/'factor_effects.json'); matched = read(RAW/'correct_minus_wrong.json')
    with np.load(RAW/'predictions.npz', allow_pickle=False) as a:
        vectors, names = a['predictions'].copy(), a['case_names'].tolist()
    assert len(rows) == len(vectors) == len(names) == 864
    by_key = {}; originals = ROOT/'build/literature-diagnosis-20261022/reader-local-v2'
    with np.load(originals/'predictions.npz', allow_pickle=False) as a:
        old_vectors = {(r['episode_id'], r['keeps_cue']): p.copy()
                       for r, p in zip(read(originals/'episode_rows.json'), a['predictions']) if r['arm'] == 'original'}
    old = PortableSnapshot(ROOT/'build/hpc-20261020/development-assets')
    fresh = FreshGateDataset(ROOT/'build/literature-baselines-20261022/fresh-data')
    fresh.verify_old_normalization(old.norm)
    metric_error = 0.; endpoint_count = 0
    for row, vector, name in zip(rows, vectors, names):
        arm = row['content'], row['age'], row['past_action']
        assert name == f"{row['episode_id']}:{arm[0]}{arm[1]}{arm[2]}"
        key = (row['episode_id'], arm); assert key not in by_key; by_key[key] = row
        rec = fresh.by_id[row['episode_id']]; z, _ = fresh.episode(rec)
        sibling, _ = fresh.episode(fresh.by_id[rec['sibling']]); times = rec['query']+np.array([3, 6])
        target = z[times].astype(np.float64); sibling = sibling[times].astype(np.float64)
        error = vector.astype(np.float64)-target
        direction = (target-sibling)/2; unit = direction/np.linalg.norm(direction, axis=-1, keepdims=True)
        parallel = np.sum(error*unit, axis=-1, keepdims=True)*unit
        checks = {'mse': np.mean(error**2), 'contrast_mse': np.mean(parallel**2),
                  'orthogonal_mse': np.mean((error-parallel)**2),
                  'raw_feature_mse': np.mean((error*old.norm['std'])**2)}
        for k, value in checks.items():
            metric_error = max(metric_error, abs(float(value)-row[k])); assert abs(float(value)-row[k]) < 1e-12
        if arm in ((0, 0, 0), (1, 1, 1)):
            assert np.array_equal(vector, old_vectors[rec['id'], arm[0] == 1]); endpoint_count += 1
    allocation_error = 0.
    for e in effects:
        values = {arm: by_key[e['episode_id'], arm][e['metric']] for arm in product((0, 1), repeat=3)}
        marginal = np.zeros(3)
        for order in permutations(range(3)):
            point = [0, 0, 0]
            for i in order:
                before = values[tuple(point)]; point[i] = 1
                marginal[i] += (values[tuple(point)]-before)/6
        for i, name in enumerate(('content', 'age', 'past_action')):
            allocation_error = max(allocation_error, abs(e['endpoint_allocation_'+name]-marginal[i]))
            assert abs(e['endpoint_allocation_'+name]-marginal[i]) < 1e-12
            balanced = np.mean([values[tuple([1 if j == i else arm[j] for j in range(3)])]-
                                values[tuple([0 if j == i else arm[j] for j in range(3)])]
                                for arm in product((0, 1), repeat=3)])
            assert abs(e['main_'+name]-balanced) < 1e-12
    directions = {}
    for cohort in ('all', 'original_bad'):
        part = [r for r in matched if r['age'] == r['past_action'] == 1 and (cohort == 'all' or r['original_bad'])]
        directions[cohort] = {'episodes': len(part), 'correct_better_mse': sum(r['mse'] < -1e-8 for r in part),
                              'correct_worse_mse': sum(r['mse'] > 1e-8 for r in part),
                              'correct_improves_branch_direction': sum(r['contrast_amplitude_mean'] > 0 for r in part),
                              'correct_worsens_branch_direction': sum(r['contrast_amplitude_mean'] < 0 for r in part)}
    reader_effects = []
    for episode in sorted({r['episode_id'] for r in rows}):
        base = by_key[episode, (0, 0, 0)]
        for metric in base['reader']:
            measure = factor_effects({arm: by_key[episode, arm]['reader'][metric] for arm in product((0, 1), repeat=3)})
            reader_effects.append({k: base[k] for k in ('episode_id', 'pair', 'condition', 'original_bad')} |
                                  {'reader_metric': metric, **measure})
    save(RAW/'reader_path_effects.json', reader_effects)
    case_cards = {}
    for title, chosen in read(RAW/'representatives.json').items():
        case_cards[title] = chosen | {'arms': [r for r in rows if r['episode_id'] == chosen['episode_id']]}
    save(RAW/'case_cards.json', case_cards)
    audit = {'status': 'passed', 'new_model_calls': 0, 'prediction_rows': len(rows),
             'endpoint_vectors_bitwise_equal': endpoint_count, 'max_metric_discrepancy': metric_error,
             'factor_metrics_audited': len(effects), 'max_six_order_allocation_discrepancy': allocation_error,
             'fixed_cue_age_action_direction_counts': directions, 'source_sha256': summary['source_sha256']}
    save(RAW/'independent_audit.json', audit)
    for name in ('protocol.json', 'summary.json', 'episode_rows.json', 'factor_effects.json',
                 'correct_minus_wrong.json', 'cases.json', 'representatives.json', 'case_cards.json',
                 'reader_path_effects.json', 'independent_audit.json'):
        shutil.copy2(RAW/name, OUT/name)
    shutil.copytree(RAW/'source', OUT/'source', dirs_exist_ok=True)
    shutil.copy2(__file__, OUT/'source/export_cue_factors.py')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for i, (key, label) in enumerate(zip(('content', 'age', 'past_action'), ('Cue content', 'Age', 'Past-action suffix'))):
        value = summary['factor_effects']['original_bad']['mse']['endpoint_allocation_'+key]['all']
        axes[0].errorbar(value['mean'], i, xerr=[[value['mean']-value['ci95'][0]], [value['ci95'][1]-value['mean']]],
                         fmt='o', capsize=4, color='#35618d')
    axes[0].set_yticks(range(3), ['Cue content', 'Age', 'Past-action suffix']); axes[0].invert_yaxis()
    axes[0].set_title('A  Original 27 failures: endpoint allocation', loc='left', fontsize=11)
    axes[0].set_xlabel('Contribution to cue−noncue MSE gap\nSix-order average; scene-pair 95% CI')
    for i, cohort in enumerate(('all', 'original_bad')):
        value = summary['correct_minus_wrong'][cohort]['age1_action1']['mse']['all']
        axes[1].errorbar(value['mean'], i, xerr=[[value['mean']-value['ci95'][0]], [value['ci95'][1]-value['mean']]],
                         fmt='o', capsize=4, color='#35618d' if i == 0 else '#d07a30')
    axes[1].set_yticks(range(2), ['All 72', 'Original 27 failures']); axes[1].invert_yaxis()
    axes[1].set_title('B  Matched content: correct vs wrong cue', loc='left', fontsize=11)
    axes[1].set_xlabel('Correct−wrong cue MSE\nIdentical cue age/actions; scene-pair 95% CI')
    for ax in axes:
        ax.axvline(0, color='0.6', lw=1); ax.spines[['top', 'right']].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    (OUT/'figures').mkdir(exist_ok=True)
    fig.savefig(OUT/'figures/cue_factors.png', dpi=180); fig.savefig(OUT/'figures/cue_factors.pdf'); plt.close(fig)
    manifest = {'status': 'completed_controlled_factorial_diagnostic', 'no_new_predictions_during_export': True,
                'excluded_large_artifacts': {'predictions.npz': {'path': str((RAW/'predictions.npz').relative_to(ROOT)),
                    'sha256': sha(RAW/'predictions.npz'), 'bytes': (RAW/'predictions.npz').stat().st_size}},
                'files': {str(p.relative_to(OUT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                          for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json'}}
    save(OUT/'manifest.json', manifest)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__': main()
