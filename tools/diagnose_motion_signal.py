"""Read-only scalar diagnosis of the locked Motion result. No model imports."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'results/motion_locked_test/native_completed'
OUTPUT = REPO / 'results/motion_signal_diagnosis'
SPLITS = ('test_id', 'test_long_gap', 'test_distractor_shift')
SEEDS = (17, 29, 43)
BASELINES = ('uniform', 'fifo', 'reject')
METRICS = ('feature_mse', 'task_error')
TOL = 1e-12


def interval(values):
    """Paired-scene percentile interval; models/seeds are never bootstrap units."""
    values = np.asarray(values, dtype=float)
    assert values.shape == (128,) and np.isfinite(values).all()
    ix = np.random.default_rng(0).integers(0, 128, (10000, 128))
    low, high = np.quantile(values[ix].mean(axis=1), [.025, .975])
    return dict(mean=float(values.mean()), low=float(low), high=float(high))


def sign_counts(feature_gain, task_gain):
    """Input axes: seed, pair. Counts concern pool-averaged outcomes only."""
    feature_gain, task_gain = map(np.asarray, (feature_gain, task_gain))
    assert feature_gain.shape == task_gain.shape and feature_gain.ndim == 2
    f, t = feature_gain.mean(0), task_gain.mean(0)
    return dict(
        pairs=len(f),
        seed_mean_both_positive=int(((f > TOL) & (t > TOL)).sum()),
        all_seeds_both_positive=int(((feature_gain > TOL) & (task_gain > TOL)).all(0).sum()),
        seed_mean_feature_positive_task_negative=int(((f > TOL) & (t < -TOL)).sum()),
        seed_mean_feature_negative_task_positive=int(((f < -TOL) & (t > TOL)).sum()),
        seed_mean_either_metric_tied=int(((abs(f) <= TOL) | (abs(t) <= TOL)).sum()),
        by_seed=[dict(seed=seed, both_positive=int(((ff > TOL) & (tt > TOL)).sum()),
                      feature_positive_task_negative=int(((ff > TOL) & (tt < -TOL)).sum()))
                 for seed, ff, tt in zip(SEEDS, feature_gain, task_gain)],
    )


def load_verified():
    protocol = json.loads((OUTPUT / 'protocol.json').read_text())
    for name, wanted in protocol['inputs'].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == wanted, name
    transfer = json.loads((SOURCE / 'transfer_integrity.json').read_text())
    assert transfer['native_before_local_native_after_all_match']
    for name, record in transfer['files'].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == record['sha256'], name
    audit = json.loads((SOURCE / 'output/job_8349/independent_audit.json').read_text())
    assert audit['status'] == 'PASS_INDEPENDENT_LOCKED_MOTION_AUDIT'
    with (SOURCE / 'output/job_8349/paired_deletions.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2304
    index = {}
    for row in rows:
        assert row['task'] == 'motion' and row['pool_group'] == 'clock_all'
        key = row['split'], int(row['horizon']), int(row['seed']), row['pair']
        assert key not in index
        index[key] = row
        for policy in (*BASELINES, 'oracle_same', 'oracle_transfer'):
            for metric in METRICS:
                assert np.isfinite(float(row[f'{policy}_{metric}']))
        assert float(row['oracle_transfer_feature_mse']) >= float(row['oracle_same_feature_mse']) - TOL
    cells = []
    for split in SPLITS:
        pair_sets = [{p for s, h, seed, p in index if (s, h, seed) == (split, hh, ss)}
                     for hh in (3, 6) for ss in SEEDS]
        assert all(x == pair_sets[0] and len(x) == 128 for x in pair_sets)
        for h in (3, 6):
            ordered = [[index[split, h, seed, p] for p in sorted(pair_sets[0])] for seed in SEEDS]
            values = {f'{policy}_{metric}': np.array([[float(r[f'{policy}_{metric}']) for r in rr] for rr in ordered])
                      for policy in (*BASELINES, 'oracle_same', 'oracle_transfer') for metric in METRICS}
            cells.append((split, h, values))
    return cells


def main():
    cells = load_verified()
    prediction = json.loads((SOURCE / 'output/job_8349/prediction_components.json').read_text())
    summary = json.loads((SOURCE / 'output/job_8349/summary.json').read_text())
    comparisons, transfer = [], []
    for split, h, v in cells:
        regret = dict(split=split, horizon=h, selection_horizon=9-h)
        for metric in METRICS:
            same = v[f'oracle_same_{metric}'].mean(0)
            other = v[f'oracle_transfer_{metric}'].mean(0)
            item = dict(same_horizon_mean=float(same.mean()), transferred_mean=float(other.mean()),
                        transfer_minus_same=interval(other-same))
            if metric == 'feature_mse':
                headroom = float((v[f'uniform_{metric}'].mean(0)-same).mean())
                item['uniform_to_same_headroom'] = headroom
                item['fraction_of_headroom_lost'] = float((other-same).mean()/headroom) if headroom > TOL else None
            regret[metric] = item
        transfer.append(regret)
        for baseline in BASELINES:
            gains = {m: v[f'{baseline}_{m}']-v[f'oracle_transfer_{m}'] for m in METRICS}
            entry = dict(split=split, horizon=h, baseline=baseline,
                         signs=sign_counts(gains['feature_mse'], gains['task_error']))
            for metric in METRICS:
                entry[metric] = dict(baseline_mean=float(v[f'{baseline}_{metric}'].mean()),
                                    transferred_mean=float(v[f'oracle_transfer_{metric}'].mean()),
                                    paired_gain=interval(gains[metric].mean(0)))
            comparisons.append(entry)
    long_gap = [x for x in summary['prediction']['fixed_reference_checks']
                if x['kind'] == 'critical_absolute_mse_reduction' and x['slice'].endswith('seedall_seeds_mean')]
    result = dict(status='LOCAL_SCALAR_DIAGNOSIS_COMPLETE', protocol_sha256=hashlib.sha256((OUTPUT/'protocol.json').read_bytes()).hexdigest(),
                  source_verified_compact_files=20, input_rows=2304,
                  prediction_components=prediction['cells'], fixed_long_gap_seed_mean=long_gap,
                  transfer_regret=transfer, paired_baseline_comparisons=comparisons,
                  original_prediction='40/51', original_fixed_reference='13/32', original_critical='0/8',
                  original_Phase1_GO=False, fresh9137_opened=False, candidate_changed=False,
                  GPU_added=0, CPU_allocation_added=0, training=False, policy=False)
    (OUTPUT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    plot(result)
    report(result)
    print(json.dumps(dict(status=result['status'], rows=2304, comparisons=len(comparisons), regret_cells=len(transfer), GPU_added=0)))


def plot(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Ordinary\nh3', 'Ordinary\nh6', 'Long gap\nh3', 'Long gap\nh6', 'Distractors\nh3', 'Distractors\nh6']
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), layout='constrained')
    seed_mean = [x for x in r['prediction_components'] if x['seed'] == 'all_seeds_mean']
    lookup = {(x['split'], x['horizon']): x for x in seed_mean}
    for i, (metric, label, color) in enumerate([('normalized_mse','Full features','.65'), ('state_position_mse','Position readout','.1')]):
        values = [lookup[s,h][metric] for s in SPLITS for h in (3,6)]
        x = np.arange(6)+(i-.5)*.36
        axes[0].bar(x, [v['mean']*100 for v in values], .36, label=label, color=color, edgecolor='black')
        axes[0].errorbar(x, [v['mean']*100 for v in values],
                        yerr=np.array([[v['mean']-v['low'], v['high']-v['mean']] for v in values]).T*100,
                        fmt='none', color='black', capsize=2)
    axes[0].axhline(10, ls='--', color='black', lw=1)
    axes[0].set(xticks=np.arange(6), xticklabels=labels, ylabel='Error reduction vs recent only (%)', title='A  Position recovery is stronger than feature recovery')
    axes[0].legend(fontsize=8)
    matrix = np.array([[x['signs']['all_seeds_both_positive'] for x in r['paired_baseline_comparisons'] if x['baseline']==b] for b in BASELINES])
    axes[1].imshow(matrix, cmap='Greys', vmin=0, vmax=128, aspect='auto')
    for row in range(3):
        for col in range(6):
            v=matrix[row,col]; axes[1].text(col,row,f'{v}/128',ha='center',va='center',color='white' if v>80 else 'black',fontsize=9)
    axes[1].set(xticks=np.arange(6), xticklabels=labels, yticks=range(3), yticklabels=['Uniform','FIFO','Reject newest'], title='B  Pairs with both errors reduced in ALL three seeds')
    axes[1].set_xlabel('Future-informed transfer choice vs baseline; pool-averaged pairs')
    for ax in axes: ax.tick_params(axis='x',labelsize=8)
    fig.savefig(OUTPUT/'prediction_and_pair_stability.png',dpi=180);plt.close(fig)


def report(r):
    name = dict(test_id='Ordinary',test_long_gap='Long gap',test_distractor_shift='Distractors')
    lines=['# Motion: prediction quality versus deletion-score consistency', '',
           '## Question and scope', '',
           'Why can deletion choices show useful relative gains while the locked Motion candidate still fails the original requirements? This diagnosis reuses audited compact results only. It makes no predictor, encoder, fitting or policy calls and uses no HPC allocation or additional GPU. The analysis protocol was saved before computing these new summaries.', '',
           'There are two separate questions: (1) Does the predictor meet the original absolute and relative prediction requirements? (2) When choosing among equal-capacity deletions, do lower future-feature errors also give lower position errors across horizons and seeds? Success in the second question cannot replace the first.', '',
           '## 1. The measured prediction deficit', '',
           'Original registered coverage remains **40/51 prediction checks**, **13/32 fixed-reference checks**, and **0/8 critical long-gap checks**. Position benefit meets **24/24 components**, while complete-feature benefit meets **13/24**. These counts share data and are not independent successes. The position readout is only one aspect of a 5,120-dimensional visual feature vector; this comparison does not identify which other visual factors cause the remaining error.', '',
           '![Prediction and paired stability](../results/motion_signal_diagnosis/prediction_and_pair_stability.png)', '',
           '**Figure.** A: seed-mean error reduction against recent-only prediction; whiskers are the already audited paired 95% intervals. The dashed line is the original 10% effect requirement; passing also requires the original confidence rule. B: for each baseline and test condition, a pair counts only if the transferred choice reduces BOTH errors in ALL three seeds. Each pair value averages two sibling episodes and six pools. This is a stability description, not a new pass rule or individual-deletion accuracy.', '',
           '| Condition | Horizon | Full-feature reduction vs recent | Position reduction vs recent |', '|---|---:|---:|---:|']
    for s in SPLITS:
        for h in (3,6):
            x=next(x for x in r['prediction_components'] if x['split']==s and x['horizon']==h and x['seed']=='all_seeds_mean')
            lines.append(f"|{name[s]}|{h}|{x['normalized_mse']['mean']*100:.2f}%|{x['state_position_mse']['mean']*100:.2f}%|")
    lines += ['', '| Long-gap horizon | Fixed same-capacity reference MSE | Candidate MSE | Absolute reduction, 95% CI |', '|---|---:|---:|---:|']
    for x in r['fixed_long_gap_seed_mean']:
        v=x['absolute_mse_reduction'];lines.append(f"|{x['slice'].split('/')[2]}|{x['reference_privileged_mse']:.6f}|{x['candidate_privileged_mse']:.6f}|{v['mean']:.6f} [{v['low']:.6f}, {v['high']:.6f}]|")
    lines += ['', 'Negative reductions mean that the candidate has MORE complete-feature error than the unchanged original same-capacity reference. These are not comparisons against recent-only or the previous repair.', '',
              '## 2. What transfers across prediction horizons?', '',
              'Both choices use future targets: the same-horizon oracle minimizes feature error at the evaluated horizon; the transfer oracle selects at the other horizon. Their difference measures the cost of transferring this future-informed choice. Neither is a deployable past-only policy. The feature regret is nonnegative by construction; position regret may have either sign because feature minimization is not position minimization.', '',
              '| Condition | Evaluated horizon | Feature regret (transfer minus same) | Uniform-oracle headroom lost | Position difference, 95% CI |', '|---|---:|---:|---:|---:|']
    for x in r['transfer_regret']:
        f=x['feature_mse'];v=x['task_error']['transfer_minus_same'];g=f['transfer_minus_same'];ratio=f['fraction_of_headroom_lost']
        lines.append(f"|{name[x['split']]}|{x['horizon']}|{g['mean']:.6f} [{g['low']:.6f}, {g['high']:.6f}]|{100*ratio:.2f}%|{v['mean']:.6f} [{v['low']:.6f}, {v['high']:.6f}]|")
    lines += ['', 'The headroom fraction divides mean transfer regret by mean uniform-deletion error minus mean same-horizon-oracle error. It describes this future-informed benchmark; it is not an achievable learned-policy estimate. Here transfer loses only 2.03–2.93% of that feature headroom. This makes horizon transfer a relatively small measured feature-cost penalty in these averages; it does not establish the cause of the separate absolute prediction failure.', '',
              '## 3. Aggregate benefit does not make every pair reliable', '',
              '| Condition | Horizon | Baseline | Both gains positive, seed mean | Both gains positive in all seeds | Feature gain positive but position gain negative, seed mean |', '|---|---:|---|---:|---:|---:|']
    for x in r['paired_baseline_comparisons']:
        s=x['signs'];lines.append(f"|{name[x['split']]}|{x['horizon']}|{x['baseline']}|{s['seed_mean_both_positive']}/128|{s['all_seeds_both_positive']}/128|{s['seed_mean_feature_positive_task_negative']}/128|")
    lines += ['', 'The previously registered aggregate cross-horizon result remains **71/72**. It includes individual seeds and seed means of the same scenes. A positive aggregate confidence interval is compatible with negative or seed-sensitive pair outcomes. Against reject-newest, 56–69/128 pairs per condition/horizon have lower feature error but higher position error after seed averaging. Only 37–58/128 pairs reduce both errors in every seed. These are real pair-average disagreements, not proof that every fork ranking is wrong. The table counts strict signs with a numerical tie tolerance of 1e-12; no new scientific acceptance threshold is introduced.', '',
              'All new intervals use 10,000 bootstrap draws, seed 0, resampling matched scene pairs after averaging the three seeds within each pair. They are descriptive 95% percentile intervals without multiplicity adjustment. Each split has 128 pairs; no pooled independent-dataset claim is made.', '',
              '## 4. What this can and cannot establish', '',
              '- **Established:** Motion has a measured full-feature prediction deficit despite strong position improvement. The original long-gap reference requirements still fail.',
              '- **Established:** the future-informed transferred deletion choice has useful average gains, but its pair-level effects need not agree across metrics and seeds.',
              '- **Not identified:** whether a particular feature component, background, readout null space or model branch causes the residual error. That requires complete vector witnesses or a controlled intervention.',
              '- **Not identified:** all-deletion rank correlation, task-optimal regret, exact selected-fork agreement or past-only learnability. Compact rows average six pools and two siblings; they cannot reconstruct those quantities.',
              '- **Scope:** revealed synthetic development data, a simulator-specific RGB guard and unmatched auxiliary RGB validity information. These are not independent confirmation or a feature-only memory-byte comparison.', '',
              '## Conclusion and next bounded experiment', '',
              '**Overall Phase1_GO remains unestablished.** Do not change the candidate, thresholds, failed cases or confirmation boundary on the strength of these summaries. Do not train a retention policy from a claim that this stage passed.', '',
              'If continuing the diagnosis, the smallest useful addition is a read-only analysis of existing HPC fork records: compare feature-error and position-error orderings WITHIN each original triple, compute the task-optimal lower bound, and inspect all tied/zero-signal pools. This needs no predictor retraining or new encoder. It would separate score alignment from transfer and seed averaging more directly. A separately frozen vector-residual analysis is required before attributing the absolute prediction deficit to a feature subspace. Neither analysis should change a model or open fresh9137.', '',
              '## Reproduction and verification', '',
              'Run `python tools/diagnose_motion_signal.py` and `python tools/audit_motion_signal.py`. Input SHA checks bind this diagnosis to all 20 previously transferred compact artifacts. A separate implementation reconstructs 48 paired confidence intervals using bootstrap count weights and checks 18 sign-count cells using scalar pair loops. Three adversarial software fixtures check feature/task disagreement, seed-mean masking and numerical ties. The figure was rendered and visually inspected. The local artifact ledger is a derived-file checksum record, not a new native transfer proof.', '',
              'The present diagnosis adds **0 GPU seconds**, **0 HPC allocations**, no model calls, no policy training and no GitHub update. Cumulative GPU use stays 12,723/28,800 seconds, with 16,077 seconds remaining.']
    (REPO/'docs/MOTION_SIGNAL_DIAGNOSIS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
