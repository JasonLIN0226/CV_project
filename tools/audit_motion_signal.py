"""Independent compact-row reconstruction of Motion scalar diagnosis."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    repo = Path(__file__).resolve().parents[1]
    folder = repo / 'results/motion_signal_diagnosis'
    source = repo / 'results/motion_locked_test/native_completed'
    result = json.loads((folder / 'summary.json').read_text())
    protocol = json.loads((folder / 'protocol.json').read_text())
    for filename, digest in protocol['inputs'].items():
        assert hashlib.sha256((source / filename).read_bytes()).hexdigest() == digest
    with (source / 'output/job_8349/paired_deletions.csv').open() as f:
        raw = list(csv.DictReader(f))
    # Reconstruct every pair from filtered records, independently of the primary index.
    seeds = [17, 29, 43]
    maxima, interval_count = [], 0
    draws = np.random.default_rng(0).integers(0, 128, (10000, 128))
    counts = np.stack([np.bincount(row, minlength=128) for row in draws]).astype(float)/128

    def check_interval(values, wanted):
        nonlocal interval_count
        distribution = counts @ values
        low, high = np.percentile(distribution, [2.5, 97.5])
        got = [sum(values)/128, low, high]
        diff = max(abs(float(a)-wanted[k]) for a, k in zip(got, ['mean','low','high']))
        assert diff < 2e-12
        maxima.append(diff)
        interval_count += 1

    def arrays(split, horizon, policy, metric):
        group = [r for r in raw if r['split'] == split and int(r['horizon']) == horizon]
        keys = sorted({r['pair'] for r in group})
        assert len(keys) == 128 and len(group) == 384
        return np.array([[float(next(r for r in group if r['pair']==key and int(r['seed'])==seed)[policy+'_'+metric])
                          for key in keys] for seed in seeds])

    for entry in result['transfer_regret']:
        for metric in ['feature_mse','task_error']:
            same = arrays(entry['split'], entry['horizon'], 'oracle_same', metric).sum(0)/3
            other = arrays(entry['split'], entry['horizon'], 'oracle_transfer', metric).sum(0)/3
            check_interval(other-same, entry[metric]['transfer_minus_same'])
            assert abs(float(same.mean())-entry[metric]['same_horizon_mean']) < 1e-12
            assert abs(float(other.mean())-entry[metric]['transferred_mean']) < 1e-12
            if metric == 'feature_mse':
                uniform = arrays(entry['split'], entry['horizon'], 'uniform', metric).sum(0)/3
                ratio = np.sum(other-same)/np.sum(uniform-same)
                assert abs(ratio-entry[metric]['fraction_of_headroom_lost']) < 1e-12
    count_cells = 0
    for entry in result['paired_baseline_comparisons']:
        g = []
        for metric in ['feature_mse','task_error']:
            a = arrays(entry['split'], entry['horizon'], entry['baseline'], metric)
            b = arrays(entry['split'], entry['horizon'], 'oracle_transfer', metric)
            g.append(a-b)
            check_interval((a-b).sum(0)/3, entry[metric]['paired_gain'])
        feature, task = g
        by_pair = []
        for j in range(128):
            f, t = sum(feature[:,j])/3, sum(task[:,j])/3
            robust = all(feature[i,j]>1e-12 and task[i,j]>1e-12 for i in range(3))
            by_pair.append((f>1e-12 and t>1e-12, robust, f>1e-12 and t < -1e-12,
                            f < -1e-12 and t>1e-12, abs(f)<=1e-12 or abs(t)<=1e-12))
        counts_here = [sum(x[i] for x in by_pair) for i in range(5)]
        fields = ['seed_mean_both_positive','all_seeds_both_positive','seed_mean_feature_positive_task_negative',
                  'seed_mean_feature_negative_task_positive','seed_mean_either_metric_tied']
        assert counts_here == [entry['signs'][key] for key in fields]
        for i, row in enumerate(entry['signs']['by_seed']):
            assert row['seed'] == seeds[i]
            assert row['both_positive'] == sum(feature[i,j]>1e-12 and task[i,j]>1e-12 for j in range(128))
            assert row['feature_positive_task_negative'] == sum(feature[i,j]>1e-12 and task[i,j]<-1e-12 for j in range(128))
        count_cells += 1
    components=json.loads((source/'output/job_8349/prediction_components.json').read_text())
    assert result['prediction_components']==components['cells']
    assert sum(x['normalized_mse']['passed'] for x in components['cells'])==13
    assert sum(x['state_position_mse']['passed'] for x in components['cells'])==24
    # Adversarial software fixtures exercise the primary summaries, not scientific evidence.
    from diagnose_motion_signal import sign_counts
    assert sign_counts(np.ones((3,2)), -np.ones((3,2)))['seed_mean_feature_positive_task_negative']==2
    hidden = np.array([[2.,2.],[2.,2.],[-1.,-1.]])
    v=sign_counts(hidden, hidden)
    assert v['seed_mean_both_positive']==2 and v['all_seeds_both_positive']==0
    assert sign_counts(np.zeros((3,2)), np.ones((3,2)))['seed_mean_either_metric_tied']==2
    audit=dict(status='PASS_LOCAL_INDEPENDENT_RECONSTRUCTION', intervals=interval_count,
               sign_cells=count_cells, max_difference=max(maxima), software_fixtures=3,
               scientific_acceptance_changed=False, GPU_added=0)
    assert interval_count==48 and count_cells==18
    (folder/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    names=['protocol.json','summary.json','prediction_and_pair_stability.png','independent_audit.json']
    ledger={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}
    for name in ['tools/diagnose_motion_signal.py','tools/audit_motion_signal.py','docs/MOTION_SIGNAL_DIAGNOSIS.md']:
        ledger[name]=hashlib.sha256((repo/name).read_bytes()).hexdigest()
    (folder/'local_artifact_ledger.json').write_text(json.dumps(dict(scope='Local derived artifacts; not a new native transfer',sha256=ledger),indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':
    main()
