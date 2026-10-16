"""Recompute the 32 fixed Motion latent checks without project statistics code.

Historical mode checks software compatibility using already exposed v3 scalars.
Default mode refuses to read new v5 evaluation scalars until all six actual
checkpoint completion proofs exist. This is a root arithmetic audit, not a new
experiment, delegated review, or replacement for the original task rules.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_SHA = '2b86a173aa20b772a5c08ed9c19ac1edb952e13a4f980f37628f9027e4c0672d'
KEYS = ('seed', 'task', 'split', 'horizon', 'condition', 'pair', 'episode')
SEEDS = ('17', '29', '43', 'all_seeds_mean')
SPLITS = ('test_id', 'test_long_gap', 'test_distractor_shift')
KINDS = ('candidate_vs_reference_recent', 'critical_absolute_mse_reduction')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return tuple(row[k] for k in KEYS)


def read(path):
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    lookup = {key(row): row for row in rows}
    assert len(rows) == len(lookup) == 34560
    assert Counter(r['task'] for r in rows) == {'motion': 34560}
    assert Counter(r['split'] for r in rows) == {
        'validation': 6912, **{split: 9216 for split in SPLITS}}
    assert Counter(r['seed'] for r in rows) == {s: 11520 for s in SEEDS[:3]}
    assert Counter(r['horizon'] for r in rows) == {'3': 17280, '6': 17280}
    assert Counter(r['condition'] for r in rows) == {
        c: 5760 for c in ('recent_only', 'full_history', 'privileged_small_budget',
                          'random_small_budget', 'wrong_evidence', 'persistence')}
    assert all(np.isfinite(float(r['normalized_mse'])) for r in rows)
    return rows, lookup


def replay(raw, reference, checks):
    wanted = {
        (KINDS[0], f'motion/{split}/h{h}/seed{seed}')
        for split in SPLITS for h in (3, 6) for seed in SEEDS}
    wanted |= {(KINDS[1], f'motion/test_long_gap/h{h}/seed{seed}')
               for h in (3, 6) for seed in SEEDS}
    actual_keys = [(s['kind'], s['slice']) for s in checks]
    assert len(actual_keys) == len(set(actual_keys)) == 32
    assert set(actual_keys) == wanted, 'Missing, duplicate, or wrong critical split'
    results, maximum = [], 0.0
    for saved in checks:
        _, split, h, seed = saved['slice'].split('/')
        h, seed = h[1:], seed[4:]
        selected = [r for r in raw if r['split'] == split and r['horizon'] == h
                    and r['condition'] == 'privileged_small_budget'
                    and (seed == 'all_seeds_mean' or r['seed'] == seed)]
        assert len(selected) == 256 * (3 if seed == 'all_seeds_mean' else 1)
        grouped = {}
        for row in selected:
            k = key(row)
            recent = list(k)
            recent[4] = 'recent_only'
            grouped.setdefault((row['pair'], row['episode']), []).append([
                float(reference[k]['normalized_mse']), float(row['normalized_mse']),
                float(reference[tuple(recent)]['normalized_mse'])])
        episodes = sorted(grouped)
        assert len(episodes) == 256
        episode_values = np.array([np.mean(grouped[k], axis=0) for k in episodes])
        paired = {}
        for k, value in zip(episodes, episode_values):
            paired.setdefault(k[0], []).append(value)
        assert len(paired) == 128 and all(len(v) == 2 for v in paired.values())
        matrix = np.array([np.mean(v, axis=0) for v in paired.values()])
        draws = np.random.default_rng(0).integers(0, 128, (2000, 128))
        if saved['kind'] == KINDS[0]:
            base, alt = matrix[:, 2], matrix[:, 1]
            absolute = base.mean() < .02
            def effect(a, b):
                return a - b if absolute else (a - b) / np.maximum(a, .02)
            mean = float(effect(base.mean(), alt.mean()))
            boot = effect(base[draws].mean(1), alt[draws].mean(1))
            original = saved['improvement']
            passed = mean >= (.01 if absolute else .1) and np.quantile(boot, .025) > 0
            assert original['scale'] == ('absolute_error_reduction' if absolute else 'relative_error_reduction')
        else:
            values = matrix[:, 0] - matrix[:, 1]
            mean, boot = float(values.mean()), values[draws].mean(1)
            original = saved['absolute_mse_reduction']
            passed = mean > 0 and np.quantile(boot, .025) > 0
        result = {'mean': mean, 'low': float(np.quantile(boot, .025)),
                  'high': float(np.quantile(boot, .975))}
        difference = max(abs(result[k] - original[k]) for k in result)
        maximum = max(maximum, difference)
        assert difference < 1e-14 and bool(passed) == saved['passed']
        assert original['pairs'] == 128
        results.append({'slice': saved['slice'], 'kind': saved['kind'],
                        'passed': bool(passed), 'replayed': result,
                        'max_scalar_difference': difference})
    return results, maximum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--historical-software-check', action='store_true')
    args = parser.parse_args()
    refpath = ROOT/'runs/phase1_v3_dev/motion/reference/evaluation/raw_records.csv'
    assert sha(refpath) == REFERENCE_SHA
    if args.historical_software_check:
        base = ROOT/'runs/phase1_v3_dev/motion'
        comparison_path = base/'development_comparison.json'
        candidate_path = base/'motion_wide_span/evaluation/raw_records.csv'
        comparison = json.loads(comparison_path.read_text())
        assert comparison['status'] == 'REJECTED'
        assert sha(candidate_path) == comparison['raw_sha256']['candidate']
        checks = [x for x in comparison['checks'] if x['kind'] in KINDS]
        output = Path(__file__).with_name('historical_scalar_software_check.json')
    else:
        base = ROOT/'runs/phase1_v5_motion/evaluation_results'
        completion = json.loads((base/'all_models_completion.json').read_text())
        assert completion['complete'] and completion['required_epochs'] == 120
        assert set(completion['models']) == {'motion/'+s for s in SEEDS[:3]}
        for s in SEEDS[:3]:
            proof = completion['models']['motion/'+s]
            assert proof['complete'] and proof['completed_epochs'] == 120 and proof['final_step'] == 7680
            assert proof['signature']['task'] == 'motion' and proof['signature']['reader'] == 'SyntheticCausalReader'
            assert proof['signature']['source_lock_sha256'] == 'f230af5ce3b0f2bf85d88eeecca886db218970547975e34629b61eabc843896c'
            assert proof['issues'] == [] and len(proof['best_sha256']) == len(proof['latest_sha256']) == 64
        comparison_path = base/'development_comparison.json'
        comparison = json.loads(comparison_path.read_text())
        candidate_path = base/'evaluation/raw_records.csv'
        metrics = json.loads((base/'evaluation/metrics.json').read_text())
        assert sha(candidate_path) == metrics['raw_sha256']
        checks = comparison['fixed_reference_checks']
        output = base/'fixed_reference_scalar_replay.json'
    rr, reference = read(refpath)
    raw, candidate = read(candidate_path)
    assert reference.keys() == candidate.keys()
    results, maximum = replay(raw, reference, checks)
    proof = {
        'status': 'PASS_HISTORICAL_SCALAR_SOFTWARE_CHECK_NOT_NEW_V5_RESULTS'
                  if args.historical_software_check else 'PASS_SCALAR_REPLAY_NOT_NEW_SCIENCE',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'root_audit_not_delegated_review': True, 'historical_software_check': args.historical_software_check,
        'candidate_rows': len(raw), 'reference_rows': len(rr),
        'complete_unique_matching_keys': True, 'original_paired_base_pair_2000_resampling': True,
        'original_floors_and_thresholds': True, 'required_24_recent_and_8_long_gap_checks': True,
        'checks': results, 'max_scalar_difference': maximum,
        'candidate_raw_sha256': sha(candidate_path), 'reference_raw_sha256': sha(refpath),
        'comparison_sha256': sha(comparison_path), 'script_sha256': sha(__file__),
        'development_adoption': comparison['status'], 'phase1_go': False,
    }
    assert not output.exists(), 'Preserve the prior audit rather than overwrite it'
    output.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps({'status': proof['status'], 'replayed': len(results), 'max_difference': maximum}))


if __name__ == '__main__':
    main()
