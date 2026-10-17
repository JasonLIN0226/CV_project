#!/usr/bin/env python3
"""Post-hoc pair-level signal and baseline-headroom diagnosis; no model access."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/deletion'
SEEDS = (17, 29, 43)
SPLITS = ('test_id', 'test_long_gap', 'test_distractor_shift')
EPS = 1e-12  # Descriptive sign tolerance, not an acceptance threshold.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(successes, total):
    if not 0 <= successes <= total or total == 0:
        raise ValueError('Nonempty independent-pair denominator required')
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return [max(0., center-half), min(1., center+half)]


def sign_summary(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Finite independent-pair values required')
    positive = int(np.sum(a > EPS))
    tied = int(np.sum(np.abs(a) <= EPS))
    return dict(pairs=len(a), positive=positive, tied=tied,
                negative=len(a)-positive-tied, positive_fraction=positive/len(a),
                positive_fraction_wilson95=wilson(positive, len(a)))


def ranks(values):
    a = np.asarray(values, dtype=float)
    order = np.argsort(a, kind='stable')
    answer = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i+1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        answer[order[i:j]] = (i+j-1)/2
        i = j
    return answer


def spearman(a, b):
    x, y = ranks(a), ranks(b)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pair_cube(rows):
    """Align repeated seeds by pair ID; never treat seeds as independent pairs."""
    cells = {}
    seen = set()
    for r in rows:
        if r['kind'] != 'critical_vs_nuisance':
            continue
        key = (r['task'], r['split'], int(r['horizon']), r['role'])
        identity = key + (int(r['seed']), r['pair'])
        if identity in seen:
            raise ValueError('Duplicate seed/pair contrast')
        seen.add(identity)
        value = [float(r['critical_'+m])-float(r['nuisance_'+m])
                 for m in ('feature', 'task')]
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite contrast')
        cells.setdefault(key, {}).setdefault(int(r['seed']), {})[r['pair']] = value
    expected = {(task, split, h, role) for task, roles in (('gate', ('cue',)), ('motion', ('start', 'end')))
                for role in roles for split in SPLITS for h in (3, 6)}
    if set(cells) != expected:
        raise ValueError('Missing or unexpected task/split/horizon/role cell')
    answer = {}
    for key, by_seed in sorted(cells.items()):
        if set(by_seed) != set(SEEDS):
            raise ValueError('Missing training seed')
        pair_ids = sorted(by_seed[SEEDS[0]])
        if len(pair_ids) != 128 or any(set(by_seed[s]) != set(pair_ids) for s in SEEDS):
            raise ValueError('Pair axis differs across seeds')
        answer[key] = np.array([[by_seed[s][p] for p in pair_ids] for s in SEEDS])
    return answer


def headroom_rows(baseline_records):
    grouped = {}
    for r in baseline_records:
        key = (r['task'], r['split'], int(r['horizon']), int(r['seed']))
        if r['baseline'] in grouped.setdefault(key, {}):
            raise ValueError('Duplicate baseline')
        grouped[key][r['baseline']] = r
    output = []
    required = {'uniform_delete', 'FIFO_delete_oldest', 'reject_newest', 'future_feature_oracle'}
    for (task, split, horizon, seed), values in sorted(grouped.items()):
        if set(values) != required:
            raise ValueError('Missing comparison baseline')
        for metric in ('feature_mse', 'task_error'):
            means = {b: float(r['metrics'][metric]['mean']) for b, r in values.items()}
            reject, oracle = means['reject_newest'], means['future_feature_oracle']
            if not np.isfinite(list(means.values())).all() or reject < 0:
                raise ValueError('Invalid baseline mean')
            # The feature-selected oracle need not minimize task error; preserve negative gaps.
            output.append(dict(task=task, split=split, horizon=horizon, seed=seed,
                metric=metric, reject_newest=reject, future_feature_oracle=oracle,
                uniform_delete=means['uniform_delete'], FIFO_delete_oldest=means['FIFO_delete_oldest'],
                absolute_gap=reject-oracle,
                relative_gap=None if reject < 1e-8 else (reject-oracle)/reject))
    return output


def age_shortcut_audit():
    spec = importlib.util.spec_from_file_location('diagnostic_assay_core', ROOT/'experiments/deletion/assay_core.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = newest_nuisance = 0
    for task in ('gate', 'motion'):
        c = json.loads((ROOT/f'results/reference/{task}/resolved_config.json').read_text())['data']
        gaps = sorted(set(range(c['gap'][0], c['gap'][1]+1)) | set(range(c['long_gap'][0], c['long_gap'][1]+1)))
        duration = c['cue_duration'] if task == 'gate' else c['motion_dt']
        for start in range(c['cue_start'][0], c['cue_start'][1]+1):
            for d in range(duration[0], duration[1]+1):
                end = start+d-1 if task == 'gate' else start+d
                evidence = [start] if task == 'gate' else [start, end]
                for gap in gaps:
                    query = end+gap+c['recent']
                    for _, parent in module.pools(task, evidence, query, c['recent'], end):
                        cases += 1
                        newest_nuisance += max(parent) not in evidence
    return dict(legal_parent_constructions=cases, newest_is_nuisance=newest_nuisance,
                newest_nuisance_fraction=newest_nuisance/cases,
                inference='Existing pool construction confounds newest timestamp with nuisance role; not a causal content-only test.')


def diagnose():
    paths = sorted(SOURCE.glob('pair_contrasts_*.csv')) + [SOURCE/'baselines.json',
        ROOT/'experiments/deletion/assay_core.py',
        ROOT/'results/reference/gate/resolved_config.json',
        ROOT/'results/reference/motion/resolved_config.json']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    recorded = {x['path']: x['sha256'] for x in json.loads((ROOT/'results/manifest.json').read_text())['files']}
    if any(recorded.get(p) != h for p, h in hashes.items()):
        raise ValueError('Input differs from the retained scientific evidence manifest')
    rows = []
    for p in paths:
        if p.suffix == '.csv':
            with p.open() as f:
                rows.extend(csv.DictReader(f))
    if len(rows) != 11520:
        raise ValueError('Expected original six compact CSVs')
    per_seed = []
    consistency = []
    for (task, split, horizon, role), cube in pair_cube(rows).items():
        for si, seed in enumerate(SEEDS):
            for mi, metric in enumerate(('feature_mse', 'task_error')):
                per_seed.append(dict(task=task, split=split, horizon=horizon, role=role,
                    seed=seed, metric=metric, **sign_summary(cube[si, :, mi])))
        consistent = np.all(cube > EPS, axis=(0, 2))
        correlations = {}
        for mi, metric in enumerate(('feature_mse', 'task_error')):
            correlations[metric] = {f'{a}:{b}': spearman(cube[i, :, mi], cube[j, :, mi])
                                   for i, a in enumerate(SEEDS) for j, b in enumerate(SEEDS) if j > i}
        count = int(consistent.sum())
        consistency.append(dict(task=task, split=split, horizon=horizon, role=role,
            pairs=128, positive_both_metrics_all_seeds=count, fraction=count/128,
            wilson95=wilson(count, 128), cross_seed_spearman=correlations))
    headroom = headroom_rows(json.loads((SOURCE/'baselines.json').read_text())['results'])
    source_ok = all(sha(ROOT/p) == h for p, h in hashes.items())
    if not source_ok:
        raise ValueError('Source evidence changed during read-only analysis')
    return dict(status='POSTHOC_SIGNAL_DIAGNOSIS_COMPLETE', input_sha256=hashes,
        analysis_source_sha256=sha(__file__), compact_pair_rows=len(rows),
        independent_pairs_per_task_split=128, sign_tolerance=EPS,
        per_seed_signs=per_seed, cross_seed_consistency=consistency,
        baseline_headroom=headroom, age_shortcut=age_shortcut_audit(),
        limitations=['Revealed development data; exploratory follow-up, not confirmation.',
            'Pair deltas average siblings and two parent variants; no per-fork reconstruction.',
            'A feature-selected oracle is future-informed and not a task-error upper bound.',
            'Marginal baseline CIs cannot yield a paired CI for their difference.',
            'Stable deletion labels do not establish learnability from past-only inputs.'],
        new_model_forward=False, new_training=False, new_GPU_seconds=0,
        original_phase1_go=False, joint_policy_stage_ready=False)


def write(result, folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'signal_diagnosis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for name, rows in [('pair_signs', result['per_seed_signs']), ('headroom', result['baseline_headroom'])]:
        with (folder/f'{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/diagnostics')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = diagnose()
    if args.check:
        saved = json.loads((args.output/'signal_diagnosis.json').read_text())
        if saved != result:
            raise ValueError('Saved diagnostic differs from deterministic recomputation')
        print('SIGNAL_DIAGNOSIS_RECOMPUTATION_PASS')
    else:
        write(result, args.output)
        print(json.dumps({k: result[k] for k in ('status', 'age_shortcut', 'new_GPU_seconds')}, indent=2))


if __name__ == '__main__':
    main()
