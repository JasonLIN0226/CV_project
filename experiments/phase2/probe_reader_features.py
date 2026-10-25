"""Frozen-feature binary probes, fitted only on TRAIN and selected on VAL.

Labels/metadata define diagnostic targets and grouping, never model inputs.
Raw encoder features and the frozen reader projection get the same weighted
ridge protocol. No reader is trained or changed by this module.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

HEADS = ('R_a', 'R_b', 'D_a', 'D_b', 'S_payload', 'S_tag')
DEFAULT = {'lambdas': [.001, .01, .1, 1., 10., 100.], 'shuffle_seed': 20261121,
           'standard_deviation_floor': 1e-4}
BOOTSTRAP = {'seed': 20261122, 'resamples': 2000}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def config_values(config):
    config = read(config) if isinstance(config, (str, Path)) else config
    value = {**DEFAULT, **(config.get('probe', config) if config else {})}
    if any(x <= 0 for x in value['lambdas']) or value['standard_deviation_floor'] <= 0:
        raise ValueError('Ridge penalties and normalization floor must be positive')
    result = {k: value[k] for k in DEFAULT}
    supplied_bootstrap = config.get('bootstrap', {}) if config else {}
    result['bootstrap'] = {k: int(supplied_bootstrap.get(k, v)) for k, v in BOOTSTRAP.items()}
    if result['bootstrap']['resamples'] <= 0:
        raise ValueError('Bootstrap resamples must be positive')
    return result


def cue_rows(data_path, split):
    """Deduplicate the physical cue image within each family and target head."""
    data_path = Path(data_path)
    index = read(data_path/'rgb_index.json')
    reverse = {int(i): h for h, i in index.items()}
    if len(reverse) != len(index):
        raise ValueError('RGB index must be one-to-one')
    rows = {}
    for r in read(data_path/'records.json'):
        if r['split'] != split:
            continue
        task = r['task_family']
        for slot, time in enumerate(r['evidence']):
            if task == 'D' and r['roles'][slot] == 'D':
                continue
            targets = [('S_payload', r['slot_payload'][slot]), ('S_tag', r['tags'][slot])] if task == 'S' else [(task+'_'+r['semantic_axes'][slot], r['slot_payload'][slot])]
            frame = r['rgb_indices'][time]
            for head, target in targets:
                if head not in HEADS or target not in (0, 1):
                    raise ValueError('Invalid binary probe target')
                key = (r['family'], reverse[frame], head)
                row = {'id': hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24],
                       'family': r['family'], 'rgb_sha256': reverse[frame], 'frame_index': frame,
                       'blank_index': r['rgb_indices'][r['query']], 'head': head, 'target': int(target),
                       'split': split, 'salient': slot == r['salient_slot'],
                       'axis': r['semantic_axes'][slot], 'conditions': {r['condition']}}
                if key in rows:
                    old = rows[key]
                    for field in ('target', 'salient', 'axis', 'blank_index'):
                        if old[field] != row[field]:
                            raise ValueError('Identical family/RGB/head has inconsistent '+field)
                    old['conditions'].add(r['condition'])
                else:
                    rows[key] = row
    result = sorted(rows.values(), key=lambda row: row['id'])
    for row in result:
        row['conditions'] = sorted(row['conditions'])
    if not result:
        raise ValueError('No probe rows for split '+split)
    return result


def assert_disjoint(*row_groups):
    sets = [{r['family'] for r in rows} for rows in row_groups]
    for i, a in enumerate(sets):
        for b in sets[i+1:]:
            if a & b:
                raise ValueError('Family overlap across splits')


def family_weights(rows):
    counts = Counter(r['family'] for r in rows)
    return np.array([1/(len(counts)*counts[r['family']]) for r in rows], dtype=np.float64)


def normalize_fit(x, rows, floor):
    w = family_weights(rows)
    mean = w@x
    std = np.sqrt(w@((x-mean)**2)).clip(min=floor)
    return mean, std


def shuffled_targets(rows, seed):
    y = np.array([r['target'] for r in rows], dtype=np.float64)
    permutation = np.arange(len(rows))
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r['head'], r['salient']].append(i)
    for key, indices in sorted(groups.items()):
        # Stable head/salience stream; shared by raw and projected probes.
        number = int(hashlib.sha256(json.dumps([seed, key]).encode()).hexdigest()[:16], 16)
        permutation[indices] = np.random.default_rng(number).permutation(indices)
    return y[permutation], permutation


def balanced_accuracy(y, score, weights=None):
    y, predicted = np.asarray(y), np.asarray(score) >= 0
    weights = np.ones(len(y)) if weights is None else np.asarray(weights)
    if not np.isfinite(score).all() or not np.isfinite(weights).all():
        raise ValueError('Nonfinite classification scores or weights')
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Balanced accuracy requires both binary classes')
    return float(np.mean([np.average(predicted[y == c] == c, weights=weights[y == c]) for c in (0, 1)]))


def ridge_path(x, targets, weights, lambdas):
    """Weighted mean squared error + lambda*||coef||^2, free intercept.

    Center before the spectral solve. Choose the smaller primal/dual Gram
    matrix and reuse its eigendecomposition across labels and penalties.
    """
    x, targets = np.asarray(x, np.float64), np.asarray(targets, np.float64)
    weights = np.asarray(weights, np.float64)
    if targets.ndim == 1:
        targets = targets[:, None]
    if not all(np.isfinite(a).all() for a in (x, targets, weights)):
        raise ValueError('Nonfinite ridge inputs')
    if not np.isclose(weights.sum(), 1) or np.min(weights) <= 0:
        raise ValueError('Weights must be positive and normalized')
    xm, ym = weights@x, weights@targets
    z = (x-xm)*np.sqrt(weights)[:, None]
    t = (targets-ym)*np.sqrt(weights)[:, None]
    primal = x.shape[1] <= x.shape[0]
    gram = z.T@z if primal else z@z.T
    if not np.isfinite(gram).all():
        raise ValueError('Nonfinite ridge Gram matrix')
    values, vectors = np.linalg.eigh(gram)
    values = np.maximum(values, 0)
    rhs = vectors.T@(z.T@t if primal else t)
    result = {}
    for penalty in lambdas:
        solved = vectors@(rhs/(values[:, None]+penalty))
        coef = solved if primal else z.T@solved
        intercept = ym-xm@coef
        if not np.isfinite(coef).all() or not np.isfinite(intercept).all():
            raise ValueError('Nonfinite ridge solution')
        result[float(penalty)] = (coef, intercept)
    return result


def normal_equation_diagnostics(x, y, weights, coef, intercept, penalty):
    """Validate the selected solution to the centered, weighted ridge system."""
    x, y, weights = (np.asarray(a, np.float64) for a in (x, y, weights))
    xc, yc = x-weights@x, y-weights@y
    rhs = xc.T@(weights*yc)
    lhs = xc.T@(weights*(xc@coef))+penalty*coef
    residual = lhs-rhs
    prediction = x@coef+intercept
    error = prediction-y
    values = [coef, intercept, prediction, residual, lhs, rhs]
    if not all(np.isfinite(a).all() for a in values):
        raise ValueError('Nonfinite selected-model normal equation')
    relative = float(np.linalg.norm(residual)/max(1., np.linalg.norm(rhs)))
    intercept_residual = float(abs(weights@error))
    report = {'solver': 'weighted_centered_primal_eigh' if x.shape[1] <= x.shape[0] else 'weighted_centered_dual_eigh',
              'all_finite': True, 'coefficient_l2': float(np.linalg.norm(coef)),
              'coefficient_equation_residual_l2': float(np.linalg.norm(residual)),
              'coefficient_equation_residual_max_abs': float(np.max(np.abs(residual))),
              'residual_l2_divided_by_max_one_rhs_l2': relative,
              'intercept_equation_residual_abs': intercept_residual,
              'weighted_mean_squared_error': float(weights@(error**2)),
              'penalized_objective': float(weights@(error**2)+penalty*np.sum(coef**2)),
              'relative_residual_tolerance': 1e-7, 'intercept_residual_tolerance': 1e-8}
    if relative > report['relative_residual_tolerance'] or intercept_residual > report['intercept_residual_tolerance']:
        raise ValueError('Selected-model normal-equation residual exceeds tolerance')
    return report


def representations(data_path, rows, basis):
    data_path = Path(data_path)
    ids = np.array([r['frame_index'] for r in rows])
    blank_ids = np.array([r['blank_index'] for r in rows])
    raw = np.load(data_path/'features_raw.npy', mmap_mode='r', allow_pickle=False)
    normalized = np.load(data_path/'features.npy', mmap_mode='r', allow_pickle=False)
    x_raw = np.asarray(raw[ids], np.float64)
    x_projected = (np.asarray(normalized[ids], np.float64)-basis['input_mean'])@basis['projection']
    x_blank_raw = np.asarray(raw[blank_ids], np.float64)
    x_blank_projected = (np.asarray(normalized[blank_ids], np.float64)-basis['input_mean'])@basis['projection']
    result = {'raw': x_raw, 'projected': x_projected, 'raw_blank': x_blank_raw, 'projected_blank': x_blank_projected}
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError('Nonfinite frozen features')
    return result


def unique_frame_rows(rows):
    unique = {}
    for i, r in enumerate(rows):
        unique.setdefault((r['family'], r['rgb_sha256']), i)
    indices = np.array(list(unique.values()), dtype=int)
    return indices, [rows[i] for i in indices]


def fit(source_run, out, config=None):
    source_run, out = Path(source_run), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Refusing to overwrite probe fit')
    c = config_values(config)
    data_path = source_run/'data'
    train, validation = cue_rows(data_path, 'train'), cue_rows(data_path, 'validation')
    assert_disjoint(train, validation)
    basis = dict(np.load(source_run/'basis.npz', allow_pickle=False))
    train_x, val_x = representations(data_path, train, basis), representations(data_path, validation, basis)
    out.mkdir(parents=True, exist_ok=True)
    (out/'models').mkdir()
    # Freeze the exact input projection in the artifact before any held-out use.
    np.savez(out/'basis.npz', input_mean=basis['input_mean'], projection=basis['projection'])
    shuffled, permutation = shuffled_targets(train, c['shuffle_seed'])
    np.save(out/'shuffle_permutation.npy', permutation)
    write(out/'train_rows.json', train); write(out/'validation_rows.json', validation)
    manifest = {'schema': 'reader-feature-probe-v1', 'config': c, 'status': 'fitting',
                'scope': 'TRAIN fit / own real VAL balanced-accuracy selection; no DEV or fresh diagnostic access in fit.',
                'representations': ['raw', 'projected'], 'heads': list(HEADS),
                'label_arms': ['real', 'shuffle', 'blank'], 'models': [], 'normalization': {},
                'blank_control': 'Keep cue-unique rows, labels and family weights; substitute associated current-frame features. Use the same TRAIN cue normalization and own VAL selection.',
                'source_basis_sha256': file_hash(source_run/'basis.npz'),
                'artifact_basis_sha256': file_hash(out/'basis.npz'),
                'source_records_sha256': file_hash(data_path/'records.json'),
                'source_rgb_index_sha256': file_hash(data_path/'rgb_index.json'),
                'fit_families': {'train': sorted({r['family'] for r in train}), 'validation': sorted({r['family'] for r in validation})},
                'inputs_hashes': {split: {name: array_hash(x) for name,x in values.items()} for split,values in [('train', train_x), ('validation', val_x)]},
                'shuffle_permutation_sha256': file_hash(out/'shuffle_permutation.npy'),
                'shuffle_targets_sha256': array_hash(shuffled),
                'train_rows_sha256': file_hash(out/'train_rows.json'), 'validation_rows_sha256': file_hash(out/'validation_rows.json'),
                'implementation_sha256': file_hash(__file__)}
    write(out/'prefit.json', manifest)
    unique, norm_rows = unique_frame_rows(train)
    normalization = {}
    for name in manifest['representations']:
        mean, std = normalize_fit(train_x[name][unique], norm_rows, c['standard_deviation_floor'])
        normalization[name+'_mean'], normalization[name+'_std'] = mean, std
        train_norm, val_norm = (train_x[name]-mean)/std, (val_x[name]-mean)/std
        blank_train_norm = (train_x[name+'_blank']-mean)/std
        blank_val_norm = (val_x[name+'_blank']-mean)/std
        manifest['normalization'][name] = {'train_unique_frames': len(unique), 'mean_sha256': array_hash(mean), 'std_sha256': array_hash(std), 'dimension': len(mean)}
        for head in HEADS:
            ti = np.array([i for i,r in enumerate(train) if r['head'] == head]); vi = np.array([i for i,r in enumerate(validation) if r['head'] == head])
            if not len(ti) or not len(vi):
                raise ValueError('Head absent from TRAIN or VAL: '+head)
            tr, va = [train[i] for i in ti], [validation[i] for i in vi]
            labels = np.stack([[r['target'] for r in tr], shuffled[ti]], 1)*2-1
            weights = family_weights(tr)
            path = ridge_path(train_norm[ti], labels, weights, c['lambdas'])
            blank_path = ridge_path(blank_train_norm[ti], labels[:, 0], weights, c['lambdas'])
            y_val, val_weights = np.array([r['target'] for r in va]), family_weights(va)
            for arm_i, arm in enumerate(manifest['label_arms']):
                this_path, this_column, this_val = (blank_path, 0, blank_val_norm) if arm == 'blank' else (path, arm_i, val_norm)
                candidates = []
                for penalty in sorted(this_path):
                    coef, intercept = this_path[penalty]
                    score = this_val[vi]@coef[:, this_column]+intercept[this_column]
                    candidates.append({'lambda': penalty, 'validation_balanced_accuracy': balanced_accuracy(y_val, score, val_weights)})
                best_accuracy = max(x['validation_balanced_accuracy'] for x in candidates)
                chosen = max((x for x in candidates if abs(x['validation_balanced_accuracy']-best_accuracy) <= 1e-12), key=lambda x:x['lambda'])
                coef, intercept = this_path[chosen['lambda']]
                this_train = blank_train_norm if arm == 'blank' else train_norm
                diagnostics = normal_equation_diagnostics(this_train[ti], labels[:, this_column], weights,
                    coef[:, this_column], intercept[this_column], chosen['lambda'])
                model_name = name+'__'+head+'__'+arm
                checkpoint = out/'models'/(model_name+'.npz')
                np.savez(checkpoint, coefficient=coef[:, this_column], intercept=intercept[this_column])
                manifest['models'].append({'name': model_name, 'representation': name, 'head': head, 'label_arm': arm,
                    'checkpoint': str(checkpoint.relative_to(out)), 'checkpoint_sha256': file_hash(checkpoint),
                    'selected_lambda': chosen['lambda'], 'validation_candidates': candidates,
                    'normal_equation_diagnostics': diagnostics,
                    'validation_balanced_accuracy': chosen['validation_balanced_accuracy'],
                    'train_rows': len(ti), 'validation_rows': len(vi),
                    'train_families': len({r['family'] for r in tr}), 'validation_families': len({r['family'] for r in va}),
                    'train_label_counts': dict(Counter(str(int(x)) for x in (labels[:, this_column]+1)/2)),
                    'train_groups': group_counts(tr), 'validation_groups': group_counts(va)})
    np.savez(out/'normalization.npz', **normalization)
    manifest['normalization_sha256'] = file_hash(out/'normalization.npz')
    manifest['status'] = 'locked-before-held-out-evaluation'
    write(out/'manifest.json', manifest)
    return manifest


def group_counts(rows):
    groups = Counter((r['family'], r['salient'], r['axis'], r['target']) for r in rows)
    return [{'family': k[0], 'salient': k[1], 'axis': k[2], 'target': k[3], 'n': v} for k,v in sorted(groups.items())]


def bootstrap_metrics(y, scores, rows, config):
    """Paired family resampling; do not resample factorial rows independently."""
    families = sorted({r['family'] for r in rows})
    if len(families) < 2:
        return {'status': 'omitted_single_family', **config}
    correct = (scores >= 0) == y
    values = []
    for family in families:
        ids = np.array([i for i,r in enumerate(rows) if r['family'] == family])
        yy, ok = y[ids], correct[ids]
        # Each family's first four components have total per-class mass one.
        values.append([np.mean(ok & (yy == 0)), np.mean(yy == 0),
                       np.mean(ok & (yy == 1)), np.mean(yy == 1), int(ok.sum()), len(ids)])
    draw = np.random.default_rng(config['seed']).integers(0, len(families), size=(config['resamples'], len(families)))
    resampled = np.asarray(values)[draw].mean(1)
    recalls = []
    for numerator, denominator in ((0, 1), (2, 3)):
        valid = resampled[:, denominator] > 0
        recall = np.full(config['resamples'], np.nan)
        recall[valid] = resampled[valid, numerator]/resampled[valid, denominator]
        recalls.append(recall)
    balanced = (recalls[0]+recalls[1])/2
    def interval(x):
        x = np.asarray(x); x = x[np.isfinite(x)]
        return {'valid_resamples': len(x), 'ci95': np.quantile(x, [.025, .975]).tolist() if len(x) else None}
    return {'status': 'complete', 'unit': 'base_family', 'families': len(families),
            'percentile': [2.5, 97.5], 'paired_draws_for_same_family_set': True, **config,
            'row_accuracy': interval(resampled[:, 4]/resampled[:, 5]),
            'family_equal_accuracy': interval(resampled[:, 0]+resampled[:, 2]),
            'family_equal_balanced_accuracy': interval(balanced),
            'family_equal_class_recall': {str(c): interval(recalls[c]) for c in (0, 1)}}


def metric(y, scores, rows, bootstrap=None):
    y, scores = np.asarray(y), np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite evaluation predictions')
    w = family_weights(rows)
    value = {'n': len(rows), 'families': len({r['family'] for r in rows}),
             'correct': int(np.sum((scores >= 0) == y)), 'accuracy': float(np.mean((scores >= 0) == y)),
             'family_equal_accuracy': float(np.average((scores >= 0) == y, weights=w)),
             'target_counts': dict(Counter(str(int(x)) for x in y))}
    value['balanced_accuracy'] = balanced_accuracy(y, scores, w) if len(np.unique(y)) == 2 else None
    value['class_recall'] = {str(c): float(np.average((scores[y == c] >= 0) == c, weights=w[y == c])) if np.any(y == c) else None for c in (0, 1)}
    value['bootstrap'] = bootstrap_metrics(y, scores, rows, bootstrap or BOOTSTRAP)
    return value


def evaluate(artifact, data_path, split, out):
    artifact, data_path, out = Path(artifact), Path(data_path), Path(out)
    if artifact.is_file():
        artifact = artifact.parent
    manifest = read(artifact/'manifest.json')
    if manifest['status'] != 'locked-before-held-out-evaluation':
        raise ValueError('Probe fitting is not locked')
    for filename, key in [('basis.npz', 'artifact_basis_sha256'), ('normalization.npz', 'normalization_sha256'), ('shuffle_permutation.npy', 'shuffle_permutation_sha256')]:
        if file_hash(artifact/filename) != manifest[key]:
            raise ValueError('Frozen probe artifact hash mismatch: '+filename)
    rows = cue_rows(data_path, split)
    fit_families = set(sum(manifest['fit_families'].values(), []))
    if {r['family'] for r in rows} & fit_families:
        raise ValueError('Evaluation overlaps fit/selection families')
    basis = dict(np.load(artifact/'basis.npz', allow_pickle=False))
    arrays = representations(data_path, rows, basis)
    normalization = dict(np.load(artifact/'normalization.npz', allow_pickle=False))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Refusing to overwrite probe evaluation')
    out.mkdir(parents=True, exist_ok=True)
    predictions, summary = {}, {}
    for model in manifest['models']:
        checkpoint = artifact/model['checkpoint']
        if file_hash(checkpoint) != model['checkpoint_sha256']:
            raise ValueError('Frozen model hash mismatch')
        m = dict(np.load(checkpoint, allow_pickle=False))
        name, rep = model['name'], model['representation']
        indices = np.array([i for i,r in enumerate(rows) if r['head'] == model['head']])
        selected = [rows[i] for i in indices]
        y = np.array([r['target'] for r in selected])
        input_rep = rep+'_blank' if model['label_arm'] == 'blank' else rep
        x = (arrays[input_rep][indices]-normalization[rep+'_mean'])/normalization[rep+'_std']
        scores = x@m['coefficient']+m['intercept']
        predictions[name+'__row_index'], predictions[name+'__score'] = indices, scores
        groups = {'all': np.arange(len(selected))}
        for i,r in enumerate(selected):
            for key in ['salient/'+str(r['salient']), 'axis/'+r['axis'], 'family/'+r['family']]+['condition/'+x for x in r['conditions']]:
                groups.setdefault(key, []).append(i)
        summary[name] = {key: metric(y[ids], scores[ids], [selected[j] for j in ids], manifest['config']['bootstrap']) for key,ids in sorted(groups.items())}
    np.savez_compressed(out/'predictions.npz', **predictions)
    write(out/'rows.json', rows)
    report = {'split': split, 'scope': 'Frozen diagnostic feature probes; descriptive held-out results, no reader or retention-policy claim.',
              'artifact_manifest_sha256': file_hash(artifact/'manifest.json'), 'rows_sha256': file_hash(out/'rows.json'),
              'predictions_sha256': file_hash(out/'predictions.npz'), 'data_records_sha256': file_hash(data_path/'records.json'),
              'feature_slice_hashes': {k: array_hash(v) for k,v in arrays.items()}, 'models': summary}
    write(out/'metrics.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    a = commands.add_parser('fit'); a.add_argument('--source-run', required=True); a.add_argument('--out', required=True); a.add_argument('--config')
    a = commands.add_parser('evaluate'); a.add_argument('--artifact', required=True); a.add_argument('--data-path', required=True); a.add_argument('--split', required=True); a.add_argument('--out', required=True)
    args = vars(parser.parse_args()); command = args.pop('command')
    value = fit(**args) if command == 'fit' else evaluate(**args)
    print(json.dumps({'command': command, 'models': len(value['models']), 'output': args['out']}))

if __name__ == '__main__':
    main()
