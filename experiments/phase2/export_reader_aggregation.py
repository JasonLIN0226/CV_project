"""Post-freeze export of observed feature vectors for a completed reader study.

Read arrays and saved coefficients only: no model loading, inference, fitting,
encoding, scene generation, checkpoint selection, or confirmation access.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

COMPLETED = 'completed_locked_reader_diagnostic_no_students_no_original_confirmation'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def entry(path):
    return {'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def validate_scope(run, parent, output):
    run, parent, output = map(lambda p: Path(p).resolve(), (run, parent, output))
    if output.is_relative_to(run) or output.is_relative_to(parent):
        raise ValueError('Export must be outside both immutable input runs')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Export output must be empty')
    status = read(run/'study_status.json')
    if status.get('status') != COMPLETED:
        raise ValueError('Only a completed reader diagnostic may be exported')
    if status.get('original_confirmation_opened') is not False or status.get('retention_students_trained') is not False:
        raise ValueError('Unexpected confirmation/student scope')
    for source in (run, parent):
        if (source/'confirmation_opened.json').exists():
            raise ValueError('Original confirmation must remain closed')
    opened = read(run/'diagnostic_opened.json')
    if opened.get('original_confirmation') is not False or opened.get('analysis_role') != 'fresh_diagnostic':
        raise ValueError('Invalid fresh diagnostic identity')
    protocol = read(run/'protocol.json')
    config = protocol['config']
    if config.get('original_confirm_open_enabled') is not False or config.get('students_enabled') is not False:
        raise ValueError('Frozen protocol contains unsupported scope')
    if opened.get('seed') != config['diagnostic']['seed']:
        raise ValueError('Fresh diagnostic seed differs from frozen protocol')
    lock = read(run/'all_models_locked.json')
    if sha(run/'protocol.json') != lock['protocol_sha256']:
        raise ValueError('Frozen protocol binding changed')
    if sha(run/'parent_binding.json') != protocol['parent_binding_sha256']:
        raise ValueError('Parent binding changed after study freeze')
    if sha(run/'all_models_locked.json') != opened['all_models_lock_sha256']:
        raise ValueError('Fresh diagnostic does not bind the frozen model lock')
    if sha(parent/'basis.npz') != lock['basis_sha256']:
        raise ValueError('Parent basis differs from the reader study')
    return run, parent, output, config


def prepare_vectors(records, features, horizons):
    """Class order is explicit; redundant same-class siblings must agree."""
    records = [r for r in records if r['split'] == 'development']
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Missing or duplicated development records')
    groups = {}; metadata = []; currents = []
    for r in records:
        q, indices = r['query'], r['rgb_indices']
        if r['target_times'] != [q+h for h in horizons]:
            raise ValueError('Record horizons differ from frozen targets')
        current = np.asarray(features[indices[q]])
        target = np.asarray(features[np.asarray(indices)[r['target_times']]])
        if not np.isfinite(current).all() or not np.isfinite(target).all():
            raise ValueError('Nonfinite current/prototype feature')
        group = groups.setdefault(r['counterfactual_group'], {'task_family': r['task_family'],
                                  'family': r['family'], 'classes': {}})
        if group['task_family'] != r['task_family'] or group['family'] != r['family']:
            raise ValueError('Counterfactual group mixes tasks/families')
        previous = group['classes'].get(r['outcome_class'])
        if previous is not None and not np.array_equal(previous, target):
            raise ValueError('Same group/class has inconsistent future prototypes')
        group['classes'][r['outcome_class']] = target.copy()
        currents.append(current.copy())
        metadata.append({'id': r['id'], 'family': r['family'], 'task_family': r['task_family'],
                         'condition': r['condition'], 'counterfactual_group': r['counterfactual_group'],
                         'target_class': r['outcome_class'], 'current_index': len(currents)-1,
                         'query': q, 'target_times': r['target_times']})
    ordered = []; prototypes = []
    for group_id, group in sorted(groups.items()):
        labels = sorted(group['classes'])
        expected = [0, 1] if group['task_family'] == 'S' else [0, 1, 2, 3]
        if labels != expected:
            raise ValueError('Incomplete or invalid future-prototype class set')
        ordered.append({'id': group_id, 'family': group['family'], 'task_family': group['task_family'],
                        'class_labels': labels, 'prototype_offset': len(prototypes), 'prototype_count': len(labels)})
        prototypes.extend(group['classes'][label] for label in labels)
    return metadata, ordered, np.stack(currents), np.stack(prototypes)


def audit_coefficients(rows, coefficients, metadata, groups, current, prototypes, basis, count=16):
    """Deterministic evenly spaced saved rows; not an outcome-selected sample."""
    if len(rows) != len(coefficients) or not len(rows):
        raise ValueError('Qualification rows and coefficient array do not align')
    by_id = {r['id']: r for r in metadata}; by_group = {g['id']: g for g in groups}
    samples = np.unique(np.linspace(0, len(rows)-1, min(count, len(rows)), dtype=int))
    max_error = 0.; checks = []
    for i in samples:
        row = rows[int(i)]; r = by_id[row['id']]; g = by_group[r['counterfactual_group']]
        cf = np.asarray(coefficients[row['coefficient_index']], np.float64)
        prediction = current[r['current_index']].astype(np.float64)[None] + basis['target_mean'] + cf@basis['output_basis'].T
        actual = prototypes[g['prototype_offset']:g['prototype_offset']+g['prototype_count']].astype(np.float64)
        distances = ((prediction[None]-actual)**2).mean((1, 2))
        target_index = g['class_labels'].index(row['target_class'])
        expected = np.asarray(row['distances'], np.float64)
        error = max(float(np.max(abs(distances-expected))), abs(float(distances[target_index])-row['mse']))
        if not np.isfinite(distances).all() or not np.allclose(distances, expected, atol=2e-5, rtol=2e-5):
            raise ValueError('Exported vectors cannot reconstruct saved prototype distances')
        if not np.isclose(distances[target_index], row['mse'], atol=2e-5, rtol=2e-5):
            raise ValueError('Exported vectors cannot reconstruct saved target MSE')
        max_error = max(max_error, error)
        checks.append({'qualification_row': int(i), 'id': row['id'], 'arm': row['arm'], 'max_abs_error': error})
    return {'sample_rule': '16 evenly spaced qualification rows per reader and split, including endpoints',
            'sample_count': len(samples), 'absolute_tolerance': 2e-5, 'relative_tolerance': 2e-5,
            'tolerance_scope': 'postprocessing FP32 reconstruction only; does not change experimental gates',
            'max_abs_error': max_error, 'checks': checks}


def export(run, parent, output):
    run, parent, output, config = validate_scope(run, parent, output)
    binding = read(run/'parent_binding.json')
    inputs = {}; output.mkdir(parents=True, exist_ok=True)
    for name in ('study_status.json', 'protocol.json', 'all_models_locked.json', 'diagnostic_opened.json', 'parent_binding.json'):
        inputs['reader_run/'+name] = entry(run/name)
    inputs['parent_run/basis.npz'] = entry(parent/'basis.npz')
    inputs['parent_run/protocol.json'] = entry(parent/'protocol.json')
    if inputs['parent_run/protocol.json'] != binding['verified_files']['protocol.json']:
        raise ValueError('Parent protocol no longer matches original binding')
    shutil.copyfile(parent/'basis.npz', output/'basis.npz')
    basis = dict(np.load(parent/'basis.npz', allow_pickle=False))
    parent_protocol = read(parent/'protocol.json'); horizons = parent_protocol['config']['horizons']
    splits = {}; family_sets = []
    for label, data in [('existing_development', parent/'data'), ('fresh_diagnostic', run/'diagnostic_data')]:
        for name in ('records.json', 'features.npy'):
            value = entry(data/name); inputs[label+'/'+name] = value
            if label == 'existing_development' and value != binding['verified_files']['data/'+name]:
                raise ValueError('Parent cache no longer matches original binding: '+name)
        if label == 'fresh_diagnostic':
            encoding = read(data/'encoding.json')
            generation = read(data/'generation.json')
            inputs[label+'/encoding.json'] = entry(data/'encoding.json')
            inputs[label+'/generation.json'] = entry(data/'generation.json')
            if sha(data/'features.npy') != encoding['features_sha256']:
                raise ValueError('Fresh feature cache no longer matches encoding receipt')
            if sha(data/'records.json') != generation['records_sha256']:
                raise ValueError('Fresh records no longer match generation receipt')
        records = read(data/'records.json'); features = np.load(data/'features.npy', mmap_mode='r', allow_pickle=False)
        metadata, groups, current, prototypes = prepare_vectors(records, features, horizons)
        if features.shape[1] != len(basis['target_mean']):
            raise ValueError('Feature/basis dimension mismatch')
        family_sets.append({r['family'] for r in metadata})
        dest = output/label; dest.mkdir()
        np.savez_compressed(dest/'vectors.npz', current=current, prototypes=prototypes)
        write(dest/'records.json', metadata); write(dest/'groups.json', groups)
        audits = {}
        for variant in config['variants']+['uniform_intervention']:
            evaluation = run/'evaluation'/label/variant
            rows_path, coefficients_path = evaluation/'reader_qualification_rows.json', evaluation/'reader_qualification_coefficients.npy'
            inputs[label+'/'+variant+'/rows'] = entry(rows_path)
            inputs[label+'/'+variant+'/coefficients'] = entry(coefficients_path)
            audits[variant] = audit_coefficients(read(rows_path), np.load(coefficients_path, mmap_mode='r', allow_pickle=False),
                                               metadata, groups, current, prototypes, basis)
        write(dest/'reconstruction_audit.json', audits)
        splits[label] = {'records': len(metadata), 'families': len(family_sets[-1]), 'groups': len(groups),
                         'current_shape': list(current.shape), 'prototype_shape': list(prototypes.shape),
                         'prototype_classes': {'R': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'S': [0, 1]},
                         'coefficient_source': 'reader_run/evaluation/'+label+'/<variant>/reader_qualification_coefficients.npy'}
    if family_sets[0] & family_sets[1]:
        raise ValueError('Existing and fresh diagnostics overlap families')
    files = {p.relative_to(output).as_posix(): entry(p) for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = {'schema': 'reader-aggregation-reconstruction-export-v1', 'status': 'completed_verified_export',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'postprocessing_after_experiment_source_freeze': True,
                'exporter_sha256': sha(Path(__file__)), 'read_only_inputs': True,
                'fit_or_inference_or_generation': False, 'confirmation_opened': False,
                'prediction_formula': 'current[record.current_index] + basis.target_mean + qualification_coefficients[row.coefficient_index] @ basis.output_basis.T',
                'prototype_indexing': 'Find group.id == record.counterfactual_group; use group.prototype_offset + position(target_class in group.class_labels)',
                'prototype_horizons': horizons, 'feature_space': 'original frozen TRAIN-normalized I-JEPA feature space',
                'inputs': inputs, 'splits': splits, 'files': files,
                'limitation': 'Join with the completed study release containing qualification rows/coefficients; this export is not new model evidence.'}
    write(output/'export_manifest.json', manifest)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    result = export(a.run, a.parent, a.output)
    print(json.dumps({'status': result['status'], 'splits': result['splits']}, indent=2))


if __name__ == '__main__':
    main()
