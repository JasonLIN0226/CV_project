"""Synthetic-only tests of the post-freeze vector export boundary/formula."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.phase2 import export_reader_aggregation as exporter


def test_unfinished_or_confirmation_run_cannot_export(tmp_path):
    run, parent = tmp_path/'run', tmp_path/'parent'; run.mkdir(); parent.mkdir()
    exporter.write(run/'study_status.json', {'status': 'running'})
    with pytest.raises(ValueError, match='completed'):
        exporter.validate_scope(run, parent, tmp_path/'out')
    exporter.write(run/'study_status.json', {'status': exporter.COMPLETED,
                   'original_confirmation_opened': True, 'retention_students_trained': False})
    with pytest.raises(ValueError, match='confirmation/student'):
        exporter.validate_scope(run, parent, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_export_cannot_write_inside_inputs_or_replace_results(tmp_path):
    run, parent = tmp_path/'run', tmp_path/'parent'; run.mkdir(); parent.mkdir()
    with pytest.raises(ValueError, match='outside'):
        exporter.validate_scope(run, parent, run/'export')
    out = tmp_path/'out'; out.mkdir(); (out/'prior').write_text('preserved')
    with pytest.raises(ValueError, match='empty'):
        exporter.validate_scope(run, parent, out)
    assert (out/'prior').read_text() == 'preserved'


def test_saved_coefficients_reconstruct_prediction_and_class_order():
    current = np.array([[.1, .2, .3]], dtype=np.float32)
    basis = {'target_mean': np.array([.1, -.1, .2]),
             'output_basis': np.array([[1., 0.], [0., 1.], [.5, -.5]])}
    coefficients = np.array([[[.2, -.1], [.4, .3]]])
    prediction = current[0]+basis['target_mean']+coefficients[0]@basis['output_basis'].T
    prototypes = np.stack([prediction+.1, prediction-.2])
    distances = ((prediction[None]-prototypes)**2).mean((1, 2))
    metadata = [{'id': 'r', 'counterfactual_group': 'g', 'current_index': 0}]
    groups = [{'id': 'g', 'class_labels': [0, 1], 'prototype_offset': 0, 'prototype_count': 2}]
    rows = [{'id': 'r', 'arm': 'full', 'target_class': 1, 'coefficient_index': 0,
             'distances': distances.tolist(), 'mse': float(distances[1])}]
    result = exporter.audit_coefficients(rows, coefficients, metadata, groups, current, prototypes, basis)
    assert result['max_abs_error'] < 1e-12
    with pytest.raises(ValueError, match='reconstruct'):
        exporter.audit_coefficients(rows, coefficients+1, metadata, groups, current, prototypes, basis)


def test_repeated_xor_outcomes_share_two_not_four_prototypes():
    features = np.arange(18, dtype=np.float32).reshape(6, 3)
    common = {'family': 'f', 'task_family': 'S', 'condition': 'neutral', 'counterfactual_group': 'g',
              'split': 'development', 'query': 0, 'target_times': [1, 2]}
    records = [{**common, 'id': str(i), 'outcome_class': i % 2,
                'rgb_indices': [0, 1, 2] if i % 2 == 0 else [0, 3, 4]} for i in range(4)]
    metadata, groups, current, prototypes = exporter.prepare_vectors(records, features, [1, 2])
    assert len(metadata) == 4 and groups[0]['class_labels'] == [0, 1]
    assert prototypes.shape == (2, 2, 3) and current.shape == (4, 3)
    records[-1]['rgb_indices'] = [0, 3, 5]
    with pytest.raises(ValueError, match='inconsistent'):
        exporter.prepare_vectors(records, features, [1, 2])
