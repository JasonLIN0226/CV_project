"""Preregistered pure native-record replay. No Slurm, science, filesystem writes.

Historical native text is SHA-checked. Any adapted text is explicitly synthetic,
and is passed to the same validators used by the real submission helper.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]
FIXTURES = {
    'pending': ('results/motion_v2/v2_3/deployment2/remote_evidence/submission/immediate_scontrol.stdout.txt',
                '850e8e394ebe06d301666c0575e3e9c5fec01289c5a17db6d5e513ca657c532d'),
    'cpu': ('results/motion_v2/v2_2/deployment1/remote_evidence/native_job_at_stop.txt',
            '3e79c9b4fd9e7e2ad172293b04f84ef73677499d05714df968af655f46ad8de7'),
    'a100': ('results/literature_diagnosis/qualified-reader-retention-20261025/conf/cluster/sacct_12374.txt',
             '509f39267779ea64e0b5f5c6b20925b753ba0a5c9eed2f4c8a42d49561e0d4f3'),
    'cancelled_sacct': ('results/motion_v2/v2_3/deployment2/remote_evidence/settlement/final_sacct.stdout.txt',
                        'c6e562f1d80ec1ea4d4e157775087ae0c305f7004c64fd34b72f40e0731ef676'),
    'cancelled_control': ('results/motion_v2/v2_3/deployment2/remote_evidence/settlement/final_scontrol.stdout.txt',
                          '87a38c8ba2d7ec1e7b7f357f8e761b8c7c49010cc583bcc8b3ee8a7344dca604'),
}


@pytest.fixture(scope='module')
def submit():
    source = Path(__file__).with_name('submit_step34.py')
    spec = importlib.util.spec_from_file_location('motion_step34_submit_software_only', source)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = obj
    spec.loader.exec_module(obj)
    return obj


def fixture_text(name):
    relative, checksum = FIXTURES[name]
    raw = (REPO / relative).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == checksum
    return raw.decode('utf-8')


def synthetic_control(submit, fixture='pending', **changes):
    fields = submit.load('slurm_records').parse_control(fixture_text(fixture))
    expected = submit.context()
    fields.update(TimeLimit='02:55:00', Command=expected['wrapper_path'], WorkDir=expected['workdir'])
    fields.update(changes)
    return ' '.join(key + '=' + value for key, value in fields.items())


def historical_row(submit):
    return submit.parse_complete_accounting(fixture_text('cancelled_sacct'), '16542')['allocation_row']


def artificial_history(seconds=0, attempt=1, job_id='90001'):
    return {'attempt': attempt, 'job_id': job_id, 'terminal': True, 'restarts': 0,
            'actual_a100_gpu_seconds': seconds, 'four_facts': {
                'no_batch_step': True, 'stdout_stderr_absent': True,
                'scientific_marker_absent': True, 'scientific_output_absent': True}}


def test_real_pending_schema_plus_declared_step34_resource_adaptation(submit):
    raw = submit.load('slurm_records').parse_control(fixture_text('pending'))
    assert raw['NumNodes'] == '1-1'
    assert raw['CPUs/Task'] == '4' and raw['ReqB:S:C:T'] == '0:0:*:*'
    fields = submit.validate_submission_control(synthetic_control(submit), '16542')
    assert fields['Requeue'] == '1' and fields['JobState'] == 'PENDING'
    assert fields['NumNodes'] == '1-1'


@pytest.mark.parametrize('change', [{'NumNodes': '1-2'}, {'Restarts': '1'}, {'TimeLimit': '03:00:00'},
                                   {'ReqTRES': 'cpu=4,gres/gpu=2,gres/gpu:a100=2'},
                                   {'Account': 'other'}, {'Command': '/wrong/wrapper'}, {'WorkDir': '/wrong/repo'}])
def test_immediate_native_contradictions_rejected(submit, change):
    with pytest.raises(ValueError):
        submit.validate_submission_control(synthetic_control(submit, **change), '16542')


def test_cpu_fixture_preserves_extra_cpus_but_is_not_a100(submit):
    parser = submit.load('slurm_records')
    fields = parser.parse_control(fixture_text('cpu'))
    assert fields['NumCPUs'] == '24' and fields['CPUs/Task'] == '4'
    assert parser.gpu_count(fields['AllocTRES']) == 0
    with pytest.raises(ValueError):
        parser.one_gpu(fields['AllocTRES'])
    adapted = synthetic_control(submit, NumCPUs='24', **{'CPUs/Task': '4'})
    assert submit.validate_submission_control(adapted, '16542')['NumCPUs'] == '24'


def test_historical_completed_a100_tres_is_valid_without_running_state(submit):
    parser = submit.load('slurm_records')
    row = parser.parse_accounting_row(fixture_text('a100'), '12374')
    assert parser.normalize_state(row['State']) == 'COMPLETED'
    assert parser.one_gpu(row['AllocTRES']) == 1


def test_real_zero_allocation_cancellation_with_step34_request_adaptation(submit):
    row = historical_row(submit)
    control = synthetic_control(submit, fixture='cancelled_control')
    result = submit.validate_terminal_accounting(row, control)
    assert result['native_state'] == 'CANCELLED'
    assert result['actual_a100_gpu_seconds'] == 0
    assert result['allocation_case'] == 'cancelled_before_allocation'
    assert result['restarts'] == 0


def allocated_pair(submit):
    row = dict(historical_row(submit), State='COMPLETED', Start='2026-11-01T00:00:00',
               End='2026-11-01T00:00:30', ElapsedRaw='30', AllocTRES='cpu=24,gres/gpu=1,gres/gpu:a100=1')
    control = synthetic_control(submit, fixture='cancelled_control', JobState='COMPLETED',
        StartTime=row['Start'], EndTime=row['End'], RunTime='00:00:30', NodeList='synthetic-a100-node',
        AllocTRES=row['AllocTRES'], NumCPUs='24')
    return row, control


def test_terminal_agreement_and_single_allocation_not_step_sum(submit):
    row, control = allocated_pair(submit)
    result = submit.validate_terminal_accounting(row, control)
    assert result['actual_a100_gpu_seconds'] == 30
    assert result['terminal_state_consistent'] is True
    assert result['final_fields']['NumCPUs'] == '24'


@pytest.mark.parametrize('change', ['state', 'restarts', 'gpu', 'nonterminal'])
def test_terminal_contradictions_rejected(submit, change):
    row, control = allocated_pair(submit)
    if change == 'state':
        control = control.replace('JobState=COMPLETED', 'JobState=FAILED')
    elif change == 'restarts':
        control = control.replace('Restarts=0', 'Restarts=1')
    elif change == 'gpu':
        fields = submit.load('slurm_records').parse_control(control)
        fields['AllocTRES'] = 'cpu=24'
        control = ' '.join(key + '=' + value for key, value in fields.items())
    else:
        control = control.replace('JobState=COMPLETED', 'JobState=RUNNING')
    with pytest.raises(ValueError):
        submit.validate_terminal_accounting(row, control)


def accounting_text(submit, rows):
    return '\n'.join('|'.join(row[name] for name in submit.ACCOUNTING_FIELDS) for row in rows) + '\n'


def test_complete_steps_batch_is_not_hidden_and_extern_does_not_double_bill(submit):
    row, _ = allocated_pair(submit)
    extern = dict(row, JobIDRaw='16542.extern')
    clean = submit.parse_complete_accounting(accounting_text(submit, [row, extern]), '16542')
    assert clean['no_batch_step'] and len(clean['all_rows']) == 2
    batch = dict(row, JobIDRaw='16542.batch')
    full = submit.parse_complete_accounting(accounting_text(submit, [row, extern, batch]), '16542')
    assert full['no_batch_step'] is False and full['batch_step_ids'] == ['16542.batch']
    assert full['allocation_row']['ElapsedRaw'] == '30'


@pytest.mark.parametrize('kind', ['duplicate_top', 'duplicate_step', 'unrelated', 'truncated'])
def test_ambiguous_complete_accounting_rejected(submit, kind):
    row, _ = allocated_pair(submit)
    if kind == 'duplicate_top':
        text = accounting_text(submit, [row, row])
    elif kind == 'duplicate_step':
        step = dict(row, JobIDRaw='16542.extern')
        text = accounting_text(submit, [row, step, step])
    elif kind == 'unrelated':
        text = accounting_text(submit, [row, dict(row, JobIDRaw='99999')])
    else:
        text = accounting_text(submit, [row]) + '16542.batch|bad\n'
    with pytest.raises(ValueError):
        submit.parse_complete_accounting(text, '16542')


def test_budget_prior3705_full10500_and_exact195_retry_margin(submit):
    first = submit.validate_admission(1, [], marker_absent=True, output_absent=True)
    assert first['projected_cumulative_a100_seconds'] == 14205
    edge = submit.validate_admission(2, [artificial_history(195)], marker_absent=True, output_absent=True)
    assert edge['projected_cumulative_a100_seconds'] == 14400
    with pytest.raises(ValueError):
        submit.validate_admission(2, [artificial_history(196)], marker_absent=True, output_absent=True)


@pytest.mark.parametrize('fact', ['no_batch_step', 'stdout_stderr_absent', 'scientific_marker_absent', 'scientific_output_absent'])
@pytest.mark.parametrize('value', [False, None, 1])
def test_each_missing_or_refuted_preentry_fact_forbids_retry(submit, fact, value):
    history = artificial_history()
    history['four_facts'][fact] = value
    with pytest.raises(ValueError):
        submit.validate_admission(2, [history], marker_absent=True, output_absent=True)


@pytest.mark.parametrize('marker,output', [(False, True), (True, False), (False, False)])
def test_any_scientific_entry_or_output_forbids_even_first_submission(submit, marker, output):
    with pytest.raises(ValueError):
        submit.validate_admission(1, [], marker_absent=marker, output_absent=output)


def test_max_three_attempts_order_unique_jobs_and_native_seconds(submit):
    first, second = artificial_history(100), artificial_history(95, 2, '90002')
    assert submit.validate_admission(3, [first, second], marker_absent=True, output_absent=True)['eligible']
    for attempt, history in ((4, [first, second]), (3, [first]), (2, [second])):
        with pytest.raises(ValueError):
            submit.validate_admission(attempt, history, marker_absent=True, output_absent=True)
    second['job_id'] = first['job_id']
    with pytest.raises(ValueError):
        submit.validate_admission(3, [first, second], marker_absent=True, output_absent=True)
    first['actual_a100_gpu_seconds'] = True
    with pytest.raises(ValueError):
        submit.validate_admission(2, [first], marker_absent=True, output_absent=True)


@pytest.mark.parametrize('filename', ['config.json', 'preprocessor_config.json', 'model.safetensors'])
@pytest.mark.parametrize('blob_length', [40, 64])
def test_only_registered_original_hf_snapshot_to_blob_links_allowed(submit, filename, blob_length):
    snapshot = f'{submit.HF_MODEL}/snapshots/{submit.HF_REVISION}/{filename}'
    target = submit.HF_MODEL + '/blobs/' + 'a' * blob_length
    inventory = {snapshot: 'b' * 64, target: 'b' * 64}
    assert submit.validate_hf_snapshot_target(snapshot, target, inventory) == target


@pytest.mark.parametrize('kind', ['other_file', 'other_revision', 'other_model', 'outside',
                                  'traversal', 'nested_blob', 'missing_blob', 'different_digest'])
def test_unregistered_or_escaping_hf_snapshot_link_rejected(submit, kind):
    snapshot = f'{submit.HF_MODEL}/snapshots/{submit.HF_REVISION}/model.safetensors'
    target = submit.HF_MODEL + '/blobs/' + 'a' * 64
    inventory = {snapshot: 'b' * 64, target: 'b' * 64}
    if kind == 'other_file':
        snapshot = snapshot.replace('model.safetensors', 'other.bin')
    elif kind == 'other_revision':
        snapshot = snapshot.replace(submit.HF_REVISION, '0' * 40)
    elif kind == 'other_model':
        target = target.replace('ijepa_vith14_1k', 'other')
    elif kind == 'outside':
        target = '/tmp/' + 'a' * 64
    elif kind == 'traversal':
        target = submit.HF_MODEL + '/blobs/../blobs/' + 'a' * 64
    elif kind == 'nested_blob':
        target = submit.HF_MODEL + '/blobs/subdir/' + 'a' * 64
    elif kind == 'missing_blob':
        inventory.pop(target)
    else:
        inventory[target] = 'c' * 64
    with pytest.raises(ValueError):
        submit.validate_hf_snapshot_target(snapshot, target, inventory)


def test_preserved_terminal_control_only_after_explicit_purge_and_exact_accounting(submit):
    row, control = allocated_pair(submit)
    lookup = {'returncode': 1, 'stdout': '', 'stderr': 'slurm_load_jobs error: Invalid job id specified'}
    checksum = hashlib.sha256(control.encode()).hexdigest()
    reuse = submit.load('resubmit_step2_v2_3_1').retained_control_after_purge
    parser = submit.load('slurm_records')
    result = reuse(lookup, row, row.copy(), control, checksum, parser)
    assert result['evidence_kind'] == 'preserved_terminal_scontrol_after_explicit_purge'
    assert submit.validate_terminal_accounting(row, result['control_text'])['actual_a100_gpu_seconds'] == 30
    with pytest.raises(ValueError):
        reuse(dict(lookup, stderr='Permission denied'), row, row, control, checksum, parser)
    with pytest.raises(ValueError):
        reuse(lookup, dict(row, ElapsedRaw='31'), row, control, checksum, parser)
    with pytest.raises(ValueError):
        reuse(lookup, row, row, control, '0' * 64, parser)
