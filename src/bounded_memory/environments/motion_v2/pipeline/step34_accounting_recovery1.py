"""Pure native accounting identity checks; retain every historical row.

No subprocess, filesystem write, torch import, or experiment occurs here.
An exact submission identity and DBIndex disambiguate recycled job numbers.
"""
from datetime import datetime
import importlib.util
from pathlib import Path
import re

FIELDS = ('JobIDRaw', 'JobName', 'User', 'Account', 'Partition', 'State',
          'ExitCode', 'Start', 'End', 'ElapsedRaw', 'AllocTRES', 'ReqTRES',
          'Cluster', 'DBIndex', 'Submit', 'Restarts', 'NodeList', 'TimelimitRaw')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def require(value, message):
    if not value:
        raise ValueError(message)


def records_module():
    spec = importlib.util.spec_from_file_location('_motion_recovery_records', Path(__file__).with_name('slurm_records.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_allocation(text, job_id, expected):
    """Return original rows plus their explicit DBIndex selection, never rewrite input."""
    require(type(text) is str and type(job_id) is str, 'Native text and job ID must be strings')
    require(re.fullmatch(r'[0-9]+', job_id), 'Invalid job ID')
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.split('|')]
        if len(cells) == len(FIELDS) + 1 and cells[-1] == '':
            cells.pop()
        require(len(cells) == len(FIELDS), 'Malformed extended sacct row')
        if tuple(cells) == FIELDS:
            continue
        row = dict(zip(FIELDS, cells))
        require(row['JobIDRaw'] == job_id or row['JobIDRaw'].startswith(job_id + '.'), 'Unrelated numeric job ID')
        require(re.fullmatch(r'[0-9]+', row['DBIndex']) and row['Cluster'], 'Native DBIndex/Cluster required for every row')
        rows.append(row)
    keys = ('JobName', 'User', 'Account', 'Partition', 'Submit')
    require(all(type(expected.get(k)) is str and expected[k] for k in keys), 'Complete expected submission identity required')
    candidates = [r for r in rows if r['JobIDRaw'] == job_id and all(r[k] == expected[k] for k in keys)]
    require(len(candidates) == 1, 'Exactly one matching submitted allocation is required')
    selected = candidates[0]
    # A later run of this same submitted task cannot be hidden as historical noise.
    related = [r for r in rows if r['JobIDRaw'] == job_id and all(r[k] == expected[k] for k in keys[:-1])
               and r['Submit'] >= expected['Submit']]
    require(len(related) == 1, 'Additional run/requeue of this task is ambiguous')
    same = [r for r in rows if (r['Cluster'], r['DBIndex']) == (selected['Cluster'], selected['DBIndex'])]
    ids = [r['JobIDRaw'] for r in same]
    require(len(ids) == len(set(ids)), 'Duplicate row within selected database allocation')
    for row in same:
        require(row['Account'] in ('', selected['Account']), 'Step account differs')
        require(row['User'] in ('', selected['User']), 'Step user differs')
    excluded = [r for r in rows if (r['Cluster'], r['DBIndex']) != (selected['Cluster'], selected['DBIndex'])]
    return {'allocation_row': selected, 'all_rows': rows, 'selected_rows': same,
            'excluded_historical_rows': excluded, 'selection_keys': {k: selected[k] for k in keys + ('Cluster', 'DBIndex')},
            'batch_step_ids': [r['JobIDRaw'] for r in same if r['JobIDRaw'] == job_id + '.batch' or r['JobIDRaw'].startswith(job_id + '.batch.')],
            'no_batch_step': not any(r['JobIDRaw'] == job_id + '.batch' or r['JobIDRaw'].startswith(job_id + '.batch.') for r in same)}


def terminal_from_accounting(row, context, startup_control, control_result):
    """Certify accounting after controller expiry without inventing terminal scontrol.

    The fallback requires an explicit Invalid job id error, native sacct
    Restarts=0, and a retained startup identity with matching resources/start.
    """
    require(type(startup_control) is str, 'Retained control must be original text')
    parser = records_module()
    state = parser.normalize_state(row['State'])
    require(state in TERMINAL and row['Restarts'] == '0', 'Terminal native sacct Restarts=0 is required')
    require(row['User'] == context['owner'] and row['Account'] == context['account']
            and row['Partition'] == context['partition'], 'Wrong terminal job owner/account/partition')
    require(row['TimelimitRaw'].isdigit() and int(row['TimelimitRaw']) * 60 == context['time_limit_seconds'], 'Native wall limit changed')
    require(row['ElapsedRaw'].isdigit(), 'Native allocation seconds missing')
    elapsed = int(row['ElapsedRaw'])
    require(elapsed <= context['time_limit_seconds'], 'Native allocation exceeds registered wall limit')
    require(re.fullmatch(r'[0-9]+:[0-9]+', row['ExitCode']), 'Native exit code missing')
    if state == 'COMPLETED':
        require(row['ExitCode'] == '0:0', 'Completed job has nonzero exit code')
    requested = parser.gpu_count(row['ReqTRES'])
    count = parser.gpu_count(row['AllocTRES'])
    require(requested == 1 and count in (0, 1), 'Unexpected requested/allocated GPU count')
    if count == 0:
        require(elapsed == 0 and state in {'CANCELLED', 'FAILED', 'BOOT_FAIL'}, 'Unallocated terminal branch must have zero elapsed')
    else:
        parser.one_gpu(row['AllocTRES'])
        require(row['NodeList'] not in ('', 'None', 'Unknown', '(null)'), 'Native node identity missing')
        start, end = datetime.fromisoformat(row['Start']), datetime.fromisoformat(row['End'])
        require(end >= start and elapsed <= (end - start).total_seconds(), 'Native timing inconsistency')
    require(type(control_result) is dict and type(control_result.get('stdout')) is str, 'Raw control result required')
    if control_result['returncode'] == 0:
        control = parser.parse_control(control_result['stdout'])
        require(parser.normalize_state(control.get('JobState', '')) == state, 'Terminal scontrol/sacct state mismatch')
        kind = 'fresh_terminal_scontrol_and_sacct'
    else:
        require(control_result['returncode'] == 1 and not control_result['stdout'].strip()
                and control_result.get('stderr', '').strip() == 'slurm_load_jobs error: Invalid job id specified',
                'Only explicit controller expiration permits retained-control fallback')
        control = parser.parse_control(startup_control)
        kind = 'terminal_sacct_restarts_with_retained_startup_control'
        require(control.get('JobState') == ('RUNNING' if count else 'PENDING'), 'Wrong retained startup state')
    require(control.get('JobId') == row['JobIDRaw'] and control.get('Restarts') == '0', 'Controller job/restart identity changed')
    require(control.get('UserId', '').split('(')[0] == row['User'] and control.get('Account') == row['Account']
            and control.get('Partition') == row['Partition'] and control.get('SubmitTime') == row['Submit'], 'Controller submission identity changed')
    require(control.get('Command') == context['wrapper_path'] and control.get('WorkDir') == context['workdir'], 'Controller command/workdir changed')
    require(parser.parse_duration(control['TimeLimit']) == context['time_limit_seconds'], 'Controller wall limit changed')
    if count:
        require(control.get('StartTime') == row['Start'] and control.get('NodeList') == row['NodeList'], 'Native start/node identity changed')
        require(parser.gpu_count(control.get('AllocTRES', '')) == 1, 'Controller allocation is not one GPU')
        require(control.get('TresPerNode') == 'gres/gpu:a100:1', 'Controller did not request one A100')
    return {'native_state': state, 'exit_code': row['ExitCode'], 'allocated_gpu_count': count,
            'actual_a100_gpu_seconds': elapsed * count, 'restarts': 0, 'accounting_row': row,
            'control_evidence': kind, 'fresh_terminal_scontrol_available': control_result['returncode'] == 0,
            'terminal_scontrol_fabricated': False, 'requested_seconds': context['time_limit_seconds']}
