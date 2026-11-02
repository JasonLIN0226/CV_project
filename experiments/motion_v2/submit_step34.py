"""Read-only preflight, one sbatch per attempt, and native Motion3/4 accounting.

No scientific entry point, encoder, dataset decoder, fitting, cancellation,
cleanup, or automatic retry is called. Importing this module performs no IO.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import subprocess
import sys
import traceback


sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/scratch/xy3257/motion-v2-20261101-step34-deployment1')
RUNTIME = ROOT / 'runtime'
OLD_REPO = Path('/scratch/xy3257/motion-v2-20261031-deployment3/repo')
DATASET = OLD_REPO / 'results/motion_v2/v2_2/step1/data/dataset'
ASSETS = OLD_REPO / 'results/motion_v2/v2_3_1/step2_gpu_attempt1'
VENV = Path('/scratch/xy3257/motion-v2-20261031-deployment1/venv')
BRANCH = 'experiments/motion-v2-20261030'
PROTOCOL = 'experiments/motion_v2/protocol_step34.json'
EXECUTION = 'experiments/motion_v2/protocol_step34_execution.json'
WRAPPER = 'experiments/motion_v2/run_step34.sbatch'
CPU_RECEIPT = 'results/motion_v2/step3_4/cpu'
SCIENCE_OUTPUT = 'results/motion_v2/step3_4/gpu'
MANIFEST_SHA = '9ac43b81eb75456e17f3542e64e8a0ab58282207752ae3712836f6d2ea20032a'
PRIOR_SECONDS, WALL_SECONDS, TOTAL_CAP = 3705, 10500, 14400
MAX_ATTEMPTS = 3
ACCOUNTING_FIELDS = ('JobIDRaw', 'JobName', 'User', 'Account', 'Partition', 'State',
                     'ExitCode', 'Start', 'End', 'ElapsedRaw', 'AllocTRES', 'ReqTRES')
FOUR_FACTS = ('no_batch_step', 'stdout_stderr_absent', 'scientific_marker_absent', 'scientific_output_absent')
HF_MODEL = 'hf_cache/models--facebook--ijepa_vith14_1k'
HF_REVISION = 'f157467ea509bc356ff9f61fd3c0d840eec5e04e'
HF_SNAPSHOT_FILES = {f'{HF_MODEL}/snapshots/{HF_REVISION}/{name}' for name in
                     ('config.json', 'preprocessor_config.json', 'model.safetensors')}
_MODULES = {}


def load(name):
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location('_motion_step34_submit_' + name,
                                                    Path(__file__).with_name(name + '.py'))
        obj = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = obj
        spec.loader.exec_module(obj)
        _MODULES[name] = obj
    return _MODULES[name]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def safe_path(path, parent, *, exists=True):
    path, parent = Path(path).absolute(), Path(parent).absolute()
    require('..' not in path.parts and path.is_relative_to(parent), 'Path escaped its registered root')
    for component in (path, *path.parents):
        require(not component.is_symlink(), 'Symlink paths are forbidden: ' + str(component))
    if exists:
        require(path.exists(), 'Required path missing: ' + str(path))
    return path


def absent(path):
    """Only an explicit ENOENT is absence; a broken symlink is present."""
    try:
        Path(path).lstat()
        return False
    except FileNotFoundError:
        return True


def validate_hf_snapshot_target(relative, target_relative, inventory):
    """Pure check for the three original, hash-registered HF snapshot links."""
    require(relative in HF_SNAPSHOT_FILES, 'Only the three frozen HF snapshot files may be symlinks')
    target = Path(target_relative)
    require(not target.is_absolute() and '..' not in target.parts
            and target.parent.as_posix() == HF_MODEL + '/blobs'
            and re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', target.name) is not None,
            'Snapshot link must resolve to the same original model blob directory')
    require(relative in inventory and target.as_posix() in inventory,
            'Snapshot and resolved blob must both be explicitly registered')
    require(inventory[relative] == inventory[target.as_posix()], 'Snapshot/blob content hashes disagree')
    return target.as_posix()


def asset_path(relative, inventory):
    """Resolve only the known snapshot symlinks; all other paths stay strict."""
    relative_path = Path(relative)
    require(not relative_path.is_absolute() and '..' not in relative_path.parts, 'Invalid asset relative path')
    path = ASSETS / relative_path
    if not path.is_symlink():
        return safe_path(path, ASSETS)
    safe_path(path.parent, ASSETS)  # No symlink directory or alternative cache tree.
    target = path.resolve(strict=True)
    require(target.is_relative_to(ASSETS), 'HF snapshot escaped the original asset directory')
    validate_hf_snapshot_target(relative, target.relative_to(ASSETS).as_posix(), inventory)
    return safe_path(target, ASSETS)


def context():
    return {'owner': 'xy3257', 'account': 'csci_ga_2271-2026fa', 'partition': 'c12m85-a100-1',
            'time_limit_seconds': WALL_SECONDS, 'hard_cap_seconds': TOTAL_CAP,
            'wrapper_path': str(REPO / WRAPPER), 'workdir': str(REPO)}


def validate_submission_control(text, job_id, expected=None):
    return load('submit_step2_v2_3_1').validate_submission_control(text, job_id, expected or context())


def parse_complete_accounting(text, job_id):
    """Require exactly one allocation row; retain and inspect every job step."""
    parser = load('slurm_records')
    columns = ','.join(ACCOUNTING_FIELDS)
    row = parser.parse_accounting_row(text, job_id, columns)
    require(row is not None, 'Exact top-level sacct allocation record is missing')
    rows, identities = [], set()
    for line in text.splitlines():
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.split('|')]
        if len(cells) == len(ACCOUNTING_FIELDS) + 1 and cells[-1] == '':
            cells.pop()
        require(len(cells) == len(ACCOUNTING_FIELDS), 'Malformed complete sacct row')
        if tuple(cells) == ACCOUNTING_FIELDS:
            continue
        record = dict(zip(ACCOUNTING_FIELDS, cells))
        identity = record['JobIDRaw']
        require(identity == job_id or identity.startswith(job_id + '.'), 'Unrelated accounting identity')
        require(identity not in identities, 'Duplicate raw identity makes accounting ambiguous')
        identities.add(identity)
        rows.append(record)
    batch = [r['JobIDRaw'] for r in rows if r['JobIDRaw'] == job_id + '.batch'
             or r['JobIDRaw'].startswith(job_id + '.batch.')]
    return {'allocation_row': row, 'all_rows': rows, 'batch_step_ids': batch, 'no_batch_step': not batch}


def validate_terminal_accounting(row, control_text, expected=None):
    expected = expected or context()
    parser = load('slurm_records')
    result = load('submit_step2_v2_3_1').validate_settlement_record(row, control_text, expected)
    fields = result['final_fields']
    require(fields is not None and fields.get('Restarts') == '0', 'Terminal native Restarts=0 evidence is required')
    require({'RunTime', 'AllocTRES'} <= fields.keys(), 'Terminal native allocation/runtime evidence is incomplete')
    state = parser.normalize_state(fields['JobState'])
    require(state == result['native_state'], 'Terminal scontrol and sacct states disagree')
    require(state in load('submit_step2_v2_3_1').TERMINAL, 'Terminal scontrol is required')
    require(parser.gpu_count(fields.get('AllocTRES', '')) == result['allocated_gpu_count'],
            'Terminal scontrol and sacct GPU allocations disagree')
    if result['allocated_gpu_count'] == 1:
        parser.one_gpu(fields.get('AllocTRES', ''))
        require(row['Start'] not in ('', 'None', 'Unknown'), 'Allocated job has no accounting start')
    result.update(restarts=0, terminal_state_consistent=True,
                  scontrol_runtime_seconds=parser.parse_duration(fields['RunTime']),
                  accounting_row=row, requested_seconds=expected['time_limit_seconds'])
    return result


def validate_admission(attempt, history, *, marker_absent, output_absent):
    """Pure budget and eligibility decision. An allocated science entry is never retried."""
    require(type(attempt) is int and attempt in (1, 2, 3), 'At most three frozen-job attempts are authorized')
    require(isinstance(history, list) and len(history) == attempt - 1, 'Predecessor attempt sequence is incomplete')
    require(marker_absent is True and output_absent is True, 'Scientific entry/output exists; retry is forbidden')
    ids, spent = set(), 0
    for index, entry in enumerate(history, 1):
        require(type(entry.get('attempt')) is int and entry['attempt'] == index and entry.get('terminal') is True
                and type(entry.get('restarts')) is int and entry['restarts'] == 0,
                'Predecessor must be the next terminal attempt with Restarts=0')
        identity = entry.get('job_id')
        require(isinstance(identity, str) and re.fullmatch(r'[0-9]+', identity) and identity not in ids,
                'Invalid or repeated predecessor job identity')
        ids.add(identity)
        seconds = entry.get('actual_a100_gpu_seconds')
        require(type(seconds) is int and 0 <= seconds <= WALL_SECONDS, 'Invalid native predecessor GPU seconds')
        require(all(entry.get('four_facts', {}).get(name) is True for name in FOUR_FACTS),
                'All four pre-entry-failure facts must be freshly proven')
        spent += seconds
    projected = PRIOR_SECONDS + spent + WALL_SECONDS
    require(projected <= TOTAL_CAP, 'Prior + settled infrastructure + full frozen wall limit exceeds four A100-hours')
    return {'eligible': True, 'attempt': attempt, 'prior_a100_seconds': PRIOR_SECONDS,
            'step34_predecessor_a100_seconds': spent, 'next_maximum_a100_seconds': WALL_SECONDS,
            'projected_cumulative_a100_seconds': projected, 'total_cap_a100_seconds': TOTAL_CAP,
            'maximum_total_attempts': MAX_ATTEMPTS}


def command(argv, timeout=60, env=None):
    try:
        result = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=timeout,
                                env=dict(os.environ, TZ='UTC') if env is None else env)
        return {'argv': list(argv), 'returncode': result.returncode, 'stdout': result.stdout,
                'stderr': result.stderr, 'captured_utc': utc()}
    except subprocess.TimeoutExpired as error:
        def text(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
        return {'argv': list(argv), 'returncode': None, 'timed_out': True,
                'stdout': text(error.stdout), 'stderr': text(error.stderr), 'captured_utc': utc()}


def capture(output, receipt, name, argv, *, require_success=True, env=None):
    result = command(argv, env=env)
    receipt.setdefault('commands', []).append({'name': name, **result})
    for stream in ('stdout', 'stderr'):
        with (output / (name + '.' + stream + '.txt')).open('x', encoding='utf-8') as handle:
            handle.write(result[stream])
    if require_success:
        require(result['returncode'] == 0, 'Native/read-only command failed: ' + name)
    return result


def receipt_file(path, expected_name):
    path = safe_path(path, RUNTIME)
    require(path.name == expected_name, 'Unexpected receipt filename')
    inventory = path.parent / 'SHA256SUMS'
    hashes = {}
    for line in inventory.read_text(encoding='utf-8').splitlines():
        digest, relative = line.split('  ', 1)
        require(re.fullmatch(r'[0-9a-f]{64}', digest) and Path(relative).name == relative and relative not in hashes,
                'Unsafe or duplicate receipt inventory entry')
        item = safe_path(path.parent / relative, path.parent)
        require(sha(item) == digest, 'Receipt inventory digest mismatch: ' + relative)
        hashes[relative] = digest
    require({expected_name, 'status.json'} <= hashes.keys(), 'Incomplete immutable receipt inventory')
    status, value = read(path.parent / 'status.json'), read(path)
    require(status.get('receipt_sha256') == hashes[expected_name], 'Status/receipt hash chain differs')
    return value, status, sha(inventory)


def validate_wrapper(path):
    directives = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.startswith('#SBATCH '):
            continue
        tokens = shlex.split(line[len('#SBATCH '):])
        require(len(tokens) in (1, 2), 'Malformed SBATCH directive')
        if '=' in tokens[0]:
            require(len(tokens) == 1, 'Ambiguous SBATCH value')
            key, value = tokens[0].split('=', 1)
        else:
            key, value = tokens[0], tokens[1] if len(tokens) == 2 else True
        require(key not in directives, 'Duplicate SBATCH directive')
        directives[key] = value
    expected = {'--account': context()['account'], '--partition': context()['partition'], '--nodes': '1',
                '--ntasks': '1', '--cpus-per-task': '4', '--mem': '16G', '--gres': 'gpu:a100:1',
                '--time': '02:55:00', '--no-requeue': True, '--open-mode': 'append',
                '--output': str(ROOT / 'scientific-job-%j.stdout.log'), '--error': str(ROOT / 'scientific-job-%j.stderr.log')}
    require(all(directives.get(key) == value for key, value in expected.items()), 'Frozen wrapper directives differ')
    require(not set(directives) & {'--requeue', '--gpus', '--gpus-per-node', '--gpus-per-task'}, 'Competing GPU/requeue directive')
    return directives


def registration(args, output, receipt):
    require(REPO == ROOT / 'repo', 'Run only in the registered deployment checkout')
    require(sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1, 'Invoke with Python -I -B')
    require(sys.version_info[:2] == (3, 12), 'Use the frozen Python3.12 runtime')
    require(Path(sys.prefix).absolute() == VENV, 'Use the unchanged deployment1 venv')
    require(pwd.getpwuid(os.getuid()).pw_name == 'xy3257', 'Unexpected authenticated owner')
    require(re.fullmatch(r'[0-9a-f]{40}', args.expected_head) is not None, 'Exact pushed 40-character HEAD required')
    head = capture(output, receipt, 'git_head', ['git', 'rev-parse', 'HEAD'])['stdout'].strip()
    branch = capture(output, receipt, 'git_branch', ['git', 'branch', '--show-current'])['stdout'].strip()
    require(head == args.expected_head and branch == BRANCH, 'Deployment HEAD/branch differs from frozen request')
    dirty = capture(output, receipt, 'git_tracked_status', ['git', 'status', '--porcelain', '--untracked-files=no'])['stdout'].strip()
    require(not dirty, 'Tracked deployment files are modified')
    runner = load('run_step34')
    protocol, config = runner.registration(REPO / PROTOCOL)
    required = {Path(__file__).relative_to(REPO).as_posix(), WRAPPER, 'experiments/motion_v2/run_step34.py',
                'experiments/motion_v2/slurm_records.py', 'experiments/motion_v2/submit_step2_v2_3_1.py',
                'experiments/motion_v2/resubmit_step2_v2_3_1.py'}
    require(required <= protocol['source_sha256'].keys(), 'Science registration omits the actual native validators/helper')
    execution_path = runner.committed(REPO / EXECUTION)
    execution = read(execution_path)
    require(execution.get('science_protocol_sha256') == sha(REPO / PROTOCOL)
            and execution.get('test_status') == 'passed', 'Committed execution lacks matching science and passed software')
    require(execution['cpu_receipt'].get('path') == CPU_RECEIPT, 'CPU receipt path differs from the wrapper')
    wrapper = runner.committed(REPO / WRAPPER)
    registered = {'head_commit': head, 'branch': branch, 'science_protocol_sha256': sha(REPO / PROTOCOL),
                  'execution_protocol_sha256': sha(execution_path), 'wrapper_sha256': sha(wrapper),
                  'wrapper_directives': validate_wrapper(wrapper)}
    receipt['registration'] = registered
    return runner, protocol, config, execution, registered


def verify_preparation(runner, protocol, config, execution):
    cpu_root = safe_path(REPO / CPU_RECEIPT, REPO)
    manifest = runner.validate_cpu_receipt(execution, cpu_root)
    for relative in ('status.json', 'audit.json', 'data/dataset/manifest.json'):
        runner.committed(cpu_root / relative)
    inventory = read(REPO / protocol['asset_inventory']['path'])['artifact_sha256']
    required_assets = {'resolved_config.json', 'features/index.json', 'features/normalization.npz',
                       'predictor/motion/basis.npz', 'decoder/train_position_decoder.npz'}
    require(required_assets <= inventory.keys(), 'Frozen inventory omits a scientific dependency')
    snapshot_links = {}
    for relative, digest in inventory.items():
        path = asset_path(relative, inventory)
        require(path.is_file() and sha(path) == digest, 'Frozen16614 asset changed: ' + relative)
        if (ASSETS / relative).is_symlink():
            snapshot_links[relative] = {'resolved_relative': path.relative_to(ASSETS).as_posix(), 'sha256': digest}
    for seed in (17, 29, 43):
        selection_relative = f'predictor/motion/seed_{seed}/selection.json'
        require(selection_relative in inventory, 'Frozen inventory omits a predictor selection receipt')
        selection = read(asset_path(selection_relative, inventory))
        checkpoint_relative = f'predictor/motion/seed_{seed}/' + Path(selection['checkpoint']).name
        require(checkpoint_relative in inventory, 'Selected predictor checkpoint is not hash-registered')
    original_manifest = safe_path(DATASET / 'manifest.json', DATASET)
    require(sha(original_manifest) == MANIFEST_SHA, 'Original dataset manifest changed')
    original = read(original_manifest)
    rows = original['records']
    require(len(rows) == 1216 and sum(r['split'] == 'train' for r in rows) == 1024
            and sum(r['split'] == 'validation' for r in rows) == 192
            and all(r['task'] == 'motion' for r in rows), 'Original manifest contains unauthorized rows')
    raw_hashes = {}
    for record in rows:
        require(re.fullmatch(r'[0-9a-f]{24}', record['id']) is not None, 'Malformed episode ID')
        for kind, directory in (('obs', 'observations'), ('eval', 'evaluation_only')):
            relative = f"{directory}/{record['split']}/{record['id']}.npz"
            require(record[kind] == relative and relative not in raw_hashes, 'Unexpected or duplicate episode path')
            path = safe_path(DATASET / relative, DATASET)
            require(sha(path) == record[kind + '_sha256'], 'Raw existing episode bytes changed')
            raw_hashes[relative] = record[kind + '_sha256']
    environment = runner.environment(config['gpu_environment'])
    # environment imports torch but neither queries device availability nor initializes CUDA.
    require(not sys.modules['torch'].cuda.is_initialized(), 'CPU preflight unexpectedly initialized CUDA')
    return {'cpu_receipt_path': CPU_RECEIPT, 'cpu_artifact_sha256': execution['cpu_receipt']['artifact_sha256'],
            'cpu_validation_records': len(manifest['records']), 'frozen_asset_sha256': inventory,
            'allowed_original_hf_snapshot_links': snapshot_links,
            'original_manifest_sha256': MANIFEST_SHA, 'original_raw_npz_sha256': raw_hashes,
            'original_raw_files_verified': len(raw_hashes), 'environment': environment,
            'dataset_arrays_decoded': False, 'scientific_entry_invocations': 0, 'cuda_initialized': False}


def accounting_capture(output, receipt, job_id, prefix):
    result = capture(output, receipt, prefix + '_sacct_duplicates',
                     ['sacct', '--duplicates', '-j', job_id, '--noheader', '--parsable2',
                      '--format=' + ','.join(ACCOUNTING_FIELDS)])
    parsed = parse_complete_accounting(result['stdout'], job_id)
    control = capture(output, receipt, prefix + '_scontrol',
                      ['scontrol', 'show', 'job', '--oneliner', job_id], require_success=False)
    return parsed, control


def path_facts(job_id):
    paths = {'stdout': ROOT / f'scientific-job-{job_id}.stdout.log',
             'stderr': ROOT / f'scientific-job-{job_id}.stderr.log',
             'scientific_marker': ROOT / 'scientific_job_started',
             'scientific_output': REPO / SCIENCE_OUTPUT}
    probes = {name: {'path': str(path), 'absent': absent(path)} for name, path in paths.items()}
    return probes, {'stdout_stderr_absent': probes['stdout']['absent'] and probes['stderr']['absent'],
                    'scientific_marker_absent': probes['scientific_marker']['absent'],
                    'scientific_output_absent': probes['scientific_output']['absent']}


def predecessor_history(args, output, receipt, registered, *, refresh):
    supplied = args.predecessor_settlements or []
    require(len(supplied) == args.attempt - 1, 'Supply each prior step34 settlement in attempt order')
    history = []
    for attempt, path in enumerate(supplied, 1):
        expected = RUNTIME / f'attempt{attempt}' / 'settlement' / 'settlement.json'
        require(Path(path).absolute() == expected, 'Unexpected predecessor settlement path/order')
        saved, status, inventory_sha = receipt_file(path, 'settlement.json')
        require(status.get('status') == 'passed' and saved.get('attempt') == attempt
                and saved.get('registration') == registered, 'Predecessor settlement is not passed under this registration')
        job_id = saved['job_id']
        terminal = saved['terminal_accounting']
        if refresh:
            parsed, lookup = accounting_capture(output, receipt, job_id, f'predecessor{attempt}')
            require(parsed['allocation_row'] == terminal['accounting_row'], 'Fresh predecessor accounting changed')
            if lookup['returncode'] == 0:
                control_text, evidence_kind = lookup['stdout'], 'fresh_terminal_scontrol'
            else:
                retained = load('resubmit_step2_v2_3_1').retained_control_after_purge(
                    lookup, parsed['allocation_row'], terminal['accounting_row'], saved['terminal_scontrol'],
                    saved['terminal_scontrol_sha256'], load('slurm_records'))
                control_text, evidence_kind = retained['control_text'], retained['evidence_kind']
            terminal = validate_terminal_accounting(parsed['allocation_row'], control_text)
            probes, facts = path_facts(job_id)
            facts['no_batch_step'] = parsed['no_batch_step']
        else:
            facts, probes = saved['four_facts'], saved['path_probes']
            evidence_kind = 'hash_verified_predecessor_settlement'
        history.append({'attempt': attempt, 'job_id': job_id, 'terminal': True,
                        'restarts': terminal['restarts'], 'actual_a100_gpu_seconds': terminal['actual_a100_gpu_seconds'],
                        'four_facts': facts, 'path_probes': probes, 'settlement_sha256': sha(path),
                        'settlement_inventory_sha256': inventory_sha, 'control_evidence': evidence_kind})
    return history


def admission(args, output, receipt, registered):
    for future in range(args.attempt + 1, MAX_ATTEMPTS + 1):
        require(absent(RUNTIME / f'attempt{future}' / 'submission'), 'Later attempt submission already exists')
    history = predecessor_history(args, output, receipt, registered, refresh=True)
    decision = validate_admission(args.attempt, history,
                                  marker_absent=absent(ROOT / 'scientific_job_started'),
                                  output_absent=absent(REPO / SCIENCE_OUTPUT))
    receipt['predecessor_history'], receipt['admission'] = history, decision
    return decision


def finish(output, name, receipt, status):
    receipt['finished_utc'] = utc()
    write(output / name, receipt)
    status['receipt_sha256'] = sha(output / name)
    write(output / 'status.json', status)
    with (output / 'SHA256SUMS').open('x', encoding='utf-8') as stream:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != 'SHA256SUMS':
                stream.write(sha(path) + '  ' + path.name + '\n')
    print(json.dumps(status, ensure_ascii=False, allow_nan=False), flush=True)


def execute(args):
    require(REPO == ROOT / 'repo', 'Run only in the registered deployment checkout')
    require(type(args.attempt) is int and args.attempt in (1, 2, 3), 'Invalid attempt number')
    folder = {'preflight': 'preflight', 'submit': 'submission', 'settle': 'settlement'}[args.mode]
    output = safe_path(args.output, RUNTIME, exists=False)
    require(output == RUNTIME / f'attempt{args.attempt}' / folder and absent(output), 'Use the exact new immutable attempt directory')
    if args.mode == 'preflight':
        require(absent(output.parent / 'submission'), 'This attempt has a submission guard; do not repeat preparation')
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'started_utc': utc(), 'mode': args.mode, 'attempt': args.attempt,
               'scientific_entry_invocations': 0, 'sbatch_attempted': False, 'job_id': None}
    status = {'status': 'stopped_before_' + args.mode, 'mode': args.mode, 'attempt': args.attempt,
              'scientific_success_inferred': False, 'automatic_retry': False}
    filename = {'preflight': 'preflight.json', 'submit': 'submission.json', 'settle': 'settlement.json'}[args.mode]
    code = 2
    try:
        runner, protocol, config, execution, registered = registration(args, output, receipt)
        if args.mode in ('preflight', 'submit'):
            if args.mode == 'submit':
                preflight_path = args.preflight or output.parent / 'preflight' / 'preflight.json'
                require(Path(preflight_path).absolute() == output.parent / 'preflight' / 'preflight.json', 'Use this attempt preflight')
                preflight, pre_status, pre_hash = receipt_file(preflight_path, 'preflight.json')
                require(pre_status.get('status') == 'passed' and preflight['registration'] == registered
                        and preflight['attempt'] == args.attempt, 'Matching passed preflight required')
                receipt['preflight_sha256'], receipt['preflight_inventory_sha256'] = sha(preflight_path), pre_hash
            receipt['preparation'] = verify_preparation(runner, protocol, config, execution)
            admission(args, output, receipt, registered)
            if args.mode == 'preflight':
                status.update(status='passed', gpu_seconds=0, cuda_initialized=False)
            else:
                argv = ['sbatch', '--parsable', '--no-requeue', str(REPO / WRAPPER)]
                removed = sorted(name for name in os.environ if name.startswith('SBATCH_'))
                environment = {name: value for name, value in os.environ.items() if not name.startswith('SBATCH_')}
                environment['TZ'] = 'UTC'
                write(output / 'submission_started.json', {'created_utc': utc(), 'argv': argv,
                      'registration': registered, 'attempt': args.attempt, 'removed_sbatch_environment_keys': removed,
                      'unknown_outcome_requires_read_only_investigation': True, 'sbatch_attempt_limit': 1})
                receipt['sbatch_attempted'] = True
                status.update(status='submission_outcome_unknown', sbatch_attempts=1)
                submitted = capture(output, receipt, 'sbatch', argv, require_success=False, env=environment)
                require(submitted['returncode'] == 0, 'sbatch failed/timed out; never resubmit this attempt')
                match = re.fullmatch(r'([0-9]+)(?:;([A-Za-z0-9_.-]+))?', submitted['stdout'].strip())
                require(match is not None, 'Unknown sbatch outcome; manual read-only investigation required')
                job_id = match.group(1)
                receipt.update(job_id=job_id, cluster=match.group(2))
                status.update(status='stopped_after_submission', job_id=job_id)
                native = capture(output, receipt, 'immediate_scontrol', ['scontrol', 'show', 'job', '--oneliner', job_id])
                fields = validate_submission_control(native['stdout'], job_id)
                receipt['immediate_fields'] = fields
                status.update(status='submitted_verified', native_state=fields['JobState'], restarts=0,
                              requeue_record_only=fields.get('Requeue'), allocated_cpus_record_only=fields.get('NumCPUs'))
        else:
            submission_path = args.submission or output.parent / 'submission' / 'submission.json'
            require(Path(submission_path).absolute() == output.parent / 'submission' / 'submission.json', 'Use this attempt submission receipt')
            submitted, submitted_status, submitted_inventory = receipt_file(submission_path, 'submission.json')
            require(submitted.get('sbatch_attempted') is True and submitted.get('registration') == registered
                    and submitted.get('attempt') == args.attempt and re.fullmatch(r'[0-9]+', str(submitted.get('job_id', ''))),
                    'Submission receipt does not identify exactly one registered job')
            job_id = submitted['job_id']
            receipt.update(job_id=job_id, submission_sha256=sha(submission_path), submission_inventory_sha256=submitted_inventory)
            status.update(status='stopped_accounting_error', job_id=job_id)
            parsed, control = accounting_capture(output, receipt, job_id, 'final')
            require(control['returncode'] == 0, 'Current terminal native scontrol is unavailable; preserve incomplete settlement')
            terminal = validate_terminal_accounting(parsed['allocation_row'], control['stdout'])
            probes, facts = path_facts(job_id)
            facts['no_batch_step'] = parsed['no_batch_step']
            history = predecessor_history(args, output, receipt, registered, refresh=False)
            cumulative = PRIOR_SECONDS + sum(row['actual_a100_gpu_seconds'] for row in history) + terminal['actual_a100_gpu_seconds']
            receipt.update(terminal_accounting=terminal, complete_accounting=parsed,
                           terminal_scontrol=control['stdout'],
                           terminal_scontrol_sha256=hashlib.sha256(control['stdout'].encode()).hexdigest(),
                           path_probes=probes, four_facts=facts, predecessor_history=history,
                           prior_a100_seconds=PRIOR_SECONDS, cumulative_a100_seconds=cumulative,
                           cumulative_a100_hours=cumulative/3600, total_cap_a100_seconds=TOTAL_CAP)
            require(cumulative <= TOTAL_CAP, 'Settled cumulative A100 allocation exceeds four hours')
            status.update(status='passed', native_state=terminal['native_state'], restarts=0,
                          actual_a100_gpu_seconds=terminal['actual_a100_gpu_seconds'],
                          cumulative_a100_seconds=cumulative, cumulative_a100_hours=cumulative/3600,
                          before_entry_failure_proven=all(facts.values()))
        code = 0
    except Exception as error:
        status.update(error_type=type(error).__name__, error=str(error))
        receipt['traceback'] = traceback.format_exc()
    finish(output, filename, receipt, status)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'submit', 'settle'), required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--attempt', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--submission', type=Path)
    parser.add_argument('--predecessor-settlements', type=Path, nargs='*', default=[])
    return execute(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
