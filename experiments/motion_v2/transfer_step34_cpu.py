"""One-shot byte-only transfer of the registered 384 validation NPZ files.

The outer ZIP is decoded only into opaque NPZ bytes. No NPZ member, image,
array, model, or scientific runner is opened. Existing metadata is verified
without replacement; a failed attempt and any partial files are retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import traceback
import zipfile


sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/scratch/xy3257/motion-v2-20261101-step34-deployment1')
BRANCH = 'experiments/motion-v2-20261030'
EXECUTION = 'experiments/motion_v2/protocol_step34_execution.json'
HELPER = 'experiments/motion_v2/transfer_step34_cpu.py'
CPU_PREFIX = 'results/motion_v2/step3_4/cpu'
NPZ_COUNT = 384
FILE_COUNT = 391
METADATA = frozenset(('SHA256SUMS', 'artifact_sha256.json', 'audit.json',
                      'data/dataset/manifest.json', 'data/generation_receipt.json',
                      'data/resolved_config.json', 'status.json'))
NPZ_PATTERN = re.compile(r'data/dataset/(observations|evaluation_only)/validation/([0-9a-f]{24})\.npz')


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def regular_path(path, parent=None, *, must_exist=True):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'An absolute path without traversal is required')
    if parent is not None:
        require(path.is_relative_to(parent), 'Path escaped its registered root')
    for component in (path, *path.parents):
        require(not component.is_symlink(), 'Symlink paths are forbidden: ' + str(component))
    if must_exist:
        require(path.exists(), 'Required path is missing: ' + str(path))
    return path


def require_absent(path):
    regular_path(path, REPO, must_exist=False)
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return
    raise FileExistsError('A target already exists; never overwrite or resume: ' + str(path))


def validate_inventory(mapping):
    """Validate the exact registration schema, without filesystem access."""
    require(isinstance(mapping, dict) and len(mapping) == FILE_COUNT, 'Expected exactly391 registered CPU files')
    blobs, metadata = {}, {}
    ids = {'observations': set(), 'evaluation_only': set()}
    for relative, checksum in mapping.items():
        require(isinstance(relative, str) and isinstance(checksum, str)
                and re.fullmatch(r'[0-9a-f]{64}', checksum), 'Invalid CPU inventory identity or SHA256')
        match = NPZ_PATTERN.fullmatch(relative)
        if match:
            blobs[CPU_PREFIX + '/' + relative] = checksum
            ids[match.group(1)].add(match.group(2))
        else:
            require(relative in METADATA, 'Unregistered CPU metadata or forbidden data path')
            metadata[relative] = checksum
    require(len(blobs) == NPZ_COUNT and set(metadata) == METADATA, 'Expected384 NPZ plus7 metadata files')
    require(len(ids['observations']) == 192 and ids['observations'] == ids['evaluation_only'],
            'Observation/evaluation NPZ identities must match all192 validation episodes')
    return blobs, metadata


def validate_archive_bytes(payload, allowed):
    """Return fully validated opaque NPZ bytes; perform no filesystem writes.

    Production first locks the outer ZIP SHA/size and validates the384-entry
    registration. Smaller artificial allowlists support isolated software tests.
    """
    require(isinstance(payload, bytes) and isinstance(allowed, dict) and bool(allowed), 'ZIP bytes and a nonempty allowlist required')
    for name, checksum in allowed.items():
        require(isinstance(name, str) and name.startswith(CPU_PREFIX + '/'), 'Allowlist is outside CPU receipt')
        require(NPZ_PATTERN.fullmatch(name[len(CPU_PREFIX) + 1:]) is not None,
                'Only validation observation/evaluation NPZ entries are allowed')
        require(isinstance(checksum, str) and re.fullmatch(r'[0-9a-f]{64}', checksum), 'Invalid allowed entry hash')
    result = {}
    with zipfile.ZipFile(io.BytesIO(payload), 'r') as archive:
        infos = archive.infolist()
        require(len(infos) == len(allowed), 'ZIP entry count differs from the registered NPZ count')
        require(len({info.filename for info in infos}) == len(infos), 'Duplicate ZIP member names are forbidden')
        require({info.filename for info in infos} == set(allowed), 'ZIP members differ from the exact preregistered allowlist')
        for info in infos:
            name = info.filename
            require(name == info.orig_filename and '\x00' not in name and '\\' not in name
                    and not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts,
                    'Unsafe ZIP member path')
            require(not info.is_dir() and not info.external_attr & 0x10
                    and stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG),
                    'ZIP directory/symlink/special entries are forbidden')
            require(not info.flag_bits & 1, 'Encrypted ZIP entries are forbidden')
            value = archive.read(info)  # Opaque .npz bytes; no nested ZIP/NumPy decoding.
            require(len(value) == info.file_size and sha_bytes(value) == allowed[name], 'ZIP member byte hash differs: ' + name)
            result[name] = value
    return result


def validate_existing_members(names, metadata):
    require(set(names) == set(metadata), 'Before transfer the CPU subtree must contain only its7 registered metadata files')


def git(argv, receipt):
    result = subprocess.run(['git', *argv], cwd=REPO, capture_output=True, timeout=60)
    receipt.setdefault('git_commands', []).append({'argv': ['git', *argv], 'returncode': result.returncode,
                                                  'stderr': result.stderr.decode(errors='replace')})
    require(result.returncode == 0, 'Read-only git command failed: ' + repr(argv))
    return result.stdout


def committed(path, receipt):
    path = regular_path(path, REPO)
    relative = path.relative_to(REPO).as_posix()
    require(git(['show', 'HEAD:' + relative], receipt) == path.read_bytes(), 'File differs from committed HEAD: ' + relative)
    return path


def existing_files(root):
    result = set()
    for path in root.rglob('*'):
        regular_path(path, root)
        if path.is_file():
            result.add(path.relative_to(root).as_posix())
        else:
            require(path.is_dir(), 'Unexpected special filesystem entry')
    return result


def execute(args):
    require(REPO == ROOT / 'repo', 'Run only in the registered deployment checkout')
    require(sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1, 'Invoke with Python -I -B')
    require(re.fullmatch(r'[0-9a-f]{40}', args.expected_head) is not None, 'Exact pushed40-character HEAD required')
    require(args.archive.is_absolute(), '--archive must be absolute')
    output = regular_path(args.receipt, ROOT, must_exist=False)
    require(output == ROOT / 'transfer1' and not output.exists(), 'Use the one new registered transfer1 receipt directory')
    output.mkdir(exist_ok=False)
    receipt = {'started_utc': utc(), 'operation': 'opaque_cpu_npz_transfer', 'archive': str(args.archive),
               'npz_write_attempts': [], 'npz_files_written': [], 'arrays_decoded': False,
               'scientific_generation_or_execution': False, 'files_deleted_or_overwritten': False,
               'retry_or_resume_permitted': False, 'gpu_seconds': 0}
    status = {'status': 'stopped', 'gpu_seconds': 0, 'arrays_decoded': False, 'retry_permitted': False}
    handlers = {}
    code = 2
    write_json(output / 'started.json', {'started_utc': receipt['started_utc'], 'expected_head': args.expected_head,
                                       'archive': str(args.archive), 'one_shot_transfer': True})
    try:
        def interrupted(signum, frame):
            raise RuntimeError('Transfer interrupted by signal' + str(signum))
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupted)
        head = git(['rev-parse', 'HEAD'], receipt).decode().strip()
        branch = git(['branch', '--show-current'], receipt).decode().strip()
        require(head == args.expected_head and branch == BRANCH, 'Deployment HEAD/branch mismatch')
        execution_path = committed(REPO / EXECUTION, receipt)
        execution = json.loads(execution_path.read_text(encoding='utf-8'))
        require(execution.get('status') == 'FROZEN_STEP34_EXECUTION' and execution.get('test_status') == 'passed',
                'Committed passed execution registration required')
        transport = execution['transport_helper']
        helper = committed(Path(__file__).resolve(), receipt)
        require(transport.get('path') == HELPER and helper == REPO / HELPER
                and sha_file(helper) == transport.get('sha256'), 'This transfer helper is not registered by SHA256')
        require(execution.get('deployment_root') == str(ROOT) and execution['cpu_receipt']['path'] == CPU_PREFIX,
                'Registered deployment/CPU receipt path differs')
        mapping = execution['cpu_receipt']['artifact_sha256']
        allowed, metadata = validate_inventory(mapping)
        transfer = execution['cpu_npz_transfer']
        require(transfer.get('npz_files') == NPZ_COUNT and type(transfer.get('bytes')) is int
                and transfer['bytes'] > 0 and re.fullmatch(r'[0-9a-f]{64}', transfer.get('sha256', '')),
                'Invalid registered ZIP size/count/SHA256')
        archive_path = regular_path(args.archive)
        require(archive_path.is_file(), 'Archive is not a regular file')
        require(archive_path.stat().st_size == transfer['bytes'], 'Outer ZIP byte size differs')
        payload = archive_path.read_bytes()
        require(len(payload) == transfer['bytes'] and sha_bytes(payload) == transfer['sha256'], 'Outer ZIP SHA256/size differs')
        validated = validate_archive_bytes(payload, allowed)
        receipt.update(head_commit=head, branch=branch, execution_protocol_sha256=sha_file(execution_path),
                       helper_sha256=sha_file(helper), archive_sha256=sha_bytes(payload), archive_bytes=len(payload),
                       archive_members_fully_verified=len(validated), archive_member_sha256=allowed)
        cpu_root = regular_path(REPO / CPU_PREFIX, REPO)
        require(cpu_root.is_dir(), 'CPU receipt metadata directory is missing')
        validate_existing_members(existing_files(cpu_root), metadata)
        for relative, checksum in metadata.items():
            path = committed(cpu_root / relative, receipt)
            require(path.is_file() and sha_file(path) == checksum, 'Existing CPU metadata hash differs: ' + relative)
        cpu_status = json.loads((cpu_root / 'status.json').read_text())
        cpu_audit = json.loads((cpu_root / 'audit.json').read_text())
        require(cpu_status.get('status') == 'passed' and cpu_audit.get('passed') is True, 'Original CPU run did not pass')
        for name in validated:
            require_absent(REPO / name)
        # Every member and every target has passed before the first file/parent is created.
        receipt['all_member_hashes_and_target_absence_verified_before_first_write'] = True
        write_json(output / 'before_write.json', {key: value for key, value in receipt.items()
                                                if key not in ('npz_files_written', 'npz_write_attempts')})
        for name, value in sorted(validated.items()):
            path = REPO / name
            require_absent(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            regular_path(path.parent, REPO)
            receipt['npz_write_attempts'].append(name)
            with path.open('xb') as stream:
                stream.write(value)
            require(sha_file(path) == allowed[name], 'Written target hash differs: ' + name)
            receipt['npz_files_written'].append(name)
        require(existing_files(cpu_root) == set(mapping), 'Final CPU subtree file inventory differs')
        verified = {}
        for relative, checksum in mapping.items():
            path = regular_path(cpu_root / relative, cpu_root)
            actual = sha_file(path)
            require(actual == checksum, 'Final391-file receipt hash mismatch: ' + relative)
            verified[relative] = actual
        receipt['complete_cpu_artifact_sha256'] = verified
        status.update(status='passed', npz_files_written=NPZ_COUNT, cpu_files_verified=FILE_COUNT,
                      archive_sha256=transfer['sha256'], metadata_overwritten=False)
        code = 0
    except BaseException as error:
        receipt['traceback'] = traceback.format_exc()
        status.update(error_type=type(error).__name__, error=str(error),
                      npz_files_written=len(receipt['npz_files_written']),
                      npz_write_attempts=len(receipt['npz_write_attempts']))
    finally:
        receipt['finished_utc'] = utc()
        write_json(output / 'receipt.json', receipt)
        status['receipt_sha256'] = sha_file(output / 'receipt.json')
        write_json(output / 'status.json', status)
        with (output / 'SHA256SUMS').open('x', encoding='utf-8') as stream:
            for path in sorted(output.iterdir()):
                if path.is_file() and path.name != 'SHA256SUMS':
                    stream.write(sha_file(path) + '  ' + path.name + '\n')
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    print(json.dumps(status, ensure_ascii=False, allow_nan=False), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    return execute(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
