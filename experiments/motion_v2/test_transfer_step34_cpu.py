"""Artificial in-memory ZIP checks only; no real dataset or file writes."""
import hashlib
import importlib.util
import io
from pathlib import Path
import stat
import sys
import warnings
import zipfile

import pytest


@pytest.fixture(scope='module')
def transfer():
    source = Path(__file__).with_name('transfer_step34_cpu.py')
    spec = importlib.util.spec_from_file_location('motion_step34_cpu_transport_software_only', source)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = obj
    spec.loader.exec_module(obj)
    return obj


def artificial_members(transfer):
    prefix = transfer.CPU_PREFIX + '/data/dataset/'
    return {prefix + 'observations/validation/' + 'a' * 24 + '.npz': b'artificial opaque observation bytes',
            prefix + 'evaluation_only/validation/' + 'a' * 24 + '.npz': b'artificial opaque evaluator bytes'}


def hashes(values):
    return {name: hashlib.sha256(value).hexdigest() for name, value in values.items()}


def artificial_zip(values, *, duplicate=None, symlink=None, directory=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in values.items():
            if name == symlink:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, value)
            elif name == directory:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFDIR | 0o755) << 16
                archive.writestr(info, value)
            else:
                archive.writestr(name, value)
        if duplicate is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                archive.writestr(duplicate, values[duplicate])
    return buffer.getvalue()


def test_opaque_members_verified_without_decoding_npz(transfer):
    members = artificial_members(transfer)
    assert transfer.validate_archive_bytes(artificial_zip(members), hashes(members)) == members


def test_corrupt_last_member_rejects_entire_archive_before_return(transfer):
    members = artificial_members(transfer)
    expected = hashes(members)
    members[next(reversed(members))] = b'different opaque bytes'
    with pytest.raises(ValueError, match='hash differs'):
        transfer.validate_archive_bytes(artificial_zip(members), expected)


def test_duplicate_missing_extra_and_unregistered_metadata_rejected(transfer):
    members = artificial_members(transfer)
    expected = hashes(members)
    with pytest.raises(ValueError):
        transfer.validate_archive_bytes(artificial_zip(members, duplicate=next(iter(members))), expected)
    with pytest.raises(ValueError):
        transfer.validate_archive_bytes(artificial_zip({next(iter(members)): members[next(iter(members))]}), expected)
    changed = dict(members)
    changed[transfer.CPU_PREFIX + '/status.json'] = b'never overwrite metadata'
    with pytest.raises(ValueError):
        transfer.validate_archive_bytes(artificial_zip(changed), expected)


@pytest.mark.parametrize('kind', ['test', 'train', 'conf', 'traversal', 'absolute', 'backslash'])
def test_forbidden_paths_rejected_even_if_listed_by_caller(transfer, kind):
    name = next(iter(artificial_members(transfer)))
    if kind in ('test', 'train', 'conf'):
        name = name.replace('/validation/', '/' + kind + '/')
    elif kind == 'traversal':
        name = name.replace('/validation/', '/validation/../validation/')
    elif kind == 'absolute':
        name = '/' + name
    else:
        name = name.replace('/validation/', '\\validation\\')
    members = {name: b'artificial'}
    with pytest.raises(ValueError):
        transfer.validate_archive_bytes(artificial_zip(members), hashes(members))


@pytest.mark.parametrize('kind', ['symlink', 'directory'])
def test_zip_symlink_and_special_directory_members_rejected(transfer, kind):
    members = artificial_members(transfer)
    payload = artificial_zip(members, **{kind: next(iter(members))})
    with pytest.raises(ValueError, match='special entries'):
        transfer.validate_archive_bytes(payload, hashes(members))


def registered_artificial_inventory(transfer):
    result = {name: 'b' * 64 for name in transfer.METADATA}
    for index in range(192):
        identity = f'{index:024x}'
        for directory in ('observations', 'evaluation_only'):
            result[f'data/dataset/{directory}/validation/{identity}.npz'] = 'a' * 64
    return result


def test_exact391_inventory_with192_matching_episode_identities(transfer):
    inventory = registered_artificial_inventory(transfer)
    allowed, metadata = transfer.validate_inventory(inventory)
    assert len(allowed) == 384 and len(metadata) == 7
    assert all(name.startswith(transfer.CPU_PREFIX + '/') for name in allowed)


@pytest.mark.parametrize('kind', ['missing', 'extra', 'different_ids', 'bad_digest', 'metadata_path'])
def test_registration_inventory_mismatch_rejected(transfer, kind):
    inventory = registered_artificial_inventory(transfer)
    if kind == 'missing':
        inventory.pop('status.json')
    elif kind == 'extra':
        inventory['extra.json'] = 'a' * 64
    elif kind == 'different_ids':
        original = 'data/dataset/observations/validation/' + '0' * 24 + '.npz'
        inventory[original.replace('0' * 24, 'f' * 24)] = inventory.pop(original)
    elif kind == 'bad_digest':
        inventory['status.json'] = 'z' * 64
    else:
        inventory['../status.json'] = inventory.pop('status.json')
    with pytest.raises(ValueError):
        transfer.validate_inventory(inventory)


def test_only_existing_metadata_allowed_before_any_write(transfer):
    transfer.validate_existing_members(set(transfer.METADATA), transfer.METADATA)
    with pytest.raises(ValueError):
        transfer.validate_existing_members(set(transfer.METADATA) | {'data/dataset/observations/validation/' + 'a' * 24 + '.npz'}, transfer.METADATA)
    with pytest.raises(ValueError):
        transfer.validate_existing_members(set(transfer.METADATA) - {'audit.json'}, transfer.METADATA)
