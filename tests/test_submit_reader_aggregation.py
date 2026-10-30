"""Local-only submission/packaging tests; scheduler and CUDA calls are mocked."""
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from experiments.phase2 import submit_reader_aggregation as s


def source_zip(path, extra=None):
    files={s.CONFIG:'{}',s.RUNNER:'print("runner")',s.WRAPPER:'print("wrapper")'}
    manifest={'files':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()}}
    with zipfile.ZipFile(path,'w') as z:
        for k,v in files.items():z.writestr(k,v)
        z.writestr('bundle_manifest.json',json.dumps(manifest))
        if extra is not None:z.writestr(*extra)
    return path


@pytest.mark.parametrize('name',['../escape','/absolute','a/../../b','a\\b','C:/b','a//b','a/./b'])
def test_archive_rejects_unsafe_paths_before_extracting(tmp_path,name):
    archive=source_zip(tmp_path/'source.zip',(name,'bad'))
    with pytest.raises(ValueError,match='Unsafe'):
        s.extract_verified(archive,tmp_path/'bundle')
    assert not (tmp_path/'bundle').exists()


def test_archive_rejects_symlink_and_unmanifested_source(tmp_path):
    archive=source_zip(tmp_path/'link.zip')
    entry=zipfile.ZipInfo('link');entry.create_system=3
    entry.external_attr=(stat.S_IFLNK | 0o777)<<16
    with zipfile.ZipFile(archive,'a') as z:z.writestr(entry,'../../outside')
    with pytest.raises(ValueError,match='Nonregular'):
        s.extract_verified(archive,tmp_path/'linked')
    archive=source_zip(tmp_path/'extra.zip',('unbound.py','print("unbound")'))
    with pytest.raises(ValueError,match='cover every file'):
        s.extract_verified(archive,tmp_path/'extra')


def test_manifest_detects_mutation_and_preserves_original_directory(tmp_path):
    archive=source_zip(tmp_path/'source.zip');bundle=tmp_path/'bundle'
    s.extract_verified(archive,bundle)
    (bundle/s.RUNNER).write_text('mutated')
    with pytest.raises(ValueError,match='mismatch'):
        s.validate_bundle(bundle)
    with pytest.raises(ValueError,match='must be new'):
        s.extract_verified(archive,bundle)
    assert (bundle/s.RUNNER).read_text()=='mutated'


def test_package_preserves_omitted_data_and_marks_log_nonfinal(tmp_path):
    work=tmp_path/'work';work.mkdir();run=work/'run';run.mkdir();bundle=work/'bundle';bundle.mkdir()
    (run/'features.npy').write_bytes(b'large-derived-array')
    (run/'study_status.json').write_text('{"status":"diagnostic"}')
    (bundle/'source.py').write_text('source')
    (work/'execution.json').write_text('{"runner_returncode":0}')
    (work/'slurm-1.log').write_text('partial log\n')
    receipt=s.package(work,bundle,run)
    assert (run/'features.npy').read_bytes()==b'large-derived-array'
    with zipfile.ZipFile(receipt['path']) as z:
        m=json.loads(z.read('package_manifest.json'))
        assert 'run/features.npy' not in z.namelist()
        assert m['derived_data_preserved_on_scratch']['run/features.npy']['sha256']==s.sha(run/'features.npy')
        assert m['slurm_log_snapshots'][0]['final_log'] is False
        assert m['slurm_log_snapshots'][0]['truncated_or_incomplete'] is True
        assert z.read('execution/slurm-1.log.snapshot')==b'partial log\n'
        assert z.testzip() is None
    with pytest.raises(ValueError,match='already exists'):
        s.package(work,bundle,run)


def setup_submit(tmp_path,monkeypatch):
    base=tmp_path/'base';(base/'venv/bin').mkdir(parents=True)
    (base/'venv/bin/python').write_text('pinned interpreter')
    (base/'cache/ijepa').mkdir(parents=True)
    source=base/'prior/run';source.mkdir(parents=True)
    monkeypatch.setattr(s,'BASE',base);monkeypatch.setattr(s,'SOURCE',source)
    monkeypatch.setattr(s,'WORK',base/'new-study')
    return source_zip(tmp_path/'source.zip')


def test_duplicate_active_job_is_blocked_before_any_output_directory(tmp_path,monkeypatch):
    archive=setup_submit(tmp_path,monkeypatch)
    monkeypatch.setattr(s.subprocess,'check_output',lambda *a,**k:'11441 RUNNING\n')
    monkeypatch.setattr(s.subprocess,'run',lambda *a,**k:pytest.fail('Must not submit a duplicate'))
    with pytest.raises(RuntimeError,match='active job'):
        s.submit(archive,s.sha(archive))
    assert not s.WORK.exists()


def test_submit_uses_locked_pool_two_hours_and_exclusive_work(tmp_path,monkeypatch):
    archive=setup_submit(tmp_path,monkeypatch);calls=[]
    monkeypatch.setattr(s.subprocess,'check_output',lambda *a,**k:'')
    def fake_run(cmd,**kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd,0,'9876\n','')
    monkeypatch.setattr(s.subprocess,'run',fake_run)
    receipt=s.submit(archive,s.sha(archive))
    assert receipt['status']=='submitted' and receipt['job_id']=='9876'
    assert len(calls)==1
    assert '--time=02:00:00' in calls[0] and '--partition=c12m85-a100-1' in calls[0]
    assert '--account=csci_ga_2271-2026fa' in calls[0]
    with pytest.raises(FileExistsError):s.submit(archive,s.sha(archive))
    assert len(calls)==1


def test_cuda_preflight_failure_still_packages_receipts(tmp_path,monkeypatch):
    work=tmp_path/'work';work.mkdir();bundle=work/'bundle'
    s.extract_verified(source_zip(tmp_path/'source.zip'),bundle)
    fake=SimpleNamespace(__version__='test',cuda=SimpleNamespace(is_available=lambda:False))
    monkeypatch.setitem(sys.modules,'torch',fake)
    with pytest.raises(RuntimeError,match='Allocated CUDA'):
        s.execute(bundle,work/'run')
    execution=json.loads((work/'execution.json').read_text())
    assert execution['status']=='execution_failed' and execution['commands']==[]
    with zipfile.ZipFile(work/'reader-aggregation-results.zip') as z:
        assert 'execution/execution.json' in z.namelist()
        assert 'execution/hardware.json' in z.namelist()


def test_failed_runner_records_one_returncode_and_its_scientific_status(tmp_path,monkeypatch):
    work=tmp_path/'work';work.mkdir();bundle=work/'bundle';run=work/'run'
    s.extract_verified(source_zip(tmp_path/'source.zip'),bundle)
    fake=SimpleNamespace(__version__='test',cuda=SimpleNamespace(is_available=lambda:True,
        get_device_name=lambda i:'NVIDIA A100-SXM4-40GB'))
    monkeypatch.setitem(sys.modules,'torch',fake)
    monkeypatch.setattr(s.subprocess,'check_output',lambda *a,**k:'A100, GPU-test, 40960 MiB\n')
    calls=[]
    def failed_runner(cmd,**kwargs):
        calls.append(cmd);run.mkdir()
        (run/'study_status.json').write_text('{"status":"failed_after_training","confirmation_opened":false}')
        return subprocess.CompletedProcess(cmd,7)
    monkeypatch.setattr(s.subprocess,'run',failed_runner)
    with pytest.raises(RuntimeError,match='runner failed'):
        s.execute(bundle,run)
    r=json.loads((work/'execution.json').read_text())
    assert r['runner_returncode']==7 and len(r['commands'])==1 and len(calls)==1
    assert r['study_status']['status']=='failed_after_training'
    assert '--source' in calls[0] and str(s.SOURCE) in calls[0]
    assert r['confirmation_opened'] is False
    assert (work/'reader-aggregation-results.zip').is_file()
