"""Fail-closed launch/resource/source checks without any actual model data."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

path=Path(__file__).parents[1]/'scripts/v5_run_frozen_real_diagnostics.py'
spec=importlib.util.spec_from_file_location('diagnostic_launch',path)
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)


def test_unallocated_or_wrong_course_refused_before_source_output_or_import(monkeypatch,tmp_path):
    monkeypatch.delenv('SLURM_JOB_ACCOUNT',raising=False)
    with pytest.raises(ValueError,match='allocated'):launch.execute(tmp_path,'0'*64,tmp_path/'output')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('changed',[
    'cpu=24,mem=60G|02:00:00',
    'cpu=24,mem=16G,gres/gpu=1|02:00:00',
    'cpu=24,mem=16G|04:00:00'])
def test_declared_diagnostic_budget_cannot_silently_expand(monkeypatch,tmp_path,changed):
    for key,value in {'SLURM_JOB_ID':'fake','SLURM_JOB_ACCOUNT':'csci_ga_2271-2026fa',
                      'SLURM_JOB_PARTITION':'n2c48m24','SLURM_CPUS_PER_TASK':'4'}.items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(launch.socket,'gethostname',lambda:'b-fixture')
    monkeypatch.setattr(launch.subprocess,'check_output',lambda *a,**k:'fake|csci_ga_2271-2026fa|n2c48m24|'+changed+'|\n')
    with pytest.raises(ValueError,match='resources differ'):launch.execute(tmp_path,'0'*64,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_changed_or_symlink_source_cannot_execute_under_same_lock(tmp_path):
    data=tmp_path/'source.py';data.write_text('original')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    lock=tmp_path/'source_lock.json';lock.write_text(json.dumps({'files':{'source.py':digest(data)}}))
    expected=digest(lock);launch.locked(tmp_path,expected)
    data.write_text('changed')
    with pytest.raises(ValueError):launch.locked(tmp_path,expected)
    data.unlink();real=tmp_path/'other.py';real.write_text('original');data.symlink_to(real)
    with pytest.raises(ValueError):launch.locked(tmp_path,expected)
