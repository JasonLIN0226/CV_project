"""Allocation/provenance boundary fixtures; no Slurm, fresh data, or model calls."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import budget_guard
import startup_client


def test_actual_57_attempt_accounting_is_not_collapsed(monkeypatch):
    rows=json.loads((Path(__file__).parent/'prior_budget_boundary.json').read_text())['rows']
    raw='\n'.join('|'.join(str(r[k]) for k in ['job','state','exitcode','elapsed_seconds','start_UTC','end_UTC','account','AllocTRES']) for r in rows)
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=raw)
    monkeypatch.setattr(budget_guard.subprocess,'run',run)
    result=budget_guard.native_allocations([8536])
    assert len(result)==1 and result[0]['attempt_count']==57
    assert result[0]['elapsed']==26520 and result[0]['state']=='CANCELLED'
    assert budget_guard.charged_seconds(result,0)==(26520.,[])
    assert '-D' in calls[0] and '-u' in calls[0] and '-A' in calls[0]
    assert calls[0][calls[0].index('-S')+1]=='2026-10-01'


def test_requeue_then_pending_cannot_hide_allocation_time(monkeypatch):
    raw=('99002|REQUEUED|0:0|180|2026-10-19T05:00:00|2026-10-19T05:03:00|csci_ga_2271-2026fa|gres/gpu=1\n'
         '99002|PENDING|0:0|0|Unknown|Unknown|csci_ga_2271-2026fa|\n')
    monkeypatch.setattr(budget_guard.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=raw))
    result=budget_guard.native_allocations([99002])
    assert budget_guard.charged_seconds(result,0)==(180.,[])


def test_reused_old_job_id_is_rejected(monkeypatch):
    raw='99002|COMPLETED|0:0|10|2024-10-19T05:00:00|2024-10-19T05:00:10|csci_ga_2271-2026fa|gres/gpu=1\n'
    monkeypatch.setattr(budget_guard.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=raw))
    with pytest.raises(ValueError,match='historical JobID'):
        budget_guard.native_allocations([99002])


def test_stale_cpu_heartbeat_forbids_gpu_work(tmp_path,monkeypatch):
    (tmp_path/'watchdog_submission.json').write_text(json.dumps(dict(job=1,source_lock_sha256='fixture')))
    (tmp_path/'watchdog_status.json').write_text(json.dumps(dict(job=1,source_lock_sha256='fixture',updated_epoch=1)))
    monkeypatch.setattr(startup_client,'native_job',lambda _:pytest.fail('Expired heartbeat must fail before native lookup'))
    with pytest.raises(ValueError,match='heartbeat'):
        startup_client.require_live(tmp_path,'fixture',clock=lambda:30)


def test_both_cpu_heartbeats_required_before_gpu(tmp_path,monkeypatch):
    (tmp_path/'watchdog_submission.json').write_text(json.dumps(dict(job=1,source_lock_sha256='fixture')))
    (tmp_path/'watchdog_status.json').write_text(json.dumps(dict(job=1,source_lock_sha256='fixture',updated_epoch=100)))
    (tmp_path/'watchdog_cleanup_submission.json').write_text(json.dumps(dict(job=2,source_lock_sha256='fixture')))
    (tmp_path/'guardian_peer_status.json').write_text(json.dumps(dict(job=2,source_lock_sha256='fixture',updated_epoch=50)))
    from cpu_incarnation import claim
    def cpu(job,restarts=0):return dict(JobId=str(job),NodeList='cpu-'+str(job),JobState='RUNNING',Account='csci_ga_2271-2026fa',Restarts=str(restarts),AllocTRES='cpu=1')
    claim(tmp_path,'primary',cpu(1,1),'fixture')
    claim(tmp_path,'peer',cpu(2),'fixture')
    monkeypatch.setattr(startup_client,'native_job',lambda job:cpu(job,1 if job==1 else 0))
    assert startup_client.require_live(tmp_path,'fixture',clock=lambda:100,require_peer=False)
    with pytest.raises(ValueError,match='Second independently'):
        startup_client.require_live(tmp_path,'fixture',clock=lambda:100)
    (tmp_path/'guardian_peer_status.json').write_text(json.dumps(dict(job=2,source_lock_sha256='fixture',updated_epoch=100)))
    assert startup_client.require_live(tmp_path,'fixture',clock=lambda:100)


def test_restarted_cpu_guardian_cannot_resume_a_gpu_sequence(tmp_path,monkeypatch):
    (tmp_path/'watchdog_submission.json').write_text(json.dumps(dict(job=1,source_lock_sha256='fixture')))
    (tmp_path/'watchdog_status.json').write_text(json.dumps(dict(job=1,source_lock_sha256='fixture',updated_epoch=100)))
    from cpu_incarnation import claim
    original=dict(JobId='1',NodeList='cpu-1',JobState='RUNNING',Account='csci_ga_2271-2026fa',Restarts='0',AllocTRES='cpu=1')
    claim(tmp_path,'primary',original,'fixture')
    monkeypatch.setattr(startup_client,'native_job',lambda _:original|{'Restarts':'1'})
    with pytest.raises(ValueError,match='incarnation changed'):
        startup_client.require_live(tmp_path,'fixture',clock=lambda:100,require_peer=False)
