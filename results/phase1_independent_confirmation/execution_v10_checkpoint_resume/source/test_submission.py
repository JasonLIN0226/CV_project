"""Submission software fixtures: no subprocess, allocation, or scientific data."""
from types import SimpleNamespace
import json
import pytest
import orchestrator


def fixture_root(tmp_path,monkeypatch,run):
    monkeypatch.setattr(orchestrator,'ROOT',tmp_path)
    monkeypatch.setattr(orchestrator,'startup_require_live',lambda *_:None)
    monkeypatch.setattr(orchestrator,'source_check',lambda: ({},{'files':{}}))
    (tmp_path/'source_lock.json').write_text('{}')
    state={'GPU_job_ids':[]}
    class Guard:
        def __init__(self,_):self.state=state
        def check(self):return {'remaining_cap_seconds':15669}
        def save(self):(tmp_path/'ledger.json').write_text(json.dumps(state))
    monkeypatch.setattr(orchestrator,'SequenceGuard',Guard)
    monkeypatch.setattr(orchestrator.subprocess,'run',run)
    return state


def test_one_registered_gpu_receipt_and_parent_dependency(tmp_path,monkeypatch):
    calls=[]
    def fake(args,**kwargs):calls.append(args);return SimpleNamespace(stdout='99001;fixture-cluster\n')
    state=fixture_root(tmp_path,monkeypatch,fake)
    receipt=orchestrator.submit('train_gpu',parent=123)
    assert receipt['job']==99001 and '--dependency=afterok:123' in calls[0]
    assert '--no-requeue' in calls[0]
    assert '--gres=gpu:1' in calls[0] and state['GPU_job_ids']==[99001]
    with pytest.raises(FileExistsError):orchestrator.submit('train_gpu',parent=123)
    assert len(calls)==1


def test_unknown_submission_outcome_cannot_be_retried(tmp_path,monkeypatch):
    calls=[]
    def fake(args,**kwargs):calls.append(args);raise RuntimeError('Fixture unknown transport outcome')
    fixture_root(tmp_path,monkeypatch,fake)
    with pytest.raises(RuntimeError):orchestrator.submit('train_gpu',parent=123)
    with pytest.raises(FileExistsError):orchestrator.submit('train_gpu',parent=123)
    assert len(calls)==1
    assert not (tmp_path/'receipts/train_gpu_submission.json').exists()


def test_unknown_stage_never_allocates(tmp_path,monkeypatch):
    fixture_root(tmp_path,monkeypatch,lambda *a,**k:pytest.fail('No unknown-stage allocation'))
    with pytest.raises(ValueError,match='Unknown current'):orchestrator.submit('preflight_gpu')


@pytest.mark.parametrize('completed_stage', ['encode_train', 'fit_cpu'])
def test_completed_stages_cannot_be_submitted_again(tmp_path, monkeypatch, completed_stage):
    fixture_root(tmp_path, monkeypatch, lambda *a, **k: pytest.fail('Completed stage replay forbidden'))
    with pytest.raises(ValueError, match='Unknown current'):
        orchestrator.submit(completed_stage)
