"""Synthetic fixtures only: no research cache, fresh9137, or scientific training."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from budget_guard import SequenceGuard, BudgetStop, charged_seconds, CAP, PRIOR
from kernel import fit, predict
from core import maps
from locked_inference import MotionInference


def test_charged_allocation_and_complete_scope_guard():
    base = dict(job=9000,state='RUNNING',exitcode='0:0',elapsed=100,
        start='2026-10-19T00:00:00',end='Unknown',account='csci_ga_2271-2026fa',tres={'gres/gpu':'1'})
    from budget_guard import start_epoch
    now=start_epoch(base['start'])+103.4
    assert charged_seconds([base],now)==(104.,[9000])
    # A failed allocation with zero useful scientific work still spends GPU time.
    failed={**base,'job':8999,'state':'FAILED','elapsed':312,'end':'2026-10-19T00:05:12'}
    assert charged_seconds([failed],now)==(312.,[])
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'sequence_budget.json'
        state=dict(prior_settled_seconds=PRIOR,cap_seconds=CAP,total_frames=198656,
            native_prior_verified=True,GPU_job_ids=[9000],epochs={},encoded_frames=0)
        p.write_text(json.dumps(state))
        guard=SequenceGuard(d,query=lambda _: [base],clock=lambda:now)
        report=guard.check()
        assert report['fits'] and report['actual_GPU_seconds']==PRIOR+104
        guard.record_epoch('motion',17,'candidate',1,3.6)
        before=guard.remaining_raw()['training']
        try:guard.record_epoch('motion',17,'candidate',3,3.6)
        except ValueError:pass
        else:raise AssertionError('Skipped epoch accepted')
        guard.state['encoder_frames_per_second']=1.
        try:guard.check()
        except BudgetStop:pass
        else:raise AssertionError('Whole-scope budget overflow accepted')
        assert (Path(d)/'budget_stop.json').exists()
        assert before>0


def test_pending_native_row_retains_empty_AllocTRES(monkeypatch):
    import budget_guard
    from types import SimpleNamespace
    row='99002|PENDING|0:0|0|Unknown|Unknown|csci_ga_2271-2026fa|\n'
    monkeypatch.setattr(budget_guard.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=row))
    rows=budget_guard.native_allocations([99002])
    assert len(rows)==1 and rows[0]['tres']=={} and rows[0]['state']=='PENDING'
    assert charged_seconds(rows,0)==(0.,[])


def test_exact_fixed_mixture_and_position_invariance():
    # The dimensions match the scientific contract; random synthetic assets only.
    rng=np.random.default_rng(118)
    q=np.linalg.qr(rng.normal(size=(5120,128)))[0]
    A=q[:,:2]
    class Decoder:
        weight=np.column_stack([rng.normal(size=5120),A])
        std=np.ones(5120)
        mean=np.zeros(5120)
        bias=np.zeros(3)
        def predict(self,x):return x@self.weight
    from null_basis_original import null_basis
    n=null_basis(q,A)
    x=rng.normal(size=(40,2));y=rng.normal(size=(40,126))
    import torch
    f=fit(torch.tensor(x),torch.tensor(y),scientific=False);f['N']=n
    decoder=Decoder();inference=MotionInference({'output_basis':q},decoder,{3:f,6:f})
    p=rng.normal(size=(2,5120));hist=rng.normal(size=(2,5120))
    sample={'current':rng.normal(size=5120)};geo=rng.normal(size=(2,2))
    out=inference.correct(p,hist,sample,True,geo)
    B,a,b,right,_=maps(q,decoder.weight,decoder.std,decoder.mean,decoder.bias)
    expected=hist+(geo-(hist@a+b))@right@B.T
    direct=expected.copy()
    for i,h in enumerate((3,6)):
        innovation=(sample['current']-expected[i])@n+predict(geo[i][None],f)[0]
        direct[i]+=innovation@n.T
    expected=.5*expected+.5*direct
    assert np.max(abs(out-expected))<1e-11
    assert np.max(abs(out@a+b-geo))<1e-11
    off=inference.correct(p,hist,sample,False,geo)
    assert np.max(abs(off@a+b-p@a-b))<1e-11


if __name__=='__main__':
    test_charged_allocation_and_complete_scope_guard()
    test_exact_fixed_mixture_and_position_invariance()
    print('PASS_SOFTWARE_ONLY: shared-budget/failure-charge/whole-scope-stop/epoch-boundary/fixed-mixture/position-invariance')
