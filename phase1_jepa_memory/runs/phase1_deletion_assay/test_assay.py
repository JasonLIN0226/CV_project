import ast
from pathlib import Path
import numpy as np
import pytest
from assay_core import pools,forks,effect,summarize

def test_legal_equal_capacity_no_reentry():
    for task,ev in [('gate',[4]),('motion',[4,8])]:
        for _,parent in pools(task,ev,23):
            assert max(parent)<20 and len(parent)==3
            for deleted,kept in forks(parent):assert len(kept)==2 and deleted not in kept

def test_gate_nuisance_excludes_all_frames_of_multiframe_cue():
    for _,parent in pools('gate',[4],23,evidence_end=6):
        assert 4 in parent and not {5,6}.intersection(parent)
        assert all(t>6 for t in parent if t!=4)

@pytest.mark.parametrize('task,ev,q',[('motion',[4,4],20),('motion',[4,17],20),('gate',[18],20),('gate',[2,3],20)])
def test_illegal_timestamps_fail(task,ev,q):
    with pytest.raises(ValueError):pools(task,ev,q)

def test_zero_denominator_and_negative_signal_cannot_pass():
    assert not effect(np.zeros(128),np.zeros(128))['passed']
    assert not effect(np.ones(128),np.ones(128)*2)['passed']
    assert effect(np.ones(128)*2,np.ones(128))['passed']

def test_model_input_excludes_truth_and_future_feature():
    tree=ast.parse((Path(__file__).parent/'run_assay.py').read_text())
    definition=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='sample')
    ns={'np':np};exec(compile(ast.Module(body=[definition],type_ignores=[]),'<sample>','exec'),ns)
    z=np.arange(30*8,dtype=np.float32).reshape(30,8);p=z[:,:3];a=np.ones((29,2),np.float32)
    x=ns['sample'](z,p,a,20,(2,10));changed=z.copy();changed[21:]=1e9
    y=ns['sample'](changed,p,a,20,(2,10))
    assert set(x)=={'z','projected_z','projected_current','current','age','type','past_actions','actions'}
    assert all(np.array_equal(x[k],y[k]) for k in x)
    with pytest.raises(AssertionError):ns['sample'](z,p,a,20,(10,18))

def test_full_fixture_positive_and_scope_flags():
    rows=[]
    for task in ('gate','motion'):
     for seed in (17,29,43):
      for split in ('test_id','test_long_gap','test_distractor_shift'):
       for pair in range(128):
        for episode in (0,1):
         for pool in ('arrival','early_nuisance'):
          for deleted in (1,2,3):
           role=('cue' if deleted==1 else 'nuisance') if task=='gate' else {1:'start',2:'end',3:'nuisance'}[deleted]
           for h in (3,6):
            rows.append(dict(task=task,seed=seed,split=split,pair=str(pair),episode=str(episode),pool=pool,deleted=deleted,
                horizon=h,deleted_role=role,history_count=2,kept=';'.join(str(i) for i in (1,2,3) if i!=deleted),
                feature_mse=1. if role=='nuisance' else 2.,task_error=.1 if role=='nuisance' else .4))
    summary,_=summarize(rows)
    assert summary['decision']=='READY_FOR_ID_BOUNDED_POLICY_PILOT'
    assert not summary['original_phase1_go'] and not summary['policy_learnability_proven']
    with pytest.raises(ValueError):summarize(rows[:-1])
