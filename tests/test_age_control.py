"""Synthetic trace and independently reconstructed scalar fixtures; no science data."""
import copy,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import pytest

BASE=Path(__file__).resolve().parents[1]/'experiments/age_control'
def load(name):
    spec=importlib.util.spec_from_file_location('age_'+name,BASE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
primary=load('analysis');auditor=load('independent_audit');pools=load('counterbalanced_pools')


def fixture(n=4):
    rows=[]
    for task in ('gate','motion'):
     for seed in (17,29,43):
      for split in primary.SPLITS:
       q=40 if split=='test_long_gap' else 20
       ev=(4,) if task=='gate' else (4,8);end=6 if task=='gate' else 8
       for i in range(n):
        pair=f'pair{i:03}'
        if task=='gate':chosen=list(pools.gate_age_counterbalance(4,end,q).items())
        else:
         encoded=json.dumps((20261017,'clock_only',pair),sort_keys=True,separators=(',',':')).encode()
         rngseed=int(hashlib.sha256(encoded).hexdigest()[:16],16)%(2**32)
         chosen=[(f'clock_{j}',p) for j,p in enumerate(pools.clock_only_static_pools(q,rngseed))]
        for ep in ('a','b'):
         for name,parent in chosen:
          for j,deleted in enumerate(parent):
           role=('cue' if deleted==4 else 'other') if task=='gate' else ('start' if deleted==4 else ('end' if deleted==8 else 'other'))
           for h in (3,6):
            feature=(2. if role=='cue' else 1.) if task=='gate' else 1.+j*.2+h*.001
            state=feature/10 if task=='gate' else 1.-feature/10
            rows.append(dict(task=task,seed=seed,split=split,pair=pair,episode=pair+ep,pool=name,
             pool_group=name if task=='gate' else 'clock_all',query=q,evidence_end=end,evidence=';'.join(map(str,ev)),
             parent=';'.join(map(str,parent)),deleted=deleted,deleted_role=role,
             kept=';'.join(str(t) for t in parent if t!=deleted),history_count=2,horizon=h,feature_dim=5120,
             feature_mse=feature,task_error=state))
    return rows


@pytest.fixture(scope='module')
def analyzed():
    rows=fixture();compact,coverage=primary.aggregate(rows,pairs=4)
    summary=primary.summarize(compact,repeats=37,pairs=4)
    return rows,compact,summary


def test_independent_full_grid_fixture(analyzed):
    rows,compact,summary=analyzed
    assert len(rows)==3888 and len(compact)==288
    proof=auditor.audit_rows(rows,compact,summary,pairs=4,repeats=37)
    assert proof['independent_scalar_statistics']==1296
    assert all(x['passed'] for x in summary['gate_critical_contrasts'])
    assert not summary['joint_policy_stage_ready'] and not summary['original_phase1_go']


def test_feature_selection_does_not_erase_negative_task_gain(analyzed):
    _,_,summary=analyzed
    matching=[x for x in summary['oracle_baseline_comparisons'] if x['task']=='motion' and x['baseline']=='reject']
    assert matching and all(x['metrics']['feature_mse']['paired_gain']['mean']>0 for x in matching)
    assert all(x['metrics']['task_error']['paired_gain']['mean']<0 and not x['positive_both_metrics'] for x in matching)


def test_primary_rejects_duplicate_and_future_and_missing_sibling(analyzed):
    rows,_,_=analyzed
    with pytest.raises(ValueError):primary.aggregate(rows+[rows[0]],pairs=4)
    bad=copy.deepcopy(rows);bad[0]['parent']='4;7;30'
    with pytest.raises(ValueError):primary.aggregate(bad,pairs=4)
    missing=[r for r in rows if not (r['task']=='motion' and r['pair']=='pair000' and r['episode'].endswith('a') and r['pool']=='clock_5')]
    with pytest.raises(ValueError):primary.aggregate(missing,pairs=4)


def test_independent_audit_detects_wrong_pool_and_scalar(analyzed):
    rows,compact,summary=analyzed
    bad=copy.deepcopy(rows);bad[0]['pool']='cue_newest'
    with pytest.raises(AssertionError):auditor.audit_rows(bad,compact,summary,pairs=4,repeats=37)
    modified=copy.deepcopy(compact);modified[0]['oracle_transfer_feature_mse']+=.2
    with pytest.raises(AssertionError):auditor.audit_rows(rows,modified,summary,pairs=4,repeats=37)


def test_seed_mean_is_pair_average_before_bootstrap(analyzed):
    _,_,summary=analyzed
    spec=next(x for x in summary['oracle_baseline_comparisons'] if x['seed']=='mean')
    assert all(v['paired_gain']['pairs']==4 for v in spec['metrics'].values())
