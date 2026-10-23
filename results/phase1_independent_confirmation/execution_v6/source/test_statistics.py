"""Synthetic scalar fixtures, not model results or fresh data generation."""
import copy
from pathlib import Path
import tempfile
import numpy as np
import pytest
from phase1.config import DEFAULT
from phase1.evaluation import summarize
from phase1.reporting import decision
from phase1.stats import paired_improvement
from phase1.v5_synthetic_evaluation import fixed_reference_checks as gate_fixed
from phase1.v5_motion_evaluation import fixed_reference_checks as motion_fixed
from phase1.utils import write_json,write_csv,file_hash
from fresh_audit import audit_task,load_raw,improvement


@pytest.mark.parametrize('task',['gate','motion'])
def test_fresh_count_original_and_fixed_statistic_replay(tmp_path,task):
    c=copy.deepcopy(DEFAULT);c.update(mode='confirm',output=str(tmp_path))
    c['encoder']['backend']='ijepa'  # Metadata for arithmetic fixture; no encoder exists or runs.
    c['data']['seed']=6301
    c['data']['pairs']={'train':512,'validation':96,'test_id':256,'test_long_gap':256,'test_distractor_shift':256}
    c['training']['seeds']=[17,29,43];c['training']['epochs']=120;c['evaluation']['bootstrap']=2000
    profiles=copy.deepcopy(c);profiles['data']['tasks']=[task]
    raw=[]
    for seed in (17,29,43):
        for split in ('validation','test_id','test_long_gap','test_distractor_shift'):
            for condition in ('recent_only','full_history','privileged_small_budget','random_small_budget','wrong_evidence','persistence'):
                for h in (3,6):
                    for pair in range(c['data']['pairs'][split]):
                        for sibling in (0,1):
                            error={'recent_only':1.,'full_history':.5,'privileged_small_budget':.5,
                                'random_small_budget':.7,'wrong_evidence':1.,'persistence':1.}[condition]
                            accuracy={'recent_only':.5,'full_history':1.,'privileged_small_budget':1.,
                                'random_small_budget':.75,'wrong_evidence':0.,'persistence':.5}[condition]
                            raw.append(dict(seed=seed,task=task,split=split,condition=condition,horizon=h,
                                episode=f'{split}-{pair}-{sibling}',pair=f'{split}-{pair}',gap=7,query=14,
                                latent_mse=error,normalized_mse=error,cosine_distance=error/10,
                                pair_accuracy=accuracy,pair_margin=1. if condition=='privileged_small_budget' else 0.,
                                target_pair_distance=1.,state_accuracy=accuracy,
                                state_position_mse=.01 if error<=.5 else .05,
                                history_count=0 if condition in ('recent_only','persistence') else 2,
                                history_feature_bytes=40960,projected_history_bytes=1024,recent_feature_bytes=81920,
                                timestamp_type_mask_bytes=78,past_action_bytes=48,future_action_bytes=h*8,evidence_hit=1.))
    reference=[{**r,'normalized_mse':.65 if r['condition']=='privileged_small_budget' else r['normalized_mse']} for r in raw]
    out=tmp_path/'evaluation/candidate'/task;out.mkdir(parents=True)
    write_csv(out/'raw_records.csv',raw)
    write_csv(tmp_path/'evaluation/reference'/task/'raw_records.csv',reference)
    write_json(tmp_path/'all_models_lock.json',{'status':'SOFTWARE_SCALAR_FIXTURE_ONLY_NOT_FRESH_MODELS'})
    metrics={**summarize(profiles,raw),'raw_sha256':file_hash(out/'raw_records.csv'),
        'all_models_lock_sha256':file_hash(tmp_path/'all_models_lock.json')}
    write_json(out/'metrics.json',metrics)
    probes=[]
    for split in ('test_id','test_long_gap','test_distractor_shift'):
        probes.extend([dict(task=task,split=split,variant='gate_cue' if task=='gate' else 'motion_two_frame',
            accuracy={'mean':1.},wilson_effective_pairs=[.98,1.]),
            dict(task=task,split=split,variant='recent_only',accuracy={'mean':.5})])
        for h in (3,6):probes.append(dict(task=task,split=split,variant='state_decoder_on_true_target',horizon=h,
            accuracy={'mean':1.},position_mse={'mean':.0001}))
    write_json(tmp_path/'probes/metrics.json',{'results':probes,'software_fixture':True})
    write_json(tmp_path/'dataset/split_audit.json',{'passed':True,'software_fixture':True})
    status,reasons,checks=decision(profiles,{'passed':True},{'results':probes},metrics)
    fixed=(gate_fixed if task=='gate' else motion_fixed)(profiles,reference,raw)
    assert status=='GO' and all(v['passed'] for v in checks+fixed)
    write_json(tmp_path/'evaluation'/task/'comparison.json',dict(original_status=status,original_checks=checks,
        fixed_reference_checks=fixed,software_fixture=True))
    proof=audit_task(c,task)
    assert proof['original_possible_go'] and proof['max_original_statistic_difference']<1e-12
    assert len(proof['fixed_checks_replay'])==32
    # Completeness must fail independently of a good aggregate score.
    write_csv(tmp_path/'missing_pair.csv',raw[:-1])
    with pytest.raises(ValueError):load_raw(tmp_path/'missing_pair.csv',task)


def test_absolute_relative_and_pair_first_statistics():
    groups=['a','a','b','b','c','c']
    for base,alt,floor in [([.01]*6,[.005]*6,.02),([1,3,2,4,3,5],[.8,2.8,1.8,3.8,2.8,4.8],.02)]:
        expected=paired_improvement(base,alt,groups,2000,floor,.01)
        got=improvement(base,alt,groups,floor)
        assert np.allclose(got[:3],[expected[k] for k in ('mean','low','high')],rtol=0,atol=1e-14)
        assert got[3]==expected['scale'].startswith('absolute')
