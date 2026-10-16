"""Artificial boundary/statistics tests; no actual features or model fitting."""
import copy
import json
import numpy as np
import pytest
import torch

from phase1.v5_real_diagnostics import (joint_oracle_cost,participant_values,
    paired_participant_interval,pooled_seed_participants,sampled_states,
    validate_final_seed,visual_health_batch,visual_health_summary,IrreversibleReplay,frozen_context_losses,
    load_frozen_fit,completion_proof)
from phase1.v5_real_reader import CausalReader,feature_fit,context_payload,policy_contexts
from phase1.v5_real_training import prepared_episodes,query_rows,training_fit,schedule_lr
from phase1.v5_real_training import payload_digest


def test_hierarchy_is_episode_recording_participant_equal_not_frame_weighted():
    rows=[{'participant':'A','recording':'a','episode':0,'x':0}]*100
    rows += [{'participant':'A','recording':'a','episode':1,'x':10},
             {'participant':'A','recording':'b','episode':0,'x':20},
             {'participant':'B','recording':'c','episode':0,'x':30}]
    got=participant_values(rows,['x'])
    assert got['A'][0]==12.5 and got['B'][0]==30
    result=paired_participant_interval(got)
    assert result['point']==[21.25] and result['participants']==2 and result['bootstrap']==10000


def test_seed_pooling_does_not_inflate_participant_sample_or_drop_seed():
    values={s:{'a':np.array([s]),'b':np.array([s+1])} for s in (17,29,43)}
    result=paired_participant_interval(pooled_seed_participants(values))
    assert result['participants']==2
    with pytest.raises(ValueError):pooled_seed_participants({17:values[17],43:values[43]})


def test_zero_or_negative_interval_does_not_pass_and_nonfinite_rejected():
    assert paired_participant_interval({'a':[0.],'b':[0.]})['supported']==[False]
    assert paired_participant_interval({'a':[-1.],'b':[-2.]})['supported']==[False]
    with pytest.raises(ValueError):paired_participant_interval({'a':[1.],'b':[float('nan')]})


def test_joint_oracle_uses_same_choice_for_all_horizons_and_exact_time_tie():
    now=np.full((17,4),10.);later=np.full((17,32,4),10.)
    now[2]=[1,1,8,1];later[2]=now[2]
    now[3]=[5,5,0,5];later[3]=now[3]
    result=joint_oracle_cost(tuple(range(17)),now,later)
    assert result['deleted_id']==2 and result['per_horizon_costs'][2][2]==8
    result=joint_oracle_cost(tuple(range(17)),np.zeros((17,4)),np.zeros((17,32,4)))
    assert result['deleted_id']==0 and result['tie_count']==17


@pytest.mark.parametrize('shape',[(17,31,4),(16,32,4),(17,32,2)])
def test_no_shortened_continuation_or_omitted_candidate_horizon(shape):
    with pytest.raises(ValueError):joint_oracle_cost(tuple(range(17)),np.zeros((17,4)),np.zeros(shape))


def test_sampling_fixed_uniform_complete_boundary_and_no_loss_interface():
    contexts=tuple((tuple(range(max(0,t-7),t+1)),tuple(range(max(0,t-8-15),t-8+1))) for t in range(256))
    one=sampled_states('recording','FIFO',{0:contexts,1:contexts})
    assert one==sampled_states('recording','FIFO',{0:contexts,1:contexts})
    assert len(set(one))==4 and all(40<=t<=159 and t+32+64<256 for _,t in one)
    with pytest.raises(ValueError):sampled_states('recording','oracle',{0:contexts})


def final_fixture():
    rows=[]
    for epoch in range(1,31):
        value=1. if epoch in (4,5) else 2.
        rows.append({'epoch':str(epoch),'steps':'323','train_full_mse':'.5',
                     'validation_full_mse':str(value),
                     'validation_slices':json.dumps({p:[value]*4 for p in ('recent','FIFO','Reservoir','full')})})
    latest={'seed':17,'epoch':30,'global_step':9690,'training_signature':'signature','best_epoch':4,
            'best_validation':1.,'model':{'weight':torch.ones(2)},
            'optimizer':{'param_groups':[{'lr':schedule_lr(9689,9690),'weight_decay':.05}],
                         'state':{0:{'step':torch.tensor(9690.),'exp_avg':torch.ones(2)}}},
            'python_rng':('artificial',),'numpy_global_rng':('artificial',),
            'retention_dropout_shuffle_rng':{},'torch_cpu_rng':torch.zeros(3,dtype=torch.uint8),
            'torch_cuda_rng':[torch.zeros(3,dtype=torch.uint8)]}
    best={'seed':17,'epoch':4,'validation':1.,'training_signature':'signature','model':{'weight':torch.ones(2)}}
    return latest,best,rows


def test_actual_final_epoch_optimizer_schedule_and_earliest_tie_contract():
    latest,best,rows=final_fixture()
    assert validate_final_seed(latest,best,rows,17,'signature')['best_epoch']==4
    best['epoch']=5
    with pytest.raises(ValueError):validate_final_seed(latest,best,rows,17,'signature')


@pytest.mark.parametrize('kind',['partial','signature','nonfinite','rng','schedule','optimizer_step','validation'])
def test_incomplete_or_changed_models_fail_before_heldout_inference(kind):
    latest,best,rows=final_fixture()
    if kind=='partial':latest['epoch']=29
    if kind=='signature':best['training_signature']='changed'
    if kind=='nonfinite':best['model']['weight'][0]=float('nan')
    if kind=='rng':del latest['torch_cuda_rng']
    if kind=='schedule':latest['optimizer']['param_groups'][0]['lr']=.0003
    if kind=='optimizer_step':latest['optimizer']['state'][0]['step']=torch.tensor(1.)
    if kind=='validation':rows[0]['validation_slices']='{}'
    with pytest.raises(ValueError):validate_final_seed(latest,best,rows,17,'signature')


def health_fixture(failed_seed=None):
    rows=[]
    for seed in (17,29,43):
        for person in ('A','B','C'):
            for h in (1,4,16,64):
                for policy in ('recent','FIFO','Reservoir','full'):
                    mse=2. if policy=='recent' else 1.
                    if policy=='full' and seed==failed_seed:mse=3.
                    rows.append({'seed':seed,'participant':person,'recording':person,'episode':0,
                                 'time':40,'horizon':h,'policy':policy,'mse':mse})
    return rows


def test_visual_pass_cannot_be_full_feasibility_when_labels_and_deletion_missing():
    queries=[(p,p,0,40) for p in ('A','B','C')]
    result=visual_health_summary(health_fixture(),queries)
    assert result['visual_history_status']=='SUPPORTED'
    assert result['real_feasibility_status']=='INCONCLUSIVE' and result['phase1_go'] is False


def test_pooled_seeds_cannot_rescue_one_failed_seed_and_partial_rows_rejected():
    queries=[(p,p,0,40) for p in ('A','B','C')]
    assert visual_health_summary(health_fixture(29),queries)['visual_history_status']=='REJECTED'
    rows=health_fixture();rows.pop()
    with pytest.raises(ValueError):visual_health_summary(rows,queries)
    with pytest.raises(ValueError):visual_health_summary(health_fixture(),queries+[('A','A',0,44)])


def test_health_is_literal_full_feature_mse_and_no_refit_or_future_model_input():
    torch.manual_seed(4)
    episodes=[{'recording':'toy','participant':'A','split':'train','episode':0,'features':torch.randn(256,8)}]
    fit,_=training_fit(episodes,query_rows(episodes,'train'),rank=4)
    prepared=prepared_episodes(episodes,fit)
    model=CausalReader(rank=4,hidden=16,layers=2,heads=4,ffn=32,dropout=0).eval()
    rows=((0,40,'full'),(0,40,'recent'))
    got=visual_health_batch(model,prepared,rows,fit)
    assert all(v.shape==(2,4) and np.isfinite(v).all() for v in got.values())
    current=prepared[0]['normalized'][40]
    for hi,h in enumerate((1,4,16,64)):
        want=(current-prepared[0]['normalized'][40+h]).square().mean().item()
        np.testing.assert_allclose(got['persistence_mse'][:,hi],want,rtol=1e-6)
    assert np.all(got['basis_oracle_mse']<=got['mse']+1e-5)


@pytest.mark.parametrize('policy',['FIFO','Reservoir','Uniform'])
def test_persistent_online_payloads_match_original_contexts_without_archive(policy):
    import hashlib
    torch.manual_seed(1);observations=torch.randn(256,4)
    seed=int.from_bytes(hashlib.sha256(f'v5-real-retention-0|toy|0|{policy}'.encode()).digest()[:8],'big')
    draws=np.random.default_rng(seed).random(248)
    expected=policy_contexts(256,policy,'toy',0)
    online=IrreversibleReplay(policy)
    discarded=set()
    for t in range(256):
        dropped=online.advance(observations[t],draws[t-8] if t>=8 else .5)
        if dropped is not None:discarded.add(dropped)
        actual,clock,ids=online.payload()
        want,ages=context_payload(observations,t,expected[t])
        torch.testing.assert_close(actual,want,rtol=0,atol=0)
        torch.testing.assert_close(clock,ages,rtol=0,atol=0)
        assert not discarded.intersection(ids) and len(online.history)<=16
    assert online.full_updates==232


def test_stream_records_do_not_alias_old_or_future_archive_and_reindex_ages():
    observed=torch.arange(4.).clone();online=IrreversibleReplay('FIFO')
    online.advance(observed,.5);observed[:]=1000
    assert torch.equal(online.payload()[0][0],torch.arange(4.))
    for i in range(1,25):online.advance(torch.full((4,),float(i)),.5)
    _,temporal,ids=online.payload()
    assert 0 not in ids and online.history.keys()==set(range(1,17))
    for j,identity in enumerate(ids):assert temporal[j,0].item()==(24-identity)/256.
    assert not hasattr(online,'archive')


def test_zero_halfway_model_boundary_cannot_hide_invalid_incoming_payload():
    online=IrreversibleReplay('Uniform')
    with pytest.raises(ValueError):online.advance(torch.tensor([float('nan')]),.5)
    online.advance(torch.ones(4),.5)
    with pytest.raises(ValueError):online.advance(torch.ones(5),.5)


def test_persistent_candidate_scores_equal_literal_original_reader_full_space_loss():
    torch.manual_seed(6)
    episodes=[{'recording':'toy','participant':'A','split':'train','episode':0,'features':torch.randn(256,8)}]
    fit,_=training_fit(episodes,query_rows(episodes,'train'),rank=4)
    prepared=prepared_episodes(episodes,fit);model=CausalReader(rank=4,hidden=16,layers=2,heads=4,ffn=32,dropout=0).eval()
    rows=((0,40,'FIFO'),(0,44,'Reservoir'))
    cases=[(ei,t,context_payload(prepared[ei]['compressed'],t,prepared[ei]['contexts'][p][t])) for ei,t,p in rows]
    got=frozen_context_losses(model,prepared,cases,fit,lambda:None)
    np.testing.assert_array_equal(got,visual_health_batch(model,prepared,rows,fit)['mse'])
    with pytest.raises(ValueError):frozen_context_losses(model,prepared,[(0,192,cases[0][2])],fit,lambda:None)


def test_frozen_fit_loaded_without_refit_and_changed_bytes_or_basis_rejected(tmp_path):
    import hashlib
    torch.manual_seed(4);fit=feature_fit(torch.randn(50,8),torch.randn(60,8),rank=4)
    path=tmp_path/'fit.npz';np.savez(path,**{k:v.numpy() for k,v in fit.items()})
    digest=lambda:hashlib.sha256(path.read_bytes()).hexdigest()
    loaded=load_frozen_fit(path,digest(),dimension=8,rank=4)
    for key in fit:torch.testing.assert_close(loaded[key],fit[key],rtol=0,atol=0)
    old=digest();np.savez(path,**{k:v.numpy()+1 for k,v in fit.items()})
    with pytest.raises(ValueError):load_frozen_fit(path,old,dimension=8,rank=4)
    with pytest.raises(ValueError):load_frozen_fit(path,digest(),dimension=8,rank=4)


def test_all6_actual_serialized_files_and_executed_proof_required_before_inference(tmp_path):
    import csv,hashlib
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    signature_fields={k:'a'*64 for k in ('source_lock_sha256','protocol_sha256','fit_proof_sha256','cache_proof_sha256')}
    signature=payload_digest(signature_fields);completed=[];expected={}
    for seed in (17,29,43):
        root=tmp_path/f'seed_{seed}';root.mkdir()
        latest,best,rows=final_fixture();latest.update(seed=seed,training_signature=signature);best.update(seed=seed,training_signature=signature)
        torch.save(latest,root/'latest.pt');torch.save(best,root/'best.pt')
        with (root/'training.csv').open('w',newline='') as file:
            out=csv.DictWriter(file,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
        proof={'seed':seed,'training_signature':signature,'best_epoch':4,
               'condition_samples_per_epoch':{p:5168 for p in ('recent','FIFO','Reservoir','full')},
               'files':{name:{'bytes':(root/name).stat().st_size,'sha256':digest(root/name)}
                        for name in ('best.pt','latest.pt','training.csv')}}
        (root/'checkpoint_proof.json').write_text(json.dumps(proof));completed.append(proof)
        for name in ('best.pt','latest.pt','training.csv','checkpoint_proof.json'):
            expected[f'seed_{seed}/{name}']=digest(root/name)
    report={**signature_fields,'training_signature':signature,'job_id':'7209','course':'csci_ga_2271-2026fa',
        'status':'ALL3_REAL_READERS_30EPOCHS_CURRENT_WEIGHTS_PROVEN_NOT_PHASE1_GO','seeds':[17,29,43],
        'epochs_required':30,'train_queries':5168,'validation_queries':2356,
        'labels_or_sealed_observations_read':False,'first_epoch_projection_passed':True,'completed_seeds':completed}
    (tmp_path/'training_report.json').write_text(json.dumps(report))
    got=completion_proof(tmp_path,expected)
    assert len(got['actual_seeds'])==3 and got['new_model_inference_fit_or_training'] is False
    # A locally rehashed changed weight still conflicts with the executed proof.
    state=torch.load(tmp_path/'seed_29/best.pt',weights_only=False);state['model']['weight']+=1
    torch.save(state,tmp_path/'seed_29/best.pt');expected['seed_29/best.pt']=digest(tmp_path/'seed_29/best.pt')
    with pytest.raises(ValueError,match='Executed current model file proof'):
        completion_proof(tmp_path,expected)
