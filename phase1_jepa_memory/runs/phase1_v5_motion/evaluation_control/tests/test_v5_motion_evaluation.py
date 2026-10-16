import copy
from pathlib import Path
import numpy as np
import pytest
import torch

from phase1.v5_motion_evaluation import (SELECTION, validate_checkpoint_pair,
    load_current_reader, fixed_reference_checks)
from phase1.training import TRAIN_CONDITIONS
from phase1.config import DEFAULT, CONDITIONS, SPLITS


def snapshots():
    signature={'task':'motion','reader':'SyntheticCausalReader','selection':SELECTION}
    logs=[{'epoch':i,'step':2*i,'validation_mse':score,'mix_counts':str({k:4 for k in TRAIN_CONDITIONS}),
           **{'val_'+k+'_mse':score for k in TRAIN_CONDITIONS}} for i,score in enumerate((2.,1.,1.),1)]
    def snapshot(epoch):
        return {'signature':signature,'seed':17,'task':'motion','resume_granularity':'epoch_boundary',
                'checkpoint_selection':SELECTION,'model':{'x':torch.ones(2)},
                'optimizer':{'state':{0:{'step':torch.tensor(2*epoch)}},'param_groups':[]},
                'scheduler':{'last_epoch':epoch},'rng':dict.fromkeys(('python','numpy','torch','cuda')),
                'epoch':epoch,'step':2*epoch,'logs':copy.deepcopy(logs[:epoch]),'best':min(x['validation_mse'] for x in logs[:epoch])}
    return signature,snapshot(3),snapshot(2),{'signature':signature,'seed':17,'logs':logs}


def test_completion_accepts_earliest_tie_and_actual_prefix():
    sig,last,best,log=snapshots()
    proof=validate_checkpoint_pair(17,last,best,log,sig,epochs=3,samples=4,steps=2)
    assert proof['complete'] and proof['best_epoch']==2
    assert proof['signature']==sig and 'scientific_hash' not in proof['signature']


@pytest.mark.parametrize('failure',('partial','late_tie','imbalance','rng','optimizer','scheduler','basis_binding','validation_mean'))
def test_invalid_actual_proofs_are_rejected(failure):
    sig,last,best,log=snapshots()
    if failure=='partial':last['epoch']=2
    elif failure=='late_tie':best=copy.deepcopy(last)
    elif failure=='imbalance':last['logs'][0]['mix_counts']=str({k:3 for k in TRAIN_CONDITIONS});log['logs']=copy.deepcopy(last['logs'])
    elif failure=='rng':last['rng'].pop('numpy')
    elif failure=='optimizer':last['optimizer']['state'][0]['step']=torch.tensor(5)
    elif failure=='scheduler':last['scheduler']['last_epoch']=2
    elif failure=='basis_binding':last['signature']={**sig,'basis_sha256':'changed'}
    elif failure=='validation_mean':last['logs'][0]['val_recent_only_mse']=3.;log['logs']=copy.deepcopy(last['logs'])
    with pytest.raises(ValueError):validate_checkpoint_pair(17,last,best,log,sig,epochs=3,samples=4,steps=2)


def test_no_reader_loading_before_all_three_complete(tmp_path):
    with pytest.raises(ValueError,match='All three'):
        load_current_reader(tmp_path,tmp_path,17,None,{'complete':False})


def test_fixed_reference_rejects_one_bad_seed_even_when_pooled_passes():
    c=copy.deepcopy(DEFAULT);c['evaluation']['bootstrap']=40
    reference=[];candidate=[]
    for split in SPLITS[1:]:
        for h in (3,6):
            for seed in (17,29,43):
                for pair in range(8):
                    for sibling in range(2):
                        for condition in ('recent_only','privileged_small_budget'):
                            row={'task':'motion','seed':seed,'split':split,'horizon':h,'condition':condition,
                                 'pair':f'{split}_{pair}','episode':f'{split}_{pair}_{sibling}','normalized_mse':1.}
                            reference.append(row)
                            error=1.01 if seed==43 and split=='test_long_gap' and h==3 else .4
                            candidate.append({**row,'normalized_mse':error if condition=='privileged_small_budget' else 1.})
    checks=fixed_reference_checks(c,reference,candidate)
    assert len(checks)==32
    critical=[r for r in checks if r['slice']=='motion/test_long_gap/h3/seed43']
    assert len(critical)==2 and not any(r['passed'] for r in critical)
    pooled=[r for r in checks if r['slice']=='motion/test_long_gap/h3/seedall_seeds_mean']
    assert all(r['passed'] for r in pooled)
    assert not all(r['passed'] for r in checks)


def test_v5_metric_loop_preserves_old_metric_semantics_and_output_separation(tmp_path,monkeypatch):
    import phase1.v5_motion_evaluation as v5
    torch.set_num_threads(1)
    c=copy.deepcopy(DEFAULT);c['output']=str(tmp_path/'old_data_contract')
    c['experiment']={'stage':'development','arm':'reference','tasks':['motion']}
    c['training']['seeds']=[17];c['training']['batch_size']=2;c['data']['pairs']={s:1 for s in SPLITS}
    c['evaluation'].update(bootstrap=5,latency_warmup=0,latency_repeats=1)
    records=[{'id':f'{s}_{i}','sibling':f'{s}_{1-i}','pair':s,'split':s,'task':'motion','query':3,'gap':5} for s in SPLITS[1:] for i in (0,1)]
    class FakeStore:
        def __init__(self,*a,**kw):
            self.manifest={'records':records};self.records={r['id']:r for r in records};self.ix={'signature':'artificial'}
            self.norm={'mean':np.zeros(8,np.float32),'std':np.ones(8,np.float32)*2}
        def features(self,rec):return np.zeros((10,8),np.float32)
        def sample(self,rec,condition):
            sample={'z':np.zeros((4,8),np.float32),'age':np.array([3,2,1,0],np.float32),'type':np.zeros(4,np.int64),
                    'past_actions':np.zeros((4,2),np.float32),'actions':np.zeros((6,2),np.float32),'current':np.zeros(8,np.float32)}
            return sample,np.zeros((2,8),np.float32),{'history_count':0,'evidence_hit':False}
    class FakeModel(torch.nn.Module):
        def __init__(self):super().__init__();self.register_buffer('projection',torch.zeros(8,2))
        def forward(self,batch):return torch.ones((len(batch['z']),2,8))
    class FakeDecoder:
        def predict(self,p):return np.zeros((len(p),3))
    training=tmp_path/'current_training';weight=training/'predictor/motion/seed_17/best.pt';weight.parent.mkdir(parents=True);weight.write_bytes(b'artificial checkpoint marker')
    monkeypatch.setattr(v5,'FeatureStore',FakeStore)
    monkeypatch.setattr(v5,'all_models_completion',lambda *a:{'complete':True,'models':{},'scope':'ARTIFICIAL_SOFTWARE_ONLY'})
    monkeypatch.setattr(v5,'load_current_reader',lambda *a:(FakeModel().eval(),{'epoch':2,'seconds':1.}))
    monkeypatch.setattr(v5,'load_decoder',lambda *a:FakeDecoder())
    monkeypatch.setattr(v5,'load_episode',lambda *a:{'classes':np.zeros(10),'states':np.zeros((10,2))})
    output=tmp_path/'new_evaluation'
    metrics,raw,proof=v5.evaluate_v5(c,tmp_path,training,output)
    assert len(raw)==96 and metrics['phase1_go'] is False
    assert all(r['normalized_mse']==1. and r['latent_mse']==4. for r in raw if r['condition']!='persistence')
    assert all(r['normalized_mse']==0. for r in raw if r['condition']=='persistence')
    assert {r['condition'] for r in raw}==set(CONDITIONS+['persistence'])
    assert (output/'evaluation/raw_records.csv').is_file()
    assert not Path(c['output']).exists()
    assert 'legacy_metric_helpers_code_hash' in metrics and 'v5_evaluation_code_hash' in metrics
