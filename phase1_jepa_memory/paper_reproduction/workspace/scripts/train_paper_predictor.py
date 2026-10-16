"""Guarded SPACE predictor training, with transparent restored data integration.

Model, loss, optimizer and schedule are imported from the frozen author release.
Adds resource/time guards and complete checkpoints; no acceptance thresholds are
changed. CPU-only execution is deliberately refused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from memory_jepa.data.latent_dataset import PredictorTrainingDataset, collate_predictor
from memory_jepa.losses import jepa_loss
from memory_jepa.utils import load_config, seed_everything
from memory_jepa.utils.hashing import canonical_hash, state_dict_hash
from memory_jepa.utils.io import atomic_torch_save, write_json
from memory_jepa.utils.modeling import build_predictor
from memory_jepa.utils.parameter_count import count_parameters, expected_predictor_parameters
from memory_jepa.utils.training import autocast_context, cosine_warmup_scheduler

COURSE = 'csci_ga_2271-2026fa'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def resource_guard():
    jid=os.environ.get('SLURM_JOB_ID','')
    if not jid.isdigit():raise RuntimeError('actual GPU batch required')
    s=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',jid],text=True)
    d=dict(x.split('=',1) for x in s.split() if '=' in x)
    if d.get('Account')!=COURSE or d.get('Partition')!='c12m85-a100-1' or d.get('JobState')!='RUNNING':
        raise RuntimeError('unexpected GPU course allocation')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('one actual A100 required')
    if torch.__version__ != '2.5.1+cu121' or torch.version.cuda != '12.1':
        raise RuntimeError('isolated author-reference Torch/CUDA libraries required')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    return {'job_id':int(jid),'account':d['Account'],'partition':d['Partition'],
            'allocated_tres':d.get('AllocTRES'),'gpu':torch.cuda.get_device_name(0),
            'visible_gpus':1,'torch_version':str(torch.__version__),'science_threads':4}


def verify_inputs(root, protocol):
    lock=json.loads((root/'source_lock.json').read_text())
    for name,expected in lock['files'].items():
        f=root/name
        if f.is_symlink() or digest(f)!=expected:raise RuntimeError('training source changed')
    audit=Path(protocol['data_report'])
    if digest(audit)!=protocol['data_report_sha256']:raise RuntimeError('data audit binding changed')
    report=json.loads(audit.read_text())
    if report['status']!='OFFICIAL_TSM_TRAIN_VAL_PREPARED_AND_AUDITED' or report['test_payload_deserialized']:
        raise RuntimeError('complete test-isolated official data proof required')
    actual=subprocess.check_output(['/opt/slurm/bin/sacct','-j',str(report['job_id']),'-X','-n','-P',
                                    '--format=JobIDRaw,State,ExitCode,Account'],text=True).strip()
    lines=[r.split('|') for r in actual.splitlines()]
    if not any(r[:4]==[str(report['job_id']),'COMPLETED','0:0',COURSE] for r in lines):
        raise RuntimeError('data parent has not completed successfully')
    manifest=Path(report['manifest'])
    if digest(manifest)!=protocol['manifest_sha256']:raise RuntimeError('manifest changed')
    for row in report['features']:
        path=manifest.parent/(row['video_id']+'.pt')
        if path.is_symlink() or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
            raise RuntimeError('official feature proof changed')
    return manifest, report


def finite_state(model,optimizer):
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('nonfinite model')
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():
                raise RuntimeError('nonfinite optimizer')


def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],
            'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()}


@torch.no_grad()
def evaluate(model,loader,device,config):
    model.eval(); full=[]; recent=[]
    weights=torch.tensor(config['predictor']['horizon_weights'],device=device)
    for batch in loader:
        batch=batch.to(device)
        with autocast_context(device,config['training']['precision']):
            pred=model(batch.local,batch.memory,batch.memory_valid,batch.local_age,batch.memory_age).predictions
            no=model(batch.local,batch.memory,torch.zeros_like(batch.memory_valid),batch.local_age,batch.memory_age).predictions
            f=jepa_loss(pred,batch.targets,batch.target_valid,weights,'cosine')
            r=jepa_loss(no,batch.targets,batch.target_valid,weights,'cosine')
        full.append(f.per_horizon.float().cpu());recent.append(r.per_horizon.float().cpu())
    f,r=torch.cat(full),torch.cat(recent)
    if not torch.isfinite(f).all() or not torch.isfinite(r).all():raise RuntimeError('nonfinite validation loss')
    return {'loss':float(f.mean()),'local_only_loss':float(r.mean()),
            'loss_by_horizon':f.mean(0).tolist(),'local_only_by_horizon':r.mean(0).tolist(),
            'queries':len(f)}


def train_one(seed,root,output,manifest,config,protocol,started_global,resources):
    seed_everything(seed);device=torch.device('cuda:0')
    args=dict(manifest_path=manifest,local_context_len=8,memory_slots=17,
              horizons=[1,4,16,64],memory_dropout=.15,seed=seed)
    train=PredictorTrainingDataset(split='train',**args)
    val=PredictorTrainingDataset(split='val',**args)
    tr=DataLoader(train,batch_size=128,shuffle=True,num_workers=0,pin_memory=True,collate_fn=collate_predictor)
    va=DataLoader(val,batch_size=256,shuffle=False,num_workers=0,pin_memory=True,collate_fn=collate_predictor)
    d_in=json.loads(manifest.read_text())['d_in']
    model=build_predictor(config,d_in).to(device)
    if count_parameters(model)!=expected_predictor_parameters(d_in=d_in,d_model=512,d_ff=2048,n_layers=4,horizons=4):
        raise RuntimeError('released parameter count mismatch')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05)
    scheduler=cosine_warmup_scheduler(optimizer,30*len(tr),.05)
    weights=torch.full((4,),.25,device=device)
    dest=output/f'seed_{seed}';dest.mkdir(exist_ok=False)
    history=[];best=float('inf');best_epoch=None;steps=0
    for epoch in range(1,31):
        epoch_start=time.monotonic();model.train();total=torch.zeros(4,device=device);seen=0
        for batch in tr:
            if time.monotonic()-started_global>13800:raise RuntimeError('original four-hour science soft budget reached')
            batch=batch.to(device);optimizer.zero_grad(set_to_none=True)
            with autocast_context(device,'bf16'):
                prediction=model(batch.local,batch.memory,batch.memory_valid,batch.local_age,batch.memory_age).predictions
                losses=jepa_loss(prediction,batch.targets,batch.target_valid,weights,'cosine')
            if not torch.isfinite(losses.mean):raise RuntimeError('nonfinite training loss')
            losses.mean.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step();scheduler.step();steps+=1
            total+=losses.per_horizon.detach().sum(0);seen+=len(batch.local)
        validation=evaluate(model,va,device,config);finite_state(model,optimizer)
        elapsed=time.monotonic()-epoch_start
        row={'epoch':epoch,'steps':steps,'train_loss':float(total.mean()/seen),
             'train_by_horizon':(total/seen).tolist(),'validation':validation,
             'epoch_seconds':elapsed,'lr':optimizer.param_groups[0]['lr']}
        history.append(row)
        checkpoint={'format_version':1,'stage':'predictor','seed':seed,'d_in':d_in,
                    'config':config,'config_hash':canonical_hash(config),'model':model.state_dict(),
                    'predictor_hash':state_dict_hash(model),'trainable_parameters':count_parameters(model),
                    'manifest':str(manifest),'manifest_sha256':digest(manifest),
                    'protocol_sha256':protocol['_actual_sha256'],'source_lock_sha256':digest(root/'source_lock.json'),
                    'validation':validation,'epoch':epoch,'global_step':steps,'optimizer':optimizer.state_dict(),
                    'scheduler':scheduler.state_dict(),'rng':rng_state(),
                    'resources':resources,'complete':epoch==30}
        atomic_torch_save(checkpoint,dest/'predictor_latest.pt')
        if validation['loss']<best:
            best=validation['loss'];best_epoch=epoch
            atomic_torch_save(checkpoint,dest/'predictor_best.pt')
        write_json(history,dest/'predictor_history.json')
        print(json.dumps({'seed':seed,'epoch':epoch,'steps':steps,'best_epoch':best_epoch,
                          'validation_cosine_loss':validation['loss'],'elapsed_seconds':time.monotonic()-started_global}),flush=True)
        if seed==0 and epoch==1:
            complete_epoch_elapsed=time.monotonic()-epoch_start
            projection=(time.monotonic()-started_global)-complete_epoch_elapsed+90*complete_epoch_elapsed
            budget={'setup_seconds':time.monotonic()-started_global-complete_epoch_elapsed,'first_full_epoch_seconds':complete_epoch_elapsed,
                    'full_90_epoch_projection_seconds':projection,'budget_seconds':13800,'passed':projection<=13800}
            write_json(budget,output/'first_epoch_budget.json')
            if projection>13800:raise RuntimeError('complete ninety-epoch projection exceeds fixed budget')
    proof={'seed':seed,'epochs':30,'steps':steps,'expected_steps':30*len(tr),'train_queries':len(train),
           'validation_queries':len(val),'best_epoch':best_epoch,'best_validation_cosine_loss':best,
           'best_sha256':digest(dest/'predictor_best.pt'),'latest_sha256':digest(dest/'predictor_latest.pt'),
           'history_sha256':digest(dest/'predictor_history.json'),'finite_model_optimizer':True}
    if proof['steps']!=proof['expected_steps']:raise RuntimeError('incomplete steps')
    write_json(proof,dest/'completion_proof.json')
    del model,optimizer,train,val,tr,va;torch.cuda.empty_cache()
    return proof


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();started=time.monotonic();resources=resource_guard()
    protocol=json.loads(args.protocol.read_text());protocol['_actual_sha256']=digest(args.protocol)
    manifest,data=verify_inputs(args.root,protocol)
    args.output.mkdir(parents=True,exist_ok=False)
    config=load_config(str(args.root/'configs/assembly101.yaml'))
    config['training']['num_workers']=0
    proofs=[train_one(seed,args.root,args.output,manifest,config,protocol,started,resources) for seed in [0,1,2]]
    write_json({'status':'PAPER_PREDICTOR_TRAINING_COMPLETE_RECONSTRUCTED_DATA_MODULE',
                'resources':resources,'source_lock_sha256':digest(args.root/'source_lock.json'),
                'protocol_sha256':protocol['_actual_sha256'],'data_report_sha256':protocol['data_report_sha256'],
                'completed_models':proofs,'elapsed_seconds':time.monotonic()-started,
                'full_paper_reproduction_complete':False,'acceptance_pass_claim':False},args.output/'training_report.json')


if __name__=='__main__':main()
