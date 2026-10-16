"""One guarded A100 stage: train-only task decoder and fixed causal benchmark.

This fills author-release interfaces, not unpublished author implementation.
No completed-model/artifact/label guard may be bypassed by a historical mode.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from completed_predictor_proof import completed_proof, sha
from completed_artifact_proof import completed_artifact_proof
from memory_jepa.utils.io import atomic_torch_save
from memory_jepa.utils.training import cosine_warmup_scheduler
from space.artifacts import load_slow_basis, load_utility_bank
from space.config import load_config
from space.model import load_predictor_checkpoint
from space.slow_basis import predictive_state_signatures
from paper_benchmark import (METHODS, HORIZONS, OFFSETS, CachedRetrieval, OnlineMethods,
    ranking_scores, recovery_anchors, memory_distance, normalized_recovery, participant_pair_inference)


def save(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.new')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def finished(job):
    rows=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j',str(job),'-n','-P',
        '--format=JobIDRaw,State,ExitCode,Account'],text=True)
    if not any(r.split('|')[:4]==[str(job),'COMPLETED','0:0','csci_ga_2271-2026fa'] for r in rows.splitlines()):
        raise RuntimeError('required actual course parent is not COMPLETED0:0')


def verify(root, protocol):
    if protocol.get('actual_binding_complete') is not True or protocol.get('seeds')!=[0,1,2] or protocol.get('methods')!=list(METHODS):
        raise RuntimeError('a template or incomplete fixed comparison cannot run')
    lock=json.loads((root/'source_lock.json').read_text())
    for name,value in lock['files'].items():
        f=root/name
        if f.is_symlink() or sha(f)!=value:raise RuntimeError('benchmark frozen source changed')
    for key in ('artifact_report','label_report','label_independent_audit','completed_predictor_proof','completed_artifact_numeric_proof'):
        f=Path(protocol[key])
        if f.is_symlink() or sha(f)!=protocol[key+'_sha256']:raise RuntimeError('bound evidence changed: '+key)
    artifact=json.loads(Path(protocol['artifact_report']).read_text())
    finished(artifact['job_id'])
    artifact_root=Path(protocol['artifact_root'])
    if sha(artifact_root/'source_lock.json')!=protocol['artifact_source_lock_sha256'] or sha(artifact_root/'artifact_protocol.json')!=protocol['artifact_protocol_sha256']:
        raise RuntimeError('artifact executed source/protocol binding differs')
    if artifact['source_lock_sha256']!=protocol['artifact_source_lock_sha256'] or artifact['protocol_sha256']!=protocol['artifact_protocol_sha256']:
        raise RuntimeError('artifact report differs from actual frozen source')
    for name,value in json.loads((artifact_root/'source_lock.json').read_text())['files'].items():
        if (artifact_root/name).is_symlink() or sha(artifact_root/name)!=value:raise RuntimeError('completed artifact source changed')
    if artifact['status']!='PAPER_CONSISTENT_BASIS_AND_BANK_COMPLETE' or artifact['test_data_used'] or artifact['elapsed_seconds']>13800:
        raise RuntimeError('complete bounded train-only basis/bank proof required')
    if artifact['counts']['bank_states']!=11136:raise RuntimeError('all fixed bank states required')
    for row in artifact['outputs']:
        f=Path(row['path'])
        if f.is_symlink() or f.stat().st_size!=row['bytes'] or sha(f)!=row['sha256']:raise RuntimeError('current artifact differs')
    expected=json.loads(Path(protocol['completed_predictor_proof']).read_text())
    proof=completed_proof(protocol['training_submission'],protocol['training_root'])
    if proof!=expected or artifact['completed_predictor_proof_sha256']!=protocol['completed_predictor_proof_sha256']:
        raise RuntimeError('all six current predictor proofs differ')
    numeric=completed_artifact_proof(artifact,proof,Path(protocol['artifact_report']).parent)
    if numeric!=json.loads(Path(protocol['completed_artifact_numeric_proof']).read_text()):
        raise RuntimeError('six current numeric basis/bank proofs differ')
    labels=json.loads(Path(protocol['label_report']).read_text());finished(labels['job_id'])
    if labels['sealed_numeric_labels_interpreted'] or labels['validation_missing_labels']!=0 or labels['recordings']!=294:
        raise RuntimeError('full known validation and test isolation required')
    if labels['manifest_sha256']!=proof['manifest_sha256']:raise RuntimeError('label feature axes differ')
    for row in labels['label_artifacts']:
        f=Path(row['path'])
        if f.is_symlink() or f.stat().st_size!=row['bytes'] or sha(f)!=row['sha256']:raise RuntimeError('current label artifact differs')
    audit=json.loads(Path(protocol['label_independent_audit']).read_text())
    if audit.get('all_arrays_and_masks_equal',audit.get('all_match',False)) is not True:
        # Formal binding also stores the exact audit payload: accept only the
        # explicitly named complete independent-replay state, never a flag guess.
        if audit.get('status')!='ALL294_INDEXED_LABEL_ARTIFACTS_AND_MASKS_REPLAYED_EXACTLY':
            raise RuntimeError('complete independent label-axis replay required')
    if audit['alignment_report_sha256']!=protocol['label_report_sha256'] or audit['sealed_numeric_labels_read'] or not audit['all75_validation_recordings_have_labels']:
        raise RuntimeError('independent axis replay source/coverage differs')
    return proof,artifact,labels


def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],
            'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()}


def finite(model,optimizer):
    tensors=list(model.state_dict().values())+[v for state in optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor)]
    if not all(torch.isfinite(t).all() for t in tensors):raise RuntimeError('nonfinite decoder/optimizer')


def fit_decoders(records, output, guard, started, binding):
    # The same immutable observed TSM vectors and availability mask are used
    # for every seed. No validation record is accepted by this function.
    if any(r['split']!='train' for r in records):raise ValueError('train-only decoder fitting required')
    x=torch.cat([r['features'][r['valid']] for r in records]).to('cuda')
    y=torch.cat([r['labels'][r['valid']] for r in records]).float().to('cuda')
    if len(x)!=172887 or len(records)!=219:raise RuntimeError('full known supervised training coverage differs')
    x=F.normalize(x.float(),dim=-1)
    sample_digest=hashlib.sha256('\n'.join(r['video_id']+':'+str(int(r['valid'].sum())) for r in records).encode()).hexdigest()
    proof={'train_recordings':219,'labelled_recordings':218,'known_observations':len(x),'unknown_observations_excluded':411,
           'validation_fit_rows':0,'test_fit_rows':0,'availability_rows_digest':sample_digest,
           'input_dimension':2048,'output_verbs':24,'model':'linear','selection':'earliest minimum complete training BCE',
           'label_report_sha256':binding['label_report_sha256'],'predictor_proof_sha256':binding['completed_predictor_proof_sha256']}
    save(output/'decoder_fit_proof.json',proof)
    models=[];reports=[];steps_per_epoch=(len(x)+127)//128
    for seed in (0,1,2):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=torch.nn.Linear(2048,24).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05)
        schedule=cosine_warmup_scheduler(optimizer,total_steps=steps_per_epoch*30,warmup_ratio=.05)
        folder=output/f'decoder_seed_{seed}';folder.mkdir();history=[];best=float('inf');best_epoch=None;step=0
        for epoch in range(1,31):
            tick=time.monotonic();model.train();order=torch.randperm(len(x),device=x.device)
            for first in range(0,len(x),128):
                guard();ids=order[first:first+128];optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=F.binary_cross_entropy_with_logits(model(x[ids]),y[ids])
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();schedule.step();step+=1
            model.eval();total=0.
            with torch.no_grad():
                for first in range(0,len(x),2048):
                    guard();logits=model(x[first:first+2048])
                    total+=float(F.binary_cross_entropy_with_logits(logits,y[first:first+2048],reduction='sum'))
            mean=total/(len(x)*24);finite(model,optimizer)
            history.append({'epoch':epoch,'step':step,'training_BCE_mean':mean})
            ck={'seed':seed,'epoch':epoch,'step':step,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'scheduler':schedule.state_dict(),'rng':rng_state(),'fit_proof_sha256':sha(output/'decoder_fit_proof.json'),
                'selection_loss':mean,'complete':epoch==30}
            atomic_torch_save(ck,folder/'latest.pt')
            if mean<best:
                best=mean;best_epoch=epoch;atomic_torch_save(ck,folder/'best.pt')
            save(folder/'history.json',history)
            if seed==0 and epoch==1:
                estimate=time.monotonic()-started+(time.monotonic()-tick)*89
                save(output/'first_decoder_epoch_budget.json',{'full_three_seed_training_projection_seconds':estimate,'budget':13800})
                if estimate>13800:raise RuntimeError('fixed complete decoder training projection exceeds stage budget')
        expected=min(range(30),key=lambda k:history[k]['training_BCE_mean'])+1
        if expected!=best_epoch or step!=steps_per_epoch*30:raise RuntimeError('decoder completion selection/count differs')
        for tag in ('best','latest'):
            f=folder/(tag+'.pt');before=sha(f);ck=torch.load(f,map_location='cpu',weights_only=True)
            if ck['seed']!=seed or ck['epoch']!=(best_epoch if tag=='best' else 30) or ck['step']!=steps_per_epoch*ck['epoch']:
                raise RuntimeError('decoder current checkpoint/history boundary differs')
            tensors=list(ck['model'].values())+[v for state in ck['optimizer']['state'].values() for v in state.values() if isinstance(v,torch.Tensor)]
            if not all(torch.isfinite(t).all() for t in tensors) or sha(f)!=before or set(ck['rng'])!={'python','numpy','torch_cpu','torch_cuda'}:
                raise RuntimeError('decoder final current proof failed')
            if ck['scheduler']['last_epoch']!=ck['step'] or not ck['optimizer']['state'] or ck['optimizer']['param_groups'][0]['weight_decay']!=.05:
                raise RuntimeError('decoder complete optimizer/schedule proof failed')
        bestck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(bestck['model'],strict=True)
        model.eval()
        for param in model.parameters():param.requires_grad_(False)
        models.append(model);reports.append({'seed':seed,'epochs':30,'steps':step,'earliest_training_best':best_epoch,
            'best_sha256':sha(folder/'best.pt'),'latest_sha256':sha(folder/'latest.pt'),'history_sha256':sha(folder/'history.json')})
    save(output/'all_six_decoder_completion.json',{'fit_proof':proof,'models':reports,'heldout_inference_before_all_six_complete':False})
    return models


@torch.no_grad()
def logits_and_probabilities(decoder,predictions):
    logits=decoder(F.normalize(predictions.float(),dim=-1))
    return logits,torch.sigmoid(logits)


@torch.no_grad()
def recovery_for_anchor(engine, donor, features, labels, anchor, decoder, seed, guard):
    clean=engine.clone();perturbed=[]
    rng=np.random.default_rng(stable_seed(engine.video_id,seed,anchor))
    order=rng.permutation(16)
    for count in (4,8):
        bad=engine.clone();bad.corrupt(donor,list(map(int,order[:count])));perturbed.append(bad)
    distances={severity:{m:{'state':[],'memory':[],'task':[]} for m in METHODS[1:]} for severity in (25,50)}
    for offset in range(17):
        guard();t=anchor+offset;local=features[t-7:t+1]
        if offset:
            for policy in (clean,*perturbed):policy.step(features[t-8],t-8,local,t,t-8)
        else:
            for policy in (clean,*perturbed):policy.hold_predictions(local,t)
        if offset not in OFFSETS:continue
        cp=clean.last_predictions;_,probs=logits_and_probabilities(decoder,cp)
        cs=predictive_state_signatures(clean.predictor.project_inputs(cp.reshape(-1,2048)).reshape(6,4,-1),clean.basis)
        for severity,bad in zip((25,50),perturbed):
            bp=bad.last_predictions;_,bprobs=logits_and_probabilities(decoder,bp)
            bs=predictive_state_signatures(bad.predictor.project_inputs(bp.reshape(-1,2048)).reshape(6,4,-1),bad.basis)
            for j,m in enumerate(METHODS[1:],1):
                d=distances[severity][m];d['state'].append(float((1-F.cosine_similarity(cs[j:j+1],bs[j:j+1])).clamp(0,2)))
                d['memory'].append(memory_distance(clean,bad,m));d['task'].append(float((probs[j]-bprobs[j]).abs().mean()))
    rows=[]
    for severity,methods in distances.items():
        for method,metrics in methods.items():
            for metric,values in metrics.items():
                rows.append({'seed':seed,'video_id':engine.video_id,'anchor':anchor,'severity_percent':severity,
                             'method':method,'distance_type':metric,**normalized_recovery(values)})
    return rows


def stable_seed(video,seed,anchor):
    return int.from_bytes(hashlib.sha256(f'{video}\0{seed}\0{anchor}'.encode()).digest()[:8],'little')


@torch.no_grad()
def evaluate_record(record, predictor, basis, bank, decoder, config, seed, guard, output):
    features=record['features'].cuda();labels=record['labels'].numpy().astype(np.uint8)
    if not record['valid'].all():raise RuntimeError('all validation labels must be known')
    engine=OnlineMethods(predictor,basis,bank,config,seed=seed,video_id=record['video_id'])
    anchors=recovery_anchors(labels);donor_times={d for a,d in anchors};anchor_donors=dict(anchors);donors={}
    costs=[];probabilities=[];signatures=[];queries=[];targets=[];traces=[];recovery=[];decoder_oracle_costs=[]
    for t in range(8,len(features)-64):
        guard();local=features[t-7:t+1];pred=engine.step(features[t-8],t-8,local,t,t-8)
        if t<23:continue
        logits,prob=logits_and_probabilities(decoder,pred)
        target=torch.as_tensor(labels[np.asarray(HORIZONS)+t],device=logits.device,dtype=torch.float32)
        oracle_logits=decoder(F.normalize(features[torch.as_tensor(HORIZONS,device=features.device)+t],dim=-1))
        decoder_oracle_costs.append(F.binary_cross_entropy_with_logits(oracle_logits,target,reduction='none').sum(-1).cpu().numpy())
        loss=F.binary_cross_entropy_with_logits(logits,target.expand(6,-1,-1),reduction='none').sum(-1)
        sig=predictive_state_signatures(predictor.project_inputs(pred.reshape(-1,2048)).reshape(6,4,-1),basis)
        queries.append(t);costs.append(loss.cpu().numpy());probabilities.append(prob.cpu().numpy())
        signatures.append(sig.cpu().numpy());targets.append(target.cpu().numpy());traces.append([engine.origins[m][:] for m in METHODS])
        if t in donor_times:donors[t]=engine.clone()
        if t in anchor_donors:
            donor=donors[anchor_donors[t]]
            recovery.extend(recovery_for_anchor(engine,donor,features,labels,t,decoder,seed,guard))
    if len(queries)!=len(features)-87:raise RuntimeError('full fixed validation query count differs')
    cost=np.asarray(costs);prob=np.asarray(probabilities);target=np.asarray(targets);sig=np.asarray(signatures);q=np.asarray(queries)
    rows=[]
    for j,method in enumerate(METHODS):
        current=labels[q];within=(current[1:]==current[:-1]).all(-1)&current[1:].any(-1)
        boundary=(current[1:]!=current[:-1]).any(-1)&current[1:].any(-1)
        drift=np.clip(1.-(sig[1:,j]*sig[:-1,j]).sum(-1),0,2)
        wd=float(drift[within].mean()) if within.any() else None
        bd=float(drift[boundary].mean()) if boundary.any() else None
        row={'seed':seed,'video_id':record['video_id'],'participant':record['participant'],'method':method,
             'queries':len(q),'Action_NLL':float(cost[:,j].mean()),'horizon_NLL':cost[:,j].mean(0).tolist(),
             'true_future_feature_decoder_NLL':float(np.asarray(decoder_oracle_costs).mean()),
             'within_regime_drift':wd,'boundary_displacement':bd,'within_transitions':int(within.sum()),'boundary_transitions':int(boundary.sum()),
             'log_boundary_within_ratio':float(np.log((bd+1e-8)/(wd+1e-8))) if wd is not None and bd is not None else None,
             'ranking':ranking_scores(target,prob[:,j]),
             'horizon_ranking':[ranking_scores(target[:,h],prob[:,j,h]) for h in range(4)],
             'override_count':engine.overrides[method],'boundary_release_count':engine.overrides[method+'_boundary']}
        rows.append(row)
    filename=output/f'{record["video_id"]}_seed{seed}_raw.npz'
    np.savez(filename,query=q,nll=cost,probabilities=prob,target=target,signature=sig,payload_origin_trace=np.asarray(traces,dtype=np.int32))
    for r in recovery:r['participant']=record['participant']
    return rows,recovery,{'path':str(filename),'bytes':filename.stat().st_size,'sha256':sha(filename),'arrays_remain_HPC':True}


def summaries(rows,recovery):
    scores={};participant_scores={}
    for method in METHODS:
        by_seed=[]
        for seed in (0,1,2):
            selected=[r for r in rows if r['method']==method and r['seed']==seed]
            people=sorted(set(r['participant'] for r in selected))
            if len(people)!=10:raise RuntimeError('all ten validation participants required')
            results=[]
            for p in people:
                group=[r for r in selected if r['participant']==p]
                score={'Action_NLL':float(np.mean([r['Action_NLL'] for r in group])),
                       'true_future_feature_decoder_NLL':float(np.mean([r['true_future_feature_decoder_NLL'] for r in group])),
                       **{k:float(np.mean([r['ranking'][k] for r in group])) for k in ('mAP','LRAP','Recall_at_5','background_fraction')}}
                for key in ('foreground_LRAP','foreground_Recall_at_5'):
                    known=[r['ranking'][key] for r in group if r['ranking'][key] is not None]
                    score[key]=float(np.mean(known)) if known else None
                results.append(score);participant_scores[(method,seed,p)]=score
            by_seed.append({'seed':seed,**{k:float(np.mean([r[k] for r in results if r[k] is not None]))
                if any(r[k] is not None for r in results) else None for k in results[0]}})
        horizon_seeds=[];state_seeds=[]
        for seed in (0,1,2):
            selected=[r for r in rows if r['method']==method and r['seed']==seed]
            people=sorted(set(r['participant'] for r in selected))
            hrows=[];srows=[]
            for p in people:
                group=[r for r in selected if r['participant']==p]
                hrows.append(np.mean([r['horizon_NLL'] for r in group],axis=0))
                state={}
                for key in ('within_regime_drift','boundary_displacement','log_boundary_within_ratio'):
                    known=[r[key] for r in group if r[key] is not None]
                    state[key]=float(np.mean(known)) if known else None
                srows.append(state)
            horizon_seeds.append({'seed':seed,'NLL':np.mean(hrows,axis=0).tolist()})
            state_seeds.append({'seed':seed,**{key:float(np.mean([r[key] for r in srows if r[key] is not None]))
                if any(r[key] is not None for r in srows) else None for key in srows[0]}})
        scores[method]={'seeds':by_seed,'mean':{k:float(np.mean([r[k] for r in by_seed])) for k in ('Action_NLL','mAP','LRAP','Recall_at_5')},
                        'sample_SD':{k:float(np.std([r[k] for r in by_seed],ddof=1)) for k in ('Action_NLL','mAP','LRAP','Recall_at_5')}}
        scores[method]['horizon_seeds']=horizon_seeds
        scores[method]['horizon_NLL_mean']=np.mean([r['NLL'] for r in horizon_seeds],axis=0).tolist()
        scores[method]['horizon_NLL_sample_SD']=np.std([r['NLL'] for r in horizon_seeds],axis=0,ddof=1).tolist()
        scores[method]['semantic_state_seeds']=state_seeds
        scores[method]['background_fraction']=float(np.mean([r['background_fraction'] for r in by_seed]))
        scores[method]['decoder_true_future_feature_NLL']=float(np.mean([r['true_future_feature_decoder_NLL'] for r in by_seed]))
        scores[method]['foreground_ranking']={k:float(np.mean([r[k] for r in by_seed if r[k] is not None]))
            if any(r[k] is not None for r in by_seed) else None for k in ('foreground_LRAP','foreground_Recall_at_5')}
        scores[method]['state_transition_counts']={'within':sum(r['within_transitions'] for r in rows if r['method']==method),
                                                 'boundary':sum(r['boundary_transitions'] for r in rows if r['method']==method)}
    comparisons={};people=sorted(set(r['participant'] for r in rows))
    for method in METHODS:
        if method=='reservoir':continue
        gains=[np.mean([participant_scores[('reservoir',seed,p)]['Action_NLL']-participant_scores[(method,seed,p)]['Action_NLL'] for seed in (0,1,2)]) for p in people]
        comparisons[method]=participant_pair_inference(gains)
        base=scores['reservoir']['mean']['Action_NLL'];comparisons[method]['relative_NLL_gain_percent']=100*comparisons[method]['mean_gain']/base
    rec={}
    for method in METHODS[1:]:
        for severity in (25,50):
            for metric in ('state','memory','task'):
                selected=[r for r in recovery if r['method']==method and r['severity_percent']==severity and r['distance_type']==metric]
                key=f'{method}/{severity}/{metric}'
                if not selected:rec[key]={'status':'NO_ELIGIBLE_DONOR_CASES'};continue
                participants=sorted(set(r['participant'] for r in selected));videos=set(r['video_id'] for r in selected)
                # Anchor means within recording, recordings within participant,
                # seed means within participant, then participants equally.
                means=[]
                for p in participants:
                    seed_means=[]
                    for seed in (0,1,2):
                        records=[]
                        for video in sorted(videos):
                            group=[r for r in selected if r['participant']==p and r['seed']==seed and r['video_id']==video]
                            if group:records.append({k:float(np.mean([r[k] for r in group])) for k in ('initial_distance','final_distance','normalized_auc')})
                        if records:seed_means.append({k:float(np.mean([r[k] for r in records])) for k in records[0]})
                    if seed_means:means.append({k:float(np.mean([r[k] for r in seed_means])) for k in seed_means[0]})
                rec[key]={k:float(np.mean([r[k] for r in means])) for k in means[0]}
                rec[key].update({'participants':len(participants),'recordings':len(videos),'anchor_seed_cases':len(selected),
                    'denominator_floor_cases':sum(r['denominator_floored'] for r in selected),'missing_case_fabrication':False})
    return {'methods':scores,'paired_NLL_against_Reservoir':comparisons,'development_not_confirmatory':True,
            'records_per_seed':75,'participants':10,'complete_seeds':3},rec


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();started=time.monotonic()
    job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit():raise RuntimeError('actual course A100 job required')
    native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',job],text=True)
    fields=dict(x.split('=',1) for x in native.split() if '=' in x)
    if fields.get('Account')!='csci_ga_2271-2026fa' or fields.get('Partition')!='c12m85-a100-1' or fields.get('JobState')!='RUNNING':raise RuntimeError('course A100 allocation guard')
    if torch.__version__!='2.5.1+cu121' or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0):raise RuntimeError('one reference-environment A100 required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    protocol=json.loads(args.protocol.read_text());args.output.mkdir(parents=True,exist_ok=False)
    progress={'decoder_complete':False,'completed_record_seed_pairs':0,'full_validation_records':0,'query_method_horizon_scores':0}
    def guard():
        if time.monotonic()-started>13800:raise RuntimeError('fixed benchmark science soft budget reached')
    try:
        proof,artifact,labelreport=verify(args.root,protocol);save(args.output/'executed_input_proof.json',{'predictor':proof,'artifact_report_sha256':protocol['artifact_report_sha256'],'label_report_sha256':protocol['label_report_sha256']})
        manifest=Path(proof['manifest']);metadata=json.loads(manifest.read_text())['videos'];labelrows={r['video_id']:r for r in labelreport['label_artifacts']}
        records=[]
        for row in metadata:
            guard();v=row['video_id'];payload=torch.load(manifest.parent/(v+'.pt'),map_location='cpu',weights_only=True)
            f=Path(labelrows[v]['path']);before=sha(f)
            with np.load(f,allow_pickle=False) as z:
                lab=z['verb_labels'].copy();valid=z['label_valid'].copy();axis=z['frame_indices'].copy()
            if not np.array_equal(np.asarray(payload['clip_indices']),axis) or sha(f)!=before:raise RuntimeError('current indexed feature/label boundary differs')
            records.append(dict(row,participant=payload['metadata']['participant_id'],features=payload['latents'].float(),
                                labels=torch.from_numpy(lab),valid=torch.from_numpy(valid).bool()))
        decoders=fit_decoders([r for r in records if r['split']=='train'],args.output,guard,started,protocol);progress['decoder_complete']=True
        # Release training feature references before validation streaming.
        validation=[r for r in records if r['split']=='val'];del records
        if len(validation)!=75:raise RuntimeError('all 75 validation recordings required')
        configs=load_config(args.root/'configs/space_k16.yaml');models=[];bases=[];banks=[]
        artifact_dir=Path(protocol['artifact_report']).parent
        for seed in (0,1,2):
            ck=next(r for r in proof['models'] if r['seed']==seed and r['tag']=='best')
            model,_=load_predictor_checkpoint(ck['path'],'cuda');models.append(model)
            seed_dir=artifact_dir/f'seed_{seed}';bases.append(load_slow_basis(seed_dir/'slow_basis.pt','cuda'))
            bank=load_utility_bank([seed_dir/'banks'/f'{m}_features.pt' for m in ('fifo','reservoir')],
                [seed_dir/'banks'/f'{m}_r32_delta.pt' for m in ('fifo','reservoir')],device='cuda')
            banks.append(CachedRetrieval(bank,32,32))
        rows=[];recovery=[];outputs=[];raw=args.output/'raw';raw.mkdir()
        total_queries=sum(int(r['valid_length'])-87 for r in validation)
        for index,record in enumerate(validation):
            tick=time.monotonic()
            for seed in (0,1,2):
                guard();a,b,c=evaluate_record(record,models[seed],bases[seed],banks[seed],decoders[seed],configs,seed,guard,raw)
                rows.extend(a);recovery.extend(b);outputs.append(c);progress['completed_record_seed_pairs']+=1
                progress['query_method_horizon_scores']+=sum(r['queries'] for r in a)*4
            progress['full_validation_records']+=1
            if index==0:
                estimate=time.monotonic()-started+(time.monotonic()-tick)*(total_queries/(int(record['valid_length'])-87)-1)
                save(args.output/'first_complete_record_all_seeds_methods_budget.json',{'full_stage_projection_seconds':estimate,'budget_seconds':13800,'passed':estimate<=13800,
                     'includes_recovery_for_actual_eligible_anchors':True,'actual_completed_first_record_reused':True})
                if estimate>13800:raise RuntimeError('full fixed benchmark projection exceeds stage budget')
            save(args.output/'progress.json',dict(progress,elapsed_seconds=time.monotonic()-started))
            print(json.dumps({'stage':'benchmark_record_complete','records':index+1,'total':75,'elapsed_seconds':time.monotonic()-started}),flush=True)
        if progress['query_method_horizon_scores']!=total_queries*3*6*4:raise RuntimeError('complete per-method query counts differ')
        scores,recovery_summary=summaries(rows,recovery)
        save(args.output/'record_scores.json',rows);save(args.output/'recovery_cases.json',recovery)
        save(args.output/'main_scores.json',scores);save(args.output/'recovery_summary.json',recovery_summary)
        save(args.output/'benchmark_report.json',{'status':'FIXED_PAPER_BASED_BENCHMARK_COMPLETE','job_id':int(job),'course':fields['Account'],
            'gpu':torch.cuda.get_device_name(0),'gpus':1,'science_threads':4,'counts':progress,'elapsed_seconds':time.monotonic()-started,
            'protocol_sha256':sha(args.protocol),'source_lock_sha256':sha(args.root/'source_lock.json'),
            'all_six_decoder_completion_sha256':sha(args.output/'all_six_decoder_completion.json'),
            'main_scores_sha256':sha(args.output/'main_scores.json'),'record_scores_sha256':sha(args.output/'record_scores.json'),
            'recovery_summary_sha256':sha(args.output/'recovery_summary.json'),'recovery_cases_sha256':sha(args.output/'recovery_cases.json'),
            'raw_outputs':outputs,'exact_unpublished_author_implementation_claim':False,'sealed_test_opened':False,
            'decoder_and_benchmark_details_reconstructed':True,'old_IJEPA_raw_axis_solved':False,'Phase1_GO_claim':False})
    except BaseException as exc:
        save(args.output/'failure_boundary.json',{'exception':type(exc).__name__,'message':str(exc),'counts':progress,
            'elapsed_seconds':time.monotonic()-started,'no_seed_data_method_or_recovery_reduction':True,'budget_extension':False})
        raise


if __name__=='__main__':main()
