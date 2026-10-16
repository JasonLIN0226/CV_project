"""Unsubmitted v5 frozen diagnostics primitives. No fitting or policy learning.

Task-label alignment remains unproved. Visual results cannot replace verb NLL,
activity strata, or the registered different-activity corruption intervention.
"""
import csv
import hashlib
import json
from collections import defaultdict
from collections import deque
from pathlib import Path

import numpy as np
import torch
from v5_retention import HistoryState, insert

from .v5_real_reader import full_feature_prediction,pad_payloads
from .v5_real_training import (HORIZONS, POLICIES, check_regular, full_batch_loss,
                               make_batch, payload_digest, schedule_lr, sha_file)

SEEDS=(17,29,43)


def finite_tree(value):
    if isinstance(value,torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite_tree(v) for v in value)
    return True


def validate_final_seed(latest,best,rows,seed,signature):
    """Current checkpoint tensors and original earliest-best CSV, before inference."""
    if (seed not in SEEDS or latest.get('seed')!=seed or best.get('seed')!=seed
            or latest.get('epoch')!=30 or latest.get('global_step')!=9690
            or latest.get('training_signature')!=signature or best.get('training_signature')!=signature
            or len(rows)!=30 or [int(r['epoch']) for r in rows]!=list(range(1,31))
            or any(int(r['steps'])!=323 for r in rows)):
        raise ValueError('Actual complete seed/epoch/step/signature proof differs')
    for row in rows:
        if not np.isfinite([float(row['train_full_mse']),float(row['validation_full_mse'])]).all():
            raise ValueError('Nonfinite original training record')
        slices=json.loads(row['validation_slices'])
        if set(slices)!=set(POLICIES) or any(len(slices[p])!=4 for p in POLICIES):
            raise ValueError('Incomplete original equal-condition validation')
        values=np.asarray([slices[p] for p in POLICIES],dtype=np.float64)
        if not np.isfinite(values).all() or abs(values.mean()-float(row['validation_full_mse']))>1e-10:
            raise ValueError('Original validation selection record differs')
    chosen=min(rows,key=lambda r:(float(r['validation_full_mse']),int(r['epoch'])))
    if (best.get('epoch')!=int(chosen['epoch']) or latest.get('best_epoch')!=int(chosen['epoch'])
            or abs(float(best.get('validation',float('inf')))-float(chosen['validation_full_mse']))>1e-12
            or abs(float(latest.get('best_validation',float('inf')))-float(chosen['validation_full_mse']))>1e-12):
        raise ValueError('Actual earliest validation best differs')
    keys=('optimizer','python_rng','numpy_global_rng','retention_dropout_shuffle_rng','torch_cpu_rng','torch_cuda_rng')
    if not all(k in latest for k in keys) or not finite_tree(latest['model']) or not finite_tree(best['model']) or not finite_tree(latest['optimizer']):
        raise ValueError('Actual model/optimizer/RNG boundary incomplete or nonfinite')
    if (latest['torch_cpu_rng'].dtype!=torch.uint8 or len(latest['torch_cuda_rng'])!=1
            or latest['torch_cuda_rng'][0].dtype!=torch.uint8):
        raise ValueError('Actual registered CPU/singleGPU RNG state differs')
    optimizer=latest['optimizer'];groups=optimizer['param_groups']
    if not groups or any(abs(g['lr']-schedule_lr(9689,9690))>1e-15 or g['weight_decay']!=.05 for g in groups):
        raise ValueError('Actual final optimizer/schedule differs')
    if not optimizer['state'] or any(int(v['step'])!=9690 for v in optimizer['state'].values()):
        raise ValueError('Actual optimizer boundary step differs')
    return {'seed':seed,'epochs':30,'steps':9690,'best_epoch':int(chosen['epoch']),
            'actual_finite_model_optimizer_RNG_schedule_earliestbest_verified':True}


def completion_proof(training_root,expected_files):
    """Require all six source-bound current files before any model inference.

    Caller separately verifies formal protocol/source/cache/Slurm bindings.
    expected_files must be a final hash binding produced after job7209 completes.
    """
    root=Path(training_root);path=root/'training_report.json';check_regular(path)
    report=json.loads(path.read_text())
    if (report.get('status')!='ALL3_REAL_READERS_30EPOCHS_CURRENT_WEIGHTS_PROVEN_NOT_PHASE1_GO'
            or report.get('job_id')!='7209' or report.get('course')!='csci_ga_2271-2026fa'
            or report.get('seeds')!=list(SEEDS) or report.get('epochs_required')!=30
            or report.get('train_queries')!=5168 or report.get('validation_queries')!=2356
            or report.get('labels_or_sealed_observations_read') is not False
            or report.get('first_epoch_projection_passed') is not True
            or len(report.get('completed_seeds',[]))!=3):
        raise ValueError('Actual training is not all-three-seed complete')
    signature=payload_digest({k:report[k] for k in ('source_lock_sha256','protocol_sha256','fit_proof_sha256','cache_proof_sha256')})
    if signature!=report['training_signature']:raise ValueError('Actual training provenance signature differs')
    expected={f'seed_{s}/{n}' for s in SEEDS for n in ('best.pt','latest.pt','training.csv','checkpoint_proof.json')}
    if set(expected_files)!=expected:raise ValueError('All actual checkpoint files must be formally bound')
    proofs=[]
    for seed,saved in zip(SEEDS,report['completed_seeds']):
        hashes={}
        for name in ('best.pt','latest.pt','training.csv','checkpoint_proof.json'):
            rel=f'seed_{seed}/{name}';file=root/rel;check_regular(file);hashes[name]=sha_file(file)
            if hashes[name]!=expected_files[rel]:raise ValueError('Actual checkpoint/source boundary changed')
        seed_root=root/f'seed_{seed}'
        latest=torch.load(seed_root/'latest.pt',map_location='cpu',weights_only=False)
        best=torch.load(seed_root/'best.pt',map_location='cpu',weights_only=False)
        rows=list(csv.DictReader((seed_root/'training.csv').open()))
        proof=validate_final_seed(latest,best,rows,seed,signature)
        executed=json.loads((seed_root/'checkpoint_proof.json').read_text())
        if executed!=saved or executed['training_signature']!=signature or executed['best_epoch']!=proof['best_epoch']:
            raise ValueError('Current model proof does not match executed training proof')
        for name in ('best.pt','latest.pt','training.csv'):
            if executed['files'][name]['sha256']!=hashes[name] or executed['files'][name]['bytes']!=(seed_root/name).stat().st_size:
                raise ValueError('Executed current model file proof differs')
        if executed['condition_samples_per_epoch']!={p:5168 for p in POLICIES}:
            raise ValueError('Actual balanced condition counts differ')
        if any(sha_file(seed_root/name)!=value for name,value in hashes.items()):
            raise ValueError('Completed model files changed during proof')
        proofs.append({**proof,'actual_files':hashes})
    return {'status':'ACTUAL_ALL6_CURRENT_MODELS_VERIFIED_BEFORE_DIAGNOSTIC_INFERENCE',
            'actual_training_report_sha256':sha_file(path),'training_signature':signature,
            'actual_seeds':proofs,'new_model_inference_fit_or_training':False}


def load_frozen_fit(path,expected_hash,dimension=5120,rank=128):
    check_regular(path)
    if sha_file(path)!=expected_hash:raise ValueError('Actual fixed train-only fit bytes changed')
    with np.load(path,allow_pickle=False) as saved:
        shapes={'mean':(dimension,),'scale':(dimension,),'input_basis':(dimension,rank),
                'output_mean':(dimension,),'output_basis':(dimension,rank)}
        if set(saved.files)!=set(shapes):raise ValueError('Fixed fit payload differs')
        arrays={key:np.array(saved[key],copy=True) for key in shapes}
    if any(a.shape!=shapes[key] or a.dtype!=np.float32 or not np.isfinite(a).all() for key,a in arrays.items()):
        raise ValueError('Invalid fixed fit shape/dtype/finite values')
    if (arrays['scale']<=0).any():raise ValueError('Invalid fixed normalization scale')
    if not np.allclose(arrays['output_basis'].T@arrays['output_basis'],np.eye(rank),atol=2e-5,rtol=2e-5):
        raise ValueError('Fixed output basis is not the trained orthonormal basis')
    if sha_file(path)!=expected_hash:raise ValueError('Fit changed during read-only load')
    return {key:torch.from_numpy(array) for key,array in arrays.items()}


def participant_values(rows,value_keys):
    """Query→episode→recording→participant; never treat frames as independent."""
    if not rows:raise ValueError('No actual matched records')
    episode=defaultdict(list)
    for row in rows:
        value=np.array([float(row[key]) for key in value_keys],dtype=np.float64)
        if not np.isfinite(value).all():raise ValueError('Nonfinite scalar metric')
        episode[(str(row['participant']),row['recording'],int(row['episode']))].append(value)
    recording=defaultdict(list)
    for (person,record,ep),values in episode.items():recording[(person,record)].append(np.mean(values,axis=0))
    people=defaultdict(list)
    for (person,record),values in recording.items():people[person].append(np.mean(values,axis=0))
    return {person:np.mean(values,axis=0) for person,values in sorted(people.items())}


def paired_participant_interval(values):
    """Original real protocol: 10000 participant resamples, seed0, 95% CI."""
    matrix=np.asarray([values[p] for p in sorted(values)],dtype=np.float64)
    if len(matrix)<2 or matrix.ndim!=2 or not np.isfinite(matrix).all():
        raise ValueError('Actual participant-paired sample required')
    draws=np.random.default_rng(0).integers(0,len(matrix),size=(10000,len(matrix)))
    boot=matrix[draws].mean(axis=1)
    lower,upper=np.quantile(boot,[.025,.975],axis=0)
    return {'participant_ids':sorted(values),'participants':len(matrix),'bootstrap':10000,'seed':0,
            'point':matrix.mean(axis=0).tolist(),'ci95_lower':lower.tolist(),'ci95_upper':upper.tolist(),
            'supported':[bool(x>0) for x in lower]}


def pooled_seed_participants(seed_values):
    if set(seed_values)!=set(SEEDS):raise ValueError('All three seeds required; no seed deletion')
    persons=set(seed_values[SEEDS[0]])
    if any(set(v)!=persons for v in seed_values.values()):raise ValueError('Seed participant sets differ')
    return {person:np.mean([seed_values[s][person] for s in SEEDS],axis=0) for person in sorted(persons)}


def joint_oracle_cost(candidates,immediate,continuation):
    """One joint oracle choice; the same choice reports h16/h64. No retuning."""
    immediate=np.asarray(immediate,dtype=np.float64);continuation=np.asarray(continuation,dtype=np.float64)
    if (len(candidates)!=17 or tuple(sorted(set(candidates)))!=tuple(candidates)
            or immediate.shape!=(17,4) or continuation.shape!=(17,32,4)
            or not np.isfinite(immediate).all() or not np.isfinite(continuation).all()):
        raise ValueError('Registered all17/four-horizon/common32 cost required')
    per_horizon=.5*immediate+.5*continuation.mean(axis=1)
    joint=per_horizon.mean(axis=1)
    selected=min(range(17),key=lambda i:(joint[i],candidates[i]))
    return {'chosen_index':selected,'deleted_id':candidates[selected],
            'joint_costs':joint.tolist(),'per_horizon_costs':per_horizon.tolist(),
            'tie_count':int(np.sum(joint==joint[selected])),
            'oracle_is_future_informed_not_deployable':True}


def sampled_states(recording,source_policy,contexts_by_episode):
    """Four uniform eligible states per recording/source; independent of losses."""
    if source_policy not in ('FIFO','Reservoir'):raise ValueError('Unregistered source trajectory')
    pool=[]
    for episode,contexts in sorted(contexts_by_episode.items()):
        if len(contexts)!=256:raise ValueError('Original episode length changed')
        for t in range(40,160):
            if len(contexts[t-1][1])==16:pool.append((episode,t))
    if len(pool)<4:raise ValueError('Insufficient registered full-memory states')
    seed=int.from_bytes(hashlib.sha256(f'v5-deletion-states-0|{recording}|{source_policy}'.encode()).digest()[:8],'big')
    selected=np.random.default_rng(seed).choice(len(pool),size=4,replace=False)
    return tuple(sorted(pool[i] for i in selected))


@torch.no_grad()
def visual_health_batch(model,episodes,rows,fit):
    """Frozen common reader and literal full-space errors; no future inputs."""
    model.eval()
    batch=make_batch(episodes,rows,False,None)
    _,losses=full_batch_loss(model,batch,fit)
    current,target=batch[3],batch[4]
    persistence=(current[:,None,:]-target).square().mean(dim=-1)
    oracle=[]
    for hi,_ in enumerate(HORIZONS):
        residual=target[:,hi]-current-fit['output_mean']
        coefficients=residual@fit['output_basis']
        predicted=full_feature_prediction(coefficients,current,fit)
        oracle.append((predicted-target[:,hi]).square().mean(dim=1))
    values={'mse':losses,'persistence_mse':persistence,'basis_oracle_mse':torch.stack(oracle,dim=1)}
    if any(not torch.isfinite(x).all() for x in values.values()):raise ValueError('Nonfinite full-space diagnostic')
    return {key:value.cpu().double().numpy() for key,value in values.items()}


@torch.no_grad()
def frozen_context_losses(model,episodes,cases,fit,deadline):
    """Literal original four-horizon loss for supplied causal persistent payloads."""
    if not cases:raise ValueError('No actual frozen contexts')
    model.eval();errors=[]
    for start in range(0,len(cases),64):
        deadline();chunk=cases[start:start+64]
        if any(t<0 or t+64>=256 for ei,t,p in chunk):raise ValueError('Future target outside original episode')
        x,clock,padding=pad_payloads([p for ei,t,p in chunk])
        currents=torch.stack([episodes[ei]['normalized'][t] for ei,t,p in chunk])
        scores=[]
        for h in HORIZONS:
            coeff=model(x,clock,padding,x.new_full((len(x),),h))
            pred=full_feature_prediction(coeff,currents,fit)
            targets=torch.stack([episodes[ei]['normalized'][t+h] for ei,t,p in chunk])
            scores.append((pred-targets).square().mean(dim=1))
        value=torch.stack(scores,dim=1)
        if not torch.isfinite(value).all():raise ValueError('Nonfinite frozen diagnostic score')
        errors.append(value.cpu().double().numpy())
    return np.concatenate(errors)


def visual_health_summary(rows,expected_queries):
    """Feature-only registered contrast. Missing verb checks stay unavailable."""
    expected=set(expected_queries)
    if len(expected)!=len(expected_queries) or not expected:
        raise ValueError('Complete predetermined health query set required')
    joined={}
    for row in rows:
        key=(int(row['seed']),str(row['participant']),row['recording'],int(row['episode']),int(row['time']),int(row['horizon']))
        if row['policy'] not in POLICIES or row['policy'] in joined.setdefault(key,{}):
            raise ValueError('Duplicate or unregistered condition')
        if key[1:5] not in expected or not 40<=key[4]<=188 or key[4]%4 or key[4]+key[5]>=256:
            raise ValueError('Unexpected or illegal registered health query')
        joined[key][row['policy']]=float(row['mse'])
    contrasts=defaultdict(list)
    for (seed,participant,recording,episode,time,h),values in joined.items():
        if set(values)!=set(POLICIES) or seed not in SEEDS or h not in HORIZONS:
            raise ValueError('Incomplete registered paired health result')
        contrasts[(seed,h)].append({'participant':participant,'recording':recording,'episode':episode,
                                    'delta':values['recent']-values['full']})
    if set(contrasts)!={(s,h) for s in SEEDS for h in HORIZONS}:
        raise ValueError('Missing seed/horizon; no partial science decision')
    if len(joined)!=len(expected)*len(SEEDS)*len(HORIZONS):
        raise ValueError('Actual health queries are incomplete')
    checks={};passed=True
    for h in HORIZONS:
        each={s:participant_values(contrasts[(s,h)],['delta']) for s in SEEDS}
        checks[str(h)]={str(s):paired_participant_interval(each[s]) for s in SEEDS}
        checks[str(h)]['pooled']=paired_participant_interval(pooled_seed_participants(each))
        if h in (16,64) and not all(result['supported'][0] for result in checks[str(h)].values()):passed=False
    return {'visual_history_status':'SUPPORTED' if passed else 'REJECTED','full_vs_recent_paired_checks':checks,
            'real_feasibility_status':'INCONCLUSIVE' if passed else 'REJECTED',
            'verb_readout_NLL_activity_strata_and_recovery_status':'UNAVAILABLE_ANNOTATION_FRAME_PHASE_UNPROVED',
            'deletion_and_replay_not_established_by_health':True,'phase1_go':False,
            'all_original_Gate_Motion_rejections_preserved':True}


class IrreversibleReplay:
    """Online records receive one observation at a time; no archive interface.

    Evaluation targets and any historical corruption donor stay outside this
    object. Only current recent/history payloads and retention counters persist.
    """
    def __init__(self,policy,capacity=16):
        if policy not in ('FIFO','Reservoir','Uniform') or capacity not in (8,16,32):
            raise ValueError('Unregistered online policy/capacity')
        self.policy=policy
        self.state=HistoryState(capacity)
        self.recent=deque()
        self.history={}
        self.step=-1
        self.full_updates=0

    def advance(self,observed,draw):
        if self.step>=255:
            raise ValueError('Original fixed episode ended; reset required')
        if observed.ndim!=1 or not torch.isfinite(observed).all():
            raise ValueError('One actual observed feature vector required')
        if self.recent and observed.shape!=self.recent[-1][1].shape:
            raise ValueError('Observed feature dimension changed')
        self.step+=1
        deleted=None
        if len(self.recent)==8:
            incoming,payload=self.recent.popleft()
            if incoming!=self.step-8:raise ValueError('Observation clock changed')
            was_full=len(self.state.retained)==self.state.capacity
            self.state,deleted=insert(self.state,incoming,self.policy,draw)
            if incoming in self.state.retained:self.history[incoming]=payload
            if deleted is not None:self.history.pop(deleted,None)
            self.full_updates+=int(was_full)
        # Store an independent persistent payload; no reference to an archive.
        self.recent.append((self.step,observed.detach().clone()))
        if set(self.history)!=set(self.state.retained) or set(self.history)&{i for i,_ in self.recent}:
            raise ValueError('Retained payload/clock or recent-history boundary differs')
        return deleted

    def payload(self):
        if not self.recent:raise ValueError('Empty observed stream')
        ids=tuple(self.state.retained)+tuple(i for i,_ in self.recent)
        visuals=[self.history[i] for i in self.state.retained]+[p for _,p in self.recent]
        observed=torch.stack(visuals)
        ages=observed.new_tensor([(self.step-i)/256. for i in ids])
        roles=observed.new_tensor([0]*len(self.history)+[1]*len(self.recent))
        return observed,torch.stack((ages,roles),dim=1),ids
