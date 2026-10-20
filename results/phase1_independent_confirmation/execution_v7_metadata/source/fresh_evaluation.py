"""Full fresh prediction evaluation with the original metrics and fixed gates.

All reference AND candidate weights are new. No prior development scalar row is
copied. Expected rows:62208 per task per role,248832 across both tasks and roles.
"""
import collections
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch

from phase1.config import CONDITIONS,SPLITS
from phase1.data import load_episode
from phase1.evaluation import summarize,pair_diagnostic,pair_calibration,summarize_calibration
from phase1.inputs import FeatureStore,collate
from phase1.model import DynamicsPredictor
from phase1.centered_transport import CenteredTransportPredictor
from phase1.probes import LinearReadout
from phase1.reporting import decision
from phase1.v5_synthetic_evaluation import fixed_reference_checks as gate_fixed
from phase1.v5_motion_evaluation import fixed_reference_checks as motion_fixed
from phase1.utils import read_json,write_json,write_csv,file_hash,digest
from staged_data import verify_permit
from locked_inference import MotionInference,gate_pair_with_transport


def model_for(c,task,seed,role,basis,permit):
    verify_permit(c,permit,models_locked=True)
    root=Path(c['output']);lock=read_json(root/'all_models_lock.json')
    path=root/'models'/role/task/f'seed_{seed}/best.pt'
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if (ck['signature']['canonical_config_digest']!=digest(c)
            or ck['signature']['permit_sha256']!=file_hash(permit)
            or file_hash(path)!=lock['models'][f'{role}/{task}/{seed}']['best_sha256']):
        raise ValueError('Actual freshly locked checkpoint/permit mismatch')
    ref=DynamicsPredictor(len(basis['input_mean']),c['model'],(3,6),basis)
    if role=='candidate':
        rp=root/'models/reference'/task/f'seed_{seed}/best.pt'
        if ck['signature']['fresh_reference_best_sha256']!=file_hash(rp):
            raise ValueError('Candidate must use its fresh same-seed reference')
        ref.load_state_dict(torch.load(rp,map_location='cpu',weights_only=False)['model'],strict=True)
        model=CenteredTransportPredictor(ref,basis)
    else:model=ref
    model.load_state_dict(ck['model'],strict=True)
    return model.eval().requires_grad_(False),ck


class ExposedObservations:
    """Assembly-only provenance for the original wrong-content intervention.

    Evidence-role metadata chooses which upstream input is replaced in that
    diagnostic condition. Only supplied RGB/latents/actions reach inference.
    """
    def __init__(self,c,store):self.c,self.store,self.lru,self.trace=c,store,collections.OrderedDict(),{}
    def observed(self,rec):
        eid=rec['id']
        if eid not in self.lru:
            self.lru[eid]=load_episode(self.c,rec,keys=('rgb','actions'))
            if len(self.lru)>8:self.lru.popitem(last=False)
        self.lru.move_to_end(eid)
        return self.lru[eid]
    def context(self,rec,sample,info,condition,inference):
        kept=tuple(map(int,info['indices']));sources=[rec,rec]
        if condition=='wrong_evidence':
            evidence=set(map(int,load_episode(self.c,rec,True,keys=('evidence',))['evidence']))
            sources=[self.store.records[rec['sibling']] if t in evidence else rec for t in kept]
        if len(kept)!=2:raise ValueError('Only actual K2 samples have repaired contexts')
        for j,(source,t) in enumerate(zip(sources,kept)):
            if not np.array_equal(sample['z'][4+j],self.store.features(source)[t]):
                raise ValueError('Exposed RGB source does not match actual retained latent')
        own=self.observed(rec)
        rgb=np.stack([self.observed(source)['rgb'][t] for source,t in zip(sources,kept)])
        enabled,geo=inference.context(sample,kept,rgb,own['rgb'][rec['query']],own['actions'],rec['query'])
        key=(rec['id'],condition,kept,tuple(r['id'] for r in sources))
        self.trace[key]=dict(episode=rec['id'],condition=condition,kept=';'.join(map(str,kept)),
            source_ids=';'.join(r['id'] for r in sources),query=rec['query'],enabled=enabled,
            latent_input_sha256=hashlib.sha256(np.ascontiguousarray(sample['z'][4:]).tobytes()).hexdigest(),
            current_input_sha256=hashlib.sha256(np.ascontiguousarray(sample['current']).tobytes()).hexdigest(),
            actions_used_sha256=hashlib.sha256(np.ascontiguousarray(own['actions'][kept[1]:rec['query']+6]).tobytes()).hexdigest())
        return enabled,geo


def evaluate_role(c,task,role,permit):
    verify_permit(c,permit,models_locked=True)
    root=Path(c['output']);out=root/'evaluation'/role/task
    out.mkdir(parents=True,exist_ok=False)
    store=FeatureStore(c,max_cached=16)
    with np.load(root/f'predictor/{task}/basis.npz',allow_pickle=False) as f:basis={k:f[k] for k in f.files}
    store.bases={task:basis}
    decoder=LinearReadout.load(root/f'probes/{task}_state_decoder.npz')
    inference=None
    if task=='motion' and role=='candidate':
        heads={}
        for h in (3,6):
            with np.load(root/f'observation_maps/fitted/head_{h}.npz',allow_pickle=False) as f:
                heads[h]={k:f[k] for k in f.files}
        inference=MotionInference(basis,decoder,heads)
    exposed=ExposedObservations(c,store)
    raw,calibration,pending,witnesses,metadata=[],[],{},[],[]
    for seed in (17,29,43):
        model,ck=model_for(c,task,seed,role,basis,permit)
        before={k:v.clone() for k,v in model.state_dict().items()}
        for split in SPLITS[1:]:
            records=[r for r in store.manifest['records'] if r['task']==task and r['split']==split]
            if len(records)!=2*c['data']['pairs'][split]:raise ValueError('Full fresh split required')
            for condition in CONDITIONS+['persistence']:
                for start in range(0,len(records),8):
                    part=records[start:start+8]
                    samples=[store.sample(r,condition if condition!='persistence' else 'recent_only') for r in part]
                    contexts=[None]*len(part)
                    if condition=='persistence':
                        base=np.stack([np.tile(s[0]['current'],(2,1)) for s in samples]);predictions=base
                        history=base
                        transported=np.zeros_like(base)
                    else:
                        with torch.inference_mode():
                            batch=collate([s[0] for s in samples],'cpu')
                            if task=='gate' and role=='candidate':
                                base,predictions,transported=gate_pair_with_transport(model,batch)
                            else:
                                base=model(batch).numpy();predictions=base.copy();transported=np.zeros_like(base)
                            history=model.reference(batch).numpy() if inference is not None else base
                        if inference is not None:
                            predictions=predictions.astype(float)
                            for j,(rec,(sample,target,info)) in enumerate(zip(part,samples)):
                                if info['history_count']==2:
                                    contexts[j]=exposed.context(rec,sample,info,condition,inference)
                                    enabled,geo=contexts[j]
                                    predictions[j]=inference.correct(base[j],history[j],sample,enabled,geo)
                    if predictions.shape!=(len(part),2,5120) or not np.isfinite(predictions).all():
                        raise ValueError('Full finite5120 feature prediction required')
                    for j,(rec,(_,target,info)) in enumerate(zip(part,samples)):
                        meta=load_episode(c,rec,True,keys=('states','classes'))
                        q=rec['query'];other=store.features(store.records[rec['sibling']])[q+np.array((3,6))]
                        state=decoder.predict(predictions[j]);K=info['history_count']
                        for hi,h in enumerate((3,6)):
                            pred=predictions[j,hi];truth=target[hi]
                            P,Y=pred*store.norm['std']+store.norm['mean'],truth*store.norm['std']+store.norm['mean']
                            accuracy,margin,separation=pair_diagnostic(pred,truth,other[hi])
                            row=dict(seed=seed,task=task,split=split,condition=condition,horizon=h,
                                episode=rec['id'],pair=rec['pair'],gap=rec['gap'],query=q,
                                latent_mse=float(np.mean((P-Y)**2)),normalized_mse=float(np.mean((pred-truth)**2)),
                                cosine_distance=1-float(np.dot(P,Y)/max(np.linalg.norm(P)*np.linalg.norm(Y),1e-12)),
                                pair_accuracy=accuracy,pair_margin=margin,target_pair_distance=separation,
                                state_accuracy=float((state[hi,0]>0)==(meta['classes'][q+h]>0)),
                                state_position_mse=float(np.mean((state[hi,1:]-meta['states'][q+h])**2)),
                                history_count=K,history_feature_bytes=K*5120*4,recent_feature_bytes=4*5120*4,
                                projected_history_bytes=K*128*4,timestamp_type_mask_bytes=(K+4)*13,
                                past_action_bytes=(K+4)*8,future_action_bytes=h*2*4,evidence_hit=float(info['evidence_hit']),
                                state_pred_x=float(state[hi,1]),state_pred_y=float(state[hi,2]),
                                state_true_x=float(meta['states'][q+h,0]),state_true_y=float(meta['states'][q+h,1]))
                            raw.append(row)
                            if task=='gate':
                                key=(seed,task,split,condition,h,rec['pair'])
                                pending.setdefault(key,{})[rec['id']]=(pred,truth)
                                if len(pending[key])==2:
                                    vv=[pending[key][e] for e in sorted(pending[key])]
                                    calibration.append(dict(seed=seed,task=task,split=split,condition=condition,horizon=h,
                                        pair=rec['pair'],**pair_calibration([v[0] for v in vv],[v[1] for v in vv])))
                                    del pending[key]
                            if start==0 and j<2:
                                context=contexts[j]
                                enabled,geo=(context if context is not None else (False,np.zeros((2,2))))
                                metadata.append(dict(seed=seed,task=task,split=split,condition=condition,episode=rec['id'],horizon=h,
                                    kept=';'.join(map(str,info['indices'])),enabled=enabled,
                                    geometry_x=float(geo[hi,0]),geometry_y=float(geo[hi,1]),
                                    feature_mse=row['normalized_mse'],task_error=row['state_position_mse']))
                                witnesses.append((base[j,hi],pred,truth,meta['states'][q+h],history[j,hi],transported[j,hi]))
                write_json(out/'progress.json',dict(role=role,task=task,seed=seed,split=split,
                    condition_completed=condition,rows=len(raw)))
        if any(not torch.equal(before[k],v) for k,v in model.state_dict().items()) or any(v.grad is not None for v in model.parameters()):
            raise ValueError('Evaluation changed frozen predictor')
        del model
    if len(raw)!=62208 or pending:raise ValueError('Incomplete fresh role/task raw or sibling pair')
    write_csv(out/'raw_records.csv',raw)
    profile=copy.deepcopy(c);profile['data']['tasks']=[task]
    metrics={**summarize(profile,raw), 'raw_sha256':file_hash(out/'raw_records.csv'),
        'all_models_lock_sha256':file_hash(root/'all_models_lock.json'),
        'fresh_role':role,'full_raw_count':62208}
    write_json(out/'metrics.json',metrics)
    if calibration:write_json(out/'calibration_summary.json',summarize_calibration(profile,calibration))
    write_csv(out/'witness_metadata.csv',metadata)
    np.savez_compressed(out/'witness_vectors.npz',baseline=np.stack([v[0] for v in witnesses]),
        prediction=np.stack([v[1] for v in witnesses]),target=np.stack([v[2] for v in witnesses]),
        position=np.stack([v[3] for v in witnesses]),reference_history=np.stack([v[4] for v in witnesses]),
        gate_transport=np.stack([v[5] for v in witnesses]))
    if exposed.trace:write_csv(out/'input_trace.csv',list(exposed.trace.values()))
    return raw,metrics


def compare_task(c,task,reference,candidate,metrics):
    root=Path(c['output']);profile=copy.deepcopy(c);profile['data']['tasks']=[task]
    status,reasons,checks=decision(profile,read_json(root/'dataset/split_audit.json'),
        read_json(root/'probes/metrics.json'),metrics)
    fixed=(gate_fixed if task=='gate' else motion_fixed)(profile,reference,candidate)
    if len(checks)!=51 or len(fixed)!=32:raise ValueError('Original51+32 requirement count changed')
    result=dict(status='SUPPORTED' if status=='GO' and all(v['passed'] for v in fixed) else 'REJECTED',
        original_status=status,reasons=reasons,original_checks=checks,fixed_reference_checks=fixed,
        original_passed=sum(v['passed'] for v in checks),fixed_passed=sum(v['passed'] for v in fixed),
        task=task,fresh9137=True,phase1_go=False)
    write_json(root/'evaluation'/task/'comparison.json',result)
    return result
