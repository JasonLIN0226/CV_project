"""Actual-node compatibility proof using only previously revealed seed6301.

No fresh9137 generation, no new encoder calls, and no accepted model is trained.
The optional GPU epoch replicas are disposable execution-equivalence fixtures;
their entire allocation wall is charged and none of their epochs count toward
the twelve fresh confirmation models.
"""
import collections
import copy
import csv
import hashlib
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from phase1 import data,cache
from phase1.inputs import FeatureStore,collate
from phase1.model import DynamicsPredictor
from phase1.probes import LinearReadout
from phase1.training import paired_rows,TRAIN_CONDITIONS,regression_loss
from phase1.repair_trial import train_epoch,actual_allocation_start
from phase1.utils import read_json,write_json,file_hash,seed_all,rng_state
from staged_data import (StageCachedFeatureStore,audit_stage,audit_stage_features,TRAIN_SPLITS)
from fresh_training import make_task_store
from locked_inference import MotionInference,gate_pair_with_transport
from gate_intervention import both,state_digest
from pool_inputs import sample as fork_sample

OLD=Path('/scratch/xl4358/phase1_predictor_repair_v2/execution')


def encoder_cache_proof():
    from huggingface_hub.constants import HF_HUB_CACHE
    revision='f157467ea509bc356ff9f61fd3c0d840eec5e04e'
    snapshot=Path(HF_HUB_CACHE)/'models--facebook--ijepa_vith14_1k'/'snapshots'/revision
    files={}
    paths=[p for p in sorted(snapshot.iterdir()) if p.suffix in ('.json','.safetensors','.bin')]
    if not paths or not any(p.suffix in ('.safetensors','.bin') for p in paths):
        raise ValueError('Previously approved official cached I-JEPA weights are absent; no implicit download')
    for path in paths:
        before=path.stat();resolved=path.resolve();object_id=resolved.name
        sha256=hashlib.sha256();git=hashlib.sha1();git.update(f'blob {before.st_size}\0'.encode())
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):sha256.update(chunk);git.update(chunk)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Encoder file changed while hashing')
        expected=sha256.hexdigest() if len(object_id)==64 else git.hexdigest() if len(object_id)==40 else None
        if expected!=object_id:raise ValueError('Cached official Git/LFS object content hash does not match its fixed object ID')
        files[str(path)]=dict(sha256=sha256.hexdigest(),resolved_path=str(resolved),bytes=after.st_size,
            mtime_ns=after.st_mtime_ns,official_object_id=object_id)
    return dict(revision=revision,files=files,official_object_content_verified=True,no_download=True)


def stage_view(root,task):
    original=Path('/scratch/xl4358/phase1_v3_dev')/task/'reference'
    c=read_json(original/'resolved_config.json')
    if c['data']['seed']!=6301:raise ValueError('Only existing development data allowed in equivalence proof')
    view=root/'dev_equivalence'/task;view.mkdir(parents=True,exist_ok=False)
    s=copy.deepcopy(c);s['output']=str(view)
    m=read_json(original/'dataset/manifest.json')
    m['records']=[r for r in m['records'] if r['split'] in TRAIN_SPLITS]
    for rec in m['records']:
        for key in ('obs','eval'):rec[key]=str(original/'dataset'/rec[key])
    write_json(view/'dataset/manifest.json',m)
    ix=read_json(original/'features/index.json');allowed={r['id'] for r in m['records']}
    ix['episodes']={eid:info for eid,info in ix['episodes'].items() if eid in allowed}
    for info in ix['episodes'].values():info['path']=str(original/'features'/info['path'])
    write_json(view/'features/index.json',ix)
    for name in ('normalization.npz','normalization.json'):
        shutil.copy2(original/'features'/name,view/'features'/name)
    basisdir=view/'predictor'/task;basisdir.mkdir(parents=True)
    for name in ('basis.npz','basis.json'):shutil.copy2(original/'predictor'/task/name,basisdir/name)
    write_json(view/'resolved_config.json',s)
    audit_stage(s);audit_stage_features(s,m,ix)
    return c,s


def motion_cached_equivalence():
    root=Path('/scratch/xl4358/phase1_motion_observation_mix_locked_test/output/job_8522')
    summary=read_json(root/'summary.json');audit=read_json(root/'independent_audit.json')
    if summary['status']!='MOTION_DEVELOPMENT_REPAIR_SUPPORTED' or audit['status']!='PASS_INDEPENDENT_LOCKED_MOTION_AUDIT':
        raise ValueError('Accepted Motion native result/audit required')
    with (root/'witness_metadata.csv').open() as f:meta=list(csv.DictReader(f))
    if file_hash(root/'witness_metadata.csv')!=summary['witness_metadata_sha256']:
        raise ValueError('Accepted primitive metadata changed')
    if file_hash(root/'witness_vectors.npz')!=summary['witness_sha256']:raise ValueError('Accepted primitive vectors changed')
    with np.load(root/'witness_vectors.npz',allow_pickle=False) as f:old={k:f[k] for k in f.files}
    c=read_json('/scratch/xl4358/phase1_v3_dev/motion/reference/resolved_config.json')
    store=FeatureStore(c,max_cached=8)
    with np.load(Path(c['output'])/'predictor/motion/basis.npz',allow_pickle=False) as f:basis={k:f[k] for k in f.files}
    store.bases={'motion':basis};decoder=LinearReadout.load(Path(c['output'])/'probes/motion_state_decoder.npz')
    heads={}
    accepted=read_json(root.parent.parent/'candidate_lock.json')
    for h in (3,6):
        path=Path('/scratch/xl4358/phase1_motion_observation_map/gpu_output')/f'head_{h}.npz'
        if file_hash(path)!=accepted['observation_heads'][str(path)]:raise ValueError('Accepted map changed')
        with np.load(path,allow_pickle=False) as f:heads[h]={k:f[k] for k in f.files}
    inference=MotionInference(basis,decoder,heads)
    from fresh_evaluation import ExposedObservations
    exposed=ExposedObservations(c,store);worst=0.
    for i in range(0,len(meta),2):
        a,b=meta[i:i+2]
        if (a['episode'],a['condition'],a['kept'],int(a['horizon']))!=(b['episode'],b['condition'],b['kept'],3) or int(b['horizon'])!=6:
            raise ValueError('Accepted vector horizon pairing changed')
        rec=store.records[a['episode']];kept=tuple(map(int,a['kept'].split(';')))
        if a['condition']=='deletion':
            sample=fork_sample(store.features(rec),store.projected_features(rec),exposed.observed(rec)['actions'],rec['query'],kept)
            info={'indices':kept}
        else:sample,_,info=store.sample(rec,a['condition'])
        enabled,geo=exposed.context(rec,sample,info,a['condition'],inference)
        out=inference.correct(old['baseline'][i:i+2],old['reference_history'][i:i+2],sample,enabled,geo)
        worst=max(worst,float(abs(out-old['prediction'][i:i+2]).max()))
    if len(meta)!=468 or worst>1e-7:raise ValueError('Fresh inference differs from accepted Motion method')
    return dict(vectors=468,max_difference=worst,no_model_forward=True)


def gate_forward_equivalence():
    from phase1.proposal_trial import verify_control,reference_model
    from phase1.centered_transport import CenteredTransportPredictor
    p,_,configs=verify_control(OLD/'control');c=configs['gate'];store=FeatureStore(c,max_cached=16)
    with np.load(Path(c['output'])/'predictor/gate/basis.npz',allow_pickle=False) as f:basis={k:f[k] for k in f.files}
    store.bases={'gate':basis}
    records=[r for r in store.manifest['records'] if r['task']=='gate' and r['split']=='validation'][:8]
    count=0;worst=0.
    accepted=read_json('/scratch/xl4358/phase1_gate_locked_test/candidate_lock.json')
    for seed in (17,29,43):
        reference,_=reference_model(c,p,'gate',seed,basis)
        model=CenteredTransportPredictor(reference,basis)
        ck=torch.load(OLD/f'training/job_7574/predictor/gate/seed_{seed}/best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False);before=state_digest(model)
        if before!=accepted['state_digest_by_seed'][str(seed)]:raise ValueError('Accepted Gate checkpoint changed')
        for condition in (*TRAIN_CONDITIONS,'wrong_evidence'):
            samples=[store.sample(r,condition) for r in records];batch=collate([s[0] for s in samples],'cpu')
            expected,residual=both(model,batch)
            base,off,transport=gate_pair_with_transport(model,batch)
            worst=max(worst,float(abs(base-expected['before']).max()),float(abs(off-expected['transport_off']).max()))
            if worst!=0 or residual>=3e-5:raise ValueError('Accepted Gate ablation changed')
            count+=len(samples)*2
        if state_digest(model)!=before:raise ValueError('Gate fixture changed an accepted weight')
    return dict(full_feature_predictions=count,max_difference=worst,all_seeds=True,training=False)


def cpu_equivalence(root,protocol):
    proofs={};samples=validation_batches=0
    for task in ('gate','motion'):
        c,s=stage_view(root,task)
        original=FeatureStore(c,max_cached=1216);store=StageCachedFeatureStore(s,max_cached=1216)
        with np.load(Path(s['output'])/f'predictor/{task}/basis.npz',allow_pickle=False) as f:basis={k:f[k] for k in f.files}
        original.bases=store.bases={task:basis}
        for key in original.norm:
            if not np.array_equal(original.norm[key],store.norm[key]):raise ValueError('Stage normalization differs')
        rows=[r for r in store.manifest['records'] if r['task']==task]
        for i,rec in enumerate(rows):
            for condition in TRAIN_CONDITIONS:
                ra,rb=np.random.default_rng(90210+i),np.random.default_rng(90210+i)
                a=original.sample(original.records[rec['id']],condition,ra);b=store.sample(rec,condition,rb)
                for aa,bb in ((a[0],b[0]),(a[2],b[2])):
                    if set(aa)!=set(bb) or any(not np.array_equal(aa[k],bb[k]) or np.asarray(aa[k]).dtype!=np.asarray(bb[k]).dtype for k in aa):
                        raise ValueError('Native full original/staged input or info differs')
                if not np.array_equal(a[1],b[1]) or ra.bit_generator.state!=rb.bit_generator.state:
                    raise ValueError('Native full original/staged target or RNG differs')
                samples+=1
        pairs=paired_rows([r for r in rows if r['split']=='validation'])
        store.prepare_validation(pairs,'cpu');batches=store.verify_validation_inputs(pairs)
        validation_batches+=batches
        proofs[task]=dict(samples=len(rows)*4,validation_batches=batches,
            metadata_fingerprint=store.metadata_fingerprint,all_exact=True,seed=6301)
    if samples!=9728 or validation_batches!=24:raise ValueError('Native development proof incomplete')
    gate=gate_forward_equivalence();motion=motion_cached_equivalence()
    write_json(root/'preflight_cpu.json',dict(all_exact=True,samples=samples,validation_batches=validation_batches,
        tasks=proofs,gate=gate,motion=motion,encoder_cache=encoder_cache_proof(),
        fresh9137_accessed=False,CPU_training=False,GPU_seconds=0))


def original_reference_epoch(c,store,model,opt,pairs,seed):
    """Original reference inner loop, independently executed with no anchor."""
    model.train();rng=np.random.default_rng(seed);order=rng.permutation(len(pairs))
    counts={k:0 for k in TRAIN_CONDITIONS};totals={k:np.zeros(4,float) for k in TRAIN_CONDITIONS}
    for start in range(0,len(order),c['training']['batch_size']//2):
        selected=[pairs[i] for i in order[start:start+c['training']['batch_size']//2]]
        for condition in rng.permutation(list(TRAIN_CONDITIONS)):
            samples=[]
            for pair in selected:
                sample_seed=int(rng.integers(2**32))
                samples.extend(store.sample(r,str(condition),np.random.default_rng(sample_seed)) for r in pair)
            batch=collate([s[0] for s in samples],'cuda');target=torch.from_numpy(np.stack([s[1] for s in samples])).cuda()
            opt.zero_grad(set_to_none=True);prediction=model(batch)
            loss,mse,paired=regression_loss(prediction,target,c['training']['pair_weight'],c['training']['rank_weight'])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['training']['clip'],error_if_nonfinite=True);opt.step()
            counts[str(condition)]+=len(samples)
            totals[str(condition)]+=np.asarray([float(loss.detach()),float(mse.detach()),float(paired.detach()),0.])*len(samples)
    return {'steps':64,'counts':counts,'means':{k:(totals[k]/counts[k]).tolist() for k in TRAIN_CONDITIONS}}


def equal_tree(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def gpu_equivalence(root,protocol,guard):
    start,native=actual_allocation_start();guard.check(start)
    if torch.cuda.get_device_name(0)!=protocol['expected_GPU'] or torch.__version__!=protocol['expected_torch']:
        raise ValueError('Inherited actual rate proof requires the same A100/Torch runtime')
    proofs={}
    for task in ('gate','motion'):
        c=read_json(root/f'dev_equivalence/{task}/resolved_config.json')
        store,basis,train,validation=make_task_store(c,task)
        seed_all(17,4);one=DynamicsPredictor(5120,c['model'],(3,6),basis).cuda()
        seed_all(17,4);two=DynamicsPredictor(5120,c['model'],(3,6),basis).cuda()
        a=torch.optim.AdamW(one.parameters(),lr=.001,weight_decay=.0001)
        b=torch.optim.AdamW(two.parameters(),lr=.001,weight_decay=.0001)
        seed_all(17,4)
        left=train_epoch(c,store,one,a,train,17,0,'cuda','reference_no_anchor',{'requested':False},float('inf'))
        left_rng=rng_state()
        seed_all(17,4);right=original_reference_epoch(c,store,two,b,train,17);right_rng=rng_state()
        if not (equal_tree(left,right) and equal_tree(one.state_dict(),two.state_dict())
                and equal_tree(a.state_dict(),b.state_dict()) and equal_tree(left_rng,right_rng)):
            raise ValueError('Native fresh reference kernel differs from the original complete epoch/update/RNG')
        proofs[task]=dict(complete_epoch=1,steps=64,input_conditions=4,all_logs_model_optimizer_rng_exact=True)
        guard.check(start);del one,two,a,b,store;torch.cuda.empty_cache()
    write_json(root/'preflight_gpu.json',dict(all_exact=True,tasks=proofs,allocation=native,
        fresh9137_accessed=False,disposable_reference_fixtures_not_confirmation_models=True,
        all_fixture_allocation_wall_charged=True,GPU=torch.cuda.get_device_name(0),torch=torch.__version__))
    guard.state['native_GPU_equivalence_done']=True;guard.save();guard.check(start)
