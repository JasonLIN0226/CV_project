"""Inference-only, frozen job7574; equal-capacity deletion forks on cached6301."""
import csv,datetime,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from assay_core import pools,forks,summarize,baseline_summary

ROOT=Path('/scratch/xl4358/phase1_deletion_assay')
OLD=Path('/scratch/xl4358/phase1_predictor_repair_v2/execution')
sys.path.insert(0,str(OLD/'control/src'))
from phase1.proposal_trial import verify_control,reference_model,verify_completed_model
from phase1.centered_transport import CenteredTransportPredictor
from phase1.inputs import FeatureStore,collate
from phase1.data import load_episode
from phase1.probes import LinearReadout
from phase1.training import paired_rows

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)
def native(job):
    text=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True)
    d=dict(v.split('=',1) for v in text.split() if '=' in v)
    assert d['JobState']=='RUNNING' and d['Account']=='csci_ga_2271-2026fa'
    assert d['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in d['AllocTRES']
    return text
def sample(z,p,actions,q,kept):
    R=4;idx=np.r_[np.arange(q-R+1,q+1),np.array(kept,dtype=np.int64)]
    assert len(kept)==2 and len(set(kept))==2 and max(kept)<q-R+1 and min(kept)>=0
    prefix=np.vstack([np.zeros((1,2),np.float32),np.cumsum(actions[:q],0)])
    return dict(z=z[idx].copy(),projected_z=p[idx].copy(),projected_current=p[q].copy(),current=z[q].copy(),
       age=(q-idx).astype(np.float32),type=np.r_[np.zeros(R,np.int64),np.ones(2,np.int64)],
       past_actions=prefix[q]-prefix[idx],actions=actions[q:q+6].copy())

def main():
    begin=time.monotonic();job=int(os.environ['SLURM_JOB_ID']);allocation=native(job)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES','') in ('','-1')
    out=ROOT/'output'/f'job_{job}';out.mkdir(parents=True,exist_ok=False)
    lock=read(ROOT/'source_lock.json');assert all(sha(ROOT/k)==v for k,v in lock['files'].items())
    protocol=read(ROOT/'protocol.json')
    parents=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-n','-P','-j','7574,7575,7576','--format=JobID,State,ExitCode'],text=True)
    assert set(parents.strip().splitlines())=={f'{j}|COMPLETED|0:0' for j in (7574,7575,7576)}
    assert sha(OLD/'final_native_verification.json')=='dc8eaf5ce1276698e8fc31539167da1d0b57511aaf7bb9a4a2e7107531eaa967'
    p,old_lock,configs=verify_control(OLD/'control')
    models=read(OLD/'training/job_7574/all_models_completion.json')['models']
    assert len(models)==6
    ledgertext=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-n','-P','-j','7477,7480,7485,7524,7535,7574,7582','--format=JobID,State,ExitCode,ElapsedRaw,Account,AllocTRES'],text=True)
    ledger={int(v[0]):int(v[3]) for line in ledgertext.splitlines() if (v:=line.split('|')) and v[0].isdigit()}
    assert len(ledger)==7 and sum(ledger.values())==12723 and sum(ledger.values())<=28800
    proofs={};assets={};software=[];allrows=[]
    for task in ('gate','motion'):
        c=configs[task];store=FeatureStore(c,max_cached=4)
        basispath=Path(c['output'])/'predictor'/task/'basis.npz'
        assert sha(basispath)==p['immutable_inputs'][task]['basis_sha256']
        with np.load(basispath,allow_pickle=False) as f:basis={k:f[k] for k in f.files}
        store.bases[task]=basis
        dp=Path(c['output'])/'probes'/f'{task}_state_decoder.npz'
        assert sha(dp)==read(Path(c['output'])/'probes/metrics.json')['checkpoints'][dp.name]
        decoder=LinearReadout.load(dp)
        for asset in (basispath,dp,Path(c['output'])/'features/normalization.npz',Path(c['output'])/'features/index.json'):
            if asset.is_file():assets[str(asset)]=sha(asset)
        for seed in (17,29,43):
            reference,_=reference_model(c,p,task,seed,basis)
            folder=OLD/'training/job_7574/predictor'/task/f'seed_{seed}'
            proof=verify_completed_model(folder,models[f'{task}/{seed}']['signature'],reference.state_dict())
            assert proof==models[f'{task}/{seed}'];cp=folder/'best.pt';assets[str(cp)]=sha(cp)
            ck=torch.load(cp,map_location='cpu',weights_only=False)
            model=CenteredTransportPredictor(reference,basis);model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False)
            proofs[f'{task}/{seed}']=proof
            for split in ('test_id','test_long_gap','test_distractor_shift'):
                pairs=paired_rows([r for r in store.manifest['records'] if r['task']==task and r['split']==split])
                assert len(pairs)==128
                rows=[r for pair in pairs for r in pair]
                for offset in range(0,len(rows),8):
                    samples=[];descriptions=[];targetlist=[]
                    for rec in rows[offset:offset+8]:
                        z=store.features(rec);projected=store.projected_features(rec);q=rec['query']
                        ev=load_episode(c,rec,True,keys=('evidence',))['evidence']
                        actions=load_episode(c,rec,keys=('actions',))['actions']
                        # Hash actually used input caches once, then rehash every used asset at exit.
                        cache=Path(c['output'])/'features'/store.ix['episodes'][rec['id']]['path']
                        if str(cache) not in assets:assets[str(cache)]=sha(cache)
                        # Gate cue spans1-3frames; exclude the entire cue window,
                        # not just its annotated first evidence timestamp.
                        visible_end=q-int(rec['gap'])-4
                        for pname,parent in pools(task,ev,q,4,visible_end):
                            roles={int(t):('cue' if task=='gate' else ('start' if i==0 else 'end')) for i,t in enumerate(ev)}
                            for deleted,kept in forks(parent):
                                s=sample(z,projected,actions,q,kept);samples.append(s)
                                descriptions.append((rec,pname,parent,deleted,kept,roles.get(deleted,'nuisance')))
                                targetlist.append(z[q+np.array([3,6])])
                    batch=collate(samples,'cpu')
                    with torch.inference_mode():pred=model(batch)
                    if offset==0 and split=='test_id':
                        with torch.inference_mode():
                            repeat=model(batch)
                            perm={k:(v[:,[0,1,2,3,5,4]] if k in ('z','projected_z','age','type','past_actions','mask') else v) for k,v in batch.items()}
                            shuffled=model(perm)
                            changed={k:v.clone() for k,v in batch.items()};changed['actions'][:,3:]+=10
                            h3=model(changed)[:,0]
                        errs=dict(repeat=float((pred-repeat).abs().max()),permutation=float((pred-shuffled).abs().max()),h3_suffix=float((pred[:,0]-h3).abs().max()))
                        assert errs['repeat']<1e-6 and errs['permutation']<3e-5 and errs['h3_suffix']<1e-6
                        software.append(dict(task=task,seed=seed,errors=errs,passed=True))
                    pred=pred.numpy();target=np.asarray(targetlist)
                    assert pred.shape==target.shape and pred.shape[-1]==5120 and np.isfinite(pred).all()
                    state=decoder.predict(pred.reshape(-1,5120)).reshape(len(pred),2,-1)
                    for i,(rec,pname,parent,deleted,kept,role) in enumerate(descriptions):
                        # All truth is evaluation-only and first read AFTER the model call.
                        meta=load_episode(c,rec,True,keys=('states','classes'))
                        for hi,h in enumerate((3,6)):
                            taskerr=float((state[i,hi,0]>0)!=(meta['classes'][rec['query']+h]>0)) if task=='gate' else float(np.mean((state[i,hi,1:]-meta['states'][rec['query']+h])**2))
                            allrows.append(dict(task=task,seed=seed,split=split,pair=rec['pair'],episode=rec['id'],pool=pname,
                                parent=';'.join(map(str,parent)),deleted=deleted,deleted_role=role,kept=';'.join(map(str,kept)),history_count=2,horizon=h,
                                feature_mse=float(np.mean((pred[i,hi]-target[i,hi])**2)),task_error=taskerr))
                write(out/'progress.json',dict(task=task,seed=seed,split_completed=split,raw_rows=len(allrows),GPU_seconds=0,elapsed=time.monotonic()-begin))
                print(f'{task} seed{seed} {split} complete; raw_rows={len(allrows)}',flush=True)
            assert sha(cp)==proof['best_sha256'];del ck,model,reference
    raw=out/'raw_deletions.csv'
    with raw.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    result,compact=summarize(allrows)
    write(out/'baselines.json',baseline_summary(allrows))
    for task in ('gate','motion'):
      for split in ('test_id','test_long_gap','test_distractor_shift'):
        sub=[r for r in compact if r['task']==task and r['split']==split]
        path=out/f'pair_contrasts_{task}_{split}.csv'
        with path.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(sub[0]));w.writeheader();w.writerows(sub)
    assert all(sha(Path(k))==v for k,v in assets.items())
    verify_control(OLD/'control');assert all(sha(ROOT/k)==v for k,v in lock['files'].items())
    write(out/'asset_ledger.json',assets)
    result.update(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=job,allocation=allocation,parents=parents,
        no_training_or_fitting=True,GPU_seconds=0,prior_GPU_seconds=12723,remaining_GPU_seconds=16077,
        GPU_native_ledger=ledger,GPU_native_sacct=ledgertext,models=proofs,software_checks=software,
        protocol_sha256=sha(ROOT/'protocol.json'),source_lock_sha256=sha(ROOT/'source_lock.json'),raw_sha256=sha(raw),
        asset_ledger_sha256=sha(out/'asset_ledger.json'),elapsed_seconds=time.monotonic()-begin,
        interpretation='Post-hoc development mechanism assay; privileged pools and future-selected oracle are not online policies; previous predictor rejection remains.')
    write(out/'summary.json',result)
    write(ROOT/'latest_output.json',dict(job=job,output=str(out),summary_sha256=sha(out/'summary.json')))
    print(json.dumps(dict(status='ASSAY_COMPLETE',decision=result['decision'],tasks=result['tasks'],seconds=result['elapsed_seconds'])),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write(ROOT/'failure.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=os.environ.get('SLURM_JOB_ID'),error=traceback.format_exc()))
        raise
