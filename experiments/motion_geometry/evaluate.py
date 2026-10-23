"""All96 original Motion validation pairs, frozen checkpoints/cache, zero fitting."""
import csv, datetime, hashlib, json, os, socket, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from geometry import observed_forecast, propagate

ROOT = Path(__file__).resolve().parent
AGE = Path('/scratch/xl4358/phase1_age_control_a1')
sys.path.insert(0, str(AGE))
import run as old

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
def savecsv(p, rows):
    with Path(p).open('x', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items(): h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main(job):
    begin=time.monotonic()
    assert str(job)==os.environ['SLURM_JOB_ID']
    native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True)
    fields=dict(t.split('=',1) for t in native.split() if '=' in t)
    assert fields['JobState']=='RUNNING' and fields['Account']=='csci_ga_2271-2026fa' and fields['Partition']=='n2c48m24'
    assert fields['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in fields['AllocTRES']
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    for root, expected in [(ROOT,None),(AGE,'80336bfde8fd9da47d31060e7725e34486dc042df5f0c83e9f2a3161055bba5f')]:
        if expected: assert sha(root/'source_lock.json')==expected
        assert all(sha(root/k)==v for k,v in read(root/'source_lock.json')['files'].items())
    p,_,configs=old.verify_control(old.OLD/'control');c=configs['motion']
    assert c['data']['seed']==6301 and c['data']['horizons']==[3,6]
    parent=read(AGE/'output/job_8275/summary.json');parentassets=read(AGE/'output/job_8275/asset_ledger.json')
    assert sha(AGE/'output/job_8275/summary.json')=='ed9232c2503b76030eea05ad4b54cf565774f952200f69a85306b873b131e35f'
    out=ROOT/'output'/f'job_{job}';out.mkdir(parents=True,exist_ok=False)
    write(out/'attempt.json',dict(job=job,allocation=native,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU=0,training=False,validation_only=True,protocol_sha256=sha(ROOT/'protocol.json')))
    assets={}
    def check(path):
        path=str(path)
        if path not in assets: assets[path]=sha(path)
        if path in parentassets: assert assets[path]==parentassets[path]
    store=old.FeatureStore(c,max_cached=192)
    basispath=Path(c['output'])/'predictor/motion/basis.npz';check(basispath)
    with np.load(basispath,allow_pickle=False) as f: basis={k:f[k] for k in f.files}
    store.bases['motion']=basis
    dp=Path(c['output'])/'probes/motion_state_decoder.npz';check(dp)
    decoder=old.LinearReadout.load(dp)
    for path in [Path(c['output'])/'features/normalization.npz',Path(c['output'])/'features/index.json']:check(path)
    pairs=old.paired_rows([r for r in store.manifest['records'] if r['task']=='motion' and r['split']=='validation'])
    assert len(pairs)==96
    records=[r for pair in pairs for r in pair]
    goldpath=old.OLD/'evaluation/job_7575/motion/evaluation/raw_records.csv';check(goldpath)
    assert sha(goldpath)==read(ROOT/'protocol.json')['baseline_raw_sha256']
    with goldpath.open() as f: gold={(int(r['seed']),r['episode'],r['condition'],int(r['horizon'])):r for r in csv.DictReader(f) if r['split']=='validation'}
    conditions=['recent_only','privileged_small_budget','full_history']
    raw=[];vectors=[];targets=[];proofs={};stateproof={};maxreplay=0.;maxoracle=0.
    for seed in (17,29,43):
        ref,refsha=old.reference_model(c,p,'motion',seed,basis)
        check(Path(c['output'])/'predictor/motion'/f'seed_{seed}/best.pt')
        folder=old.OLD/'training/job_7574/predictor/motion'/f'seed_{seed}'
        proof=old.verify_completed_model(folder,parent['models'][f'motion/{seed}']['signature'],ref.state_dict())
        assert proof==parent['models'][f'motion/{seed}'];proofs[str(seed)]=proof;check(folder/'best.pt')
        ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        model=old.CenteredTransportPredictor(ref,basis);model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False);initial=digest(model)
        for offset in range(0,len(records),8):
            part=records[offset:offset+8]
            samples=[store.sample(rec,condition)[0] for rec in part for condition in conditions]
            with torch.inference_mode(): pred=model(old.collate(samples,'cpu')).numpy().reshape(len(part),3,2,5120)
            assert np.isfinite(pred).all()
            decoded=decoder.predict(pred.reshape(-1,5120)).reshape(len(part),3,2,3)[:,:,:,1:]
            for i,rec in enumerate(part):
                q=int(rec['query']);check(Path(c['output'])/'features'/store.ix['episodes'][rec['id']]['path'])
                z=store.features(rec)
                # Legal, prescribed evidence timestamps; no simulator states used for these forecasts.
                evidence=tuple(map(int,old.load_episode(c,rec,True,keys=('evidence',))['evidence']))
                assert len(evidence)==2 and evidence[1]<q-c['data']['recent']+1
                actions=old.load_episode(c,rec,keys=('actions',))['actions']
                observed=decoder.predict(z[list(evidence)])[:,1:]
                forecasts=[]
                for j,h in enumerate((3,6)):
                    legal,v,nref=observed_forecast(observed,evidence,actions[evidence[1]:q+h],q+h,True)
                    nocontrol,_,_=observed_forecast(observed,evidence,actions[evidence[1]:q+h],q+h,False)
                    forecasts.append((legal,nocontrol,v,nref))
                # Scoring and isolated oracle attribution begin only after all normal predictions exist.
                truth=old.load_episode(c,rec,True,keys=('states','visible'))
                states=truth['states'];assert truth['visible'][list(evidence)].all()
                truev=float((states[evidence[1],0]-states[evidence[0],0])/(evidence[1]-evidence[0]))
                for j,h in enumerate((3,6)):
                    target=z[q+h];xy=states[q+h];legal,nocontrol,v,nref=forecasts[j]
                    oraclepos,_=propagate(states[evidence[1]],v,actions[evidence[1]:q+h],q+h-evidence[1])
                    oraclevel,_=propagate(observed[1],truev,actions[evidence[1]:q+h],q+h-evidence[1])
                    oracleboth,_=propagate(states[evidence[1]],truev,actions[evidence[1]:q+h],q+h-evidence[1])
                    maxoracle=max(maxoracle,float(np.max(np.abs(oracleboth-xy))))
                    assert maxoracle<2e-6
                    r=dict(seed=seed,pair=rec['pair'],episode=rec['id'],horizon=h,query=q,start=evidence[0],end=evidence[1],dt=evidence[1]-evidence[0],steps=q+h-evidence[1],target_x=float(xy[0]),target_y=float(xy[1]),observed_start_x=float(observed[0,0]),observed_start_y=float(observed[0,1]),observed_end_x=float(observed[1,0]),observed_end_y=float(observed[1,1]),true_start_x=float(states[evidence[0],0]),true_start_y=float(states[evidence[0],1]),true_end_x=float(states[evidence[1],0]),true_end_y=float(states[evidence[1],1]),true_velocity=truev,estimated_velocity=v,reflections=nref,observed_x_outside_arena=bool(np.any((observed[:,0]<.08)|(observed[:,0]>.92))))
                    positions={'kinematic_actions':legal,'kinematic_no_actions':nocontrol,'hold_last_visible':observed[1],'oracle_position_only':oraclepos,'oracle_velocity_only':oraclevel,'oracle_both':oracleboth,'true_target_readout':decoder.predict(target[None])[0,1:]}
                    for k,x in positions.items():r[k+'_x']=float(x[0]);r[k+'_y']=float(x[1]);r[k+'_mse']=float(np.mean((x-xy)**2))
                    for k,condition in enumerate(conditions):
                        loss=float(np.mean((pred[i,k,j]-target)**2));posloss=float(np.mean((decoded[i,k,j]-xy)**2));g=gold[(seed,rec['id'],condition,h)]
                        maxreplay=max(maxreplay,abs(loss-float(g['normalized_mse'])),abs(posloss-float(g['state_position_mse'])))
                        assert maxreplay<2e-6
                        r[condition+'_feature_mse']=loss;r[condition+'_x']=float(decoded[i,k,j,0]);r[condition+'_y']=float(decoded[i,k,j,1]);r[condition+'_mse']=posloss
                    raw.append(r);vectors.append(pred[i,:,j]);targets.append(target)
            write(out/'progress.json',dict(seed=seed,validation_episodes_completed=offset+len(part),rows=len(raw),expected_rows=1152))
            if time.monotonic()-begin>1200:raise RuntimeError('CPU soft bound; preserve partial; no retry')
        assert digest(model)==initial and all(x.grad is None for x in model.parameters());stateproof[str(seed)]=initial
        del model,ref,ck
    assert len(raw)==1152 and len({r['pair'] for r in raw})==96
    savecsv(out/'validation_geometry.csv',raw)
    np.savez_compressed(out/'prediction_witnesses.npz',predictions=np.stack(vectors),targets=np.stack(targets))
    assert all(sha(path)==value for path,value in assets.items())
    assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
    old.verify_control(old.OLD/'control')
    write(out/'asset_ledger.json',assets)
    write(out/'summary.json',dict(status='VALIDATION_DIAGNOSIS_COMPLETE_NOT_ACCEPTANCE',job=job,allocation=native,rows=1152,pairs=96,seeds=[17,29,43],horizons=[3,6],model_proofs=proofs,state_proofs=stateproof,baseline_replay_max_difference=maxreplay,oracle_transition_max_difference=maxoracle,raw_sha256=sha(out/'validation_geometry.csv'),witness_sha256=sha(out/'prediction_witnesses.npz'),asset_ledger_sha256=sha(out/'asset_ledger.json'),protocol_sha256=sha(ROOT/'protocol.json'),source_lock_sha256=sha(ROOT/'source_lock.json'),training=False,fitting=False,GPU=0,test_forward=False,fresh9137_opened=False,original_gates_changed=False,elapsed=time.monotonic()-begin))

if __name__=='__main__':
    try:main(int(sys.argv[1]))
    except Exception:
        import traceback
        p=ROOT/'output'/f"job_{os.environ.get('SLURM_JOB_ID','unknown')}";p.mkdir(parents=True,exist_ok=True);write(p/'failure.json',dict(error=traceback.format_exc(),no_retry=True));raise
