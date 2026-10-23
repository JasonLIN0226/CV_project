"""One fixed scoring formula; unchanged predictors, encoder caches and outcomes."""
import collections,csv,datetime,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from core import maps,constrain
from guard import allow_update
from geometry import observed_forecast
from analysis import aggregate,summarize
ROOT=Path(__file__).resolve().parent;PARENT=Path('/scratch/xl4358/phase1_motion_locked_test');AGE=Path('/scratch/xl4358/phase1_age_control_a1')
sys.path.insert(0,str(AGE));import run as old
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvread(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def savecsv(p,rr):
    with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def digest(m):
    x=hashlib.sha256()
    for k,v in m.state_dict().items():x.update(k.encode());x.update(v.cpu().contiguous().numpy().tobytes())
    return x.hexdigest()
def decoded_score(pred,target,A):
    e=np.asarray(pred,float)-np.asarray(target,float);return float(np.mean((e@A)**2))
def main(job):
    begin=time.monotonic();native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);n=dict(t.split('=',1) for t in native.split() if '=' in t)
    assert n['JobState']=='RUNNING' and n['Account']=='csci_ga_2271-2026fa' and n['Partition']=='n2c48m24' and n['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in n['AllocTRES']
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    protocol=read(ROOT/'protocol.json');parent=Path(protocol['parent']);val=Path(protocol['validation_root'])
    assert sha(parent/'summary.json')==protocol['parent_summary_sha256'] and sha(parent/'raw_deletions.csv')==protocol['parent_raw_sha256']
    assert sha(val/'summary.json')==protocol['validation_summary_sha256'] and sha(val/'independent_audit.json')==protocol['validation_audit_sha256']
    assert read(val/'independent_audit.json')['status']=='PASS_INDEPENDENT_SVD_RESIDUAL_AND_RANK_AUDIT'
    lock=read(PARENT/'candidate_lock.json');assert sha(PARENT/'candidate_lock.json')==protocol['candidate_lock_sha256']
    for root in (ROOT,PARENT,AGE):assert all(sha(root/k)==v for k,v in read(root/'source_lock.json')['files'].items())
    assert all(sha(ROOT/k)==sha(PARENT/k) for k in ('core.py','geometry.py','guard.py'))
    assert all(sha(k)==v for k,v in lock['weights_and_transforms'].items())
    p,_,cc=old.verify_control(old.OLD/'control');c=cc['motion'];assert c['data']['seed']==6301 and c['data']['budget']==2 and c['data']['recent']==4
    store=old.FeatureStore(c,max_cached=16)
    with np.load(Path(c['output'])/'predictor/motion/basis.npz',allow_pickle=False) as f:basis={k:f[k] for k in f.files}
    store.bases['motion']=basis;decoder=old.LinearReadout.load(Path(c['output'])/'probes/motion_state_decoder.npz');B,A,b,right,_=maps(basis['output_basis'],decoder.weight,decoder.std,decoder.mean,decoder.bias)
    source=csvread(parent/'raw_deletions.csv');assert len(source)==82944
    index={(int(r['seed']),r['split'],r['episode'],r['pool'],int(r['deleted']),int(r['horizon'])):r for r in source};assert len(index)==82944
    out=ROOT/'output'/f'job_{job}';out.mkdir(parents=True,exist_ok=False);rows=[];W=[];M=[];assets=dict(lock['weights_and_transforms']);models={};maximum=0.;newscore_identity=0.
    def check(path):
        name=str(path)
        if name not in assets:assets[name]=sha(path)
    for seed in (17,29,43):
        ref,_=old.reference_model(c,p,'motion',seed,basis);folder=old.OLD/'training/job_7574/predictor/motion'/f'seed_{seed}'
        proof=old.verify_completed_model(folder,read(parent/'summary.json')['model_proofs'][str(seed)]['signature'],ref.state_dict());assert proof==read(parent/'summary.json')['model_proofs'][str(seed)]
        ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);m=old.CenteredTransportPredictor(ref,basis);m.load_state_dict(ck['model'],strict=True);m.eval().requires_grad_(False);initial=digest(m);assert initial==lock['state_digest_by_seed'][str(seed)];models[str(seed)]=proof
        for split in protocol['scope_grid']['splits']:
            pairs=old.paired_rows([r for r in store.manifest['records'] if r['task']=='motion' and r['split']==split]);assert len(pairs)==128;records=[r for pair in pairs for r in pair]
            for start in range(0,len(records),8):
                samples=[];desc=[];context=[]
                for rec in records[start:start+8]:
                    q=int(rec['query']);z=store.features(rec);proj=store.projected_features(rec);obs=old.load_episode(c,rec,keys=('rgb','actions'));check(Path(c['output'])/'dataset'/rec['obs']);check(Path(c['output'])/'features'/store.ix['episodes'][rec['id']]['path'])
                    for pi,pool in enumerate(old.clock_only_static_pools(q,old.stable_seed(20261017,'clock_only',rec['pair']),count=6)):
                        for deleted in pool:
                            kept=tuple(t for t in pool if t!=deleted);samples.append(old.sample(z,proj,obs['actions'],q,kept));desc.append((rec,pi,pool,deleted,kept))
                            enabled=allow_update(obs['rgb'][list(kept)],obs['rgb'][q]);xy=decoder.predict(z[list(kept)])[:,1:]
                            geometry=np.stack([observed_forecast(xy,kept,obs['actions'][kept[1]:q+h],q+h)[0] for h in (3,6)]);context.append((enabled,geometry))
                with torch.inference_mode():base=m(old.collate(samples,'cpu')).numpy()
                assert base.shape==(len(samples),2,5120) and np.isfinite(base).all()
                for j,((rec,pi,pool,deleted,kept),(enabled,geo)) in enumerate(zip(desc,context)):
                    # Parent float32 arithmetic is preserved for the disabled-guard branch.
                    pred=constrain(base[j],geo,B,A,b,right)[0] if enabled else base[j]
                    q=int(rec['query']);z=store.features(rec);truth=old.load_episode(c,rec,True,keys=('states',));xy=decoder.predict(pred)[:,1:]
                    for hi,h in enumerate((3,6)):
                        original=index[seed,split,rec['id'],f'clock_{pi}',deleted,h];assert enabled==(original['guard_enabled']=='True')
                        target=z[q+h];gt=truth['states'][q+h];fm=float(np.mean((pred[hi]-target)**2));te=float(np.mean((xy[hi]-gt)**2));maximum=max(maximum,abs(fm-float(original['feature_mse'])),abs(te-float(original['task_error'])));assert maximum<2e-6
                        target_xy=decoder.predict(target)[1:];score=decoded_score(pred[hi],target,A);identity=float(np.mean((xy[hi]-target_xy)**2));newscore_identity=max(newscore_identity,abs(score-identity));assert newscore_identity<1e-10
                        row=dict(original,decoded_score=score,pred_x=float(xy[hi,0]),pred_y=float(xy[hi,1]),target_decoded_x=float(target_xy[0]),target_decoded_y=float(target_xy[1]),true_x=float(gt[0]),true_y=float(gt[1]));rows.append(row)
                        if start==0 and j<18:
                            W.append((base[j,hi],pred[hi],target,gt));M.append(dict(seed=seed,split=split,episode=rec['id'],pair=rec['pair'],pool=f'clock_{pi}',deleted=deleted,kept=';'.join(map(str,kept)),horizon=h,enabled=enabled,geometry_x=float(geo[hi,0]),geometry_y=float(geo[hi,1]),decoded_score=score))
                if time.monotonic()-begin>1200:raise RuntimeError('CPU soft budget stop; preserve partial, no automatic retry')
            write(out/'progress.json',dict(seed=seed,split_completed=split,rows=len(rows),expected=82944,model_calls='frozen CPU only',GPU=0))
        assert digest(m)==initial and all(x.grad is None for x in m.parameters());del m,ref,ck
    assert len(rows)==82944 and len(W)==324
    savecsv(out/'raw_scoring.csv',rows);paired=aggregate(rows);savecsv(out/'paired_scoring.csv',paired);stats=summarize(paired)
    np.savez_compressed(out/'witness_vectors.npz',baseline=np.stack([w[0] for w in W]),prediction=np.stack([w[1] for w in W]),target=np.stack([w[2] for w in W]),truth=np.stack([w[3] for w in W]));savecsv(out/'witness_metadata.csv',M)
    assert all(sha(k)==v for k,v in assets.items()) and sha(parent/'raw_deletions.csv')==protocol['parent_raw_sha256'];write(out/'asset_ledger.json',assets)
    write(out/'summary.json',dict(status='COMPLETE_LOCKED_TASK_SPECIFIC_SCORING_REGRESSION',raw_rows=82944,paired_rows=2304,statistics=stats,parent_outcomes_unchanged=True,original_prediction='40/51',original_fixed_reference='13/32',critical_long_gap='0/8',original_Phase1_GO=False,raw_sha256=sha(out/'raw_scoring.csv'),paired_sha256=sha(out/'paired_scoring.csv'),witness_sha256=sha(out/'witness_vectors.npz'),witness_metadata_sha256=sha(out/'witness_metadata.csv'),models=models,max_original_scalar_replay_difference=maximum,max_score_identity_difference=newscore_identity,source_lock_sha256=sha(ROOT/'source_lock.json'),elapsed_seconds=time.monotonic()-begin,training=False,fitting=False,encoding=False,GPU=0,fresh9137_opened=False,policy_training=False,test_driven_switch=False))
if __name__=='__main__':main(int(sys.argv[1]))
