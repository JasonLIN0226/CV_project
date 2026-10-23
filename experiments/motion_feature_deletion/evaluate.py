"""Frozen validation-only equal-capacity forks, same prelocked feature constraint."""
import csv,datetime,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from core import maps,constrain
from geometry import observed_forecast
import legacy_analysis as analysis
analysis.GROUPS={'motion':('clock_all',)};analysis.SPLITS=('validation',)
ROOT=Path(__file__).resolve().parent;AGE=Path('/scratch/xl4358/phase1_age_control_a1');PARENT=Path('/scratch/xl4358/phase1_motion_feature_constraint')
sys.path.insert(0,str(AGE));import run as old
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def savecsv(p,rows):
    with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(job):
    begin=time.monotonic();native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);fields=dict(t.split('=',1) for t in native.split() if '=' in t)
    assert str(job)==os.environ['SLURM_JOB_ID'] and fields['JobState']=='RUNNING' and fields['Account']=='csci_ga_2271-2026fa' and fields['Partition']=='n2c48m24'
    assert fields['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in fields['AllocTRES'] and os.environ['CUDA_VISIBLE_DEVICES']==''
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    protocol=read(ROOT/'protocol.json');assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
    assert sha(PARENT/'output/job_8339/summary.json')==protocol['feasibility_summary_sha256']
    assert read(PARENT/'output/job_8339/summary.json')['conditional_deletion_eligible'] and read(PARENT/'execution_receipt.json')['status']=='COMPLETE_WITH_INDEPENDENT_AUDIT'
    assert sha(ROOT/'core.py')==sha(PARENT/'core.py') and sha(ROOT/'geometry.py')==sha(Path('/scratch/xl4358/phase1_motion_geometry_validation/geometry.py'))
    assert sha(ROOT/'legacy_analysis.py')==sha(AGE/'analysis.py')
    assert sha(AGE/'source_lock.json')=='80336bfde8fd9da47d31060e7725e34486dc042df5f0c83e9f2a3161055bba5f'
    assert all(sha(AGE/k)==v for k,v in read(AGE/'source_lock.json')['files'].items())
    p,_,configs=old.verify_control(old.OLD/'control');c=configs['motion'];assert c['data']['seed']==6301 and c['data']['budget']==2
    parent=read(AGE/'output/job_8275/summary.json');assets=read(PARENT/'output/job_8339/asset_ledger.json');assert all(sha(k)==v for k,v in assets.items())
    def check(path):
        path=str(path);value=sha(path)
        if path in assets:assert value==assets[path]
        else:assets[path]=value
    store=old.FeatureStore(c,max_cached=192);basispath=Path(c['output'])/'predictor/motion/basis.npz';check(basispath)
    with np.load(basispath,allow_pickle=False) as f:basis={k:f[k] for k in f.files}
    store.bases['motion']=basis;dp=Path(c['output'])/'probes/motion_state_decoder.npz';decoder=old.LinearReadout.load(dp);check(dp)
    B,A,b,right,gram=maps(basis['output_basis'],decoder.weight,decoder.std,decoder.mean,decoder.bias)
    pairs=old.paired_rows([r for r in store.manifest['records'] if r['task']=='motion' and r['split']=='validation']);assert len(pairs)==96
    records=[r for pair in pairs for r in pair];out=ROOT/'output'/f'job_{job}';out.mkdir(parents=True,exist_ok=False)
    write(out/'attempt.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=job,allocation=native,training=False,GPU=0))
    raw=[];primitive=[];models={};states={};constraint_residual=0.
    for seed in (17,29,43):
        reference,_=old.reference_model(c,p,'motion',seed,basis);folder=old.OLD/'training/job_7574/predictor/motion'/f'seed_{seed}'
        proof=old.verify_completed_model(folder,parent['models'][f'motion/{seed}']['signature'],reference.state_dict());assert proof==parent['models'][f'motion/{seed}'];models[str(seed)]=proof;check(folder/'best.pt')
        ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model=old.CenteredTransportPredictor(reference,basis);model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False)
        def digest():
            h=hashlib.sha256()
            for k,v in model.state_dict().items():h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        state=digest()
        for offset in range(0,len(records),8):
            samples=[];desc=[];geometries=[]
            for rec in records[offset:offset+8]:
                q=int(rec['query']);z=store.features(rec);projected=store.projected_features(rec);actions=old.load_episode(c,rec,keys=('actions',))['actions'];check(Path(c['output'])/'features'/store.ix['episodes'][rec['id']]['path'])
                pools=old.clock_only_static_pools(q,old.stable_seed(20261017,'clock_only',rec['pair']),count=6)
                for pi,pool in enumerate(pools):
                    for deleted in pool:
                        kept=tuple(t for t in pool if t!=deleted);samples.append(old.sample(z,projected,actions,q,kept));desc.append((rec,pi,pool,deleted,kept))
                        observed=decoder.predict(z[list(kept)])[:,1:]
                        geometries.append([observed_forecast(observed,kept,actions[kept[1]:q+h],q+h)[0] for h in (3,6)])
            # Normal model and geometry use only the two retained past frames, never importance labels.
            with torch.inference_mode():pred=model(old.collate(samples,'cpu')).numpy()
            assert pred.shape==(len(desc),2,5120) and np.isfinite(pred).all()
            g=np.asarray(geometries);updated,coeff=constrain(pred.reshape(-1,5120),g.reshape(-1,2),B,A,b,right);updated=updated.reshape(pred.shape)
            constraint_residual=max(constraint_residual,float(np.max(np.abs((updated.reshape(-1,5120)@A+b)-g.reshape(-1,2)))))
            assert constraint_residual<1e-8
            positions=[decoder.predict(pred.reshape(-1,5120))[:,1:].reshape(len(desc),2,2),(updated.reshape(-1,5120)@A+b).reshape(len(desc),2,2)]
            # Only after normal forecasts exist, load scoring states and evidence annotations.
            for i,(rec,pi,pool,deleted,kept) in enumerate(desc):
                q=int(rec['query']);z=store.features(rec);truth=old.load_episode(c,rec,True,keys=('states','evidence','visible'));ev=tuple(map(int,truth['evidence']))
                for hi,h in enumerate((3,6)):
                    witness=len(primitive);primitive.append(pred[i,hi]);target=z[q+h]
                    for arm,forecast,xy in [('baseline',pred[i,hi],positions[0][i,hi]),('geometry_constraint',updated[i,hi],positions[1][i,hi])]:
                        raw.append(dict(arm=arm,task='motion',seed=seed,split='validation',pair=rec['pair'],episode=rec['id'],pool=f'clock_{pi}',pool_group='clock_all',parent=';'.join(map(str,pool)),deleted=deleted,kept=';'.join(map(str,kept)),query=q,evidence_end=q-int(rec['gap'])-4,evidence=';'.join(map(str,ev)),deleted_role='start' if deleted==ev[0] else 'end' if deleted==ev[1] else 'other',history_count=2,feature_dim=5120,horizon=h,feature_mse=float(np.mean((forecast-target)**2)),task_error=float(np.mean((xy-truth['states'][q+h])**2)),witness_index=witness,retained_visible_count=int(truth['visible'][list(kept)].sum()),geometry_x=float(g[i,hi,0]),geometry_y=float(g[i,hi,1])))
            write(out/'progress.json',dict(seed=seed,episodes_completed=offset+min(8,len(records)-offset),raw_rows=len(raw),expected_rows=41472))
            if time.monotonic()-begin>900:raise RuntimeError('CPU soft bound; partial only, no retry')
        assert digest()==state and all(x.grad is None for x in model.parameters());states[str(seed)]=state;del model,reference,ck
    assert len(raw)==41472 and len(primitive)==20736
    savecsv(out/'raw_deletion_constraint.csv',raw);np.savez_compressed(out/'baseline_vectors.npz',predictions=np.stack(primitive))
    summaries={};paired=[]
    for arm in ('baseline','geometry_constraint'):
        compact,cov=analysis.aggregate([r for r in raw if r['arm']==arm],pairs=96);result=analysis.summarize(compact,repeats=10000,pairs=96)
        summaries[arm]={'statistics':result,'coverage':cov};paired.extend([dict(arm=arm,**r) for r in compact])
    savecsv(out/'paired_deletion_constraint.csv',paired)
    assert all(sha(k)==v for k,v in assets.items());old.verify_control(old.OLD/'control');write(out/'asset_ledger.json',assets)
    write(out/'summary.json',dict(status='VALIDATION_DELETION_DIAGNOSIS_COMPLETE_NOT_ACCEPTANCE',job=job,allocation=native,raw_rows=41472,witness_vectors=20736,paired_rows=len(paired),basepairs=96,clock_pools=6,summaries=summaries,model_proofs=models,state_proofs=states,normal_position_constraint_max_difference=constraint_residual,raw_sha256=sha(out/'raw_deletion_constraint.csv'),vectors_sha256=sha(out/'baseline_vectors.npz'),paired_sha256=sha(out/'paired_deletion_constraint.csv'),asset_ledger_sha256=sha(out/'asset_ledger.json'),source_lock_sha256=sha(ROOT/'source_lock.json'),protocol_sha256=sha(ROOT/'protocol.json'),training=False,fitting=False,encoding=False,GPU=0,model_forward=True,policy_training=False,future_informed_oracles_are_diagnostic_only=True,original_gates_changed=False,fresh9137_opened=False,test_forward=False,elapsed=time.monotonic()-begin,original_Phase1_GO=False))

if __name__=='__main__':
    try:main(int(sys.argv[1]))
    except Exception:
        import traceback
        out=ROOT/'output'/f"job_{os.environ.get('SLURM_JOB_ID','unknown')}";out.mkdir(parents=True,exist_ok=True);write(out/'failure.json',dict(error=traceback.format_exc(),no_automatic_retry=True));raise
