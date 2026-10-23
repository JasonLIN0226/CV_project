"""CPU validation calibration then one unchanged Gate deletion assay."""
import collections,csv,datetime,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from calibration import GRID,blend,choose
ROOT=Path(__file__).resolve().parent
AGE=Path('/scratch/xl4358/phase1_age_control_a1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert sha(AGE/'source_lock.json')=='80336bfde8fd9da47d31060e7725e34486dc042df5f0c83e9f2a3161055bba5f'
sys.path.insert(0,str(AGE));import run as old
from phase1.training import TRAIN_CONDITIONS
from phase1.common_anchor import recent_only_teacher_batch

def main():
 begin=time.monotonic();job=int(sys.argv[1]);assert job==int(os.environ['SLURM_JOB_ID'])
 state=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True)
 ff=dict(v.split('=',1) for v in state.split() if '=' in v)
 assert ff['JobState']=='RUNNING' and ff['Account']=='csci_ga_2271-2026fa' and ff['Partition']=='n2c48m24' and 'gres/gpu' not in ff['AllocTRES'] and ff['NodeList']==socket.gethostname().split('.')[0]
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 for root in (ROOT,AGE):assert all(sha(root/k)==v for k,v in read(root/'source_lock.json')['files'].items())
 native=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-n','-P','-j','7574,7575,7576,8275','--format=JobID,State,ExitCode'],text=True)
 assert set(native.strip().splitlines())=={f'{j}|COMPLETED|0:0' for j in (7574,7575,7576,8275)}
 out=ROOT/'output'/f'job_{job}';out.mkdir(parents=True,exist_ok=False)
 write(out/'attempt.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=job,allocation=state,training=False,GPU=0,source_lock_sha256=sha(ROOT/'source_lock.json')))
 prior=AGE/'output/job_8275';parent=read(prior/'summary.json');assert sha(prior/'summary.json')=='ed9232c2503b76030eea05ad4b54cf565774f952200f69a85306b873b131e35f'
 assert sha(prior/'raw_deletions.csv')==parent['raw_sha256']=='cc53f6cd73f9fcaed5e6a25f50edf46257fb652eafd5625b06d000297ab15a45'
 parentassets=read(prior/'asset_ledger.json');assets={}
 def check(path):
  path=str(path)
  if path not in assets:
   assets[path]=sha(path)
   if path in parentassets:assert assets[path]==parentassets[path]
 p,_,configs=old.verify_control(old.OLD/'control');c=configs['gate'];store=old.FeatureStore(c,max_cached=4);assert c['data']['seed']==6301
 bp=Path(c['output'])/'predictor/gate/basis.npz';check(bp)
 with np.load(bp,allow_pickle=False) as f:basis={k:f[k] for k in f.files}
 store.bases['gate']=basis
 for path in (Path(c['output'])/'features/normalization.npz',Path(c['output'])/'features/index.json'):check(path)
 dp=Path(c['output'])/'probes/gate_state_decoder.npz';check(dp);decoder=old.LinearReadout.load(dp)
 def load(seed):
  ref,_=old.reference_model(c,p,'gate',seed,basis);check(Path(c['output'])/'predictor/gate'/f'seed_{seed}/best.pt')
  folder=old.OLD/'training/job_7574/predictor/gate'/f'seed_{seed}';proof=old.verify_completed_model(folder,parent['models'][f'gate/{seed}']['signature'],ref.state_dict());assert proof==parent['models'][f'gate/{seed}'];cp=folder/'best.pt';check(cp)
  model=old.CenteredTransportPredictor(ref,basis);model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'],strict=True);return model.eval().requires_grad_(False),ref.eval().requires_grad_(False)
 def cachecheck(rec):check(Path(c['output'])/'features'/store.ix['episodes'][rec['id']]['path'])
 # Selection runs first on original validation only. No test data is loaded here.
 val=old.paired_rows([r for r in store.manifest['records'] if r['task']=='gate' and r['split']=='validation']);assert len(val)==96
 validation=[]
 for seed in (17,29,43):
  model,ref=load(seed)
  for condition in TRAIN_CONDITIONS:
   for offset in range(0,96,8):
    rr=[r for pair in val[offset:offset+8] for r in pair]
    for r in rr:cachecheck(r)
    ss=[store.sample(r,condition) for r in rr];batch=old.collate([s[0] for s in ss],'cpu');target=np.stack([s[1] for s in ss])
    with torch.inference_mode():pred=model(batch).numpy();anchor=ref(recent_only_teacher_batch(batch)).numpy()
    for alpha in GRID:
     pp=blend(anchor,pred,alpha);mse=np.mean((pp-target)**2,axis=-1)
     assert np.isfinite(mse).all()
     for i,r in enumerate(rr):
      for hi,h in enumerate((3,6)):validation.append(dict(seed=seed,condition=condition,pair=r['pair'],episode=r['id'],horizon=h,alpha=alpha,feature_mse=float(mse[i,hi])))
  del model,ref
 scores={a:float(np.mean([r['feature_mse'] for r in validation if r['alpha']==a])) for a in GRID};selected=choose(scores)
 def savecsv(name,rows):
  with (out/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 savecsv('validation_scores.csv',validation)
 selection=dict(selected_alpha=selected,score_by_alpha={str(a):v for a,v in scores.items()},criterion='Original equal condition x horizon feature MSE, equal seed weights',tie_rule='largest alpha on exact score ties',rows=len(validation),validation_pairs=96,chosen_before_test=True,validation_csv_sha256=sha(out/'validation_scores.csv'))
 write(out/'selection_lock.json',selection)
 print(json.dumps(selection),flush=True)
 # Only the selected value is evaluated; alpha1 is a provenance reference, never selected on test.
 with (prior/'raw_deletions.csv').open() as f:gold=[r for r in csv.DictReader(f) if r['task']=='gate']
 key=lambda r:(int(r['seed']),r['split'],r['episode'],r['pool'],int(r['deleted']),int(r['horizon']))
 gold={key(r):r for r in gold};assert len(gold)==41472
 rows=[];seen=set();maxdiff=0.;taskdiff=0;emptydiff=0.;refpairs=0
 for seed in (17,29,43):
  model,ref=load(seed)
  for split in ('test_id','test_long_gap','test_distractor_shift'):
   pairs=old.paired_rows([r for r in store.manifest['records'] if r['task']=='gate' and r['split']==split]);assert len(pairs)==128
   records=[r for pair in pairs for r in pair]
   for offset in range(0,len(records),8):
    inputs=[];desc=[]
    for rec in records[offset:offset+8]:
     cachecheck(rec);z=store.features(rec);proj=store.projected_features(rec);q=int(rec['query']);ev=tuple(map(int,old.load_episode(c,rec,True,keys=('evidence',))['evidence']));act=old.load_episode(c,rec,keys=('actions',))['actions'];end=q-int(rec['gap'])-4
     for group,pool in old.gate_age_counterbalance(ev[0],end,q).items():
      for deleted in pool:inputs.append(old.sample(z,proj,act,q,tuple(t for t in pool if t!=deleted)));desc.append((rec,group,pool,deleted,ev[0]))
    batch=old.collate(inputs,'cpu')
    with torch.inference_mode():pred=model(batch).numpy();rb=recent_only_teacher_batch(batch);anchor=ref(rb).numpy()
    if offset==0:
     with torch.inference_mode():emptydiff=max(emptydiff,float((model(rb)-ref(rb)).abs().max()))
    oldscore=decoder.predict(pred.reshape(-1,5120))[:,0].reshape(len(pred),2);cal=blend(anchor,pred,selected);score=decoder.predict(cal.reshape(-1,5120))[:,0].reshape(len(cal),2)
    assert np.isfinite(cal).all()
    for i,(rec,group,pool,deleted,cue) in enumerate(desc):
     z=store.features(rec);q=int(rec['query']);classes=old.load_episode(c,rec,True,keys=('classes',))['classes']
     for hi,h in enumerate((3,6)):
      k=(seed,split,rec['id'],group,deleted,h);assert k not in seen;seen.add(k);expected=gold[k]
      orig=float(np.mean((pred[i,hi]-z[q+h])**2));maxdiff=max(maxdiff,abs(orig-float(expected['feature_mse'])));taskdiff+=int(float((oldscore[i,hi]>0)!=(classes[q+h]>0))!=float(expected['task_error']))
      assert maxdiff<=1e-7 and taskdiff==0
      rows.append(dict(seed=seed,split=split,pair=rec['pair'],episode=rec['id'],pool_group=group,parent=';'.join(map(str,pool)),deleted=deleted,cue=cue,horizon=h,alpha=selected,history_count=2,feature_mse=float(np.mean((cal[i,hi]-z[q+h])**2)),task_error=float((score[i,hi]>0)!=(classes[q+h]>0)),old_feature_mse=orig,old_task_error=float(expected['task_error'])))
    assert time.monotonic()-begin<1500
   write(out/'progress.json',dict(seed=seed,split=split,raw_rows=len(rows),selected_alpha=selected))
  del model,ref
 assert len(rows)==41472 and maxdiff==0 and emptydiff==0
 savecsv('raw_selected.csv',rows)
 assert all(sha(k)==v for k,v in assets.items());old.verify_control(old.OLD/'control')
 write(out/'asset_ledger.json',assets)
 write(out/'summary.json',dict(job=job,selected_alpha=selected,selection=selection,raw_rows=len(rows),max_original_feature_mse_difference=maxdiff,original_task_mismatches=taskdiff,empty_history_max_difference=emptydiff,selection_sha256=sha(out/'selection_lock.json'),raw_sha256=sha(out/'raw_selected.csv'),asset_ledger_sha256=sha(out/'asset_ledger.json'),elapsed=time.monotonic()-begin,source_lock_sha256=sha(ROOT/'source_lock.json'),GPU=0,network_training=False,scalar_calibration=True,original_gates_changed=False,original_Phase1_GO=False,fresh9137_opened=False))
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  folder=ROOT/'output'/f"job_{os.environ.get('SLURM_JOB_ID','unknown')}";folder.mkdir(parents=True,exist_ok=True);write(folder/'failure.json',dict(error=traceback.format_exc(),no_retry_unknown=True));raise
