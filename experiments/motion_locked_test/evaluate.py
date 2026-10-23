"""One locked Motion development recheck; cached encoder, no fitting or training."""
import collections,copy,csv,datetime,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from guard import allow_update
from core import maps,constrain
from geometry import observed_forecast
import deletion_analysis as analysis
analysis.GROUPS={'motion':('clock_all',)}
ROOT=Path(__file__).resolve().parent;AGE=Path('/scratch/xl4358/phase1_age_control_a1');VAL=Path('/scratch/xl4358/phase1_motion_observation_guard_validation')
sys.path.insert(0,str(AGE));import run as old
from phase1.evaluation import summarize,pair_diagnostic
from phase1.reporting import decision,_development_artifacts,_raw_records
from phase1.v5_motion_evaluation import fixed_reference_checks
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def savecsv(p,rows):
 with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def digest(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main(job):
 begin=time.monotonic();assert str(job)==os.environ['SLURM_JOB_ID'];native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);nf=dict(t.split('=',1) for t in native.split() if '=' in t)
 assert nf['JobState']=='RUNNING' and nf['Account']=='csci_ga_2271-2026fa' and nf['Partition']=='n2c48m24' and nf['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in nf['AllocTRES'] and os.environ['CUDA_VISIBLE_DEVICES']==''
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 lock=read(ROOT/'candidate_lock.json');protocol=read(ROOT/'protocol.json')
 for r,expected in ((ROOT,None),(AGE,'80336bfde8fd9da47d31060e7725e34486dc042df5f0c83e9f2a3161055bba5f'),(VAL,'678d236e88b638641ac29f7ee3932e433c697618efc5bb5651611a6b00becc4f')):
  if expected:assert sha(r/'source_lock.json')==expected
  assert all(sha(r/k)==v for k,v in read(r/'source_lock.json')['files'].items())
 assert sha(VAL/'output/job_8347/summary.json')==lock['validation_summary_sha256'] and sha(VAL/'output/job_8347/independent_audit.json')==lock['validation_audit_sha256'] and read(VAL/'final_audit_receipt.json')['status']=='COMPLETE_WITH_INDEPENDENT_AUDIT'
 assert sha(ROOT/'guard.py')==sha(VAL/'guard.py')==lock['guard_sha256']
 assert all(sha(k)==v for k,v in lock['weights_and_transforms'].items())
 p,_,configs=old.verify_control(old.OLD/'control');c=configs['motion'];assert c['data']['seed']==6301 and c['data']['budget']==2 and c['data']['recent']==4
 parent=read(AGE/'output/job_8275/summary.json');out=ROOT/'output'/f'job_{job}';out.mkdir(parents=True,exist_ok=False);write(out/'attempt.json',dict(job=job,allocation=native,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU=0,training=False,candidate_lock=lock))
 assets=dict(lock['weights_and_transforms']);priorassets=read(AGE/'output/job_8275/asset_ledger.json');store=old.FeatureStore(c,max_cached=16)
 def check(path):
  path=str(path)
  if path not in assets:assets[path]=sha(path)
  if path in priorassets:assert assets[path]==priorassets[path]
 with np.load(Path(c['output'])/'predictor/motion/basis.npz',allow_pickle=False) as f:basis={k:f[k] for k in f.files}
 store.bases['motion']=basis;decoder=old.LinearReadout.load(Path(c['output'])/'probes/motion_state_decoder.npz');B,A,b,right,_=maps(basis['output_basis'],decoder.weight,decoder.std,decoder.mean,decoder.bias)
 obs_cache=collections.OrderedDict();trace={};modelproof={};stateproof={};replay=0.;W=[];D=[];M=[]
 def observed(rec):
  if rec['id'] not in obs_cache:
   check(Path(c['output'])/'dataset'/rec['obs']);obs_cache[rec['id']]=old.load_episode(c,rec,keys=('rgb','actions'))
   if len(obs_cache)>16:obs_cache.popitem(last=False)
  obs_cache.move_to_end(rec['id']);return obs_cache[rec['id']]
 def context(rec,sample,info,condition):
  idx=tuple(map(int,info['indices']));q=int(rec['query']);assert len(idx)==2 and max(idx)<q-3
  own=observed(rec);source=[rec,rec]
  if condition=='wrong_evidence':
   ev=set(map(int,old.load_episode(c,rec,True,keys=('evidence',))['evidence']))
   source=[store.records[rec['sibling']] if t in ev else rec for t in idx]
  rgb=np.stack([observed(sr)['rgb'][t] for sr,t in zip(source,idx)])
  assert all(np.array_equal(sample['z'][4+j],store.features(sr)[t]) for j,(sr,t) in enumerate(zip(source,idx)))
  enabled=allow_update(rgb,own['rgb'][q]);positions=decoder.predict(sample['z'][4:])[:,1:]
  geo=np.stack([observed_forecast(positions,idx,own['actions'][idx[1]:q+h],q+h)[0] for h in (3,6)])
  key=(rec['id'],condition,idx,tuple(sr['id'] for sr in source));trace[key]=dict(episode=rec['id'],condition=condition,kept=';'.join(map(str,idx)),source_ids=';'.join(sr['id'] for sr in source),query=q,enabled=enabled)
  return enabled,geo
 def corrected(pred,geo,enabled):
  if not enabled:return pred
  return constrain(pred,geo,B,A,b,right)[0]
 def model(seed):
  ref,_=old.reference_model(c,p,'motion',seed,basis);folder=old.OLD/'training/job_7574/predictor/motion'/f'seed_{seed}';proof=old.verify_completed_model(folder,parent['models'][f'motion/{seed}']['signature'],ref.state_dict());assert proof==parent['models'][f'motion/{seed}'];modelproof[str(seed)]=proof;check(folder/'best.pt')
  ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);m=old.CenteredTransportPredictor(ref,basis);m.load_state_dict(ck['model'],strict=True);m.eval().requires_grad_(False);assert digest(m)==lock['state_digest_by_seed'][str(seed)];return m
 goldpath=old.OLD/'evaluation/job_7575/motion/evaluation/raw_records.csv';check(goldpath);assert sha(goldpath)==protocol['parent_prediction_raw_sha256'];gold=_raw_records(goldpath);assert len(gold)==34560
 index={(r['seed'],r['split'],r['condition'],r['episode'],r['horizon']):r for r in gold};candidate={k:dict(v) for k,v in index.items()};assert all(r['history_count']!=2 for r in gold if r['condition'] in ('recent_only','full_history','persistence'));changed={'privileged_small_budget','random_small_budget','wrong_evidence'}
 # Unchanged conditions are copied from the audited parent; candidate is identity for K!=2.
 for seed in (17,29,43):
  m=model(seed);initial=digest(m)
  for split in ('validation','test_id','test_long_gap','test_distractor_shift'):
   records=[r for r in store.manifest['records'] if r['task']=='motion' and r['split']==split];assert len(records)==(192 if split=='validation' else 256)
   for start in range(0,len(records),8):
    part=records[start:start+8]
    for condition in sorted(changed):
     samples=[store.sample(r,condition) for r in part];contexts=[context(r,x[0],x[2],condition) for r,x in zip(part,samples)]
     with torch.inference_mode():base=m(old.collate([x[0] for x in samples],'cpu')).numpy()
     assert base.shape==(len(part),2,5120) and np.isfinite(base).all()
     for j,(rec,sample,(enabled,geo)) in enumerate(zip(part,samples,contexts)):
      q=int(rec['query']);z=store.features(rec);check(Path(c['output'])/'features'/store.ix['episodes'][rec['id']]['path']);pred=corrected(base[j],geo,enabled);truth=old.load_episode(c,rec,True,keys=('states','classes'));state=decoder.predict(pred);oldstate=decoder.predict(base[j]);other=store.features(store.records[rec['sibling']]);target=sample[1]
      for hi,h in enumerate((3,6)):
       key=(seed,split,condition,rec['id'],h);g=index[key];bm=float(np.mean((base[j,hi]-target[hi])**2));bt=float(np.mean((oldstate[hi,1:]-truth['states'][q+h])**2));replay=max(replay,abs(bm-g['normalized_mse']),abs(bt-g['state_position_mse']));assert replay<2e-6
       if enabled:
        P,Y=pred[hi]*store.norm['std']+store.norm['mean'],target[hi]*store.norm['std']+store.norm['mean'];acc,margin,sep=pair_diagnostic(pred[hi],target[hi],other[q+h]);r=candidate[key]
        r.update(latent_mse=float(np.mean((P-Y)**2)),normalized_mse=float(np.mean((pred[hi]-target[hi])**2)),cosine_distance=1-float(np.dot(P,Y)/max(np.linalg.norm(P)*np.linalg.norm(Y),1e-12)),pair_accuracy=acc,pair_margin=margin,target_pair_distance=sep,state_accuracy=float((state[hi,0]>0)==(truth['classes'][q+h]>0)),state_position_mse=float(np.mean((state[hi,1:]-truth['states'][q+h])**2)),state_pred_x=float(state[hi,1]),state_pred_y=float(state[hi,2]))
       else:assert candidate[key]==g
       if start==0 and j<2:W.append((base[j,hi],pred[hi],target[hi],truth['states'][q+h]));M.append(dict(stage='prediction',seed=seed,split=split,condition=condition,episode=rec['id'],horizon=h,enabled=enabled,kept=';'.join(map(str,sample[2]['indices'])),geometry_x=float(geo[hi,0]),geometry_y=float(geo[hi,1]),feature_mse=candidate[key]['normalized_mse'],task_error=candidate[key]['state_position_mse']))
    if time.monotonic()-begin>1500:raise RuntimeError('CPU soft budget; preserve partial')
   write(out/'progress.json',dict(stage='original_prediction',seed=seed,split_completed=split,baseline_max_difference=replay))
  assert digest(m)==initial;stateproof[str(seed)]=initial;del m
 raw=[candidate[(r['seed'],r['split'],r['condition'],r['episode'],r['horizon'])] for r in gold];original=out/'original';(original/'evaluation').mkdir(parents=True);savecsv(original/'evaluation/raw_records.csv',raw)
 metrics={'raw_sha256':sha(original/'evaluation/raw_records.csv'),**summarize(c,raw)};write(original/'evaluation/metrics.json',metrics)
 _,oldmetrics,refraw,dataaudit,probes=_development_artifacts(c['output'],'motion');profile=copy.deepcopy(c);profile['data']['tasks']=['motion'];profile.pop('experiment',None)
 verdict,reasons,checks=decision(profile,dataaudit,probes,metrics);fixed=fixed_reference_checks(c,refraw,raw);assert len(checks)==51 and len(fixed)==32
 prediction=dict(task='motion',original_task_status=verdict,original_checks=checks,fixed_reference_checks=fixed,reasons=reasons,original_checks_passed=sum(x['passed'] for x in checks),fixed_checks_passed=sum(x['passed'] for x in fixed),status='SUPPORTED' if verdict=='GO' and all(x['passed'] for x in fixed) else 'REJECTED',candidate_raw_sha256=sha(original/'evaluation/raw_records.csv'),reference_raw_sha256=oldmetrics['raw_sha256'],phase1_go=False)
 write(original/'development_comparison.json',prediction);write(original/'outcome.json',dict(tasks={'motion':prediction},phase1_go=False));print('Original Motion checks complete; locked candidate unchanged',flush=True)
 # Complete original clock-only deletion tests. No role annotations enter candidate.
 gp=AGE/'output/job_8275/raw_deletions.csv';check(gp)
 with gp.open() as f:prior={(int(r['seed']),r['split'],r['episode'],r['pool'],int(r['deleted']),int(r['horizon'])):r for r in csv.DictReader(f) if r['task']=='motion'}
 assert len(prior)==82944;deletions=[]
 for seed in (17,29,43):
  m=model(seed);initial=digest(m)
  for split in ('test_id','test_long_gap','test_distractor_shift'):
   pairs=old.paired_rows([r for r in store.manifest['records'] if r['task']=='motion' and r['split']==split]);assert len(pairs)==128;records=[r for pair in pairs for r in pair]
   for start in range(0,len(records),8):
    samples=[];desc=[];contexts=[]
    for rec in records[start:start+8]:
     q=int(rec['query']);z=store.features(rec);proj=store.projected_features(rec);actions=observed(rec)['actions'];check(Path(c['output'])/'features'/store.ix['episodes'][rec['id']]['path'])
     for pi,pool in enumerate(old.clock_only_static_pools(q,old.stable_seed(20261017,'clock_only',rec['pair']),count=6)):
      for deleted in pool:
       kept=tuple(t for t in pool if t!=deleted);s=old.sample(z,proj,actions,q,kept);samples.append(s);desc.append((rec,pi,pool,deleted,kept));contexts.append(context(rec,s,dict(indices=kept),'deletion'))
    with torch.inference_mode():base=m(old.collate(samples,'cpu')).numpy()
    for j,((rec,pi,pool,deleted,kept),(enabled,geo)) in enumerate(zip(desc,contexts)):
     pred=corrected(base[j],geo,enabled);q=int(rec['query']);z=store.features(rec);truth=old.load_episode(c,rec,True,keys=('states','evidence'));xy=decoder.predict(pred)[:,1:];bxy=decoder.predict(base[j])[:,1:]
     for hi,h in enumerate((3,6)):
      r=prior[seed,split,rec['id'],f'clock_{pi}',deleted,h];fm=float(np.mean((base[j,hi]-z[q+h])**2));te=float(np.mean((bxy[hi]-truth['states'][q+h])**2));replay=max(replay,abs(fm-float(r['feature_mse'])),abs(te-float(r['task_error'])));assert replay<2e-6
      row=dict(r);row.update(old_feature_mse=float(r['feature_mse']),old_task_error=float(r['task_error']),guard_enabled=enabled)
      if enabled:row.update(feature_mse=float(np.mean((pred[hi]-z[q+h])**2)),task_error=float(np.mean((xy[hi]-truth['states'][q+h])**2)))
      deletions.append(row)
      if start==0 and j<18:W.append((base[j,hi],pred[hi],z[q+h],truth['states'][q+h]));M.append(dict(stage='deletion',seed=seed,split=split,condition='deletion',episode=rec['id'],horizon=h,enabled=enabled,kept=';'.join(map(str,kept)),geometry_x=float(geo[hi,0]),geometry_y=float(geo[hi,1]),feature_mse=float(row['feature_mse']),task_error=float(row['task_error'])))
    if time.monotonic()-begin>1500:raise RuntimeError('CPU soft budget; preserve partial')
   write(out/'progress.json',dict(stage='deletion',seed=seed,split_completed=split,rows=len(deletions),expected=82944))
  assert digest(m)==initial==stateproof[str(seed)] and all(x.grad is None for x in m.parameters());del m
 assert len(deletions)==82944 and len(M)==468
 savecsv(out/'raw_deletions.csv',deletions);compact,coverage=analysis.aggregate(deletions);stats=analysis.summarize(compact);savecsv(out/'paired_deletions.csv',compact);assert len(compact)==2304
 savecsv(out/'input_trace.csv',list(trace.values()));savecsv(out/'witness_metadata.csv',M);np.savez_compressed(out/'witness_vectors.npz',baseline=np.stack([x[0] for x in W]),prediction=np.stack([x[1] for x in W]),target=np.stack([x[2] for x in W]),position=np.stack([x[3] for x in W]))
 # Compact original fixed-reference pair axes for local re-computation.
 fixedpairs=[];refidx={(r['seed'],r['split'],r['condition'],r['episode'],r['horizon']):r for r in refraw}
 for r in raw:
  if r['condition']=='privileged_small_budget' and r['split'] in ('test_id','test_long_gap','test_distractor_shift'):
   k=(r['seed'],r['split'],r['condition'],r['episode'],r['horizon']);rk=list(k);rk[2]='recent_only';fixedpairs.append(dict(seed=r['seed'],split=r['split'],episode=r['episode'],pair=r['pair'],horizon=r['horizon'],candidate=r['normalized_mse'],reference=refidx[k]['normalized_mse'],reference_recent=refidx[tuple(rk)]['normalized_mse']))
 savecsv(out/'fixed_reference_pairs.csv',fixedpairs)
 assert all(sha(k)==v for k,v in assets.items()) and all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items());old.verify_control(old.OLD/'control');write(out/'asset_ledger.json',assets)
 transfer=[x for x in stats['oracle_baseline_comparisons'] if x['choice']=='oracle_transfer'];prediction_ok=prediction['status']=='SUPPORTED';deletion_ok=all(x['positive_both_metrics'] for x in transfer)
 write(out/'summary.json',dict(status='MOTION_DEVELOPMENT_REPAIR_SUPPORTED' if prediction_ok and deletion_ok else 'MOTION_DEVELOPMENT_REPAIR_REJECTED',job=job,allocation=native,prediction=prediction,deletion=stats,coverage=coverage,raw_prediction_rows=34560,new_prediction_rows=17280,copied_unchanged_prediction_rows=17280,raw_deletion_rows=82944,paired_deletion_rows=2304,deletion_positive=sum(x['positive_both_metrics'] for x in transfer),deletion_total=len(transfer),raw_prediction_sha256=sha(original/'evaluation/raw_records.csv'),raw_deletion_sha256=sha(out/'raw_deletions.csv'),paired_deletion_sha256=sha(out/'paired_deletions.csv'),trace_sha256=sha(out/'input_trace.csv'),witness_sha256=sha(out/'witness_vectors.npz'),witness_metadata_sha256=sha(out/'witness_metadata.csv'),asset_ledger_sha256=sha(out/'asset_ledger.json'),max_baseline_replay_difference=replay,model_proofs=modelproof,state_proofs=stateproof,source_lock_sha256=sha(ROOT/'source_lock.json'),protocol_sha256=sha(ROOT/'protocol.json'),elapsed=time.monotonic()-begin,training=False,fitting=False,encoding=False,GPU=0,original_gates_changed=False,original_Phase1_GO=False,fresh9137_opened=False,test_driven_switching=False,synthetic_renderer_specific=True))
if __name__=='__main__':
 try:main(int(sys.argv[1]))
 except Exception:
  import traceback
  o=ROOT/'output'/f"job_{os.environ.get('SLURM_JOB_ID','unknown')}";o.mkdir(parents=True,exist_ok=True);write(o/'failure.json',dict(error=traceback.format_exc(),no_automatic_retry=True));raise
