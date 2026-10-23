"""Frozen Gate-only replay and evaluation-only error decomposition, no fitting."""
import collections,csv,datetime,hashlib,importlib.util,json,os,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
AGE=Path('/scratch/xl4358/phase1_age_control_a1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
protocol=read(ROOT/'protocol.json')
assert sha(AGE/'source_lock.json')==protocol['parent_source_lock_sha256']
age_lock=read(AGE/'source_lock.json');assert all(sha(AGE/k)==v for k,v in age_lock['files'].items())
sys.path.insert(0,str(AGE))
import run as old
sys.path.insert(0,str(ROOT))
from geometry import decompose,deletion_tradeoff
spec=importlib.util.spec_from_file_location('gate_diagnostic_analysis',ROOT/'analysis.py');analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
import numpy as np
import torch

def main():
 begin=time.monotonic();job=int(os.environ['SLURM_JOB_ID']);assert job==int(sys.argv[1])
 state=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True)
 fields=dict(t.split('=',1) for t in state.split() if '=' in t)
 assert fields['JobState']=='RUNNING' and fields['Account']=='csci_ga_2271-2026fa' and fields['Partition']=='n2c48m24'
 assert fields['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in fields['AllocTRES']
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 out=ROOT/'output'/f'job_{job}';out.mkdir(parents=True,exist_ok=False)
 write(out/'attempt.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=job,allocation=state,source_lock_sha256=sha(ROOT/'source_lock.json'),GPU=0,training=False))
 native_parents=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-n','-P','-j','7574,7575,7576,8275','--format=JobID,State,ExitCode'],text=True)
 assert set(native_parents.strip().splitlines())=={f'{j}|COMPLETED|0:0' for j in (7574,7575,7576,8275)}
 prior=AGE/'output/job_8275';assert sha(prior/'summary.json')==protocol['parent_summary_sha256']
 assert sha(prior/'independent_audit.json')==protocol['parent_audit_sha256']
 parent=read(prior/'summary.json');assert sha(prior/'raw_deletions.csv')==parent['raw_sha256']==protocol['parent_raw_sha256']
 with (prior/'raw_deletions.csv').open() as f:gold=[r for r in csv.DictReader(f) if r['task']=='gate']
 key=lambda r:(int(r['seed']),r['split'],r['pair'],r['episode'],r['pool'],int(r['deleted']),int(r['horizon']))
 gold={key(r):r for r in gold};assert len(gold)==41472
 parent_assets=read(prior/'asset_ledger.json');assert sha(prior/'asset_ledger.json')==parent['asset_ledger_sha256']
 p,_,configs=old.verify_control(old.OLD/'control');c=configs['gate'];store=old.FeatureStore(c,max_cached=4)
 assert c['data']['seed']==6301
 assets={};check=lambda path:check_asset(path,assets,parent_assets)
 basispath=Path(c['output'])/'predictor/gate/basis.npz';check(basispath)
 with np.load(basispath,allow_pickle=False) as f:basis={k:f[k] for k in f.files}
 store.bases['gate']=basis;Q=np.linalg.qr(basis['output_basis'].astype(np.float64),mode='reduced')[0]
 dp=Path(c['output'])/'probes/gate_state_decoder.npz';check(dp);decoder=old.LinearReadout.load(dp)
 w=decoder.weight[:,0]/decoder.std;assert w.shape==(5120,)
 for path in (Path(c['output'])/'features/normalization.npz',Path(c['output'])/'features/index.json'):check(path)
 rows=[];seen=set();target_scores={};witness_p=[];witness_t=[];witness_metrics=[];witness_ids=[];max_error=0.;max_outside_delta=0.;max_decomposition=0.
 metric_names=('total','parallel','orthogonal','common','contrast','basis_inside','basis_outside','target_pair_signal','prediction_pair_signal','signal_gain','prediction_sibling_maxabs')
 for seed in (17,29,43):
  reference,ref_sha=old.reference_model(c,p,'gate',seed,basis)
  check(Path(c['output'])/'predictor/gate'/f'seed_{seed}'/'best.pt')
  folder=old.OLD/'training/job_7574/predictor/gate'/f'seed_{seed}';proof=old.verify_completed_model(folder,parent['models'][f'gate/{seed}']['signature'],reference.state_dict())
  assert proof==parent['models'][f'gate/{seed}'];cp=folder/'best.pt';check(cp)
  ck=torch.load(cp,map_location='cpu',weights_only=False);model=old.CenteredTransportPredictor(reference,basis);model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False)
  for split in protocol['splits']:
   pairs=old.paired_rows([r for r in store.manifest['records'] if r['task']=='gate' and r['split']==split]);assert len(pairs)==128
   records=[r for pair in pairs for r in pair];first_pair=records[0]['pair']
   for offset in range(0,len(records),8):
    inputs=[];descriptions=[]
    for rec in records[offset:offset+8]:
     z=store.features(rec);projected=store.projected_features(rec);q=int(rec['query'])
     ev=tuple(map(int,old.load_episode(c,rec,True,keys=('evidence',))['evidence']))
     actions=old.load_episode(c,rec,keys=('actions',))['actions'];end=q-int(rec['gap'])-4
     check(Path(c['output'])/'features'/store.ix['episodes'][rec['id']]['path'])
     for group,pool in old.gate_age_counterbalance(ev[0],end,q).items():
      for deleted in pool:
       kept=tuple(t for t in pool if t!=deleted);inputs.append(old.sample(z,projected,actions,q,kept));descriptions.append((rec,group,pool,deleted,ev[0]))
    batch=old.collate(inputs,'cpu')
    with torch.inference_mode():prediction=model(batch)
    assert prediction.shape==(len(inputs),2,5120) and torch.isfinite(prediction).all()
    pred=prediction.numpy();scores=decoder.predict(pred.reshape(-1,5120))[:,0].reshape(len(pred),2)
    cells=collections.defaultdict(list)
    for i,(rec,group,pool,deleted,cue) in enumerate(descriptions):
     # All targets and class labels enter scoring only after forward.
     z=store.features(rec);q=int(rec['query']);meta=old.load_episode(c,rec,True,keys=('classes',))
     for hi,h in enumerate((3,6)):
      target=z[q+h];label=2*int(meta['classes'][q+h])-1;error=float((scores[i,hi]>0)!=(label>0))
      k=(seed,split,rec['pair'],rec['id'],group,deleted,h);assert k not in seen;seen.add(k)
      expected=gold[k];diff=abs(float(np.mean((pred[i,hi]-target)**2))-float(expected['feature_mse']));max_error=max(max_error,diff)
      assert diff<=1e-7 and error==float(expected['task_error'])
      target_score=float(decoder.predict(target[None])[0,0]);target_scores[(split,rec['id'],h)]=(target_score,label)
      cells[(rec['pair'],group,pool,deleted,cue,h)].append((rec['id'],pred[i,hi].copy(),target.copy(),scores[i,hi],target_score,label,error,float(expected['feature_mse'])))
    parents=collections.defaultdict(list)
    for (pair,group,pool,deleted,cue,h),rr in cells.items():
     rr=sorted(rr,key=lambda x:x[0]);assert len(rr)==2 and set(x[5] for x in rr)=={-1,1}
     pp=np.stack([x[1] for x in rr]);tt=np.stack([x[2] for x in rr]);v=decompose(pp,tt,w,Q)
     v.update(task_error=float(np.mean([x[6] for x in rr])),margin_loss=-float(np.mean([x[3]*x[5] for x in rr])),score_error=float(np.mean([(x[3]-x[4])**2 for x in rr])),label_score_error=float(np.mean([(x[3]-x[5])**2 for x in rr])),original_feature_mse=float(np.mean([x[7] for x in rr])))
     max_decomposition=max(max_decomposition,abs(v['total']-v['parallel']-v['orthogonal']),abs(v['total']-v['common']-v['contrast']))
     if pair==first_pair:
      witness_p.append(pp);witness_t.append(tt);witness_metrics.append([v[m] for m in metric_names]);witness_ids.append([str(seed),split,pair,group,str(h),str(deleted)])
     parents[(pair,group,pool,cue,h)].append((deleted,v))
    for (pair,group,pool,cue,h),vv in parents.items():
     assert len(vv)==3 and {d for d,_ in vv}==set(pool)
     crit=next(v for d,v in vv if d==cue);others=[v for d,v in vv if d!=cue]
     row=dict(seed=seed,split=split,pool_group=group,pair=pair,horizon=h,parent=';'.join(map(str,pool)),cue=cue)
     for name in (*analysis.METRICS,'target_pair_signal','prediction_sibling_maxabs','original_feature_mse'):
      row['critical_'+name]=crit[name];row['other_'+name]=float(np.mean([v[name] for v in others]))
     max_outside_delta=max(max_outside_delta,max(v['basis_outside'] for _,v in vv)-min(v['basis_outside'] for _,v in vv))
     row.update(deletion_tradeoff((crit['original_feature_mse'],crit['task_error']),[(v['original_feature_mse'],v['task_error']) for v in others]))
     rows.append(row)
    if time.monotonic()-begin>1200:raise RuntimeError('CPU diagnostic soft limit; preserve partial, no resubmit')
   write(out/'progress.json',dict(seed=seed,split_completed=split,reproduced_raw_rows=len(seen),expected_raw_rows=41472,elapsed=time.monotonic()-begin,GPU=0));print(f'Gate{seed} {split}: reproduced{len(seen)}/41472',flush=True)
  assert sha(cp)==proof['best_sha256'];del model,reference,ck
 assert len(seen)==41472 and len(rows)==6912 and len(target_scores)==1536 and len(witness_p)==162
 assert max_outside_delta<1e-6
 path=out/'geometry_pairs.csv'
 with path.open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 np.savez_compressed(out/'vector_witnesses.npz',prediction=np.stack(witness_p),target=np.stack(witness_t),w=w,Q=Q,metrics=np.asarray(witness_metrics),metric_names=np.asarray(metric_names),witness_ids=np.asarray(witness_ids))
 result=analysis.summarize(rows)
 target_pred=np.array([x[0] for x in target_scores.values()]);target_y=np.array([x[1] for x in target_scores.values()])
 result.update(job=job,allocation=state,parent_states=native_parents,raw_reproduced=len(seen),max_original_feature_error_difference=max_error,max_decomposition_error=max_decomposition,max_across_fork_output_floor_difference=max_outside_delta,
  unique_real_target_readout_accuracy=float(np.mean((target_pred>0)==(target_y>0))),unique_real_targets=len(target_scores),true_target_signed_margin_quantiles=np.quantile(target_pred*target_y,[0,.25,.5,.75,1]).tolist(),
  paired_csv_sha256=sha(path),witness_sha256=sha(out/'vector_witnesses.npz'),source_lock_sha256=sha(ROOT/'source_lock.json'),protocol_sha256=sha(ROOT/'protocol.json'),elapsed_seconds=time.monotonic()-begin,
  parent_summary_sha256=protocol['parent_summary_sha256'],parent_raw_sha256=protocol['parent_raw_sha256'],assets_stable=True)
 assert all(sha(k)==v for k,v in assets.items());old.verify_control(old.OLD/'control')
 assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
 write(out/'asset_ledger.json',assets);result['asset_ledger_sha256']=sha(out/'asset_ledger.json');write(out/'summary.json',result)
 print(json.dumps({k:v for k,v in result.items() if k!='cells' and k!='allocation'}),flush=True)

def check_asset(path,assets,expected):
 path=str(path)
 if path not in assets:
  assets[path]=sha(path);assert assets[path]==expected[path]

if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  folder=ROOT/'output'/f"job_{os.environ.get('SLURM_JOB_ID','unknown')}";folder.mkdir(parents=True,exist_ok=True)
  write(folder/'failure.json',dict(error=traceback.format_exc(),no_automatic_retry=True));raise
