"""Frozen Gate: original validation contexts versus same-capacity deletion inputs."""
import collections,csv,datetime,hashlib,importlib.util,json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from boundary import pack,remove,same,paired_errors
ROOT=Path(__file__).resolve().parent;AGE=Path('/scratch/xl4358/phase1_age_control_a1');CAL=Path('/scratch/xl4358/phase1_gate_calibration')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert sha(AGE/'source_lock.json')=='80336bfde8fd9da47d31060e7725e34486dc042df5f0c83e9f2a3161055bba5f'
sys.path.insert(0,str(AGE));import run as old
from phase1.training import TRAIN_CONDITIONS
spec=importlib.util.spec_from_file_location('val_deletion_analysis',ROOT/'analysis.py');analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)

def main():
 begin=time.monotonic();job=int(sys.argv[1]);assert job==int(os.environ['SLURM_JOB_ID'])
 native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);ff=dict(x.split('=',1) for x in native.split() if '=' in x)
 assert ff['JobState']=='RUNNING' and ff['Partition']=='n2c48m24' and ff['Account']=='csci_ga_2271-2026fa' and ff['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in ff['AllocTRES']
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 for root in (ROOT,AGE,CAL):assert all(sha(root/k)==v for k,v in read(root/'source_lock.json')['files'].items())
 assert sha(CAL/'source_lock.json')=='ece18ff37c6ef0f4e6f86add5f066cf3453d07a9bf448f55f13bd3254bee1335'
 assert sha(CAL/'execution_receipt.json')=='df7c17d042bccd9381716fe4e0a367b5250a9339bdc0822ac2bf064a4140b7d2'
 receipt=read(CAL/'execution_receipt.json');assert receipt['diagnostic_returncode']==receipt['independent_returncode']==0
 goldpath=CAL/'output/job_8289/validation_scores.csv';assert sha(goldpath)=='c3e59091004c1a2636f602a396d42599042f053e54abee6f15fdb88784f6e879'
 with goldpath.open() as f:gold={(int(r['seed']),r['condition'],r['episode'],int(r['horizon'])):float(r['feature_mse']) for r in csv.DictReader(f) if float(r['alpha'])==1.}
 assert len(gold)==4608
 states=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-n','-P','-j','7574,7575,7576,8275','--format=JobID,State,ExitCode'],text=True);assert set(states.strip().splitlines())=={f'{j}|COMPLETED|0:0' for j in (7574,7575,7576,8275)}
 out=ROOT/'output'/f'job_{job}';out.mkdir(parents=True,exist_ok=False);write(out/'attempt.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),allocation=native,job=job,source_lock_sha256=sha(ROOT/'source_lock.json'),training=False,GPU=0))
 p,_,configs=old.verify_control(old.OLD/'control');c=configs['gate'];assert c['data']['seed']==6301 and c['data']['budget']==2 and c['data']['recent']==4
 store=old.FeatureStore(c,max_cached=4);pairs=old.paired_rows([r for r in store.manifest['records'] if r['task']=='gate' and r['split']=='validation']);assert len(pairs)==96
 records=[r for pair in pairs for r in pair];parent=read(AGE/'output/job_8275/summary.json');assert sha(AGE/'output/job_8275/summary.json')=='ed9232c2503b76030eea05ad4b54cf565774f952200f69a85306b873b131e35f'
 assets={};oldassets=read(AGE/'output/job_8275/asset_ledger.json')
 def check(path):
  path=str(path)
  if path not in assets:assets[path]=sha(path)
  if path in oldassets:assert assets[path]==oldassets[path]
 bp=Path(c['output'])/'predictor/gate/basis.npz';check(bp)
 with np.load(bp,allow_pickle=False) as f:basis={k:f[k] for k in f.files}
 store.bases['gate']=basis;dp=Path(c['output'])/'probes/gate_state_decoder.npz';check(dp);decoder=old.LinearReadout.load(dp)
 for path in (Path(c['output'])/'features/normalization.npz',Path(c['output'])/'features/index.json'):check(path)
 raw=[];paired=[];paths=0;anchors=0;maxold=0.;maxanchor=0.;maxpathforward=0.;maxtotal=0.;targetreadout={};proofs={};witness_p=[];witness_t=[];witness_metrics=[]
 for seed in (17,29,43):
  ref,refsha=old.reference_model(c,p,'gate',seed,basis);check(Path(c['output'])/'predictor/gate'/f'seed_{seed}/best.pt')
  folder=old.OLD/'training/job_7574/predictor/gate'/f'seed_{seed}';proof=old.verify_completed_model(folder,parent['models'][f'gate/{seed}']['signature'],ref.state_dict());assert proof==parent['models'][f'gate/{seed}'];check(folder/'best.pt');proofs[str(seed)]=proof
  model=old.CenteredTransportPredictor(ref,basis);model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model'],strict=True);model.eval().requires_grad_(False)
  for offset in range(0,len(records),16):
   rr=records[offset:offset+16];predictions={};descriptions={};standard_samples={}
   for rec in rr:check(Path(c['output'])/'features'/store.ix['episodes'][rec['id']]['path'])
   # Original conditions are replayed in exactly the old batch shape and order.
   for condition in TRAIN_CONDITIONS:
    samples=[store.sample(rec,condition) for rec in rr];batch=old.collate([s[0] for s in samples],'cpu')
    with torch.inference_mode():pred=model(batch).numpy()
    for i,(rec,ss) in enumerate(zip(rr,samples)):
     assert rec['split']=='validation';idx=tuple(map(int,ss[2]['indices']));predictions[(rec['id'],condition)]=pred[i].copy();standard_samples[(rec['id'],condition)]=ss[0]
     descriptions[(rec['id'],condition)]=dict(variant=condition,family='original',pool_group='',parent='',deleted=-1,kept=';'.join(map(str,idx)),history_count=len(idx),native_privileged=int(condition=='privileged_small_budget'))
   inputs=[];removed_inputs=[];descs=[]
   for rec in rr:
    z=store.features(rec);proj=store.projected_features(rec);q=int(rec['query']);cue=int(old.load_episode(c,rec,True,keys=('evidence',))['evidence'][0]);actions=old.load_episode(c,rec,keys=('actions',))['actions'];end=q-int(rec['gap'])-4
    # All newly packed arrays derive from historical observations and legal actions.
    for group,pool in old.gate_age_counterbalance(cue,end,q).items():
     parent_sample=pack(z,proj,actions,q,pool)
     for j,deleted in enumerate(pool):
      kept=tuple(t for t in pool if t!=deleted);direct=old.sample(z,proj,actions,q,kept);removed=remove(parent_sample,j)
      assert same(direct,removed);paths+=1
      original_anchor=standard_samples[(rec['id'],'privileged_small_budget')];anchor=same(direct,original_anchor)
      if anchor:anchors+=1
      variant=f'{group}/delete_{j}';meta=dict(variant=variant,family='deletion',pool_group=group,parent=';'.join(map(str,pool)),deleted=deleted,kept=';'.join(map(str,kept)),history_count=2,native_privileged=int(anchor));inputs.append(direct);removed_inputs.append(removed);descs.append((rec,meta,cue))
   batch=old.collate(inputs,'cpu')
   with torch.inference_mode():pred=model(batch).numpy()
   assert pred.shape==(len(inputs),2,5120) and np.isfinite(pred).all()
   if offset==0:
    with torch.inference_mode():alternate=model(old.collate(removed_inputs,'cpu')).numpy()
    maxpathforward=max(maxpathforward,float(np.abs(pred-alternate).max()));assert maxpathforward==0
   for i,(rec,meta,cue) in enumerate(descs):
    variant=meta['variant'];predictions[(rec['id'],variant)]=pred[i].copy();descriptions[(rec['id'],variant)]=meta
    if meta['native_privileged']:maxanchor=max(maxanchor,float(np.abs(pred[i]-predictions[(rec['id'],'privileged_small_budget')]).max()))
   assert maxanchor<3e-5
   # Future targets and task truth are used exclusively for scoring after all forwards.
   by=collections.defaultdict(list)
   for rec in rr:
    z=store.features(rec);q=int(rec['query']);cue=int(old.load_episode(c,rec,True,keys=('evidence',))['evidence'][0]);classes=old.load_episode(c,rec,True,keys=('classes',))['classes']
    variants=[v for eid,v in predictions if eid==rec['id']];assert len(variants)==13
    for variant in variants:
     pp=predictions[(rec['id'],variant)];scores=decoder.predict(pp)[:,0];meta=descriptions[(rec['id'],variant)]
     for hi,h in enumerate((3,6)):
      target=z[q+h];truth=int(classes[q+h]>0);loss=float(np.mean((pp[hi]-target)**2));task=float((scores[hi]>0)!=truth)
      if variant in TRAIN_CONDITIONS:maxold=max(maxold,abs(loss-gold[(seed,variant,rec['id'],h)]))
      targetscore=float(decoder.predict(target[None])[0,0]);targetreadout[(rec['id'],h)]=(int((targetscore>0)==bool(truth)),targetscore)
      row=dict(seed=seed,split='validation',pair=rec['pair'],episode=rec['id'],query=q,gap=rec['gap'],cue=cue,horizon=h,**meta,cue_present=int(cue in tuple(map(int,meta['kept'].split(';'))) if meta['kept'] else False),feature_mse=loss,task_error=task)
      raw.append(row);by[(rec['pair'],variant,h)].append((rec['id'],pp[hi].copy(),target.copy(),row))
   for (pair,variant,h),vv in by.items():
    vv=sorted(vv,key=lambda x:x[0]);assert len(vv)==2
    pp=np.stack([v[1] for v in vv]);tt=np.stack([v[2] for v in vv]);v=paired_errors(pp,tt);r0=vv[0][3]
    for field in ('kept','parent','deleted','cue_present','native_privileged'):assert vv[0][3][field]==vv[1][3][field]
    if r0['family']=='deletion' and not r0['cue_present']:assert np.array_equal(pp[0],pp[1])
    pairrow={k:r0[k] for k in ('seed','split','pair','variant','family','pool_group','parent','deleted','kept','history_count','native_privileged','cue_present','cue','horizon','query','gap')};pairrow.update(v,feature_mse=float(np.mean([v0[3]['feature_mse'] for v0 in vv])),task_error=float(np.mean([v0[3]['task_error'] for v0 in vv])));paired.append(pairrow)
    maxtotal=max(maxtotal,abs(pairrow['feature_mse']-pairrow['total']))
    if pair==pairs[0][0]['pair']:
     witness_p.append(pp);witness_t.append(tt);witness_metrics.append([pairrow[k] for k in ('total','common','contrast','prediction_sibling_maxabs')])
   if time.monotonic()-begin>1500:raise RuntimeError('CPU soft bound; preserve partial, no retry')
   write(out/'progress.json',dict(seed=seed,completed_validation_pairs=offset//2+len(rr)//2,raw_rows=len(raw),expected_raw=14976,elapsed=time.monotonic()-begin))
  print(f'Gate{seed}: all96validationpairs /13contexts complete',flush=True)
  assert not model.training and all(p.grad is None for p in model.parameters());del model,ref
 assert len(raw)==14976 and len(paired)==7488 and paths==5184 and anchors==576 and len(targetreadout)==384 and maxold<=1e-7 and maxtotal<1e-7
 def csvwrite(name,rr):
  with (out/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
 csvwrite('raw_validation.csv',raw);csvwrite('paired_validation.csv',paired)
 assert all(sha(k)==v for k,v in assets.items());old.verify_control(old.OLD/'control');assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
 assert len(witness_p)==78
 np.savez_compressed(out/'vector_witnesses.npz',prediction=np.stack(witness_p),target=np.stack(witness_t),metrics=np.asarray(witness_metrics))
 write(out/'asset_ledger.json',assets)
 result=analysis.summarize(paired);result.update(job=job,allocation=native,parent_states=states,raw_rows=len(raw),paired_rows=len(paired),validation_pairs=96,models=proofs,only_validation_features_loaded=True,max_prior_validation_MSE_difference=maxold,physical_delete_path_input_equal=paths,max_path_forward_difference=maxpathforward,native_privileged_anchor_input_equal=anchors,max_anchor_forward_difference=maxanchor,max_float32_geometry_difference=maxtotal,true_target_decoder_correct=sum(v[0] for v in targetreadout.values()),true_target_decoder_total=384,assets_stable=True,raw_sha256=sha(out/'raw_validation.csv'),paired_sha256=sha(out/'paired_validation.csv'),assets_sha256=sha(out/'asset_ledger.json'),source_lock_sha256=sha(ROOT/'source_lock.json'),protocol_sha256=sha(ROOT/'protocol.json'),witness_sha256=sha(out/'vector_witnesses.npz'),witness_pairs=78,elapsed=time.monotonic()-begin,GPU=0,new_training=False,new_fitting=False,original_gates_changed=False,original_Phase1_GO=False,fresh9137_opened=False)
 write(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('cells','models','allocation','comparisons')}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  out=ROOT/'output'/f"job_{os.environ.get('SLURM_JOB_ID','unknown')}";out.mkdir(parents=True,exist_ok=True);write(out/'failure.json',dict(error=traceback.format_exc(),no_automatic_retry=True));raise
