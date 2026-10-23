"""One prelocked transport-off original development test; no selection or training."""
import collections,copy,csv,datetime,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;AGE=Path('/scratch/xl4358/phase1_age_control_a1');VAL=Path('/scratch/xl4358/phase1_gate_transport_ablation')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sys.path.insert(0,str(AGE));import run as old
import original_evaluation as original
from intervention import both,state_digest
from deletion_primary import analyze
from phase1.reporting import decision,_development_artifacts
from phase1.v5_synthetic_evaluation import fixed_reference_checks

def savecsv(path,rows):
 with Path(path).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
 begin=time.monotonic();job=int(sys.argv[1]);assert job==int(os.environ['SLURM_JOB_ID'])
 allocation=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);fields=dict(t.split('=',1) for t in allocation.split() if '=' in t)
 assert fields['JobState']=='RUNNING' and fields['Account']=='csci_ga_2271-2026fa' and fields['Partition']=='n2c48m24' and fields['NodeList']==socket.gethostname() and 'gres/gpu' not in fields['AllocTRES']
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 for root,expected in ((ROOT,None),(AGE,'80336bfde8fd9da47d31060e7725e34486dc042df5f0c83e9f2a3161055bba5f'),(VAL,'14d39167c9b59a2b64d1b2886305509b75a37c8271d903da226b9b0184962c5f')):
  if expected:assert sha(root/'source_lock.json')==expected
  assert all(sha(root/k)==v for k,v in read(root/'source_lock.json')['files'].items())
 assert sha(VAL/'output/job_8327/summary.json')==read(ROOT/'protocol.json')['validation_summary_sha256'] and read(VAL/'execution_receipt.json')['status']=='COMPLETE_WITH_INDEPENDENT_AUDIT'
 assert sha(ROOT/'intervention.py')==sha(VAL/'intervention.py')==read(ROOT/'protocol.json')['intervention_sha256']
 states=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-n','-P','-j','7574,7575,7576,8275','--format=JobID,State,ExitCode'],text=True);assert set(states.strip().splitlines())=={f'{j}|COMPLETED|0:0' for j in (7574,7575,7576,8275)}
 out=ROOT/'output'/f'job_{job}';out.mkdir(parents=True,exist_ok=False);write(out/'attempt.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=job,allocation=allocation,candidate_lock=read(ROOT/'protocol.json'),GPU=0,training=False))
 p,_,configs=old.verify_control(old.OLD/'control');c=configs['gate'];assert c['data']['seed']==6301 and c['data']['budget']==2 and c['data']['recent']==4
 parent=read(AGE/'output/job_8275/summary.json');assert sha(AGE/'output/job_8275/summary.json')=='ed9232c2503b76030eea05ad4b54cf565774f952200f69a85306b873b131e35f'
 assert all(sha(k)==v for k,v in read(ROOT/'candidate_lock.json')['weights_and_transforms'].items())
 assets={};parentassets=read(AGE/'output/job_8275/asset_ledger.json')
 def check(path):
  path=str(path)
  if path not in assets:assets[path]=sha(path)
  if path in parentassets:assert assets[path]==parentassets[path]
 bp=Path(c['output'])/'predictor/gate/basis.npz';check(bp)
 with np.load(bp,allow_pickle=False) as f:basis={k:f[k] for k in f.files}
 for path in (Path(c['output'])/'features/normalization.npz',Path(c['output'])/'features/index.json',Path(c['output'])/'probes/gate_state_decoder.npz'):check(path)
 proofs={}
 def load(seed,store):
  store.bases['gate']=basis
  ref,_=old.reference_model(c,p,'gate',seed,basis);check(Path(c['output'])/'predictor/gate'/f'seed_{seed}/best.pt')
  folder=old.OLD/'training/job_7574/predictor/gate'/f'seed_{seed}';proof=old.verify_completed_model(folder,parent['models'][f'gate/{seed}']['signature'],ref.state_dict());assert proof==parent['models'][f'gate/{seed}'];check(folder/'best.pt');proofs[str(seed)]=proof
  ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model=old.CenteredTransportPredictor(ref,basis);model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False);assert state_digest(model)==read(ROOT/'candidate_lock.json')['state_digest_by_seed'][str(seed)];return model,ck
 # First execute the exact original prediction metric/input loop, candidate fixed OFF.
 original.LOADER=load;original.COMPLETION=dict(complete=True,models=parent['models'],scope='Frozen all6 parent proof; Gate3 actual checkpoint proofs reverified in current execution')
 metrics,raw,_=original.evaluate_candidate(c,None,old.OLD/'training/job_7574',out/'original')
 assert len(raw)==34560 and all(r['task']=='gate' for r in raw)
 _,oldmetrics,reference_raw,audit,probes=_development_artifacts(c['output'],'gate')
 verification=read(p['immutable_inputs']['gate']['frozen_v3_verification'])
 for rel in ('evaluation/raw_records.csv','evaluation/metrics.json','probes/metrics.json'):
  check(Path(c['output'])/rel);assert sha(Path(c['output'])/rel)==verification['actual_hpc_compact_sha256']['gate/reference/'+rel]
 profile=copy.deepcopy(c);profile['data']['tasks']=['gate'];profile.pop('experiment',None)
 original_status,reasons,checks=decision(profile,audit,probes,metrics);fixed=fixed_reference_checks(c,reference_raw,raw)
 assert len(checks)==51 and len(fixed)==32
 prediction_result=dict(task='gate',original_task_status=original_status,original_checks=checks,fixed_reference_checks=fixed,reasons=reasons,original_checks_passed=sum(x['passed'] for x in checks),fixed_checks_passed=sum(x['passed'] for x in fixed),status='SUPPORTED' if original_status=='GO' and all(x['passed'] for x in fixed) else 'REJECTED',candidate_raw_sha256=sha(out/'original/evaluation/raw_records.csv'),reference_raw_sha256=oldmetrics['raw_sha256'],phase1_go=False)
 write(out/'original/development_comparison.json',prediction_result);write(out/'original/outcome.json',dict(tasks={'gate':prediction_result},phase1_go=False));print('Original prediction evaluation finished; candidate remains locked.',flush=True)
 if time.monotonic()-begin>1500:raise RuntimeError('CPU bound; partial only, no retry')
 # Exact original age-controlled Gate deletion test, all three128pair splits.
 store=old.FeatureStore(c,max_cached=4);store.bases['gate']=basis;decoder=old.LinearReadout.load(Path(c['output'])/'probes/gate_state_decoder.npz')
 goldpath=AGE/'output/job_8275/raw_deletions.csv';assert sha(goldpath)==parent['raw_sha256']=='cc53f6cd73f9fcaed5e6a25f50edf46257fb652eafd5625b06d000297ab15a45'
 with goldpath.open() as f:gold={(int(r['seed']),r['split'],r['episode'],r['pool'],int(r['deleted']),int(r['horizon'])):r for r in csv.DictReader(f) if r['task']=='gate'}
 assert len(gold)==41472
 rows=[];maxreplay=0.;maxbranch=0.;stateproof={};witnessP=[];witnessT=[];witnessM=[]
 for seed in (17,29,43):
  model,_=load(seed,store);initial=state_digest(model)
  for split in ('test_id','test_long_gap','test_distractor_shift'):
   pairs=old.paired_rows([r for r in store.manifest['records'] if r['task']=='gate' and r['split']==split]);assert len(pairs)==128;records=[r for pair in pairs for r in pair]
   for offset in range(0,len(records),8):
    inputs=[];desc=[]
    for rec in records[offset:offset+8]:
     check(Path(c['output'])/'features'/store.ix['episodes'][rec['id']]['path']);z=store.features(rec);proj=store.projected_features(rec);q=int(rec['query']);cue=int(old.load_episode(c,rec,True,keys=('evidence',))['evidence'][0]);actions=old.load_episode(c,rec,keys=('actions',))['actions'];end=q-int(rec['gap'])-4
     for group,pool in old.gate_age_counterbalance(cue,end,q).items():
      for d in pool:
       kept=tuple(t for t in pool if t!=d);inputs.append(old.sample(z,proj,actions,q,kept));desc.append((rec,group,pool,d,kept,cue))
    arms,residual=both(model,old.collate(inputs,'cpu'));maxbranch=max(maxbranch,residual);pred=arms['transport_off'];before=arms['before'];score=decoder.predict(pred.reshape(-1,5120))[:,0].reshape(len(pred),2);oldscore=decoder.predict(before.reshape(-1,5120))[:,0].reshape(len(before),2)
    for i,(rec,g,pool,d,kept,cue) in enumerate(desc):
     q=int(rec['query']);z=store.features(rec);classes=old.load_episode(c,rec,True,keys=('classes',))['classes']
     for j,h in enumerate((3,6)):
      target=z[q+h];loss=float(np.mean((pred[i,j]-target)**2));oldloss=float(np.mean((before[i,j]-target)**2));key=(seed,split,rec['id'],g,d,h);expected=gold[key];maxreplay=max(maxreplay,abs(oldloss-float(expected['feature_mse'])));assert maxreplay<=1e-7 and float((oldscore[i,j]>0)!=(classes[q+h]>0))==float(expected['task_error'])
      rows.append(dict(seed=seed,split=split,pair=rec['pair'],episode=rec['id'],pool_group=g,parent=';'.join(map(str,pool)),deleted=d,kept=';'.join(map(str,kept)),cue=cue,query=q,horizon=h,history_count=2,feature_mse=loss,task_error=float((score[i,j]>0)!=(classes[q+h]>0)),old_feature_mse=oldloss,old_task_error=float(expected['task_error'])))
      if offset==0 and rec['pair']==pairs[0][0]['pair']:
       witnessP.append(pred[i,j]);witnessT.append(target);witnessM.append(loss)
    if time.monotonic()-begin>1500:raise RuntimeError('CPU bound; preserve partial, no automatic repeat')
   write(out/'progress.json',dict(stage='deletion',seed=seed,split=split,raw_rows=len(rows),expected_rows=41472));print(f'Deletion Gate{seed} {split} complete',flush=True)
  assert state_digest(model)==initial and all(p.grad is None for p in model.parameters());stateproof[str(seed)]=initial;del model
 assert len(rows)==41472 and maxreplay<=1e-7 and stateproof==original.STATE_PROOFS
 savecsv(out/'raw_deletions.csv',rows);paired,cells=analyze(rows);savecsv(out/'paired_deletions.csv',paired);assert len(paired)==6912 and len(cells)==72
 for rec in [r for r in store.manifest['records'] if r['task']=='gate' and r['split'] in ('validation','test_id','test_long_gap','test_distractor_shift')]:check(Path(c['output'])/'features'/store.ix['episodes'][rec['id']]['path'])
 assert all(sha(k)==v for k,v in assets.items());old.verify_control(old.OLD/'control');assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
 np.savez_compressed(out/'deletion_vector_witnesses.npz',prediction=np.stack(witnessP),target=np.stack(witnessT),feature_mse=np.asarray(witnessM));write(out/'asset_ledger.json',assets)
 counts=lambda mean:dict(total=sum((x['seed']=='mean')==mean for x in cells),both=sum(x['passed'] for x in cells if (x['seed']=='mean')==mean),feature=sum(x['metrics']['feature_mse']['passed'] for x in cells if (x['seed']=='mean')==mean),task=sum(x['metrics']['task_error']['passed'] for x in cells if (x['seed']=='mean')==mean))
 deletion_result=dict(cells=cells,individual=counts(False),seedmean=counts(True),baseline_all41472_replayed=True,max_baseline_difference=maxreplay,raw_rows=41472,paired_rows=6912)
 complete=prediction_result['status']=='SUPPORTED' and all(x['passed'] for x in cells)
 result=dict(status='GATE_DEVELOPMENT_REPAIR_SUPPORTED' if complete else 'GATE_DEVELOPMENT_REPAIR_NOT_SUPPORTED',prediction=prediction_result,deletion=deletion_result,source_lock_sha256=sha(ROOT/'source_lock.json'),protocol_sha256=sha(ROOT/'protocol.json'),raw_prediction_sha256=sha(out/'original/evaluation/raw_records.csv'),raw_deletion_sha256=sha(out/'raw_deletions.csv'),paired_deletion_sha256=sha(out/'paired_deletions.csv'),assets_sha256=sha(out/'asset_ledger.json'),witness_sha256=sha(out/'deletion_vector_witnesses.npz'),witness_count=len(witnessM),original_prediction_state_proofs=original.STATE_PROOFS,deletion_state_proofs=stateproof,max_branch_subtraction_residual=max(maxbranch,*original.BRANCH_RESIDUALS.values()),content_reference_unchanged=True,validation_selected_then_test_locked=True,test_driven_switching=False,elapsed=time.monotonic()-begin,job=job,allocation=allocation,GPU=0,training=False,fitting=False,original_gates_changed=False,original_Phase1_GO=False,fresh9137_opened=False)
 write(out/'summary.json',result);print(json.dumps(dict(status=result['status'],original=prediction_result['original_checks_passed'],fixed=prediction_result['fixed_checks_passed'],deletion=deletion_result['individual'])),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  out=ROOT/'output'/f"job_{os.environ.get('SLURM_JOB_ID','unknown')}";out.mkdir(parents=True,exist_ok=True);write(out/'failure.json',dict(error=traceback.format_exc(),no_automatic_retry=True));raise
