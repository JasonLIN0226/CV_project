"""Cached frozen-history increment plus observed persistence/training-mean anchor; no fitting or forward."""
import csv,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
from core import maps,constrain,interval
import torch
ROOT=Path(__file__).resolve().parent
P=Path('/scratch/xl4358/phase1_motion_geometry_validation/output/job_8336')
REF=Path('/scratch/xl4358/phase1_v3_dev/motion/reference')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load():
 protocol=read(ROOT/'protocol.json');s=read(P/'summary.json');assert sha(P/'summary.json')==protocol['parent_summary_sha256']
 assert read(P/'independent_audit.json')['status']=='PASS'
 assert sha(P/'prediction_witnesses.npz')==s['witness_sha256'] and sha(P/'validation_geometry.csv')==s['raw_sha256']
 with (P/'validation_geometry.csv').open() as f:rows=list(csv.DictReader(f))
 with np.load(P/'prediction_witnesses.npz',allow_pickle=False) as f:recent=f['predictions'][:,0].astype(float);hist=f['predictions'][:,1].astype(float);target=f['targets'].astype(float)
 assert hist.shape==recent.shape==target.shape==(1152,5120)
 assets=read(P/'asset_ledger.json');assert all(sha(k)==v for k,v in assets.items())
 lock=read('/scratch/xl4358/phase1_motion_locked_test/candidate_lock.json');assert sha('/scratch/xl4358/phase1_motion_locked_test/candidate_lock.json')==protocol['candidate_lock_sha256'];assert all(sha(k)==v for k,v in lock['weights_and_transforms'].items());assets.update(lock['weights_and_transforms'])
 with np.load(REF/'predictor/motion/basis.npz',allow_pickle=False) as f:B=f['output_basis']
 with np.load(REF/'probes/motion_state_decoder.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
 B,A,b,right,_=maps(B,d['weight'],d['std'],d['mean'],d['bias'])
 # Reconstruct the current nominal two-evidence candidate using ONLY observed-input geometry.
 g=np.array([[float(r['kinematic_actions_x']),float(r['kinematic_actions_y'])] for r in rows])
 sys.path.insert(0,'/scratch/xl4358/phase1_age_control_a1');import run as old
 _,_,cfg=old.verify_control(old.OLD/'control');c=cfg['motion'];store=old.FeatureStore(c,max_cached=1);recs=store.records;guarded={}
 from guard import allow_update
 for r in rows:
  eid=r['episode']
  if eid not in guarded:
   rec=recs[eid];obs=old.load_episode(c,rec,keys=('rgb',));guarded[eid]=allow_update(obs['rgb'][[int(r['start']),int(r['end'])]],obs['rgb'][int(rec['query'])]);path=Path(c['output'])/'dataset'/rec['obs'];assets[str(path)]=sha(path)
 assert len(guarded)==192 and all(guarded.values()),'Nominal cached inputs must reproduce current enabled guard; preserve failure otherwise'
 current,_=constrain(hist,g,B,A,b,right)
 # Swap ONLY the learned recent anchor; preserve the best historical increment.
 HV=Path('/scratch/xl4358/phase1_motion_history_reference_validation/output/job_8449');vs=read(HV/'summary.json');assert sha(HV/'summary.json')==protocol['history_validation_summary_sha256'] and sha(HV/'vectors.npz')==vs['vectors_sha256']
 assert sha(HV/'independent_audit.json')==protocol['history_validation_audit_sha256'] and read(HV/'independent_audit.json')['status']=='PASS_INDEPENDENT_QR_AND_SCALAR_AUDIT'
 with np.load(HV/'vectors.npz',allow_pickle=False) as f:
  assert np.array_equal(target,f['target']) and np.max(abs(recent-f['recent']))<1e-7;reference=f['reference_history'].astype(float);previous=f['candidate'].astype(float)
 assert np.max(abs((previous-current)@A))<1e-10
 basispath=REF/'predictor/motion/basis.npz';metapath=basispath.with_suffix('.json');meta=read(metapath);assert meta['fit_split']=='train' and meta['sha256']==sha(basispath);assets[str(metapath)]=sha(metapath)
 with np.load(basispath,allow_pickle=False) as f:
  target_mean=f['target_mean'].astype(float);train_ids=f['train_ids'].tolist()
 assert target_mean.shape==(5120,) and len(train_ids)==1024 and all(recs[e]['split']=='train' and recs[e]['task']=='motion' for e in train_ids)
 observed=np.empty_like(reference);seen={}
 for i,r in enumerate(rows):
  eid=r['episode'];rec=recs[eid]
  if eid not in seen:
   seen[eid]=store.features(rec)[int(rec['query'])].astype(float);fp=Path(c['output'])/'features'/store.ix['episodes'][eid]['path'];assets[str(fp)]=sha(fp)
  observed[i]=seen[eid]
 assert len(seen)==192 and np.isfinite(observed).all()
 base=observed+target_mean+(reference-recent)
 candidate,_=constrain(base,current@A+b,B,A,b,right);current=previous
 assert np.max(abs((candidate-current)@A))<1e-10
 np.savez_compressed(ROOT/'anchor_inputs.npz',observed=observed,target_mean=target_mean,base=base)
 state_proofs={'no_model_forward':True,'basis_fit_split':'train','training_episodes':1024};maxreplay=0.
 return rows,recent,hist,current,candidate,target,B,A,b,assets,reference,state_proofs,maxreplay

def main(job):
 t=time.monotonic();native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);n=dict(x.split('=',1) for x in native.split() if '=' in x)
 assert n['JobState']=='RUNNING' and n['Account']=='csci_ga_2271-2026fa' and n['Partition']=='n2c48m24' and n['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in n['AllocTRES']
 assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items());out=ROOT/'output'/f'job_{job}';out.mkdir(parents=True,exist_ok=False)
 rows,recent,hist,current,candidate,target,B,A,b,assets,reference,state_proofs,maxrefreplay=load()
 truth=np.array([[float(r['target_x']),float(r['target_y'])] for r in rows]);arms={'recent':recent,'old_history':hist,'current':current,'reference_history':reference,'candidate':candidate};losses={};raw=[]
 for arm,z in arms.items():
  assert np.isfinite(z).all();f=((z-target)**2).mean(1);p=((z@A+b-truth)**2).mean(1);losses[arm]=(f,p)
  for i,r in enumerate(rows):raw.append(dict(seed=int(r['seed']),horizon=int(r['horizon']),pair=r['pair'],episode=r['episode'],arm=arm,feature_mse=float(f[i]),position_mse=float(p[i])))
 replay=max(np.max(abs(losses['old_history'][0]-[float(r['privileged_small_budget_feature_mse']) for r in rows])),np.max(abs(losses['recent'][0]-[float(r['recent_only_feature_mse']) for r in rows])))
 assert replay<1e-7 and np.max(abs(losses['current'][1]-losses['candidate'][1]))<1e-10
 cells=[]
 for seed in (17,29,43,'mean'):
  for h in (3,6):
   ix=[i for i,r in enumerate(rows) if int(r['horizon'])==h and (seed=='mean' or int(r['seed'])==seed)];groups=[rows[i]['pair'] for i in ix]
   cf,cp=losses['candidate'];rf,rp=losses['recent'];of,op=losses['current']
   cells.append(dict(seed=seed,horizon=h,feature_vs_recent=interval(rf[ix],cf[ix],groups),position_vs_recent=interval(rp[ix],cp[ix],groups,1e-5,1e-6),feature_vs_current=interval(of[ix],cf[ix],groups),position_vs_current=interval(op[ix],cp[ix],groups,1e-5,1e-6),recent_feature=float(rf[ix].mean()),current_feature=float(of[ix].mean()),candidate_feature=float(cf[ix].mean())))
 with (out/'scalar_rows.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
 np.savez_compressed(out/'vectors.npz',recent=recent,current=current,candidate=candidate,reference_history=reference,target=target,truth=truth)
 write(out/'asset_ledger.json',assets);assert all(sha(k)==v for k,v in assets.items());assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
 eligible=all(c['feature_vs_recent']['numeric_rule_met'] and c['position_vs_recent']['numeric_rule_met'] and c['feature_vs_current']['low']>0 for c in cells)
 write(out/'summary.json',dict(status='VALIDATION_SUPPORTED_NOT_ACCEPTANCE' if eligible else 'VALIDATION_REJECTED_STOP',validation_eligible=eligible,source_rows=1152,scalar_rows=len(raw),pairs=96,cells=cells,feature_checks=sum(c['feature_vs_recent']['numeric_rule_met'] for c in cells),position_checks=sum(c['position_vs_recent']['numeric_rule_met'] for c in cells),incremental_feature_positive=sum(c['feature_vs_current']['low']>0 for c in cells),checks_each=8,original_phase1_go=False,original_gates_changed=False,source_lock_sha256=sha(ROOT/'source_lock.json'),scalar_sha256=sha(out/'scalar_rows.csv'),vectors_sha256=sha(out/'vectors.npz'),position_preservation_max=float(abs((candidate-current)@A).max()),source_replay_max=float(replay),reference_replay_max=maxrefreplay,reference_state_proofs=state_proofs,GPU=0,training=False,model_forward=False,fresh9137_opened=False,test_opened=False,elapsed_seconds=time.monotonic()-t,allocation=native))
if __name__=='__main__':main(int(sys.argv[1]))
