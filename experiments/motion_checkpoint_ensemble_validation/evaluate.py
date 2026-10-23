"""Fixed equal best/final checkpoint predictions plus existing position constraint; no fitting."""
import csv,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
from core import maps,constrain,interval
import torch
import copy
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
 # Frozen old reference receives the SAME legitimate two historical observations.
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 p,_,cfg=old.verify_control(old.OLD/'control');c=cfg['motion'];store=old.FeatureStore(c,max_cached=192)
 with np.load(REF/'predictor/motion/basis.npz',allow_pickle=False) as f:basis={k:f[k] for k in f.files}
 store.bases['motion']=basis
 goldpath=REF/'evaluation/raw_records.csv';parent=read('/scratch/xl4358/phase1_motion_locked_test/output/job_8349/summary.json');assert sha(goldpath)==parent['prediction']['reference_raw_sha256'];assets[str(goldpath)]=sha(goldpath)
 with goldpath.open() as f:gold={(int(r['seed']),r['episode'],int(r['horizon'])):r for r in csv.DictReader(f) if r['split']=='validation' and r['condition']=='privileged_small_budget'}
 assert len(gold)==1152
 reference=np.zeros_like(current);best_reference=np.zeros_like(current);latest_reference=np.zeros_like(current);state_proofs={};maxreplay=0.;repeat=0.;component=read(ROOT/'component_lock.json')
 def digest(m):
  h=hashlib.sha256()
  for k,v in m.state_dict().items():h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 for seed in (17,29,43):
  m,weightsha=old.reference_model(c,p,'motion',seed,basis);m.eval().requires_grad_(False);initial=digest(m);episodes=list(dict.fromkeys(r['episode'] for r in rows if int(r['seed'])==seed));assert len(episodes)==192
  weights=REF/'predictor/motion'/f'seed_{seed}/best.pt';assert sha(weights)==weightsha==component[str(seed)]['best.pt'];assets[str(weights)]=weightsha
  training=weights.with_name('training.json');assert sha(training)==component[str(seed)]['training.json'];assets[str(training)]=sha(training)
  latest=weights.with_name('latest.pt');assert sha(latest)==component[str(seed)]['latest.pt'];assets[str(latest)]=sha(latest);ck=torch.load(latest,map_location='cpu',weights_only=False);bestck=torch.load(weights,map_location='cpu',weights_only=False);assert ck['signature']==bestck['signature'] and ck['epoch']==120 and ck['step']==7680
  last=copy.deepcopy(m);last.load_state_dict(ck['model'],strict=True);last.eval().requires_grad_(False);lastdigest=digest(last);assert all(torch.isfinite(v).all() for v in last.state_dict().values() if v.is_floating_point())
  for j in range(0,192,8):
   recs=[store.records[e] for e in episodes[j:j+8]];samples=[store.sample(r,'privileged_small_budget')[0] for r in recs];batch=old.collate(samples,'cpu')
   with torch.inference_mode():pred=m(batch).numpy();late=last(batch).numpy()
   assert pred.shape==late.shape==(len(recs),2,5120) and np.isfinite(pred).all() and np.isfinite(late).all()
   if j==0:
    with torch.inference_mode():repeat=max(repeat,float(abs(m(batch).numpy()-pred).max()),float(abs(last(batch).numpy()-late).max()))
   for q,rec in enumerate(recs):
    for hi,h in enumerate((3,6)):
     ids=[i for i,r in enumerate(rows) if int(r['seed'])==seed and r['episode']==rec['id'] and int(r['horizon'])==h];assert len(ids)==1;i=ids[0];best_reference[i]=pred[q,hi];latest_reference[i]=late[q,hi];reference[i]=.5*(pred[q,hi].astype(float)+late[q,hi].astype(float))
     got=float(np.mean((pred[q,hi]-target[i])**2));xy=pred[q,hi]@A+b;truth=np.array([float(rows[i]['target_x']),float(rows[i]['target_y'])]);pos=float(np.mean((xy-truth)**2));v=gold[seed,rec['id'],h]
     maxreplay=max(maxreplay,abs(got-float(v['normalized_mse'])),abs(pos-float(v['state_position_mse'])));assert maxreplay<2e-6
  assert digest(m)==initial and digest(last)==lastdigest and all(v.grad is None for v in last.parameters());state_proofs[str(seed)]={'state_digest':initial,'weight_sha256':weightsha,'latest_state_digest':lastdigest,'latest_weight_sha256':sha(latest)};del m,last,ck,bestck
 assert repeat==0
 previous,_=constrain(best_reference,current@A+b,B,A,b,right)
  # Keep previous exact composite as the incremental comparison.
 current=previous
 np.savez_compressed(ROOT/'ensemble_witness.npz',best=best_reference,latest=latest_reference,ensemble=reference)
 candidate,_=constrain(reference,current@A+b,B,A,b,right)
 assert np.max(abs((candidate-current)@A))<1e-10
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
 eligible=all(c['feature_vs_recent']['numeric_rule_met'] and c['position_vs_recent']['numeric_rule_met'] for c in cells) and all(c['feature_vs_current']['low']>0 for c in cells if c['seed']=='mean')
 write(out/'summary.json',dict(status='VALIDATION_SUPPORTED_NOT_ACCEPTANCE' if eligible else 'VALIDATION_REJECTED_STOP',validation_eligible=eligible,source_rows=1152,scalar_rows=len(raw),pairs=96,cells=cells,feature_checks=sum(c['feature_vs_recent']['numeric_rule_met'] for c in cells),position_checks=sum(c['position_vs_recent']['numeric_rule_met'] for c in cells),incremental_feature_positive=sum(c['feature_vs_current']['low']>0 for c in cells),checks_each=8,original_phase1_go=False,original_gates_changed=False,source_lock_sha256=sha(ROOT/'source_lock.json'),scalar_sha256=sha(out/'scalar_rows.csv'),vectors_sha256=sha(out/'vectors.npz'),position_preservation_max=float(abs((candidate-current)@A).max()),source_replay_max=float(replay),reference_replay_max=maxrefreplay,reference_state_proofs=state_proofs,GPU=0,training=False,model_forward=True,fresh9137_opened=False,test_opened=False,elapsed_seconds=time.monotonic()-t,allocation=native))
if __name__=='__main__':main(int(sys.argv[1]))
