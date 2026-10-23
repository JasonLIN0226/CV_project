"""One validation-only output ablation; existing complete vectors, no forward/fitting."""
import csv,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
from core import maps,constrain,interval
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
 # The only candidate: recent null components + current historical decoded position.
 candidate,_=constrain(recent,current@A+b,B,A,b,right)
 assert np.max(abs((candidate-current)@A))<1e-10
 return rows,recent,hist,current,candidate,target,B,A,b,assets

def main(job):
 t=time.monotonic();native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);n=dict(x.split('=',1) for x in native.split() if '=' in x)
 assert n['JobState']=='RUNNING' and n['Account']=='csci_ga_2271-2026fa' and n['Partition']=='n2c48m24' and n['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in n['AllocTRES']
 assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items());out=ROOT/'output'/f'job_{job}';out.mkdir(parents=True,exist_ok=False)
 rows,recent,hist,current,candidate,target,B,A,b,assets=load()
 truth=np.array([[float(r['target_x']),float(r['target_y'])] for r in rows]);arms={'recent':recent,'old_history':hist,'current':current,'candidate':candidate};losses={};raw=[]
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
 np.savez_compressed(out/'vectors.npz',recent=recent,current=current,candidate=candidate,target=target,truth=truth)
 write(out/'asset_ledger.json',assets);assert all(sha(k)==v for k,v in assets.items());assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
 eligible=all(c['feature_vs_recent']['numeric_rule_met'] and c['position_vs_recent']['numeric_rule_met'] and c['feature_vs_current']['low']>0 for c in cells)
 write(out/'summary.json',dict(status='VALIDATION_SUPPORTED_NOT_ACCEPTANCE' if eligible else 'VALIDATION_REJECTED_STOP',validation_eligible=eligible,source_rows=1152,scalar_rows=len(raw),pairs=96,cells=cells,feature_checks=sum(c['feature_vs_recent']['numeric_rule_met'] for c in cells),position_checks=sum(c['position_vs_recent']['numeric_rule_met'] for c in cells),incremental_feature_positive=sum(c['feature_vs_current']['low']>0 for c in cells),checks_each=8,original_phase1_go=False,original_gates_changed=False,source_lock_sha256=sha(ROOT/'source_lock.json'),scalar_sha256=sha(out/'scalar_rows.csv'),vectors_sha256=sha(out/'vectors.npz'),position_preservation_max=float(abs((candidate-current)@A).max()),source_replay_max=float(replay),GPU=0,training=False,model_forward=False,fresh9137_opened=False,test_opened=False,elapsed_seconds=time.monotonic()-t,allocation=native))
if __name__=='__main__':main(int(sys.argv[1]))
