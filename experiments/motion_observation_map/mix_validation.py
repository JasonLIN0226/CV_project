"""One prospectively fixed .5/.5 mixture, cached validation, no fit or search."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import csv,hashlib,json,socket,subprocess,sys
import numpy as np
BASE=Path('/scratch/xl4358/phase1_motion_observation_map')
ROOT=Path('/scratch/xl4358/phase1_motion_observation_mix')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def interval(base,cand,groups,floor=.02,absolute=.01):
 pairs=sorted(set(groups));x=np.array([[np.mean(base[groups==p]),np.mean(cand[groups==p])] for p in pairs]);ab=x[:,0].mean()<floor
 def effect(z):
  a=z[...,0].mean(-1);b=z[...,1].mean(-1);return a-b if ab else (a-b)/np.maximum(a,floor)
 ix=np.random.default_rng(0).integers(0,len(x),(2000,len(x)));lo,hi=np.quantile(effect(x[ix]),[.025,.975]);val=float(effect(x));return dict(mean=val,low=float(lo),high=float(hi),threshold=absolute if ab else .1,numeric_rule_met=bool(val>=(absolute if ab else .1) and lo>0))
def main(job):
 native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);n=dict(x.split('=',1) for x in native.split() if '=' in x);assert n['JobState']=='RUNNING' and n['Account']=='csci_ga_2271-2026fa' and 'gres/gpu' not in n['AllocTRES'] and n['NodeList']==socket.gethostname().split('.')[0]
 assert read(ROOT/'protocol.json')['fixed_coefficient']==.5
 with (ROOT/'execution_attempt.json').open('x') as f:json.dump({'job':job,'native':native,'unknown_no_retry':True},f)
 source=read(BASE/'validation/summary.json');assert sha(BASE/'validation/vectors.npz')==source['vectors_sha256'] and read(BASE/'fit_audit.json')['status']=='PASS_INDEPENDENT_OBSERVATION_MAP_KKT_AND_GROUPED_CROSSFIT';assert sha(BASE/'validation.npz')==read(BASE/'preparation.json')['validation_sha256']
 with np.load(BASE/'validation.npz',allow_pickle=False) as f:v={k:f[k] for k in f.files}
 with np.load(BASE/'validation/vectors.npz',allow_pickle=False) as f:mapped=f['candidate']
 old=v['candidate'];candidate=old+.5*(mapped-old);independent=(old+mapped)/2;diff=float(abs(candidate-independent).max());assert diff<1e-12;pos=float(abs((candidate-old)@v['A']).max());assert pos<1e-10
 losses={};raw=[]
 for name,z in [('recent',v['recent']),('current',old),('candidate',candidate)]:
  f=((z-v['target'])**2).mean(1);p=((z@v['A']+v['b']-v['truth'])**2).mean(1);losses[name]=(f,p)
  for i in range(1152):raw.append(dict(seed=int(v['seed'][i]),horizon=int(v['horizon'][i]),pair=str(v['pair'][i]),episode=str(v['episode'][i]),arm=name,feature_mse=float(f[i]),position_mse=float(p[i])))
 D=((mapped-old)**2).mean(1);Lold=losses['current'][0];Lnew=((mapped-v['target'])**2).mean(1);assert abs(losses['candidate'][0]-((Lold+Lnew)/2-D/4)).max()<1e-12
 cells=[]
 for seed in (17,29,43,'mean'):
  for h in (3,6):
   ids=np.where((v['horizon']==h)&((v['seed']==seed) if seed!='mean' else np.ones(1152,bool)))[0];g=v['pair'][ids];cell={'seed':seed,'horizon':h}
   for name,arm,mi,floor,absolute in [('feature_vs_recent','recent',0,.02,.01),('position_vs_recent','recent',1,1e-5,1e-6),('feature_vs_current','current',0,.02,.01),('position_vs_current','current',1,1e-5,1e-6)]:cell[name]=interval(losses[arm][mi][ids],losses['candidate'][mi][ids],g,floor,absolute)
   cells.append(cell)
 eligible=all(c['feature_vs_recent']['numeric_rule_met'] and c['position_vs_recent']['numeric_rule_met'] and c['feature_vs_current']['low']>0 for c in cells);out=ROOT/'validation';out.mkdir(exist_ok=False)
 with (out/'scalar_rows.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
 np.savez(out/'vectors.npz',candidate=candidate)
 s=dict(status='VALIDATION_SUPPORTED_NOT_PHASE1_ACCEPTANCE' if eligible else 'VALIDATION_REJECTED_STOP',validation_eligible=eligible,cells=cells,feature_checks=sum(c['feature_vs_recent']['numeric_rule_met'] for c in cells),position_checks=sum(c['position_vs_recent']['numeric_rule_met'] for c in cells),incremental_feature_positive=sum(c['feature_vs_current']['low']>0 for c in cells),fixed_coefficient=.5,coefficient_fitted=False,weight_grid=False,new_GPU=0,position_preservation_max=pos,vector_replay_max=diff,scalar_sha256=sha(out/'scalar_rows.csv'),vector_sha256=sha(out/'vectors.npz'),source_sha256=sha(__file__),parent_map_validation_sha256=sha(BASE/'validation/summary.json'),original_gates_changed=False,fresh9137=False,phase1_go=False,test_opened=False)
 write(out/'summary.json',s);print(json.dumps({k:v for k,v in s.items() if k!='cells'}))
if __name__=='__main__':main(int(sys.argv[1]))
