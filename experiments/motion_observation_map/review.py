"""Terminal GPU settlement, independent KKT, cached validation and scalar audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import csv,hashlib,json,socket,subprocess,time,traceback
import numpy as np
from kernel import predict
ROOT=Path('/scratch/xl4358/phase1_motion_observation_map')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def independent_kernel(x,y,f):
 a=(x-f['mean'])/f['scale'];b=(y-f['mean'])/f['scale'];return np.exp(-np.sum((a[:,None,:]-b[None,:,:])**2,axis=2)/float(f['bandwidth']))
def independent_predict(x,f):
 k=independent_kernel(x,f['train_X'],f);return (k-k.mean(1)[:,None]-f['row_mean'][None]+f['grand_mean'])@f['coef']+f['target_mean']
def interval(base,candidate,groups,floor=.02,absolute=.01):
 pairs=sorted(set(groups));x=np.array([[base[groups==p].mean(),candidate[groups==p].mean()] for p in pairs]);ab=x[:,0].mean()<floor
 def effect(z):
  a=z[...,0].mean(-1);b=z[...,1].mean(-1);return a-b if ab else (a-b)/np.maximum(a,floor)
 ix=np.random.default_rng(0).integers(0,len(x),(2000,len(x)));lo,hi=np.quantile(effect(x[ix]),[.025,.975]);val=float(effect(x));return {'mean':val,'low':float(lo),'high':float(hi),'threshold':absolute if ab else .1,'numeric_rule_met':bool(val>=(absolute if ab else .1) and lo>0)}
def main():
 job=int(os.environ['SLURM_JOB_ID']);print('JOB',job,'HOST',socket.gethostname(),'IP',socket.gethostbyname(socket.gethostname()),flush=True);print(Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text(),flush=True)
 ns=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);n=dict(x.split('=',1) for x in ns.split() if '=' in x);assert n['JobState']=='RUNNING' and n['Account']=='csci_ga_2271-2026fa' and 'gres/gpu' not in n['AllocTRES'] and n['NodeList']==socket.gethostname().split('.')[0]
 assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
 with (ROOT/'review_attempt.json').open('x') as f:json.dump({'job':job,'native':ns,'unknown_no_retry':True},f)
 gid=read(ROOT/'submission.json')['GPU_job'];terminal=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j',str(gid),'-n','-P','--format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,AllocTRES'],text=True);rows=[x.split('|') for x in terminal.splitlines() if x.split('|')[0]==str(gid)];assert len(rows)==1;seconds=int(rows[0][3]);write(ROOT/'GPU_budget_ledger.json',{'native':terminal,'new_job':gid,'new_GPU_seconds':seconds,'prior_settled_seconds':13104,'settled_seconds':13104+seconds,'remaining_seconds':28800-13104-seconds,'total_cap_seconds':28800});assert 13104+seconds<=28800 and rows[0][1:3]==['COMPLETED','0:0'] and seconds<=600
 prep=read(ROOT/'preparation.json');assert sha(ROOT/'train.npz')==prep['train_sha256'] and sha(ROOT/'validation.npz')==prep['validation_sha256'];d=load(ROOT/'train.npz');out=ROOT/'gpu_output';manifest=read(out/'manifest.json');assert all(sha(out/k)==v for k,v in manifest['files'].items());cross=read(out/'crossfit.json');supported=cross['status']=='GEOMETRY_INNOVATION_PREDICTABLE';maxkkt=0.;maxreplay=0.;count=0
 assert abs(d['N'].T@d['N']-np.eye(126)).max()<1e-10 and abs(d['N'].T@d['A']).max()<1e-10
 for hi,h in enumerate((3,6)):
  base=np.empty(1024);loss=np.empty(1024)
  for held,tr in [(a,d['fold']!=a) for a in (0,1)]+([('final',np.ones(1024,bool))] if supported else []):
   f=load(out/(f'head_{h}.npz' if held=='final' else f'cv_{h}_{held}.npz'));x=d['X'][tr,hi];y=d['Y'][tr,hi];assert np.array_equal(f['train_X'],x) and np.array_equal(f['N'],d['N']);K=independent_kernel(x,x,f);kc=K-K.mean(0)[None]-K.mean(1)[:,None]+K.mean();alpha=float(f['alpha']);rhs=y-y.mean(0);lhs=(kc+alpha*np.eye(len(x)))@f['coef'];err=float(abs(lhs-rhs).max()/max(1.,abs(rhs).max()));maxkkt=max(maxkkt,err);assert err<1e-8;ev=np.maximum(np.linalg.eigvalsh(kc),0);assert 1+np.sum(ev/(ev+alpha))<=32+1e-6 and (ev.max()+alpha)/alpha<=1e6+1e-2;count+=1
   if held!='final':
    te=~tr;z=independent_predict(d['X'][te,hi],f);base[te]=np.mean((d['Y'][te,hi]-f['target_mean'])**2,1);loss[te]=np.mean((d['Y'][te,hi]-z)**2,1)
  cell=cross['cells'][hi];maxreplay=max(maxreplay,abs(loss.mean()-cell['kernel_innovation_MSE']),abs(base.mean()-cell['constant_innovation_MSE']));assert maxreplay<1e-8
 write(ROOT/'fit_audit.json',{'status':'PASS_INDEPENDENT_OBSERVATION_MAP_KKT_AND_GROUPED_CROSSFIT','fits':count,'max_KKT_relative':maxkkt,'max_loss_replay':maxreplay,'scientific_CPU_fitting':False,'screen_supported':supported})
 if supported:
  v=load(ROOT/'validation.npz');old=v['candidate'];candidate=old.copy();independent=old.copy();positionmax=0.
  for h in (3,6):
   f=load(out/f'head_{h}.npz');ix=np.where(v['horizon']==h)[0];delta_coords=predict(v['geometry_X'][ix],f)+(v['current_feature'][ix]-old[ix])@v['N'];candidate[ix]+=delta_coords@v['N'].T
   want_coords=independent_predict(v['geometry_X'][ix],f)+np.einsum('ij,jk->ik',v['current_feature'][ix]-old[ix],v['N']);independent[ix]+=want_coords@v['N'].T;positionmax=max(positionmax,float(abs((candidate[ix]-old[ix])@v['A']).max()))
  vector_max=float(abs(candidate-independent).max());assert vector_max<2e-7 and positionmax<1e-10;raw=[];losses={}
  for name,z in [('recent',v['recent']),('current',old),('candidate',candidate)]:
   feature=((z-v['target'])**2).mean(1);position=((z@v['A']+v['b']-v['truth'])**2).mean(1);losses[name]=(feature,position)
   for i in range(1152):raw.append(dict(seed=int(v['seed'][i]),horizon=int(v['horizon'][i]),pair=str(v['pair'][i]),episode=str(v['episode'][i]),arm=name,feature_mse=float(feature[i]),position_mse=float(position[i])))
  folder=ROOT/'validation';folder.mkdir(exist_ok=False);cells=[]
  for seed in (17,29,43,'mean'):
   for h in (3,6):
    ix=np.where((v['horizon']==h)&((v['seed']==seed) if seed!='mean' else np.ones(1152,bool)))[0];groups=v['pair'][ix];cell={'seed':seed,'horizon':h}
    for key,arm,mi,floor,absolute in [('feature_vs_recent','recent',0,.02,.01),('position_vs_recent','recent',1,1e-5,1e-6),('feature_vs_current','current',0,.02,.01),('position_vs_current','current',1,1e-5,1e-6)]:cell[key]=interval(losses[arm][mi][ix],losses['candidate'][mi][ix],groups,floor,absolute)
    cells.append(cell)
  eligible=all(c['feature_vs_recent']['numeric_rule_met'] and c['position_vs_recent']['numeric_rule_met'] and c['feature_vs_current']['low']>0 for c in cells)
  with (folder/'scalar_rows.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
  np.savez(folder/'vectors.npz',candidate=candidate);s={'status':'VALIDATION_SUPPORTED_NOT_PHASE1_ACCEPTANCE' if eligible else 'VALIDATION_REJECTED_STOP','validation_eligible':eligible,'feature_checks':sum(c['feature_vs_recent']['numeric_rule_met'] for c in cells),'position_checks':sum(c['position_vs_recent']['numeric_rule_met'] for c in cells),'incremental_feature_positive':sum(c['feature_vs_current']['low']>0 for c in cells),'cells':cells,'vector_independent_max':vector_max,'position_preservation_max':positionmax,'scalar_sha256':sha(folder/'scalar_rows.csv'),'vectors_sha256':sha(folder/'vectors.npz'),'unique_heads':2,'same_maps_shared_by_3base_seeds':True,'paired_contrast_preservation_not_assumed':True,'test_opened':False,'fresh9137':False,'original_gates_changed':False,'phase1_go':False};write(folder/'summary.json',s);write(folder/'independent_audit.json',{'status':'PASS_INDEPENDENT_OBSERVATION_MAP_VECTOR_REPLAY','vector_max_difference':vector_max,'position_max':positionmax,'note':'Local32CI rawCSV audit additionally required before any test','validation_eligible':eligible});print(s['status'],flush=True)
 else:print(cross['status'],flush=True)
 assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items());write(ROOT/'outcome.json',{'status':'REVIEW_COMPLETE','job':job,'screen_supported':supported,'validation_eligible':eligible if supported else False,'new_fit_in_review':False,'base_models_frozen':True,'new_neural_training':False,'test_opened':False,'fresh9137':False,'source_lock_sha256':sha(ROOT/'source_lock.json')})
 for _ in range(120):
  if (ROOT/'transfer_done.json').exists():return
  time.sleep(5)
if __name__=='__main__':
 try:main()
 except Exception:
  write(ROOT/'review_failure.json',{'error':traceback.format_exc(),'no_automatic_retry':True});raise
