"""Independent CPU KKT and crossfit audit of fixed CUDA solutions; no fitting."""
import csv,hashlib,json,sys,subprocess
from pathlib import Path
import numpy as np
from kernel import apply
ROOT=Path('/scratch/xl4358/phase1_motion_context_research/kernel_round1_execfix')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def kernel(a,b,f):
 a=(a-f['mean'])/f['scale'];b=(b-f['mean'])/f['scale']
 def dist(x,y):return np.maximum(np.einsum('ij,ij->i',x,x)[:,None]+np.einsum('ij,ij->i',y,y)[None]-2*np.einsum('ik,jk->ij',x,y),0)
 return .5*np.exp(-dist(a[:,:2],b[:,:2])/f['bandwidth'][0])+.5*np.exp(-dist(a[:,2:],b[:,2:])/f['bandwidth'][1])
def main():
 gid=read(ROOT/'submission.json')['job'];native=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j',str(gid),'-n','-P','--format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,AllocTRES'],text=True);rows=[r.split('|') for r in native.splitlines() if r.split('|')[0]==str(gid)];assert len(rows)==1 and rows[0][1:3]==['COMPLETED','0:0'];seconds=int(rows[0][3]);assert 0<seconds<=600
 write(ROOT/'GPU_budget_ledger.json',dict(native=native,new_job=gid,new_GPU_seconds=seconds,prior_settled_seconds=13077,settled_seconds=13077+seconds,remaining_seconds=28800-13077-seconds,total_cap_seconds=28800))
 assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items());d=load(ROOT/'train.npz');out=ROOT/'gpu_output';manifest=read(out/'manifest.json');assert all(sha(out/k)==v for k,v in manifest['files'].items());cross=read(out/'crossfit.json');supported=cross['status']=='TRAIN_SCREEN_SUPPORTED';maxkkt=0.;df=[];maxreplay=0.;fits=0
 for si,seed in enumerate((17,29,43)):
  for hi,h in enumerate((3,6)):
   X=d['X'][:,:,hi];Y=d['R'][si,:,hi];loss=np.empty(512)
   cases=[(held,d['fold']!=held) for held in (0,1)]+([('final',np.ones(512,bool))] if supported else [])
   for held,tr in cases:
    path=out/(f'head_{seed}_{h}.npz' if held=='final' else f'cv_{seed}_{h}_{held}.npz');f=load(path);x=X[tr];y=Y[tr];n=len(x);assert np.array_equal(f['train_X'],x);assert np.max(abs(f['mean']-x.reshape(-1,130).mean(0)))<1e-10 and np.max(abs(f['scale']-np.maximum(x.reshape(-1,130).std(0),1e-6)))<1e-10
    K=kernel(x.reshape(-1,130),x.reshape(-1,130),f).reshape(n,2,n,2).mean((1,3));kc=K-K.mean(0)[None]-K.mean(1)[:,None]+K.mean();Yp=y@f['N'];rhs=Yp-Yp.mean(0);lhs=(kc+float(f['alpha'])*np.eye(n))@f['coef'];err=float(abs(lhs-rhs).max()/max(1.,abs(rhs).max()));assert err<1e-8;maxkkt=max(maxkkt,err);eig=np.maximum(np.linalg.eigvalsh(kc),0);edf=float(1+np.sum(eig/(eig+float(f['alpha']))));assert abs(edf-32)<1e-7;df.append(edf);fits+=1
    if held!='final':
     te=~tr;k=kernel(X[te].reshape(-1,130),x.reshape(-1,130),f).reshape(te.sum()*2,n,2).mean(2);k=k-k.mean(1)[:,None]-K.mean(0)[None]+K.mean();delta=(k@f['coef']+f['target_mean'])@f['N'].T;actual=apply(X[te].reshape(-1,130),f);assert abs(actual-delta).max()<1e-9;delta=delta.reshape(te.sum(),2,5120);assert abs(delta[:,0]-delta[:,1]).max()<1e-10;loss[te]=((Y[te]-delta.mean(1))**2).mean(1)
   c=next(c for c in cross['cells'] if c['seed']==seed and c['horizon']==h);diff=abs(c['after']-float(loss.mean()));assert diff<1e-10;maxreplay=max(maxreplay,diff)
 write(ROOT/'fit_audit.json',dict(status='PASS_INDEPENDENT_KERNEL_KKT_CROSSFIT_DF32',fits=fits,max_normal_equation_relative_residual=maxkkt,max_crossfit_replay=maxreplay,effective_df_min=min(df),effective_df_max=max(df),scientific_CPU_fitting=False,source_lock_sha256=sha(ROOT/'source_lock.json'),crossfit_status=cross['status']))
 print(json.dumps(read(ROOT/'fit_audit.json')))
if __name__=='__main__':main()
