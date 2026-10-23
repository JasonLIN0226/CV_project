"""CPU numerical audit of CUDA-produced heads, never a CPU fitting fallback."""
import sys
import numpy as np
from common import *
def main(job):
 native(job,False);source_check();receipt=read(ROOT/'gpu_submission.json');gid=receipt['job'];nativefinal=subprocess.check_output(['/opt/slurm/bin/sacct','-j',str(gid),'-X','-n','-P','--format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,AllocTRES'],text=True);lines=[x.split('|') for x in nativefinal.strip().splitlines() if x.split('|')[0]==str(gid)];assert len(lines)==1 and lines[0][1]=='COMPLETED' and lines[0][2]=='0:0' and 0<int(lines[0][3])<=600
 write(ROOT/'GPU_budget_ledger.json',dict(prior_settled_seconds=12723,new_job=gid,new_GPU_seconds=int(lines[0][3]),settled_seconds=12723+int(lines[0][3]),total_cap_seconds=28800,remaining_seconds=28800-12723-int(lines[0][3]),native_sacct=nativefinal))
 out=ROOT/'gpu_output';m=read(out/'manifest.json');assert all(sha(out/k)==v for k,v in m['files'].items());assert m['state_proofs']==read(ROOT/'parent_lock.json')['state_digest_by_seed']
 with np.load(out/'training_residuals.npz',allow_pickle=False) as f:R=f['residuals'];X=f['X'];N=f['N']
 assert R.shape==(3,512,2,5120) and X.shape==(512,2,134) and N.shape==(5120,126);fold=np.array(read(out/'folds.json')['fold']);assert sum(fold==0)==sum(fold==1)==256
 with np.load(ROOT/'train_pack.npz',allow_pickle=False) as f:A=f['A'];B=f['B'];assert np.array_equal(X,f['X'][::2])
 Q=np.linalg.qr(B)[0];assert np.max(abs(N.T@N-np.eye(126)))<1e-10 and np.max(abs(N-Q@(Q.T@N)))<1e-10
 leakage=float(abs(N.T@A).max());assert leakage<1e-10;cross=read(out/'crossfit.json');maxkkt=0.;effects=[]
 def check(path,x,y):
  nonlocal maxkkt
  with np.load(path,allow_pickle=False) as f:d={k:f[k] for k in ('mean','scale','coef','N','alpha')}
  assert np.max(abs(d['mean']-x.mean(0)))<1e-10 and np.max(abs(d['scale']-np.maximum(x.std(0),1e-6)))<1e-10
  Z=np.column_stack([(x-d['mean'])/d['scale'],np.ones(len(x))]);G=Z.T@Z;P=np.eye(135);P[-1,-1]=0;alpha=.01*np.trace(G[:-1,:-1])/134;assert abs(float(d['alpha'])-alpha)<1e-9
  lhs=(G+alpha*P)@d['coef'];rhs=Z.T@(y@N);err=float(np.max(abs(lhs-rhs))/max(1.,np.max(abs(rhs))));maxkkt=max(maxkkt,err);assert err<1e-8 and np.array_equal(N,d['N'])
  return d
 supported=cross['status']=='TRAIN_PREDICTABILITY_SUPPORTED'
 for si,seed in enumerate((17,29,43)):
  for hi,h in enumerate((3,6)):
   y=R[si,:,hi];x=X[:,hi];before=float(np.mean(y*y));loss=np.empty(512)
   for held in (0,1):
    tr=np.where(fold!=held)[0];te=np.where(fold==held)[0];d=check(out/f'cv_{seed}_{h}_{held}.npz',x[tr],y[tr]);pred=apply_numpy(x[te],d);loss[te]=((y[te]-pred)**2).mean(1)
   cell=next(c for c in cross['cells'] if c['seed']==seed and c['horizon']==h);assert abs(loss.mean()-cell['after'])<1e-10 and abs(before-cell['before'])<1e-10;effects.append(dict(seed=seed,horizon=h,gain=float(before-loss.mean())))
   if supported:check(out/f'head_{seed}_{h}.npz',x,y)
 means={str(h):float(np.mean([c['gain'] for c in effects if c['horizon']==h])) for h in (3,6)};assert supported==all(v>0 for v in means.values());assert all(sha(k)==v for k,v in read(ROOT/'asset_ledger.json').items());source_check()
 write(ROOT/'fit_independent_audit.json',dict(status='PASS_CUDA_PARAMETERS_KKT_AND_CROSSFIT_AUDIT',heads=6 if supported else 0,crossfit_models=12,training_pairs=512,normal_equation_max_relative_error=maxkkt,position_null_max=leakage,mean_gain=means,scientific_status=cross['status'],scientific_CPU_fitting=False,base_models_frozen=True,calibrated_validation_evaluated=False,test_opened=False))
 print(cross['status'],flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
