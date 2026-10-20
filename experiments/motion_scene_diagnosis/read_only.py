"""Audit existing calibration solutions. No fitting, forward pass or test access."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import csv,hashlib,json,socket,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path('/scratch/xl4358/phase1_motion_scene_calibration')
OUT=Path('/scratch/xl4358/phase1_motion_scene_diagnosis')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main(job):
 native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True)
 n=dict(v.split('=',1) for v in native.split() if '=' in v)
 assert n['JobState']=='RUNNING' and n['Account']=='csci_ga_2271-2026fa' and 'gres/gpu' not in n['AllocTRES'] and n['NodeList']==socket.gethostname().split('.')[0]
 OUT.mkdir(exist_ok=False)
 files=[ROOT/'gpu_output/training_residuals.npz',ROOT/'gpu_output/folds.json',ROOT/'gpu_output/manifest.json']
 manifest=json.loads(files[-1].read_text())['files']
 d=load(files[0]);fold=np.array(json.loads(files[1].read_text())['fold']);N=d['N'];rows=[];cells=[];fits=[]
 assert sha(files[0])==manifest['training_residuals.npz']
 for si,seed in enumerate((17,29,43)):
  for hi,h in enumerate((3,6)):
   X=d['X'][:,hi];Y=d['residuals'][si,:,hi];before=np.mean(Y**2,axis=1);after=np.zeros(512);trainloss=[]
   for held in (0,1):
    p=ROOT/f'gpu_output/cv_{seed}_{h}_{held}.npz';assert sha(p)==manifest[p.name];files.append(p);f=load(p)
    Z=np.column_stack([(X-f['mean'])/f['scale'],np.ones(512)]);delta=(Z@f['coef'])@N.T;loss=np.mean((Y-delta)**2,axis=1);tr=fold!=held;te=~tr;after[te]=loss[te];trainloss.append(float(loss[tr].mean()))
    gram=Z[tr].T@Z[tr];pen=np.eye(135);pen[-1,-1]=0;alpha=float(f['alpha']);eig=np.linalg.eigvalsh(gram)
    # Diagnostic inverse only; never solve coefficients or change alpha.
    df=float(np.trace(np.linalg.solve(gram+alpha*pen,gram)))
    fits.append(dict(seed=seed,horizon=h,held=held,train_before=float(before[tr].mean()),train_after=float(loss[tr].mean()),held_before=float(before[te].mean()),held_after=float(loss[te].mean()),effective_df=df,train_pairs=int(tr.sum()),normalized_input_rank=int(np.linalg.matrix_rank(Z[tr])),condition_regularized=float(np.linalg.cond(gram+alpha*pen)),alpha=alpha,correction_energy_held=float(np.mean(delta[te]**2)),alignment_held=float(np.mean(Y[te]*delta[te]))))
   gain=before-after
   cells.append(dict(seed=seed,horizon=h,improved=int((gain>0).sum()),total=512,before=float(before.mean()),after=float(after.mean()),train_after_mean=float(np.mean(trainloss)),gain_quantiles=dict(zip(('p0','p10','p50','p90','p100'),map(float,np.quantile(gain,[0,.1,.5,.9,1])))),null_reachable_energy_fraction=float(np.sum((Y@N)**2)/np.sum(Y**2))))
   for i in range(512):rows.append(dict(seed=seed,horizon=h,pair_index=i,held_fold=int(fold[i]),before=float(before[i]),after=float(after[i]),gain=float(gain[i])))
 ledger={str(p):sha(p) for p in files}
 with (OUT/'pair_losses.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 result=dict(status='READ_ONLY_EXISTING_COEFFICIENT_DIAGNOSIS',native=native,source_sha256=sha(__file__),inputs=ledger,cells=cells,folds=fits,total_improved=sum(c['improved'] for c in cells),total_pair_seed_horizon=3072,no_fit=True,no_model_forward=True,no_validation_or_test=True,GPU=0,original_gates_changed=False)
 result['pair_losses_sha256']=sha(OUT/'pair_losses.csv');(OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');assert all(sha(p)==v for p,v in ledger.items());print(json.dumps(result))
if __name__=='__main__':main(int(sys.argv[1]))
