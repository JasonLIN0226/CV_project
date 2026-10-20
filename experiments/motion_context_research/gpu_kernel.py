"""One fixed cached-kernel experiment; no encoder or base-model forward."""
import datetime,hashlib,json,socket,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from kernel import fit,apply
ROOT=Path('/scratch/xl4358/phase1_motion_context_research/kernel_round1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main(job):
 native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);n=dict(x.split('=',1) for x in native.split() if '=' in x);assert n['JobState']=='RUNNING' and n['Account']=='csci_ga_2271-2026fa' and n['Partition']=='c12m85-a100-1' and n['NodeList']==socket.gethostname().split('.')[0] and n['TimeLimit']=='00:05:00'
 assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
 with (ROOT/'gpu_attempt.json').open('x') as f:json.dump({'job':job,'native':native,'no_retry':True},f)
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True);assert torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0)
 write(ROOT/'actual_GPU.json',dict(job=job,native=native,GPU=torch.cuda.get_device_name(0),threads=torch.get_num_threads(),base_model_forward=False,encoder_forward=False))
 start=datetime.datetime.fromisoformat(n['StartTime']).replace(tzinfo=datetime.timezone.utc).timestamp()
 def guard():
  wall=time.time()-start
  if wall>240 or 12765+wall>28800:raise RuntimeError('COMPUTE_BUDGET_STOP')
  return wall
 prep=read(ROOT/'preparation.json');assert sha(ROOT/'train.npz')==prep['train_sha256'];assert sha(ROOT/'validation.npz')==prep['validation_sha256']
 with np.load(ROOT/'train.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
 out=ROOT/'gpu_output';out.mkdir(exist_ok=False);cells=[];pairgains=[]
 for si,seed in enumerate((17,29,43)):
  for hi,h in enumerate((3,6)):
   X=d['X'][:,:,hi];Y=d['R'][si,:,hi];loss=np.empty(512)
   for held in (0,1):
    tr=d['fold']!=held;te=~tr;f=fit(torch.as_tensor(X[tr],device='cuda'),torch.as_tensor(Y[tr],device='cuda'),d['N']);np.savez(out/f'cv_{seed}_{h}_{held}.npz',**f);delta=apply(X[te].reshape(-1,130),f).reshape(256,2,5120).mean(1);loss[te]=((Y[te]-delta)**2).mean(1);guard()
   before=(Y**2).mean(1);pairgains.append(before-loss);cells.append(dict(seed=seed,horizon=h,before=float(before.mean()),after=float(loss.mean()),gain=float((before-loss).mean()),relative_reduction=float(1-loss.mean()/before.mean())))
 gains=np.stack(pairgains).reshape(3,2,512).mean(0);ix=np.random.default_rng(0).integers(0,512,(2000,512));ci=[list(map(float,np.quantile(g[ix].mean(1),[.025,.975]))) for g in gains];supported=all(c['gain']>0 for c in cells) and all(c[0]>0 for c in ci)
 write(out/'crossfit.json',dict(status='TRAIN_SCREEN_SUPPORTED' if supported else 'TRAIN_SCREEN_REJECTED',cells=cells,seed_mean_horizon_CI=ci,new_acceptance_gate=False,criterion='Resource continuation only: all6positive means and both seed-mean paired lower95CIpositive; original gates unchanged'))
 if supported:
  for si,seed in enumerate((17,29,43)):
   for hi,h in enumerate((3,6)):
    f=fit(torch.as_tensor(d['X'][:,:,hi],device='cuda'),torch.as_tensor(d['R'][si,:,hi],device='cuda'),d['N']);np.savez(out/f'head_{seed}_{h}.npz',**f);guard()
 assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items());write(out/'manifest.json',dict(files={p.name:sha(p) for p in out.glob('*.npz')},source_lock_sha256=sha(ROOT/'source_lock.json')));write(ROOT/'gpu_outcome.json',dict(status='TRAIN_SCREEN_SUPPORTED' if supported else 'TRAIN_SCREEN_REJECTED',job=job,cells=cells,heads=6 if supported else 0,GPU_wall_observed=guard(),base_frozen=True,no_model_forward=True,no_test=True,fresh9137=False,source_lock_sha256=sha(ROOT/'source_lock.json')))
if __name__=='__main__':
 try:main(int(sys.argv[1]))
 except Exception:
  write(ROOT/'gpu_failure.json',{'error':traceback.format_exc(),'automatic_retry':False});raise
