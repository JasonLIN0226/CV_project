"""Shared two-horizon observation maps; no encoder/base-model execution."""
from pathlib import Path
import datetime,hashlib,json,os,socket,subprocess,sys,time,traceback
import numpy as np
import torch
from kernel import fit,predict
ROOT=Path('/scratch/xl4358/phase1_motion_observation_map')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 job=int(os.environ['SLURM_JOB_ID']);s=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);n=dict(t.split('=',1) for t in s.split() if '=' in t);assert n['JobState']=='RUNNING' and n['Account']=='csci_ga_2271-2026fa' and n['Partition']=='c12m85-a100-1' and n['NodeList']==socket.gethostname().split('.')[0] and n['TimeLimit']=='00:10:00'
 assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
 with (ROOT/'gpu_attempt.json').open('x') as f:json.dump({'job':job,'native':s,'unknown_do_not_retry':True},f)
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True);assert torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0)
 start=datetime.datetime.fromisoformat(n['StartTime']).replace(tzinfo=datetime.timezone.utc).timestamp()
 def guard():
  wall=time.time()-start
  if wall>540 or 13104+wall>28800:raise RuntimeError('COMPUTE_BUDGET_STOP')
  return wall
 write(ROOT/'actual_GPU.json',{'job':job,'native':s,'GPU':torch.cuda.get_device_name(0),'threads':4,'no_base_or_encoder_forward':True});p=read(ROOT/'preparation.json');assert sha(ROOT/'train.npz')==p['train_sha256'] and sha(ROOT/'validation.npz')==p['validation_sha256']
 with np.load(ROOT/'train.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
 parent=Path('/scratch/xl4358/phase1_motion_scene_calibration/gpu_output');assert sha(parent/'training_residuals.npz')==read(parent/'manifest.json')['files']['training_residuals.npz']
 with np.load(parent/'training_residuals.npz',allow_pickle=False) as f:assert np.array_equal(d['N'],f['N'])
 out=ROOT/'gpu_output';out.mkdir(exist_ok=False);cells=[]
 for hi,h in enumerate((3,6)):
  before=np.empty(1024);after=np.empty(1024)
  for held in (0,1):
   tr=d['fold']!=held;te=~tr;f=fit(torch.tensor(d['X'][tr,hi],device='cuda'),torch.tensor(d['Y'][tr,hi],device='cuda'));f['N']=d['N'];np.savez(out/f'cv_{h}_{held}.npz',**f);y=d['Y'][te,hi];pred=predict(d['X'][te,hi],f);before[te]=np.mean((y-f['target_mean'])**2,1);after[te]=np.mean((y-pred)**2,1);guard()
  gain=(before-after).reshape(512,2).mean(1);ix=np.random.default_rng(0).integers(0,512,(2000,512));lo,high=np.quantile(gain[ix].mean(1),[.025,.975]);cells.append({'horizon':h,'constant_innovation_MSE':float(before.mean()),'kernel_innovation_MSE':float(after.mean()),'relative_reduction':float(1-after.mean()/before.mean()),'pair_gain_lower95':float(lo),'pair_gain_upper95':float(high)})
 supported=all(c['pair_gain_lower95']>0 for c in cells);write(out/'crossfit.json',{'status':'GEOMETRY_INNOVATION_PREDICTABLE' if supported else 'GEOMETRY_INNOVATION_SCREEN_REJECTED','cells':cells,'pairs':512,'folds':'Originalfixedpairgrouped0/1','target':'126dimprojectedfuture-minus-currentfeature','comparison':'Matchedfoldtrain-meaninnovation, not insamplebasepredictor','new_original_gate':False})
 if supported:
  for hi,h in enumerate((3,6)):
   f=fit(torch.tensor(d['X'][:,hi],device='cuda'),torch.tensor(d['Y'][:,hi],device='cuda'));f['N']=d['N'];np.savez(out/f'head_{h}.npz',**f);guard()
 write(out/'manifest.json',{'files':{p.name:sha(p) for p in out.glob('*.npz')},'source_lock_sha256':sha(ROOT/'source_lock.json')});write(ROOT/'gpu_outcome.json',{'status':'GEOMETRY_INNOVATION_PREDICTABLE' if supported else 'GEOMETRY_INNOVATION_SCREEN_REJECTED','job':job,'unique_heads':2 if supported else 0,'shared_across_original_base_seeds':True,'no_new_neural_training':True,'no_model_forward':True,'elapsed_observed':guard(),'source_lock_sha256':sha(ROOT/'source_lock.json'),'test_opened':False,'fresh9137':False})
if __name__=='__main__':
 try:main()
 except Exception:
  write(ROOT/'gpu_failure.json',{'error':traceback.format_exc(),'automatic_retry':False});raise
