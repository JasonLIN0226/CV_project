"""Read-only CPU replay witnesses for the CUDA training residuals; no refitting."""
from pathlib import Path
import hashlib,json,os,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from common import native,source_check,read,write,sha,AGE
from core import constrain

def main(job):
 native(job,False);source_check();HERE=Path(__file__).resolve().parent
 with (HERE/'execution_attempt.json').open('x') as f:json.dump(dict(CPU_job=job,pair_indices=[0,255,511],new_fit=False,no_retry=True),f)
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);sys.path.insert(0,str(AGE));import run as old
 with np.load(ROOT/'train_pack.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
 with np.load(ROOT/'gpu_output/training_residuals.npz',allow_pickle=False) as f:gpu=f['residuals']
 p,_,cfg=old.verify_control(old.OLD/'control');c=cfg['motion'];lock=read(ROOT/'parent_lock.json')
 with np.load(Path(c['output'])/'predictor/motion/basis.npz',allow_pickle=False) as f:basis={k:f[k] for k in f.files}
 def digest(m):
  h=hashlib.sha256()
  for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 keys=('z','age','type','past_actions','actions','current','projected_z','projected_current','mask');indices=np.array([0,1,510,511,1022,1023]);B,A,b,right=[d[k] for k in ('B','A','b','right')];maxvector=0.;maxscalar=0.;records=[]
 for si,seed in enumerate((17,29,43)):
  ref,_=old.reference_model(c,p,'motion',seed,basis);folder=old.OLD/'training/job_7574/predictor/motion'/f'seed_{seed}';ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);m=old.CenteredTransportPredictor(ref,basis);m.load_state_dict(ck['model'],strict=True);m.eval().requires_grad_(False);initial=digest(m);assert initial==lock['state_digest_by_seed'][str(seed)]
  batch={k:torch.from_numpy(d[k][indices]) for k in keys}
  with torch.inference_mode():base=m(batch).numpy();hist=m.reference(batch).numpy()
  pred=[]
  for j,i in enumerate(indices):
   pos=constrain(base[j],d['geometry'][i],B,A,b,right)[0] if d['enabled'][i] else base[j];pred.append(constrain(hist[j],pos@A+b,B,A,b,right)[0])
  cpu=(d['target'][indices].astype(float)-np.stack(pred)).reshape(3,2,2,5120).mean(1)
  for j,pair in enumerate((0,255,511)):
   z=gpu[si,pair];maxvector=max(maxvector,float(np.max(abs(cpu[j]-z))));diff=np.max(abs((cpu[j]**2).mean(1)-(z**2).mean(1)));maxscalar=max(maxscalar,float(diff));assert diff<2e-6;records.append(dict(seed=seed,pair_index=pair,scalar_max_difference=float(diff)))
  assert digest(m)==initial and all(x.grad is None for x in m.parameters());del m,ref,ck
 source_check();write(HERE/'summary.json',dict(status='PASS_FROZEN_CPU_CUDA_TRAINING_RESIDUAL_REPLAY',model_episode_forward_cases=18,distinct_training_episodes=6,horizon_vectors=36,common_residual_vectors=18,scalar_tolerance_unchanged=2e-6,scalar_max_difference=maxscalar,vector_max_difference_descriptive=maxvector,records=records,original_weights_unchanged=True,new_fit=False,GPU=0))
 print('PASS_FROZEN_CPU_CUDA_TRAINING_RESIDUAL_REPLAY',flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
