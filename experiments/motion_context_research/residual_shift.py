"""Descriptive train-versus-existing-validation residual audit, no fitting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib
import numpy as np
P=Path('/scratch/xl4358/phase1_motion_scene_calibration');O=Path('/scratch/xl4358/phase1_motion_context_research')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
prep=json.loads((P/'preparation.json').read_text());assert sha(P/'train_pack.npz')==prep['train_pack_sha256'] and sha(P/'validation_pack.npz')==prep['validation_pack_sha256'];d=load(P/'train_pack.npz');r=load(P/'gpu_output/training_residuals.npz');v=load(P/'validation_pack.npz');cells=[]
for si,seed in enumerate((17,29,43)):
 for hi,h in enumerate((3,6)):
  ix=np.where((v['seed']==seed)&(v['horizon']==h))[0];pairs=sorted(set(v['pair'][ix]));er=[];targets=[]
  for pair in pairs:
   ids=ix[v['pair'][ix]==pair];assert len(ids)==2;er.append((v['target'][ids]-v['candidate'][ids]).mean(0));targets.append(v['target'][ids].mean(0))
  a=r['residuals'][si,:,hi];b=np.array(er);ta=d['target'][:,hi].reshape(512,2,5120).mean(1);tb=np.array(targets)
  def variance(x):return float(np.mean((x-x.mean(0))**2))
  cells.append(dict(seed=seed,horizon=h,training_pairs=512,validation_pairs=96,training_common_MSE=float(np.mean(a*a)),validation_common_MSE=float(np.mean(b*b)),residual_MSE_ratio=float(np.mean(b*b)/np.mean(a*a)),target_centred_variance_train=variance(ta),target_centred_variance_validation=variance(tb),target_variance_ratio=variance(tb)/variance(ta),common_residual_mean_cosine=float(a.mean(0)@b.mean(0)/np.linalg.norm(a.mean(0))/np.linalg.norm(b.mean(0)))))
result=dict(status='DESCRIPTIVE_RESIDUAL_DISTRIBUTION_SHIFT',cells=cells,interpretation='Base predictor saw all512trainingpairs; head-onlycrossfit does not hold out base-modeltraining. Residualshift is descriptive and not a causal proof ofoverfit or a new acceptance gate.',source_sha256=sha(__file__),new_fit=False,model_forward=False,GPU=0,test_access=False)
(O/'residual_shift.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
