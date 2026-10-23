"""One <=600s GPU allocation: frozen inference and six train-only ridge heads."""
import datetime,os,socket,sys,time,traceback
import numpy as np
import torch
from common import *
from core import constrain
from calibrator import null_basis,fit_train_only,correction

def main(job):
 begin=time.monotonic();ns,n=native(job,True);source_check();assert n['TimeLimit']=='00:10:00'
 assert read(ROOT/'authorization.json')['new_head_fitting_approved'] and read(ROOT/'protocol.json')['prior_GPU_seconds']==12723
 with (ROOT/'gpu_execution_attempt.json').open('x') as f:json.dump(dict(job=job,allocation=ns,original_source_lock=sha(ROOT/'source_lock.json'),no_retry=True),f)
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 assert torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0)
 write(ROOT/'actual_GPU.json',dict(job=job,name=torch.cuda.get_device_name(0),threads=torch.get_num_threads(),allocation=ns))
 prep=read(ROOT/'preparation.json');assert prep['status']=='READ_ONLY_PREFLIGHT_PASS' and sha(ROOT/'train_pack.npz')==prep['train_pack_sha256'];assert sha(ROOT/'validation_pack.npz')==prep['validation_pack_sha256']
 with np.load(ROOT/'train_pack.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
 sys.path.insert(0,str(AGE));import run as old
 p,_,cfg=old.verify_control(old.OLD/'control');c=cfg['motion'];lock=read(ROOT/'parent_lock.json')
 with np.load(Path(c['output'])/'predictor/motion/basis.npz',allow_pickle=False) as f:basis={k:f[k] for k in f.files}
 B,A,b,right=[data[k] for k in ('B','A','b','right')];N=null_basis(B,A);assert N.shape==(5120,126)
 out=ROOT/'gpu_output';out.mkdir(exist_ok=False);foldorder=np.random.default_rng(0).permutation(512);fold=np.empty(512,int);fold[foldorder[:256]]=0;fold[foldorder[256:]]=1
 write(out/'folds.json',dict(pair_ids=data['pair_ids'].tolist(),fold=fold.tolist(),seed=0))
 def digest(m):
  h=hashlib.sha256()
  for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 def guard():
  wall=time.time()-datetime.datetime.fromisoformat(n['StartTime']).replace(tzinfo=datetime.timezone.utc).timestamp()
  if wall>540:raise RuntimeError('COMPUTE_BUDGET_STOP: preserve partial, no retry')
  return wall
 residuals=[];state={};maxbatch=0
 keys=('z','age','type','past_actions','actions','current','projected_z','projected_current','mask')
 for si,seed in enumerate((17,29,43)):
  ref,_=old.reference_model(c,p,'motion',seed,basis);folder=old.OLD/'training/job_7574/predictor/motion'/f'seed_{seed}';ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);m=old.CenteredTransportPredictor(ref,basis);m.load_state_dict(ck['model'],strict=True);m.eval().requires_grad_(False);initial=digest(m);assert initial==lock['state_digest_by_seed'][str(seed)];m.to('cuda');preds=[]
  for j in range(0,1024,32):
   step=time.monotonic();batch={k:torch.as_tensor(data[k][j:j+32],device='cuda') for k in keys}
   with torch.inference_mode():base=m(batch).cpu().numpy();history=m.reference(batch).cpu().numpy()
   oldpred=[]
   for k in range(len(base)):
    current=constrain(base[k],data['geometry'][j+k],B,A,b,right)[0] if data['enabled'][j+k] else base[k]
    oldpred.append(constrain(history[k],current@A+b,B,A,b,right)[0])
   preds.extend(oldpred);torch.cuda.synchronize();maxbatch=max(maxbatch,time.monotonic()-step) if j>0 else maxbatch;wall=guard();remaining=(2-si)*32+(1024-j-32)//32
   if j==32 and wall+remaining*maxbatch*1.25+120>540:raise RuntimeError('COMPUTE_BUDGET_STOP: complete stage projection exceeds cap')
  pred=np.stack(preds);assert pred.shape==(1024,2,5120) and np.isfinite(pred).all();R=(data['target'].astype(float)-pred).reshape(512,2,2,5120).mean(1);residuals.append(R)
  assert digest(m)==initial and all(v.grad is None for v in m.parameters());state[str(seed)]=initial;del m,ref,ck;torch.cuda.empty_cache();write(ROOT/'progress.json',dict(stage='frozen_training_inference',completed_seeds=si+1,total_seeds=3,elapsed_GPU_wall=guard()))
 residuals=np.stack(residuals);np.savez(out/'training_residuals.npz',residuals=residuals,X=data['X'][::2],N=N)
 results=[];osplits=['train']*256
 for si,seed in enumerate((17,29,43)):
  for hi,h in enumerate((3,6)):
   X=torch.as_tensor(data['X'][::2,hi],device='cuda',dtype=torch.float64);Y=torch.as_tensor(residuals[si,:,hi],device='cuda',dtype=torch.float64);cv=np.zeros(512);before=(residuals[si,:,hi]**2).mean(1)
   for held in (0,1):
    guard()
    tr=np.where(fold!=held)[0];te=np.where(fold==held)[0];f=fit_train_only(X[tr],Y[tr],N,osplits,authorized=True,stage='fixed_train_half');delta=correction(X[te],f);cv[te]=((Y[te]-delta)**2).mean(1).cpu().numpy();np.savez(out/f'cv_{seed}_{h}_{held}.npz',**{k:v.detach().cpu().numpy() if torch.is_tensor(v) else v for k,v in f.items()})
   results.append(dict(seed=seed,horizon=h,before=float(before.mean()),after=float(cv.mean()),relative_reduction=float((before.mean()-cv.mean())/before.mean())))
 means={str(h):float(np.mean([c['before']-c['after'] for c in results if c['horizon']==h])) for h in (3,6)};supported=all(v>0 for v in means.values());write(out/'crossfit.json',dict(status='TRAIN_PREDICTABILITY_SUPPORTED' if supported else 'TRAIN_PREDICTABILITY_REJECTED',cells=results,mean_absolute_gain=means,folds='fixed0/1all512pairs',new_gate=False))
 if supported:
  for si,seed in enumerate((17,29,43)):
   for hi,h in enumerate((3,6)):
    f=fit_train_only(torch.as_tensor(data['X'][::2,hi],device='cuda'),torch.as_tensor(residuals[si,:,hi],device='cuda'),N,['train']*512,authorized=True);np.savez(out/f'head_{seed}_{h}.npz',**{k:v.detach().cpu().numpy() if torch.is_tensor(v) else v for k,v in f.items()});guard()
 assert all(sha(k)==v for k,v in read(ROOT/'asset_ledger.json').items());source_check();write(out/'manifest.json',dict(files={p.name:sha(p) for p in out.glob('*.npz')},state_proofs=state,source_lock_sha256=sha(ROOT/'source_lock.json')))
 write(ROOT/'gpu_outcome.json',dict(status='FIT_COMPLETE_PENDING_INDEPENDENT_AUDIT' if supported else 'TRAIN_PREDICTABILITY_REJECTED',job=job,full_training_pairs=512,heads=6 if supported else 0,CV=results,GPU_wall_observed=guard(),base_frozen=True,original_gates_changed=False,calibrated_validation_evaluated=False,validation_used_for_fitting=False,test_opened=False,fresh9137=False));print('GPU_STAGE_COMPLETE',flush=True)
if __name__=='__main__':
 try:main(int(sys.argv[1]))
 except Exception:
  write(ROOT/'gpu_failure.json',dict(error=traceback.format_exc(),no_automatic_retry=True));raise
