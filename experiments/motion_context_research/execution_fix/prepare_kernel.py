"""Existing cache packing only; no scientific fit or model forward."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from context import context
BASE=Path('/scratch/xl4358/phase1_motion_scene_calibration')
ROOT=Path('/scratch/xl4358/phase1_motion_context_research/kernel_round1_execfix')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items())
 with (ROOT/'preparation_attempt.json').open('x') as f:json.dump({'no_retry':True},f)
 assert sha(BASE/'train_pack.npz')==read(BASE/'preparation.json')['train_pack_sha256']
 with np.load(BASE/'train_pack.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
 X=np.stack([context(d['z'],d['past_actions'],d['actions'],d['projected_current'],d['A'],d['b'],h) for h in (3,6)],1);assert X.shape==(1024,2,130) and np.array_equal(X[::2],X[1::2]);assert d['enabled'].all()
 rpath=BASE/'gpu_output/training_residuals.npz';assert sha(rpath)==read(BASE/'gpu_output/manifest.json')['files']['training_residuals.npz']
 with np.load(rpath,allow_pickle=False) as f:R=f['residuals'];N=f['N']
 fold=np.array(read(BASE/'gpu_output/folds.json')['fold']);np.savez(ROOT/'train.npz',X=X.reshape(512,2,2,130),R=R,N=N,fold=fold)
 assert sha(BASE/'validation_pack.npz')==read(BASE/'preparation.json')['validation_pack_sha256']
 with np.load(BASE/'validation_pack.npz',allow_pickle=False) as f:v={k:f[k] for k in f.files}
 path=Path('/scratch/xl4358/phase1_motion_geometry_validation/output/job_8336/validation_geometry.csv');assert sha(path)==read(path.parent/'summary.json')['raw_sha256']
 with path.open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==1152
 sys.path.insert(0,'/scratch/xl4358/phase1_age_control_a1');import run as old
 _,_,cfg=old.verify_control(old.OLD/'control');store=old.FeatureStore(cfg['motion'],max_cached=1);actions={};VX=[]
 for i,row in enumerate(rows):
  assert row['episode']==v['episode'][i] and int(row['seed'])==v['seed'][i] and int(row['horizon'])==v['horizon'][i];eid=row['episode'];rec=store.records[eid];assert rec['split']=='validation'
  if eid not in actions:actions[eid]=old.load_episode(cfg['motion'],rec,keys=('actions',))['actions']
  q=int(rec['query']);end=int(row['end']);h=int(row['horizon']);anchor=np.array([float(row['observed_end_x']),float(row['observed_end_y'])]);prefix=np.concatenate([np.zeros((1,2),np.float32),np.cumsum(actions[eid][:q],axis=0)]);u=(prefix[q]-prefix[end])+actions[eid][q:q+h].sum(0);anchor[0]+=.0015*u[0]+.001*u[1];VX.append(np.r_[anchor,v['X'][i,:128]])
 v['context_X']=np.stack(VX);assert v['context_X'].shape==(1152,130)
 for seed in (17,29,43):
  for h in (3,6):
   for pair in sorted(set(v['pair'])):
    ix=np.where((v['seed']==seed)&(v['horizon']==h)&(v['pair']==pair))[0];assert len(ix)==2 and np.array_equal(v['context_X'][ix[0]],v['context_X'][ix[1]])
 np.savez(ROOT/'validation.npz',**v)
 write(ROOT/'preparation.json',dict(status='CACHED_CAUSAL_INPUT_PREFLIGHT_PASS',train_pairs=512,validation_pairs=96,exact_sibling_context=True,no_fit=True,no_model_forward=True,source_lock_sha256=sha(ROOT/'source_lock.json'),train_sha256=sha(ROOT/'train.npz'),validation_sha256=sha(ROOT/'validation.npz'),parent_residuals_sha256=sha(rpath)))
if __name__=='__main__':main()
