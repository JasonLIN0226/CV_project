"""Read-only CPU input/causality packing before GPU allocation; no model forward or fitting."""
import csv,sys,time
import numpy as np
from common import *
from core import maps
from guard import allow_update
from geometry import observed_forecast

def main(job):
 t=time.monotonic();ns,_=native(job,False);source_check();assert read(ROOT/'authorization.json')['new_head_fitting_approved']
 with (ROOT/'preparation_attempt.json').open('x') as f:json.dump(dict(job=job,allocation=ns,no_retry=True),f)
 sys.path.insert(0,str(AGE));import run as old
 p,_,cfg=old.verify_control(old.OLD/'control');c=cfg['motion'];store=old.FeatureStore(c,max_cached=4)
 lock=read(ROOT/'parent_lock.json');assert all(sha(k)==v for k,v in lock['weights_and_transforms'].items());assets=dict(lock['weights_and_transforms'])
 with np.load(Path(c['output'])/'predictor/motion/basis.npz',allow_pickle=False) as f:basis={k:f[k] for k in f.files}
 store.bases['motion']=basis;d=old.LinearReadout.load(Path(c['output'])/'probes/motion_state_decoder.npz');B,A,b,right,_=maps(basis['output_basis'],d.weight,d.std,d.mean,d.bias)
 records=[r for r in store.manifest['records'] if r['task']=='motion' and r['split']=='train'];groups={}
 for r in records:groups.setdefault(r['pair'],[]).append(r)
 assert len(groups)==512 and all(len(v)==2 for v in groups.values());pairs=sorted(groups);records=[r for pair in pairs for r in sorted(groups[pair],key=lambda x:x['id'])]
 S=[];T=[];G=[];E=[];X=[];ids=[]
 for rec in records:
  s,y,info=store.sample(rec,'privileged_small_budget');idx=tuple(map(int,info['indices']));q=int(rec['query']);assert len(idx)==2 and max(idx)<q-3
  obs=old.load_episode(c,rec,keys=('rgb','actions'));en=allow_update(obs['rgb'][list(idx)],obs['rgb'][q]);xy=d.predict(s['z'][4:])[:,1:]
  g=np.stack([observed_forecast(xy,idx,obs['actions'][idx[1]:q+h],q+h)[0] for h in (3,6)])
  x=np.stack([x_features(s['projected_current'][None],s['actions'][None,:h])[0] for h in (3,6)])
  S.append(s);T.append(y);G.append(g);E.append(en);X.append(x);ids.append(rec['id'])
  for path in (Path(c['output'])/'dataset'/rec['obs'],Path(c['output'])/'features'/store.ix['episodes'][rec['id']]['path']):assets[str(path)]=sha(path)
 X=np.stack(X);assert X.shape==(1024,2,134) and np.array_equal(X[::2],X[1::2]),'Adapter context must be identical between siblings'
 pack={k:np.stack([s[k] for s in S]) for k in S[0]};assert pack['z'].shape==(1024,6,5120);pack['mask']=np.ones((1024,6),bool)
 pack.update(target=np.stack(T),geometry=np.stack(G),enabled=np.array(E),X=X,episode_ids=np.array(ids),pair_ids=np.array(pairs),B=B,A=A,b=b,right=right)
 np.savez(ROOT/'train_pack.npz',**pack)
 assert sha(HV/'summary.json')==read(ROOT/'protocol.json')['baseline_validation_summary_sha256'] and sha(HV/'vectors.npz')==read(HV/'summary.json')['vectors_sha256']
 assert sha(GV/'summary.json')==read(ROOT/'protocol.json')['geometry_summary_sha256'] and sha(GV/'validation_geometry.csv')==read(GV/'summary.json')['raw_sha256']
 with (GV/'validation_geometry.csv').open() as f:rows=list(csv.DictReader(f))
 with np.load(HV/'vectors.npz',allow_pickle=False) as f:v={k:f[k] for k in ('candidate','recent','target','truth')}
 VX=[]
 for r in rows:
  rec=store.records[r['episode']];assert rec['split']=='validation';q=int(rec['query']);a=old.load_episode(c,rec,keys=('actions',))['actions'];h=int(r['horizon']);VX.append(x_features(store.projected_features(rec)[q][None],a[None,q:q+h])[0])
 v.update(X=np.stack(VX),seed=np.array([int(r['seed']) for r in rows]),horizon=np.array([int(r['horizon']) for r in rows]),pair=np.array([r['pair'] for r in rows]),episode=np.array([r['episode'] for r in rows]),A=A,b=b)
 assert v['X'].shape==(1152,134);np.savez(ROOT/'validation_pack.npz',**v)
 write(ROOT/'asset_ledger.json',assets);assert all(sha(k)==v for k,v in assets.items());source_check()
 write(ROOT/'preparation.json',dict(status='READ_ONLY_PREFLIGHT_PASS',train_pairs=512,train_episodes=1024,validation_pairs=96,validation_rows=1152,same_sibling_context=True,train_pack_sha256=sha(ROOT/'train_pack.npz'),validation_pack_sha256=sha(ROOT/'validation_pack.npz'),asset_ledger_sha256=sha(ROOT/'asset_ledger.json'),basis_train_only=True,model_forward=False,fitting=False,GPU=0,elapsed_seconds=time.monotonic()-t))
 print('READ_ONLY_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
