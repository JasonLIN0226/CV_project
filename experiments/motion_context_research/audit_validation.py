"""Separate scalar/vector implementation; original validation numeric rules unchanged."""
import sys
import numpy as np
from common import *
from audit_kernel import kernel
def main(job):
 native(job,False);source_check();out=ROOT/'validation';s=read(out/'summary.json');assert sha(out/'scalar_rows.csv')==s['scalar_sha256'] and sha(out/'vectors.npz')==s['vector_sha256']
 with np.load(ROOT/'validation.npz',allow_pickle=False) as f:v={k:f[k] for k in f.files}
 with np.load(out/'vectors.npz',allow_pickle=False) as f:candidate=f['candidate']
 independent=v['candidate'].copy()
 for seed in (17,29,43):
  for h in (3,6):
   path=ROOT/'gpu_output'/f'head_{seed}_{h}.npz';assert sha(path)==s['heads'][str(path)]
   with np.load(path,allow_pickle=False) as z:f={k:z[k] for k in z.files}
   ids=np.where((v['seed']==seed)&(v['horizon']==h))[0];k=kernel(v['context_X'][ids],f['train_X'].reshape(-1,130),f).reshape(len(ids),-1,2).mean(2);k=k-k.mean(1)[:,None]-f['row_mean'][None]+f['grand_mean'];independent[ids]+=(k@f['coef']+f['target_mean'])@f['N'].T
 diff=float(np.max(abs(independent-candidate)));assert diff<1e-10
 with (out/'scalar_rows.csv').open() as f:raw=list(csv.DictReader(f))
 assert len(raw)==3456;losses={}
 for arm,z in [('recent',v['recent']),('current',v['candidate']),('candidate',independent)]:
  feature=((z-v['target'])**2).mean(1);position=((z@v['A']+v['b']-v['truth'])**2).mean(1);losses[arm]=(feature,position);rr=[r for r in raw if r['arm']==arm]
  assert len(rr)==1152 and np.max(abs(feature-[float(r['feature_mse']) for r in rr]))<1e-11 and np.max(abs(position-[float(r['position_mse']) for r in rr]))<1e-11
 checks=0;maximum=0
 for cell in s['cells']:
  seed,h=cell['seed'],cell['horizon'];ids=np.where((v['horizon']==h)&((v['seed']==seed) if seed!='mean' else np.ones(1152,bool)))[0];pairs=sorted(set(v['pair'][ids]));assert len(pairs)==96;ix=np.random.default_rng(0).integers(0,96,(2000,96))
  for name,arm,mi,floor,absolute in [('feature_vs_recent','recent',0,.02,.01),('position_vs_recent','recent',1,1e-5,1e-6),('feature_vs_current','current',0,.02,.01),('position_vs_current','current',1,1e-5,1e-6)]:
   L,R=losses[arm][mi],losses['candidate'][mi];x=np.array([[L[ids[v['pair'][ids]==p]].mean(),R[ids[v['pair'][ids]==p]].mean()] for p in pairs]);ab=x[:,0].mean()<floor
   def eff(a):
    b=a[...,0].mean(-1);c=a[...,1].mean(-1);return b-c if ab else (b-c)/np.maximum(b,floor)
   val=float(eff(x));lo,hi=np.quantile(eff(x[ix]),[.025,.975]);got=cell[name];d=max(abs(val-got['mean']),abs(lo-got['low']),abs(hi-got['high']));maximum=max(maximum,d);assert d<1e-10 and got['numeric_rule_met']==bool(val>=(absolute if ab else .1) and lo>0);checks+=1
 eligible=all(c['feature_vs_recent']['numeric_rule_met'] and c['position_vs_recent']['numeric_rule_met'] and c['feature_vs_current']['low']>0 for c in s['cells']);assert eligible==s['validation_eligible'];assert all(sha(k)==v for k,v in s['heads'].items());source_check()
 write(out/'independent_audit.json',dict(status='PASS_INDEPENDENT_VECTOR_AND_32_INTERVALS',source_rows=1152,scalar_rows=3456,intervals=checks,vector_max_difference=diff,interval_max_difference=maximum,validation_eligible=eligible,original_gates_changed=False,phase1_go=False))
 print(s['status'],flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
