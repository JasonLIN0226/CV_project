"""Apply independently audited fixed heads to original cached validation only."""
import sys
import numpy as np
from common import *
from kernel import apply as apply_kernel
from core import interval

def main(job):
 native(job,False);source_check();fit=read(ROOT/'fit_audit.json');assert fit['status']=='PASS_INDEPENDENT_KERNEL_KKT_CROSSFIT_DF32' and fit['crossfit_status']=='TRAIN_SCREEN_SUPPORTED'
 out=ROOT/'validation';out.mkdir(exist_ok=False)
 with np.load(ROOT/'validation.npz',allow_pickle=False) as f:v={k:f[k] for k in f.files}
 X=v['context_X'];old=v['candidate'];candidate=old.copy();maximum=0;context_identity=0.;headfiles={}
 for seed in (17,29,43):
  for h in (3,6):
   path=ROOT/'gpu_output'/f'head_{seed}_{h}.npz';headfiles[str(path)]=sha(path)
   with np.load(path,allow_pickle=False) as f:d={k:f[k] for k in ('train_X','mean','scale','bandwidth','row_mean','grand_mean','coef','target_mean','N')}
   ids=np.where((v['seed']==seed)&(v['horizon']==h))[0];update=apply_kernel(X[ids],d);candidate[ids]+=update;maximum=max(maximum,float(abs(update@v['A']).max()))
   for pair in sorted(set(v['pair'][ids])):
    ix=ids[v['pair'][ids]==pair];assert len(ix)==2 and np.array_equal(X[ix[0]],X[ix[1]]);context_identity=max(context_identity,float(abs((candidate[ix[0]]-candidate[ix[1]])-(old[ix[0]]-old[ix[1]])).max()))
 assert maximum<1e-10 and context_identity<1e-10 and np.isfinite(candidate).all();losses={};raw=[]
 for name,z in [('recent',v['recent']),('current',old),('candidate',candidate)]:
  f=((z-v['target'])**2).mean(1);p=((z@v['A']+v['b']-v['truth'])**2).mean(1);losses[name]=(f,p)
  for i in range(1152):raw.append(dict(seed=int(v['seed'][i]),horizon=int(v['horizon'][i]),pair=str(v['pair'][i]),episode=str(v['episode'][i]),arm=name,feature_mse=float(f[i]),position_mse=float(p[i])))
 cells=[]
 for seed in (17,29,43,'mean'):
  for h in (3,6):
   ids=np.where((v['horizon']==h)&((v['seed']==seed) if seed!='mean' else np.ones(1152,bool)))[0];groups=v['pair'][ids];cf,cp=losses['candidate'];rf,rp=losses['recent'];of,op=losses['current']
   cells.append(dict(seed=seed,horizon=h,feature_vs_recent=interval(rf[ids],cf[ids],groups),position_vs_recent=interval(rp[ids],cp[ids],groups,1e-5,1e-6),feature_vs_current=interval(of[ids],cf[ids],groups),position_vs_current=interval(op[ids],cp[ids],groups,1e-5,1e-6)))
 with (out/'scalar_rows.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
 np.savez(out/'vectors.npz',candidate=candidate)
 eligible=all(c['feature_vs_recent']['numeric_rule_met'] and c['position_vs_recent']['numeric_rule_met'] and c['feature_vs_current']['low']>0 for c in cells)
 write(out/'summary.json',dict(status='VALIDATION_SUPPORTED_NOT_PHASE1_ACCEPTANCE' if eligible else 'VALIDATION_REJECTED_STOP',validation_eligible=eligible,pairs=96,source_rows=1152,scalar_rows=3456,feature_checks=sum(c['feature_vs_recent']['numeric_rule_met'] for c in cells),position_checks=sum(c['position_vs_recent']['numeric_rule_met'] for c in cells),incremental_feature_positive=sum(c['feature_vs_current']['low']>0 for c in cells),cells=cells,position_preservation_max=maximum,paired_contrast_preservation_max=context_identity,heads=headfiles,scalar_sha256=sha(out/'scalar_rows.csv'),vector_sha256=sha(out/'vectors.npz'),head_training_split='train',base_frozen=True,original_gates_changed=False,phase1_go=False,test_opened=False,fresh9137=False));print('VALIDATION_FINISHED',flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
