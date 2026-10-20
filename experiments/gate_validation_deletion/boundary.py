"""Observed-input-only packing and physical deletion path controls."""
import numpy as np
TOKEN_KEYS=frozenset(('z','projected_z','age','type','past_actions'))
def pack(z,projected,actions,query,kept):
 kept=tuple(kept)
 if len(set(kept))!=len(kept) or tuple(sorted(kept))!=kept or any(t<0 or t>=query-3 for t in kept):raise ValueError('Distinct chronological past observations only')
 idx=np.r_[np.arange(query-3,query+1),np.asarray(kept,dtype=np.int64)];prefix=np.vstack([np.zeros((1,2),np.float32),np.cumsum(actions[:query],0)])
 return dict(z=z[idx].copy(),projected_z=projected[idx].copy(),projected_current=projected[query].copy(),current=z[query].copy(),age=(query-idx).astype(np.float32),type=np.r_[np.zeros(4,np.int64),np.ones(len(kept),np.int64)],past_actions=prefix[query]-prefix[idx],actions=actions[query:query+6].copy())
def remove(parent,history_slot):
 if history_slot<0 or history_slot>=len(parent['z'])-4:raise ValueError('Can remove historical tokens only')
 return {k:np.delete(v,4+history_slot,axis=0) if k in TOKEN_KEYS else v.copy() for k,v in parent.items()}
def same(a,b):
 return set(a)==set(b) and all(np.array_equal(a[k],b[k]) for k in a)
def paired_errors(pred,target):
 p,t=np.asarray(pred,np.float64),np.asarray(target,np.float64)
 if p.shape!=t.shape or p.ndim!=2 or len(p)!=2:raise ValueError('Complete sibling pair required')
 e=p-t;mean=float(np.mean(e.mean(0)**2));contrast=float(np.mean(((e[0]-e[1])/2)**2));total=float(np.mean(e**2))
 assert abs(total-mean-contrast)<1e-10
 return dict(total=total,common=mean,contrast=contrast,prediction_sibling_maxabs=float(np.abs(p[0]-p[1]).max()))
