"""Diagnostic identities for a fixed decoder and fixed output subspace."""
import numpy as np

def decompose(pred,target,w,basis_q):
 p=np.asarray(pred,np.float64);t=np.asarray(target,np.float64);w=np.asarray(w,np.float64)
 assert p.shape==t.shape and p.ndim==2 and len(p)==2 and w.shape==(p.shape[1],)
 assert np.isfinite(p).all() and np.isfinite(t).all() and np.linalg.norm(w)>0
 dim=p.shape[-1];u=w/np.linalg.norm(w);e=p-t
 total=np.mean(e*e);parallel=np.mean((e@u)**2)/dim
 orthogonal=total-parallel
 common=np.mean((e.mean(0))**2);contrast=np.mean(((e[0]-e[1])/2)**2)
 inside=np.mean(np.sum((e@basis_q)**2,axis=1))/dim;outside=total-inside
 assert min(orthogonal,outside)>-1e-10 and abs(total-common-contrast)<1e-10
 td=(t[0]-t[1])/2;pd=(p[0]-p[1])/2;den=np.dot(td,td)
 return dict(total=total,parallel=parallel,orthogonal=max(0.,orthogonal),common=common,contrast=contrast,
  basis_inside=inside,basis_outside=max(0.,outside),target_pair_signal=np.mean(td**2),
  prediction_pair_signal=np.mean(pd**2),signal_gain=float(pd@td/den) if den>1e-16 else None,
  prediction_sibling_maxabs=float(np.abs(p[0]-p[1]).max()))

def deletion_tradeoff(critical,others,tolerance=1e-8):
 """Descriptive feasibility within this fixed three-fork parent, not a new gate."""
 cf,ct=critical;helpful=[(f,t) for f,t in others if t<ct-tolerance]
 return dict(task_help_available=int(bool(helpful)),joint_help_available=int(any(f<cf-tolerance for f,t in helpful)),
  all_task_help_increases_feature=int(bool(helpful) and all(f>cf+tolerance for f,t in helpful)))
