"""Exact original 3-to-2 fork input builder, no model/label/future access."""
import numpy as np

def sample(z,p,actions,q,kept):
    R=4;idx=np.r_[np.arange(q-R+1,q+1),np.array(kept,dtype=np.int64)]
    assert len(kept)==2 and len(set(kept))==2 and max(kept)<q-R+1 and min(kept)>=0
    prefix=np.vstack([np.zeros((1,2),np.float32),np.cumsum(actions[:q],0)])
    return dict(z=z[idx].copy(),projected_z=p[idx].copy(),projected_current=p[q].copy(),current=z[q].copy(),
       age=(q-idx).astype(np.float32),type=np.r_[np.zeros(R,np.int64),np.ones(2,np.int64)],
       past_actions=prefix[q]-prefix[idx],actions=actions[q:q+6].copy())
