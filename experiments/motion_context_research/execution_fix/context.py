"""Causal paired-scene centre: latest retained decoded position plus controls.

No sibling, target, simulator state, future image or metadata enters this map.
The existing fixed control coefficients match the frozen geometry operator.
"""
import numpy as np
def context(z,past_actions,actions,projected_current,A,b,h):
 z=np.asarray(z);assert z.ndim==3 and z.shape[1]==6
 anchor=z[:,5]@A+b
 controls=np.asarray(past_actions)[:,5]+np.asarray(actions)[:,:h].sum(1)
 centre=anchor.copy();centre[:,0]+=.0015*controls[:,0]+.001*controls[:,1]
 return np.column_stack([centre,np.asarray(projected_current)]).astype(np.float64)
