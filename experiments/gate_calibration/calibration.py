"""Single validation-selected residual scale; never changes a network weight."""
import numpy as np
GRID=(0.,.25,.5,.75,1.)
def blend(anchor,prediction,alpha):
    if alpha not in GRID:raise ValueError('Unregistered alpha')
    if alpha==1:return prediction
    if alpha==0:return anchor
    return anchor+alpha*(prediction-anchor)
def choose(scores):
    if set(scores)!=set(GRID) or not np.isfinite(list(scores.values())).all():raise ValueError('Incomplete/nonfinite validation')
    # Exact ties prefer the unchanged model. No test-conditioned or seed-specific choice.
    return min(GRID,key=lambda a:(scores[a],-a))
