"""Fixed observed-RGB validity check; synthetic-renderer diagnostic, not a policy."""
import numpy as np


def support_crop(frame):
    x = np.asarray(frame)
    if x.ndim != 3 or x.shape[2] != 3 or x.shape[0] != x.shape[1]:
        raise ValueError('Expected square observed RGB')
    n = x.shape[0]
    # Interior of the fixed arena; excludes external distractors. No annotations.
    return x[int(.30*n):int(.73*n), int(.10*n):int(.90*n)]


def allow_update(retained_rgb, current_rgb):
    r = np.asarray(retained_rgb)
    if r.shape[0] != 2:
        raise ValueError('Exactly two actually retained observations required')
    current = support_crop(current_rgb)
    return bool(all(not np.array_equal(support_crop(frame), current) for frame in r))
