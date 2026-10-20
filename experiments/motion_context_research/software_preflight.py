"""Synthetic fixture only, never scientific CPU fitting."""
import numpy as np
import torch
from kernel import fit,apply,point_kernel
rng=np.random.default_rng(7);x=rng.normal(size=(40,2,130));y=rng.normal(size=(40,5120));N=np.linalg.qr(rng.normal(size=(5120,6)))[0]
f=fit(torch.tensor(x),torch.tensor(y),N,scientific=False);delta=apply(x.reshape(-1,130),f).reshape(40,2,5120).mean(1);K=point_kernel(x.reshape(-1,130),x.reshape(-1,130),f).reshape(40,2,40,2).mean((1,3));kc=K-K.mean(0)[None]-K.mean(1)[:,None]+K.mean();expected=(kc@f['coef']+f['target_mean'])@N.T
assert np.max(abs(delta-expected))<1e-10
assert abs(float(f['effective_df'])-32)<1e-8
lhs=(kc+float(f['alpha'])*np.eye(40))@f['coef'];rhs=y@N-f['target_mean'];assert np.max(abs(lhs-rhs))<1e-9
assert np.max(abs(f['train_X']-x))==0
print('PASS_SYNTHETIC_KERNEL_CENTERING_KKT_DF32_NO_SCIENTIFIC_FIT')
