"""Synthetic-only check, no cache/model/research data used."""
import numpy as np
import torch
from kernel import fit,gram,predict
rng=np.random.default_rng(13);x=rng.normal(size=(40,2));y=rng.normal(size=(40,126));f=fit(torch.tensor(x),torch.tensor(y),scientific=False);K=gram(x,x,f);kc=K-K.mean(0)[None]-K.mean(1)[:,None]+K.mean();assert np.max(abs(predict(x,f)-(kc@f['coef']+f['target_mean'])))<1e-9;assert float(f['effective_df'])<=32+1e-7;assert np.linalg.cond(kc+float(f['alpha'])*np.eye(40))<=1e6+1e-2;assert np.max(abs((kc+float(f['alpha'])*np.eye(40))@f['coef']-(y-y.mean(0))))<1e-8;print('PASS_SYNTHETIC_OBSERVATION_MAP_KKT_CENTERING_CONDITION_CAP')
