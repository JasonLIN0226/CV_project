"""Narrow algebra/packing fixtures, synthetic inputs only; no CUDA or scientific fitting."""
import json
from pathlib import Path
import numpy as np
from calibrator import null_basis,fit_train_only
from common import x_features
r=np.random.default_rng(27);B=r.normal(size=(32,8));A=r.normal(size=(32,2));N=null_basis(B,A);X=r.normal(size=(80,6));X[:,0]=2.;mu=X[:40].mean(0);scale=np.maximum(X[:40].std(0),1e-6);Z=np.column_stack([(X[:40]-mu)/scale,np.ones(40)]);Y=r.normal(size=(40,6));G=Z.T@Z;P=np.eye(7);P[-1,-1]=0;alpha=.01*np.trace(G[:-1,:-1])/6;W=np.linalg.solve(G+alpha*P,Z.T@Y);assert np.max(abs((G+alpha*P)@W-Z.T@Y))<1e-11;delta=(Z@W)@N.T;assert np.max(abs(delta@A))<1e-12
base=r.normal(size=(40,32));other=r.normal(size=(40,32));assert np.max(abs(((base+delta)-(other+delta))-(base-other)))<1e-12
xx=x_features(np.zeros((2,128),np.float32),np.ones((2,6,2),np.float32));assert xx.shape==(2,134) and np.array_equal(xx[0],xx[1]) and np.array_equal(xx[0,130:132],[6,6])
try:fit_train_only(None,None,None,[],authorized=False)
except PermissionError:pass
else:raise AssertionError('Authorization guard bypassed')
Path(__file__).with_name('software_preflight.json').write_text(json.dumps(dict(status='PASS_SOFTWARE_ONLY',fixtures=['ridge_normal_equation_with_constant_feature','position_null_update','paired_contrast_identity','134legal_context_features','authorization_guard'],scientific_data=False,Cuda_science_tested=False,GPU=0),indent=2)+'\n');print('PASS_SOFTWARE_ONLY: 5fixtures')
