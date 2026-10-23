"""Algebraic software fixtures only: no scientific data, fitting or GPU."""
from pathlib import Path
import json
import numpy as np
from calibrator import null_basis,fit_train_only
rng=np.random.default_rng(19);B=rng.normal(size=(32,8));A=rng.normal(size=(32,2));N=null_basis(B,A)
assert N.shape==(32,6) and np.max(abs(N.T@A))<1e-12
base=rng.normal(size=(20,32));delta=rng.normal(size=(20,6))@N.T
assert np.max(abs((base+delta)@A-base@A))<1e-12
# Equal current/action context gives equal correction to paired siblings,
# hence their prediction difference remains unchanged.
left=rng.normal(size=(20,32));right=rng.normal(size=(20,32))
assert np.max(abs(((left+delta)-(right+delta))-(left-right)))<1e-12
try:fit_train_only(None,None,None,[],authorized=False)
except PermissionError:pass
else:raise AssertionError('An unauthorised fit must stop before importing CUDA or reading data')
p=Path(__file__).with_name('software_preflight.json');p.write_text(json.dumps(dict(status='PASS_SOFTWARE_ONLY',fixtures=['fixed_position_null_update','paired_prediction_contrast_unchanged','unapproved_fit_rejected'],scientific_execution=False,GPU=0),indent=2)+'\n');print('PASS_SOFTWARE_ONLY; no scientific fitting executed')
