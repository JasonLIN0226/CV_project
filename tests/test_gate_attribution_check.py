from pathlib import Path
import importlib.util
import numpy as np
p=Path(__file__).resolve().parents[1]/'experiments/gate_attribution_check/identities.py';s=importlib.util.spec_from_file_location('attribution_id',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_pair_mean_control_matches_scalar_identity():
 p=np.array([[3.,2.],[1.,4.]]);d=np.array([[.5,.2],[.5,.2]]);y=np.array([[1.,-1.],[-1.,1.]])
 q=m.replace_pair_mean(p,d)
 assert np.allclose(q[0]-q[1],p[0]-p[1]) and np.allclose(q.mean(0),d.mean(0))
 expected=np.mean((d.mean(0)-y.mean(0))**2)+np.mean(((p[0]-p[1]-y[0]+y[1])/2)**2)
 assert np.isclose(np.mean((q-y)**2),expected)
def test_drift_is_not_target_loss_cost():
 # A large prediction change can improve or worsen the actual target loss.
 a=np.array([1.]);b=np.array([3.]);u=m.loss_delta(a,b,np.array([3.]));v=m.loss_delta(a,b,np.array([1.]))
 assert u['drift']==v['drift']==4 and u['actual']==-4 and v['actual']==4
 assert u['actual']==u['drift']+u['interaction']
def test_ideal_and_no_response_controls():
 y=np.array([[1.,2.],[-1.,-2.]]);mean=np.repeat(y.mean(0,keepdims=True),2,axis=0)
 assert np.mean((mean-y)**2)>0 and np.mean((y-y)**2)==0
 same=m.loss_delta(mean,mean,y);assert same['actual']==same['drift']==same['interaction']==0
 # Correct binary sign does not ensure smaller feature error.
 assert np.sign(100)==np.sign(y[0,0]) and (100-y[0,0])**2>(0-y[0,0])**2
def test_removing_common_bias_cannot_fix_wrong_pair_contrast():
 y=np.array([[1.],[-1.]])
 deleted=np.zeros((2,1));bad=np.array([[3.],[7.]])
 corrected=m.replace_pair_mean(bad,deleted)
 assert np.mean((corrected-y)**2)==9 and np.mean((deleted-y)**2)==1
 assert m.feature_check(np.ones(128),np.full(128,9.),repeats=100)['passed'] is False
