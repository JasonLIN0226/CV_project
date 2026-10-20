import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1]/'experiments/gate_cost_diagnosis/geometry.py'
s=importlib.util.spec_from_file_location('gate_geometry',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_task_can_improve_while_total_mse_worsens():
 t=np.array([[-1.,0.],[1.,0.]])
 deleted=np.zeros((2,2));retained=np.array([[-.01,2.],[.01,2.]])
 a=m.decompose(deleted,t,np.array([1.,0.]),np.eye(2));b=m.decompose(retained,t,np.array([1.,0.]),np.eye(2))
 assert b['parallel']<a['parallel'] and b['total']>a['total']
 assert (retained[:,0]>0).tolist()==[False,True]
def test_pair_common_and_contrast_decomposition():
 g=np.random.default_rng(0);p=g.normal(size=(2,12));t=g.normal(size=(2,12));w=g.normal(size=12);q=np.linalg.qr(g.normal(size=(12,3)))[0]
 a=m.decompose(p,t,w,q)
 assert np.isclose(a['total'],a['common']+a['contrast'])
 assert np.isclose(a['total'],a['parallel']+a['orthogonal'])
 assert np.isclose(a['total'],a['basis_inside']+a['basis_outside'])
def test_output_floor_cancels_between_deletions():
 g=np.random.default_rng(1);t=g.normal(size=(2,6));q=np.eye(6)[:,:2];p=g.normal(size=(2,6));p2=p+g.normal(size=(2,2))@q.T
 a=m.decompose(p,t,np.ones(6),q);b=m.decompose(p2,t,np.ones(6),q)
 assert np.isclose(a['basis_outside'],b['basis_outside'])
def test_decoder_nullspace_is_not_a_background_label():
 g=np.random.default_rng(2);w=g.normal(size=8);u=w/np.linalg.norm(w);d=g.normal(size=8);null=d-(d@u)*u
 assert abs(null@w)<1e-12
 # This is only a score-invariance identity, not a claim that nullspace content is irrelevant to other tasks.

def witness_fixture():
 p=np.zeros((162,2,5120),np.float32);t=p.copy();t[:,0,0]=-1;t[:,1,0]=1;w=np.zeros(5120);w[0]=1;Q=np.eye(5120,2)
 names=['total','parallel','orthogonal','common','contrast','basis_inside','basis_outside','target_pair_signal','prediction_pair_signal','signal_gain','prediction_sibling_maxabs']
 expected=[1/5120,1/5120,0,0,1/5120,1/5120,0,1/5120,0,0,0]
 return dict(prediction=p,target=t,w=w,Q=Q,metric_names=np.array(names),metrics=np.tile(expected,(162,1)))
def load_audit():
 p=Path(__file__).resolve().parents[1]/'experiments/gate_cost_diagnosis/audit.py';s=importlib.util.spec_from_file_location('gate_geometry_audit',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a);return a
def test_independent_primitive_vector_audit():
 result=load_audit().audit_witness(witness_fixture());assert np.allclose(result['parallel'],1/5120)
def test_vector_audit_detects_tampered_semantic_energy():
 import pytest
 f=witness_fixture();f['metrics'][0,1]=2/5120
 with pytest.raises(AssertionError):load_audit().audit_witness(f)
def test_three_fork_tradeoff_not_fixed_by_better_ranker():
 v=m.deletion_tradeoff((.1,.5),[(.2,0.),(.08,.5)])
 assert v==dict(task_help_available=1,joint_help_available=0,all_task_help_increases_feature=1)
def test_three_fork_joint_choice_exists():
 v=m.deletion_tradeoff((.1,.5),[(.08,0.),(.2,.5)])
 assert v==dict(task_help_available=1,joint_help_available=1,all_task_help_increases_feature=0)
