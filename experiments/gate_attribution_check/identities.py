"""Offline algebraic controls; not deployable prediction transforms."""
import numpy as np

def replace_pair_mean(prediction,deleted_prediction):
 p=np.asarray(prediction,dtype=np.float64);d=np.asarray(deleted_prediction,dtype=np.float64)
 if p.shape!=d.shape or p.ndim!=2 or len(p)!=2:raise ValueError('Complete paired predictions required')
 return p-p.mean(axis=0)+d.mean(axis=0)

def loss_delta(anchor,candidate,target):
 a,b,y=map(lambda x:np.asarray(x,dtype=np.float64),(anchor,candidate,target))
 if a.shape!=b.shape or a.shape!=y.shape:raise ValueError('Aligned predictions and target required')
 drift=float(np.mean((b-a)**2));interaction=float(2*np.mean((a-y)*(b-a)))
 return dict(actual=float(np.mean((b-y)**2)-np.mean((a-y)**2)),drift=drift,interaction=interaction)

def feature_check(critical,other,repeats=10000):
 a,b=np.asarray(critical,dtype=float),np.asarray(other,dtype=float)
 if a.shape!=b.shape or a.ndim!=1 or len(a)!=128 or not np.isfinite([a,b]).all():raise ValueError('128 finite paired observations required')
 d=a-b;ix=np.random.default_rng(0).integers(0,len(a),(repeats,len(a)));lo,hi=np.quantile(d[ix].mean(1),[.025,.975]);rel=float(d.mean()/max(a.mean(),1e-8))
 return dict(critical_mean=float(a.mean()),other_mean=float(b.mean()),delta=float(d.mean()),low=float(lo),high=float(hi),relative=rel,passed=bool(a.mean()>=1e-8 and rel>=.1 and lo>0))
