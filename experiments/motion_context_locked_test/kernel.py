"""Pair-mean kernel ridge; legal per-episode input and fixed input-only capacity.

The fit is scientific only on CUDA. NumPy routines apply already-fitted heads
or verify synthetic fixtures. No validation-dependent parameter selection.
"""
import numpy as np
def distances(x,y):
 return np.maximum(np.sum(x*x,1)[:,None]+np.sum(y*y,1)[None,:]-2*x@y.T,0)
def point_kernel(x,y,f):
 a=(x-f['mean'])/f['scale'];b=(y-f['mean'])/f['scale']
 return .5*np.exp(-distances(a[:,:2],b[:,:2])/f['bandwidth'][0])+.5*np.exp(-distances(a[:,2:],b[:,2:])/f['bandwidth'][1])
def apply(x,f):
 k=point_kernel(np.asarray(x,float),f['train_X'].reshape(-1,130),f).reshape(len(x),-1,2).mean(2)
 z=k-k.mean(1,keepdims=True)-f['row_mean'][None]+f['grand_mean']
 return (z@f['coef']+f['target_mean'])@f['N'].T
def fit(x,y,N,*,scientific=True):
 import torch
 if scientific and (not x.is_cuda or not y.is_cuda):raise ValueError('CUDA-only scientific fitting')
 assert x.ndim==3 and x.shape[1:]==(2,130) and y.shape==(len(x),5120)
 with torch.no_grad():
  x=x.double();y=y.double();N=torch.as_tensor(N,device=x.device,dtype=x.dtype);flat=x.reshape(-1,130);mu=flat.mean(0);sd=flat.std(0,unbiased=False).clamp_min(1e-6);z=(flat-mu)/sd
  def dist(a):return ((a*a).sum(1)[:,None]+(a*a).sum(1)[None,:]-2*a@a.T).clamp_min(0)
  ds=[dist(z[:,:2]),dist(z[:,2:])];mask=torch.triu(torch.ones_like(ds[0],dtype=torch.bool),diagonal=1);bw=torch.stack([torch.median(d[mask]).clamp_min(1e-6) for d in ds]);k=.5*torch.exp(-ds[0]/bw[0])+.5*torch.exp(-ds[1]/bw[1]);k=k.reshape(len(x),2,len(x),2).mean((1,3));rm=k.mean(0);gm=k.mean();kc=k-rm[None]-rm[:,None]+gm
  eig=torch.linalg.eigvalsh(kc).clamp_min(0);lo=torch.tensor(1e-12,device=x.device,dtype=x.dtype);hi=eig.sum()+1
  # 31 variable degrees of freedom plus the unpenalized intercept =32.
  for _ in range(80):
   mid=(lo+hi)/2;df=(eig/(eig+mid)).sum()
   if df>31:lo=mid
   else:hi=mid
  alpha=hi;target=y@N;ym=target.mean(0);coef=torch.linalg.solve(kc+alpha*torch.eye(len(x),device=x.device,dtype=x.dtype),target-ym)
  assert torch.isfinite(coef).all()
  return {k:v.detach().cpu().numpy() for k,v in dict(train_X=x,mean=mu,scale=sd,bandwidth=bw,row_mean=rm,grand_mean=gm,coef=coef,target_mean=ym,N=N,alpha=alpha,effective_df=1+(eig/(eig+alpha)).sum()).items()}
