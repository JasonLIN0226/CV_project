"""Low-dimensional Gaussian observation map; fixed input-only df32, CUDA fit."""
import numpy as np
def gram(x,y,f):
 x=(np.asarray(x,float)-f['mean'])/f['scale'];y=(np.asarray(y,float)-f['mean'])/f['scale'];d=np.maximum(np.sum(x*x,1)[:,None]+np.sum(y*y,1)[None]-2*x@y.T,0);return np.exp(-d/f['bandwidth'])
def predict(x,f):
 k=gram(x,f['train_X'],f);k=k-k.mean(1)[:,None]-f['row_mean'][None]+f['grand_mean'];return k@f['coef']+f['target_mean']
def fit(x,y,*,scientific=True):
 import torch
 if scientific and (not x.is_cuda or not y.is_cuda):raise ValueError('Scientific fit is CUDA-only')
 assert x.ndim==2 and x.shape[1]==2 and y.shape==(len(x),126)
 with torch.no_grad():
  x=x.double();y=y.double();mu=x.mean(0);sd=x.std(0,unbiased=False).clamp_min(1e-6);z=(x-mu)/sd;d=((z*z).sum(1)[:,None]+(z*z).sum(1)[None]-2*z@z.T).clamp_min(0);mask=torch.triu(torch.ones_like(d,dtype=torch.bool),diagonal=1);bw=torch.median(d[mask]).clamp_min(1e-6);k=torch.exp(-d/bw);rm=k.mean(0);gm=k.mean();kc=k-rm[:,None]-rm[None]+gm;eig=torch.linalg.eigvalsh(kc).clamp_min(0);lo=torch.tensor(1e-12,device=x.device,dtype=x.dtype);hi=eig.sum()+1
  for _ in range(80):
   mid=(lo+hi)/2
   if (eig/(eig+mid)).sum()>31:lo=mid
   else:hi=mid
  # Bound numerical conditioning from inputs only; never tune against outcomes.
  alpha=torch.maximum(hi,eig.max()/(1e6-1));ym=y.mean(0);coef=torch.linalg.solve(kc+alpha*torch.eye(len(x),device=x.device,dtype=x.dtype),y-ym);assert torch.isfinite(coef).all()
  return {k:v.cpu().numpy() for k,v in dict(train_X=x,mean=mu,scale=sd,bandwidth=bw,row_mean=rm,grand_mean=gm,coef=coef,target_mean=ym,alpha=alpha,effective_df=1+(eig/(eig+alpha)).sum()).items()}
