"""Unexecuted scientific prototype: train-only, position-null common residual adapter.

Original predictor, encoder, basis and decoder remain frozen. Fitting adds NEW
parameters and requires separate human authorization; it is not a frozen ablation.
"""
import numpy as np


def null_basis(output_basis, position_map):
    Q=np.linalg.qr(np.asarray(output_basis,dtype=np.float64))[0]
    A=np.asarray(position_map,dtype=np.float64)
    _,s,V=np.linalg.svd(A.T@Q,full_matrices=True)
    if len(s)!=2 or s[-1]<=s[0]*1e-10:
        raise ValueError('The original two-coordinate readout must have full rank')
    N=Q@V[2:].T
    if np.max(np.abs(N.T@A))>=1e-10:
        raise ValueError('Position-null basis is numerically unstable')
    return N


def fit_train_only(X, paired_common_feature_residual, N, splits, *, authorized=False, stage="final_train"):
    """Only original training-pair means; scientific solve is CUDA-only.

    X: 512pairs x134, using frozen projected current128 + allowed action summary6.
    Residual: mean of the two TRAINING sibling (target - frozen prediction)
    vectors. Neither sibling identity nor target is an input to deployment.
    No validation-dependent alpha, rank, horizon or seed selection.
    """
    if not authorized:
        raise PermissionError('New fitted parameters require explicit human approval')
    import torch
    if len(splits)!=len(X) or not all(s=='train' for s in splits):
        raise ValueError('Fitting data must be exclusively training pairs')
    if not X.is_cuda or not paired_common_feature_residual.is_cuda:
        raise ValueError('No CPU scientific-training fallback')
    expected=512 if stage=='final_train' else 256 if stage=='fixed_train_half' else None
    if expected is None or len(X)!=expected or X.shape[1]!=134 or paired_common_feature_residual.shape!=(expected,5120):
        raise ValueError('Keep the original full training set and dimensions')
    with torch.no_grad():
        X=X.double();R=paired_common_feature_residual.double()
        N=torch.as_tensor(N,device=X.device,dtype=torch.float64)
        if not torch.isfinite(X).all() or not torch.isfinite(R).all():
            raise ValueError('Nonfinite inputs')
        mean=X.mean(0);scale=X.std(0,unbiased=False).clamp_min(1e-6)
        Z=torch.cat([(X-mean)/scale,torch.ones((len(X),1),device=X.device,dtype=X.dtype)],1)
        gram=Z.T@Z;penalty=torch.eye(gram.shape[0],device=X.device,dtype=X.dtype)
        penalty[-1,-1]=0
        alpha=.01*torch.trace(gram[:-1,:-1])/X.shape[1]
        coef=torch.linalg.solve(gram+alpha*penalty,Z.T@(R@N))
        if not torch.isfinite(coef).all():raise ValueError('Nonfinite calibration')
        return dict(mean=mean,scale=scale,coef=coef,N=N,alpha=alpha,fit_split='train',pairs=expected,stage=stage)


def correction(X, fitted):
    """Deployment consumes present observations and known actions only."""
    import torch
    with torch.no_grad():
        X=X.to(fitted['mean']);Z=torch.cat([(X-fitted['mean'])/fitted['scale'],torch.ones((len(X),1),device=X.device,dtype=X.dtype)],1)
        return (Z@fitted['coef'])@fitted['N'].T
