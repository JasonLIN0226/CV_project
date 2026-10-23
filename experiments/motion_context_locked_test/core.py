"""Analytical inference constraint on frozen features; no fitted parameters."""
import numpy as np


def maps(basis, weight, std, mean, bias):
    B=np.asarray(basis,dtype=np.float64)
    A=np.asarray(weight,dtype=np.float64)[:,1:]/np.asarray(std,dtype=np.float64)[:,None]
    b=np.asarray(bias,dtype=np.float64)[1:]-np.asarray(mean,dtype=np.float64)@A
    gram=B.T@B
    F=B.T@A
    giF=np.linalg.solve(gram,F)
    S=F.T@giF
    singular=np.linalg.svd(S,compute_uv=False)
    if singular[-1]<singular[0]*1e-12:raise ValueError('Position constraint is rank deficient in the frozen basis')
    right=np.linalg.solve(S,giF.T)
    return B,A,b,right,gram


def constrain(prediction, geometry, B, A, b, right):
    pred=np.asarray(prediction,dtype=np.float64)
    d=np.asarray(geometry,dtype=np.float64)-(pred@A+b)
    coeff=d@right
    updated=pred+coeff@B.T
    return updated,coeff


def oracle_feature(prediction, target, B, gram):
    # Scoring-only future-feature upper bound; never used by the normal arm.
    coeff=np.linalg.solve(gram,((np.asarray(target)-prediction)@B).T).T
    return prediction+coeff@B.T,coeff


def interval(base,alternative,groups,floor=.02,absolute=.01):
    pairs=sorted(set(groups));groups=np.asarray(groups)
    X=np.array([[np.mean(np.asarray(base)[groups==g]),np.mean(np.asarray(alternative)[groups==g])] for g in pairs])
    use_absolute=X[:,0].mean()<floor
    def effect(x):
        a=x[...,0].mean(-1);b=x[...,1].mean(-1)
        return a-b if use_absolute else (a-b)/np.maximum(a,floor)
    ix=np.random.default_rng(0).integers(0,len(X),(2000,len(X)))
    lo,hi=np.percentile(effect(X[ix]),[2.5,97.5])
    value=float(effect(X));threshold=absolute if use_absolute else .1
    return dict(mean=value,low=float(lo),high=float(hi),pairs=len(X),scale='absolute' if use_absolute else 'relative',threshold=threshold,numeric_rule_met=bool(value>=threshold and lo>0))
