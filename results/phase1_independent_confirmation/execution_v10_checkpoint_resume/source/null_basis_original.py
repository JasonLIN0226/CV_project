"""Exact accepted train-only position-null basis; no fitting routine here."""
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
