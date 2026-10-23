"""Mathematical fixtures, never research data or evidence."""
import numpy as np
from core import maps,constrain,oracle_feature

def main():
    rng=np.random.default_rng(0);B=rng.normal(size=(32,8));W=rng.normal(size=(32,3));std=rng.uniform(.2,2,32);mean=rng.normal(size=32);bias=rng.normal(size=3)
    B,A,b,right,gram=maps(B,W,std,mean,bias)
    p=rng.normal(size=(11,32));g=rng.normal(size=(11,2));y=rng.normal(size=p.shape)
    updated,c=constrain(p,g,B,A,b,right)
    Q,_=np.linalg.qr(B,mode='reduced');other=p+(g-(p@A+b))@np.linalg.pinv(Q.T@A)@Q.T
    assert np.max(np.abs(updated-other))<1e-11 and np.max(np.abs(updated@A+b-g))<1e-11
    zero,_=constrain(p,p@A+b,B,A,b,right);assert np.array_equal(zero,p)
    oracle,coeff=oracle_feature(p,y,B,gram);assert np.max(np.abs((y-oracle)@B))<1e-10
    assert np.mean((oracle-y)**2)<=np.mean((updated-y)**2)
    null=Q@rng.normal(size=(8,2));null=null-A@np.linalg.lstsq(A,null,rcond=None)[0]
    try:maps(B,np.zeros((32,3)),std,mean,bias)
    except (ValueError,np.linalg.LinAlgError):pass
    else:raise AssertionError('Rank-deficient constraint accepted')
    return dict(status='SOFTWARE_FIXTURES_PASS_NOT_SCIENTIFIC_RESULT',independent_QR_projection_exact=True,position_constraint_exact=True,zero_delta_identity=True,nonorthogonal_basis_supported=True,oracle_projection_optimality=True,rank_guard=True,research_data=False,model_forward=False,GPU=0)

if __name__=='__main__':
    import json
    print(json.dumps(main()))
