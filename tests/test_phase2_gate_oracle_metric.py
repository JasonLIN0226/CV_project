import numpy as np
import pytest

from experiments.phase2.diagnose_gate_oracle_metric import paired_components, fit_readout, decode, group_ci
from experiments.phase2.diagnose_gate_reader_local import cue_partition


def test_contrast_split_distinguishes_wrong_branch_from_orthogonal_error():
    y=np.array([[1.,0.],[1.,0.]])
    other=-y
    p=np.array([[1.,2.],[-1.,0.]])
    out=paired_components(p,y,other)
    assert out['mse']==2
    assert out['contrast_mse']==1
    assert out['orthogonal_mse']==1
    assert out['feature_branch_correct']==.5
    np.testing.assert_array_equal(out['contrast_amplitude'],[1,-1])
    with pytest.raises(ValueError): paired_components(p,y,y)


def test_fixed_readout_uses_training_statistics_and_handles_constant_features():
    x=np.array([[0.,7.],[1.,7.],[2.,7.]])
    readout=fit_readout(x,np.array([[0.],[1.],[2.]]))
    before={k:v.copy() for k,v in readout.items()}
    assert np.isfinite(decode(np.array([[100.,7.]]),readout)).all()
    for k in readout: np.testing.assert_array_equal(readout[k],before[k])
    np.testing.assert_array_equal(readout['mean'],[1,7])


def test_confidence_interval_uses_pairs_not_repeated_rows():
    rows=[{'pair':'a','x':0},{'pair':'b','x':2},{'pair':'b','x':2}]
    result=group_ci(rows,'x')
    assert result['mean']==1 and result['pairs']==2


def test_local_deletion_cannot_remove_two_equivalent_cue_copies():
    losses={(0,1):.1,(0,2):.2,(1,2):.3}
    yes,no=cue_partition([0,1,2],{0,1},losses)
    assert yes==(.1,(0,1)) and no is None
    yes,no=cue_partition([0,1,2],{0},losses)
    assert yes==(.1,(0,1)) and no==(.3,(1,2))
