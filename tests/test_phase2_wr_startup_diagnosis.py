"""No-new-information prefix and startup admission counterfactual contracts."""
import numpy as np
import pytest
from types import SimpleNamespace

from experiments.phase2.diagnose_wr_startup import initial_prefix, paired_ci
from experiments.phase2.run_literature_baselines import online_policy


def toy(cue_time=2):
    z=np.zeros((20,3),np.float32);z[cue_time]=[1,2,3]
    actions=np.arange(38,dtype=np.float32).reshape(19,2)
    return z,actions,12,np.array([cue_time])


def test_prefix_preserves_original_actions_target_and_relative_ages():
    z,a,q,e=toy()
    warm,b,shifted,cue=initial_prefix(z,a,q,e)
    np.testing.assert_array_equal(warm[:4],np.repeat(z[:1],4,axis=0))
    np.testing.assert_array_equal(warm[4:],z)
    np.testing.assert_array_equal(b[:4],np.zeros((4,2)))
    np.testing.assert_array_equal(b[4:],a)
    np.testing.assert_array_equal(warm[shifted+np.array([3,6])],z[q+np.array([3,6])])
    assert shifted-int(cue[0])==q-int(e[0])
    np.testing.assert_array_equal(b[int(cue[0]):shifted],a[int(e[0]):q])
    assert not np.shares_memory(warm,z)


def retained_cue(z,a,q,e):
    # WR and FIFO do not read model scores, so no predictor is needed here.
    def no_model(*args):
        raise AssertionError("WR must not query a predictor")
    online=online_policy("toy","wr",SimpleNamespace(attention=no_model))
    for t in range(q+1):
        online.observe(z[t],executed_action=None if t==0 else a[t-1])
    return int(e[0]) in [x.timestamp for x in online.stream.snapshot.retained]


def test_identical_prefix_enables_early_event_without_changing_wr_parameters():
    cold=toy(2)
    assert not retained_cue(*cold)
    assert retained_cue(*initial_prefix(*cold))
    late=toy(4)
    assert retained_cue(*late)
    assert retained_cue(*initial_prefix(*late))


def test_invalid_prefix_and_pair_level_statistics():
    z,a,q,e=toy()
    with pytest.raises(ValueError):initial_prefix(z,a,q,e,length=-1)
    with pytest.raises(ValueError):initial_prefix(z.astype(np.float64),a,q,e)
    result=paired_ci([0,2,10,12],["a","a","b","b"])
    assert result["pairs"]==2 and result["mean"]==6
