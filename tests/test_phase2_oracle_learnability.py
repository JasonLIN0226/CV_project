"""Causal feature, train split and exact-state ambiguity contracts."""
import numpy as np
import pytest

from experiments.phase2.diagnose_oracle_learnability import (
    RidgeProbe, ambiguity_report, causal_features, parent_from_arrays, parent_hashes, paired_summary,
)


def example():
    z = np.arange(39, dtype=np.float32).reshape(13,3)/10
    actions = np.ones((12,2), np.float32)
    basis = {"input_mean":np.zeros(3,np.float32), "projection":np.eye(3,dtype=np.float32)}
    return z, actions, basis


def test_causal_feature_and_hash_do_not_read_future():
    z, actions, basis = example()
    a,b,p = parent_from_arrays({"id":"one"},z,actions,6,[0,1,2])
    x = causal_features(a,b,6,basis,p)
    h = parent_hashes(z,actions,6,[0,1,2],x)
    future_z, future_actions = z.copy(), actions.copy()
    future_z[7:] += 99; future_actions[6:] *= -3
    c,d,q = parent_from_arrays({"id":"one"},future_z,future_actions,6,[0,1,2])
    y = causal_features(c,d,6,basis,q)
    g = parent_hashes(future_z,future_actions,6,[0,1,2],y)
    np.testing.assert_array_equal(x,y)
    for key in ("full_observed_prefix","bounded_observed_state","probe_inputs"):
        assert h[key] == g[key]
    assert h["future_targets"] != g["future_targets"]
    assert h["future_actions"] != g["future_actions"]


def test_ridge_train_only_and_candidate_equivariance():
    rng=np.random.default_rng(1)
    x=rng.normal(size=(10,3,5)); y=x[:,:,0]+2*x[:,:,2]
    m=[{"split":"train"} for _ in x]
    fit=RidgeProbe.fit(x,y,m,"cost")
    order=[2,0,1]
    np.testing.assert_allclose(fit.predict(x[0][order]),fit.predict(x[0])[order],atol=1e-12)
    with pytest.raises(ValueError,match="TRAIN"):
        RidgeProbe.fit(x,y,[{"split":"fresh_development"} for _ in x],"cost")
    assert fit.bytes == 5*8*2+8


def test_alias_conflict_requires_same_observable_state_and_nonoverlap_optima():
    def row(eid,loss,state="same"):
        return {"episode_id":eid,"candidate_times":[0,1,2],"losses":loss,
                "hashes":{"bounded_observed_state":state,"future_targets":eid,"future_actions":"fixed"}}
    conflict=ambiguity_report([row("a",[0.,1.,2.]),row("b",[1.,0.,2.])],"bounded_observed_state")
    assert conflict["strict_optimal_action_conflicts"] == 1
    assert conflict["empirical_shared_action_regret_on_repeated_rows"] == .5
    no_conflict=ambiguity_report([row("a",[0.,0.,2.]),row("b",[1.,0.,2.])],"bounded_observed_state")
    assert no_conflict["strict_optimal_action_conflicts"] == 0
    different_state=ambiguity_report([row("a",[0.,1.,2.]),row("b",[1.,0.,2.],"different")],"bounded_observed_state")
    assert different_state["cross_episode_repeated_states"] == 0


def test_future_or_invalid_candidate_is_rejected():
    z,actions,_=example()
    with pytest.raises(ValueError,match="historical"):
        parent_from_arrays({"id":"one"},z,actions,6,[0,1,4])


def test_pair_statistics_average_parents_then_siblings():
    rows=[{"pair":"a","episode_id":"one","condition":"id","value":0.},
          {"pair":"a","episode_id":"one","condition":"id","value":0.},
          {"pair":"a","episode_id":"two","condition":"id","value":9.}]
    assert paired_summary(rows,["value"],["all"])["all"]["value"]["mean"] == 4.5
