import numpy as np
import pytest

from experiments.phase2.run_literature_baselines import QueryOracle, aggregate, evidence_equivalence


class ConflictingHorizons:
    def predict(self, recent, history, past, actions, horizon):
        times = tuple(sorted(e.timestamp for e in history))
        # Different horizon minima: choosing each independently would be invalid.
        losses = {(0,1):(0.,16.), (0,2):(9.,0.), (1,2):(4.,4.)}
        return np.array([np.sqrt(losses[times][horizon == 6])], np.float32)


def test_oracle_selects_one_subset_for_both_horizons():
    q = QueryOracle("episode", np.zeros((13,1), np.float32),
                    np.zeros((12,2), np.float32), 6, ConflictingHorizons())
    result = q.best([0,1,2])
    assert result["kept_times"] == [1,2]
    assert result["mean_mse"] == 4.
    assert result["subsets_evaluated"] == 3
    assert len(q.cache) == 3
    with pytest.raises(ValueError):
        q.evaluate([2,3])  # t=3 belongs to recent, not long-term history.
    with pytest.raises(ValueError):
        q.evaluate([1,1])


def test_seed_replicates_do_not_inflate_independent_pairs():
    rows = []
    for pair in ("a","b"):
        for seed in (17,23,41):
            rows.append(dict(policy="pafu",pair=pair,seed=seed,mean_mse=2.,
                local_regret=1.,hindsight_regret=1.,evidence_survived=False,
                controller_parameter_bytes=1,online_seconds=.01,
                logical_memory_maxima={"total_logical_bytes":50}))
    result = aggregate(rows)[0]
    assert result["independent_pairs"] == 2
    assert result["episode_seed_rows"] == 6
    assert result["pair_bootstrap_hindsight_regret_ci95"] == [1.,1.]


def test_evidence_retention_counts_identical_cue_copies_not_only_one_timestamp():
    z = np.array([[0.,0.],[1.,2.],[1.,2.],[0.,0.],[1.,2.]], np.float32)
    assert evidence_equivalence(z, [1], 4) == [[1,2]]
