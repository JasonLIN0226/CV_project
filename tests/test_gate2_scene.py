"""Gate2 structural contracts needed before any encoder/reader experiment."""
from collections import Counter
import json
import numpy as np
import pytest

from experiments.phase2 import gate2_scene as g
from experiments.phase2.memory import MemoryEntry,StreamingMemory


@pytest.mark.parametrize("condition",g.CONDITIONS)
@pytest.mark.parametrize("index",range(12))
def test_each_axis_order_cell_has_independent_quartet_and_legal_fifo_parent(condition,index):
    s=g.scene(g.SPLIT_SEEDS["geometry"],index,"geometry",condition)
    episodes=g.validate_quartet(s)
    assert len({e["episode_id"] for e in episodes})==4
    assert len(set(s["semantic_axes"]))==2
    assert s["semantic_axes"].count(s["duplicate_axis"])==2
    assert all(e["actions"].dtype==np.float32 and e["rgb"].dtype==np.uint8 for e in episodes)
    # Exercise the real bounded FIFO implementation, not an arithmetic mock.
    stream=StreamingMemory(4,2,"fifo")
    for t in range(s["query"]):
        stream.push(MemoryEntry("structural-fixture",t,np.array([t],np.float32)))
    before=stream.snapshot
    candidates=before.retained+(before.recent[0],)
    assert [e.timestamp for e in candidates]==s["cue_times"]
    at_query=stream.push(MemoryEntry("structural-fixture",s["query"],np.array([s["query"]],np.float32)))
    assert [e.timestamp for e in at_query.recent]==s["recent_times"]
    assert stream.observations_seen==s["retention_stop"]+1
    assert at_query.transition.deleted_id[1] in s["cue_times"]


def test_style_is_matched_intervention_only_on_aprime():
    for index in range(12):
        benign=g.scene(g.SPLIT_SEEDS["geometry"],index,"geometry",g.CONDITIONS[0])
        distant=g.scene(g.SPLIT_SEEDS["geometry"],index,"geometry",g.CONDITIONS[1])
        assert benign["family_id"]==distant["family_id"]
        assert benign["scene_id"]!=distant["scene_id"]
        assert benign["actions"]==distant["actions"]
        j=benign["roles"].index("Aprime");changed=benign["cue_times"][j]
        a=benign["cue_times"][benign["roles"].index("A")]
        for bits in g.BIT_COMBINATIONS:
            left,right=g.episode(benign,bits),g.episode(distant,bits)
            assert np.array_equal(left["states"],right["states"])
            assert np.array_equal(left["actions"],right["actions"])
            differing=np.flatnonzero(np.any(left["rgb"]!=right["rgb"],axis=(1,2,3))).tolist()
            assert differing==[changed]
            assert np.array_equal(left["rgb"][a],left["rgb"][changed])
            assert not np.array_equal(right["rgb"][a],right["rgb"][changed])


def test_active_controls_explain_when_bits_begin_affecting_motion():
    s=g.scene(g.SPLIT_SEEDS["train"],13,"train")
    ep=g.episode(s,(0,1));q=s["query"]
    assert np.count_nonzero(ep["actions"][:q])==0
    assert np.all(ep["actions"][q:]>0)
    assert np.array_equal(ep["states"][:q+1],np.repeat(ep["states"][:1],q+1,axis=0))
    delta=np.diff(ep["states"][q:],axis=0)
    expected=np.array([-1,1],np.float32)*np.float32(s["motion_scale"])*ep["actions"][q:]
    assert np.allclose(delta,expected,atol=4e-8,rtol=0)


def test_locked_split_counts_balance_axes_orders_and_do_not_overlap():
    families=set();scenes=set();seeds=set()
    for split in g.SPLIT_SEEDS:
        m=g.manifest(split)
        assert m["quartets"]==2*g.FAMILIES_PER_SPLIT[split]
        assert m["episodes"]==8*g.FAMILIES_PER_SPLIT[split]
        counts=Counter((s["condition"],s["duplicate_axis"],s["role_permutation_index"]) for s in m["records"])
        assert len(counts)==24
        assert set(counts.values())=={g.FAMILIES_PER_SPLIT[split]//12}
        these_families={s["family_id"] for s in m["records"]}
        these_scenes={s["scene_id"] for s in m["records"]}
        assert not families&these_families and not scenes&these_scenes
        families|=these_families;scenes|=these_scenes;seeds.add(m["seed"])
        # Full public scene parameters can be frozen as JSON, without a bit.
        assert "bits" not in m["records"][0]
        assert json.loads(json.dumps(m))==m
    assert len(seeds)==4
    assert "confirmation" not in g.SPLIT_SEEDS


def test_invalid_seed_or_bits_fail_before_rendering():
    with pytest.raises(ValueError):g.scene(20261028,0,"development")
    with pytest.raises(ValueError):g.scene(20261030,-1,"geometry")
    with pytest.raises(ValueError):g.manifest("geometry",13)
    s=g.scene(20261030,0,"geometry")
    with pytest.raises(ValueError):g.episode(s,(0,2))
    with pytest.raises(ValueError):g.episode(s,(1,))
