import sys
from pathlib import Path
import random
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from v5_retention import HistoryState, insert
from v5_deletion_continuation import common_reservoir_continuations


def test_all17_equal_clock_capacity_common32_and_no_reentry():
    state=HistoryState(16,tuple(range(16)),16,15)
    draws=tuple(random.Random(17+i).random() for i in range(32))
    forks=common_reservoir_continuations(state,16,draws,24)
    assert {x[0] for x in forks}==set(range(17))
    for deleted, trajectory in forks:
        assert len(trajectory)==33
        dropped={deleted}
        for offset, snapshot in enumerate(trajectory):
            assert snapshot.seen==17+offset and snapshot.last_incoming==16+offset
            assert len(snapshot.retained)==16
            assert not dropped.intersection(snapshot.retained)
            assert all(x<=16+offset for x in snapshot.retained)
            if offset<32:
                expected, loss=insert(snapshot,17+offset,'Reservoir',draws[offset])
                assert expected==trajectory[offset+1]
                if loss is not None:dropped.add(loss)


def test_exact_common_rejection_and_supplied_draw_sequence_immutable():
    state=HistoryState(16,tuple(range(16)),16,15)
    draws=[.999]*32
    result=common_reservoir_continuations(state,16,draws,24)
    assert draws==[.999]*32
    for deleted, trajectory in result:
        assert all(s.retained==trajectory[0].retained for s in trajectory)
        assert deleted not in trajectory[-1].retained


@pytest.mark.parametrize('overrides',[
    {'current_step':25}, {'steps':31}, {'episode_length':257},
    {'recent':4}, {'current_step':224,'incoming':216},
])
def test_wrong_clocks_unregistered_lengths_and_episode_overrun_fail(overrides):
    args={'state':HistoryState(16,tuple(range(16)),16,15),
          'incoming':16,'draws':[.5]*32,'current_step':24}
    args.update(overrides)
    with pytest.raises(ValueError):common_reservoir_continuations(**args)


@pytest.mark.parametrize('draw',[float('nan'),float('inf'),-0.1,1.0])
def test_nonfinite_or_out_of_range_shared_draw_fails(draw):
    with pytest.raises(ValueError):
        common_reservoir_continuations(HistoryState(16,tuple(range(16)),16,15),16,[draw]*32,24)


def test_last_continuation_h64_stays_in_episode_without_truncation():
    # At159, the last continuation step191 has h64 target255 (last frame).
    state=HistoryState(16,tuple(range(135,151)),151,150)
    result=common_reservoir_continuations(state,151,[.999]*32,159)
    assert len(result)==17 and all(len(path)==33 for _,path in result)
    # At160, observations through192 still fit, but h64 target256 does not.
    # Reject the complete state instead of omitting/shortening that target.
    state=HistoryState(16,tuple(range(136,152)),152,151)
    with pytest.raises(ValueError):
        common_reservoir_continuations(state,152,[.999]*32,160)
