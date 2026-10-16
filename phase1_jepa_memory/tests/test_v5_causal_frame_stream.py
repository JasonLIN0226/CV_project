import sys
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v5_causal_frame_stream import causal_frames


@dataclass
class Frame:
    pts: object
    time_base: object = F(1, 60)
    is_corrupt: bool = False


def test_future_pixels_do_not_enter_selected_observation():
    first = Frame(0)
    later = Frame(1)
    chosen = list(causal_frames([first, later], [F(1, 120)]))
    assert chosen == [(F(1, 120), F(0), first)]


def test_stop_after_last_requested_clock_without_reading_full_future():
    calls = []
    def frames():
        for i in range(10000):
            calls.append(i)
            yield Frame(i)
    chosen = list(causal_frames(frames(), [F(0), F(1, 2)]))
    assert [stamp for _, stamp, _ in chosen] == [F(0), F(1, 2)]
    assert max(calls) == 31


def test_exact_last_frame_at_eof_is_valid():
    last = Frame(30)
    assert list(causal_frames([Frame(0), last], [F(1, 2)])) == [(F(1, 2), F(1, 2), last)]


@pytest.mark.parametrize("frames, clocks", [
    ([Frame(0)], [F(1, 60)]),
    ([Frame(0), Frame(60)], [F(1, 2)]),
    ([Frame(None)], [F(0)]),
    ([Frame(0, is_corrupt=True)], [F(0)]),
    ([Frame(1), Frame(0)], [F(0)]),
    ([Frame(0)], [F(0), F(0)]),
])
def test_missing_stale_corrupt_reordered_or_reindexed_input_fails(frames, clocks):
    with pytest.raises(ValueError):
        list(causal_frames(frames, clocks))
