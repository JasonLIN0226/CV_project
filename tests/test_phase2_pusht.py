"""CPU boundary checks; model training/planning is not implied by these tests."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "phase2_pusht", Path(__file__).resolve().parents[1]/"experiments/phase2/pusht.py")
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)


class PixelEnv:
    def __init__(self, terminate_at=100):
        self.t, self.terminate_at = 0, terminate_at

    def reset(self, seed):
        self.t = 0
        return np.full((12, 12, 3), 200, np.uint8), {"block_pose": np.array([9., 8., 7.])}

    def step(self, action):
        self.t += 1
        self.action = action.copy()
        return np.full((12, 12, 3), 200+self.t, np.uint8), .2, self.t >= self.terminate_at, False, {"state": 99}

    def render(self):
        raise AssertionError("Pixel observation contract must never call render")

    def close(self):
        pass


def test_mask_before_encoder_and_diagnostics_separation():
    adapter = p.PushTVisualAdapter(PixelEnv(), p.Occlusion(start=1, stop=3))
    initial, diagnostic = adapter.reset(seed=17)
    hidden, _ = adapter.step(np.array([120, 330]))
    assert vars(initial).keys() == {"timestamp", "rgb"}
    assert "block_pose" in diagnostic.info
    assert not hidden.rgb.any() and initial.rgb.min() == 200
    class SpyEncoder:
        def encode(self, rgb):
            self.received = rgb.copy()
            return rgb.mean((1, 2))
    encoder = SpyEncoder()
    result = p.encode_observations(encoder, [hidden])
    assert not encoder.received.any() and not result.any()
    with pytest.raises(TypeError):
        p.encode_observations(encoder, [diagnostic])
    with pytest.raises(ValueError):
        hidden.rgb[0, 0] = 2


def test_action_is_absolute_and_roundtrips_without_summing():
    sequence = np.array([[0., 512.], [256., 128.]], np.float32)
    normalized = p.normalize_actions(sequence)
    np.testing.assert_array_equal(normalized, [[-1., 1.], [0., -.5]])
    np.testing.assert_array_equal(p.denormalize_actions(normalized), sequence)
    for illegal in ([[513., 1.]], [[np.nan, 1.]], [1., 2., 3.]):
        with pytest.raises(ValueError):
            p.absolute_actions(illegal)
    adapter = p.PushTVisualAdapter(PixelEnv(), p.Occlusion(enabled=False))
    adapter.reset(seed=1)
    adapter.step(sequence[0])
    adapter.step(sequence[1])
    np.testing.assert_array_equal(adapter._env.action, sequence[1])


def test_mask_schedule_and_later_visible_target():
    schedule = p.Occlusion()
    rgb = np.full((8, 8, 3), 200, np.uint8)
    observed = [p.VisualObservation(t, schedule.apply(rgb, t)) for t in range(20)]
    assert observed[7].rgb.any() and not observed[8].rgb.any()
    assert not observed[12].rgb.any() and observed[13].rgb.any()
    rows = p.later_visible_targets(observed, schedule)
    assert {"query": 11, "horizon": 3, "target": 14} in rows
    assert not p.later_visible_targets(observed[:12], schedule)
    with pytest.raises(ValueError, match="Duplicate"):
        p.later_visible_targets(observed + [observed[0]], schedule)


def test_rectangular_occlusion_does_not_mutate_source():
    rgb = np.full((8, 8, 3), 200, np.uint8)
    schedule = p.Occlusion(start=0, stop=1, rectangle=(.25, .25, .75, .75), fill=17)
    masked = schedule.apply(rgb, 0)
    assert (rgb == 200).all()
    assert (masked[2:6, 2:6] == 17).all()
    assert (masked[0] == 200).all()
    with pytest.raises(ValueError):
        p.Occlusion(rectangle=(0., 0., 2., 1.))


def test_termination_and_reset():
    adapter = p.PushTVisualAdapter(PixelEnv(terminate_at=1), p.Occlusion())
    with pytest.raises(RuntimeError):
        adapter.step([100, 100])
    adapter.reset(seed=1)
    _, diagnostic = adapter.step([100, 100])
    assert diagnostic.terminated
    with pytest.raises(RuntimeError):
        adapter.step([100, 100])
    observed, _ = adapter.reset(seed=1)
    assert observed.timestamp == 0


def test_control_replay_and_no_latent_only_mask():
    factory = lambda schedule: p.PushTVisualAdapter(PixelEnv(), schedule)
    observed, control, actions, diagnostics, targets = p.collect_paired_trace(factory=factory)
    assert targets and len(actions) == 24 and len(diagnostics) == 25
    assert all(np.array_equal(observed[t].rgb, control[t].rgb) for t in (0, 7, 13, 24))
    assert control[9].rgb.any() and not observed[9].rgb.any()


def test_actual_gym_pusht_rgb_smoke_when_dependency_available():
    pytest.importorskip("gym_pusht")
    observed, control, actions, diagnostics, targets = p.collect_paired_trace()
    assert observed[0].rgb.shape == (96, 96, 3)
    assert observed[0].rgb.dtype == np.uint8
    assert actions.shape[1] == 2 and targets
    assert len(observed) == len(control) == len(diagnostics)
    assert all(o.rgb.any() for o in control)
