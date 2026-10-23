"""Collection/data-boundary tests, not predictor or planning validation."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_pusht_planning_data", Path(__file__).resolve().parents[1]
    / "experiments/phase2/prepare_pusht_planning_data.py")
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)


class PixelEnv:
    """State/reward are deliberately unrelated to the fixed collection actions."""
    def __init__(self, steps=20, diagnostic_value=1234567, pixel_offset=0):
        self.t, self.steps = 0, steps
        self.diagnostic_value, self.pixel_offset = diagnostic_value, pixel_offset
        self.closed = False
        self.actions = []

    def pixels(self):
        return np.full((96, 96, 3), 100+self.t+self.pixel_offset, dtype=np.uint8)

    def info(self):
        return {"block_pose": np.array([self.diagnostic_value, 2., 3.]),
                "coverage": self.diagnostic_value, "is_success": False}

    def reset(self, *, seed):
        self.t = 0
        return self.pixels(), self.info()

    def step(self, action):
        self.actions.append(action.copy())
        self.t += 1
        return self.pixels(), float(self.diagnostic_value), self.t == self.steps, False, self.info()

    def render(self):
        raise AssertionError("Collection must use reset/step pixels")

    def close(self):
        self.closed = True


def one_episode():
    record = copy.deepcopy(p.make_protocol()["episodes"][0])
    record["occlusion"] = {"start": 8, "stop": 13, "rectangle": [0., 0., 1., 1.],
                            "fill": 0, "enabled": True}
    return record


def test_frozen_seed_split_protocol_and_action_units():
    protocol = p.make_protocol()
    assert protocol == p.make_protocol()
    assert protocol["split_counts"] == {"train": 32, "validation": 8, "evaluation": 8}
    assert len({e["environment_seed"] for e in protocol["episodes"]}) == 48
    assert len({e["policy_seed"] for e in protocol["episodes"]}) == 48
    for record in protocol["episodes"]:
        action = p.absolute_action_schedule(record["policy_seed"])
        assert action.shape == (96, 2) and action.dtype == np.float32
        assert action.min() >= 40 and action.max() <= 472
        assert p.array_sha(action) == record["planned_actions_sha256"]
        np.testing.assert_allclose(p.normalize_actions(action), action/256 - 1)
    assert not np.array_equal(p.absolute_action_schedule(1), p.absolute_action_schedule(2))


def test_observation_mask_diagnostics_separation_and_no_reward_conditioning():
    record = one_episode()
    def collect(value):
        envs = []
        def factory(mask):
            env = PixelEnv(diagnostic_value=value)
            envs.append(env)
            return p.PushTVisualAdapter(env, mask)
        result = p.collect_episode(record, factory=factory)
        assert all(env.closed for env in envs)
        return result, envs
    low, low_envs = collect(-100)
    high, high_envs = collect(999999)
    assert set(low["observations"]) == p.FIELDS
    assert set(low["control"]) == p.FIELDS
    assert low["diagnostics"]["steps"][0]["info"]["block_pose"][0] == -100
    assert high["diagnostics"]["steps"][0]["info"]["block_pose"][0] == 999999
    assert not low["observations"]["rgb"][8:13].any()
    assert low["control"]["rgb"][8:13].min() > 0
    np.testing.assert_array_equal(low_envs[0].actions, high_envs[0].actions)
    np.testing.assert_array_equal(low["observations"]["actions_absolute"],
                                  p.absolute_action_schedule(record["policy_seed"])[:20])
    assert low["observations"]["rgb"].shape[0] == 21
    assert {"query": 11, "horizon": 6, "target": 17} in low["diagnostics"]["later_visible_targets"]


def test_changed_policy_refused_before_creating_environment():
    record = one_episode()
    record["planned_actions_sha256"] = "wrong"
    with pytest.raises(ValueError, match="frozen protocol"):
        p.collect_episode(record, factory=lambda _: pytest.fail("Environment must not be created"))


def test_roundoff_tolerance_does_not_hide_state_or_termination_mismatch():
    env = p.PushTVisualAdapter(PixelEnv(), p.Occlusion(enabled=False))
    _, diagnostic = env.reset(seed=17)
    base = replace(diagnostic, reward=.123)
    close = replace(base, reward=np.nextafter(.123, 1.))
    assert 0 < p.paired_diagnostic_difference(base, close) < 1e-12
    with pytest.raises(AssertionError, match="physical diagnostics"):
        p.paired_diagnostic_difference(base, replace(base, reward=.123+1e-8))
    with pytest.raises(AssertionError, match="physical diagnostics"):
        p.paired_diagnostic_difference(base, replace(base, terminated=True))
    with pytest.raises(AssertionError, match="physical diagnostics"):
        p.paired_diagnostic_difference(base, replace(base, info={**base.info, "coverage": 1234568}))


def test_control_divergence_rejected_and_environments_closed():
    record = one_episode()
    envs = []
    def factory(mask):
        env = PixelEnv(pixel_offset=0 if mask.enabled else 1)
        envs.append(env)
        return p.PushTVisualAdapter(env, mask)
    with pytest.raises(AssertionError, match="Paired RGB"):
        p.collect_episode(record, factory=factory)
    assert len(envs) == 2 and all(env.closed for env in envs)


@pytest.fixture
def saved_episode(tmp_path):
    record = one_episode()
    data = p.collect_episode(record, factory=lambda mask: p.PushTVisualAdapter(PixelEnv(), mask))
    for name in ("observations", "unoccluded_control", "evaluation_only"):
        (tmp_path/name).mkdir()
    obs, control = tmp_path/"observations/one.npz", tmp_path/"unoccluded_control/one.npz"
    np.savez_compressed(obs, **data["observations"])
    np.savez_compressed(control, **data["control"])
    # Deliberately invalid diagnostics demonstrate the audit never opens these as inputs.
    (tmp_path/"evaluation_only/one.json").write_text("NOT MODEL INPUT")
    index = [{**{k: record[k] for k in ("id", "split", "environment_seed", "policy_seed")},
              "frames": 21, "observations": "observations/one.npz", "observations_sha256": p.sha256(obs),
              "unoccluded_control": "unoccluded_control/one.npz", "unoccluded_control_sha256": p.sha256(control)}]
    protocol = p.make_protocol()
    protocol["episodes"] = [record]
    protocol["split_counts"] = {"train": 1, "validation": 0, "evaluation": 0}
    return tmp_path, protocol, index, data


def test_audit_excludes_diagnostics_and_catches_leaked_state(saved_episode):
    root, protocol, index, data = saved_episode
    audit = p.audit_dataset(root, protocol, index)
    assert audit["passed"] and not audit["diagnostics_loaded"]
    assert not audit["evaluation_quality_measured"]
    obs = root/index[0]["observations"]
    np.savez_compressed(obs, **data["observations"], block_pose=np.ones((21, 3)))
    index[0]["observations_sha256"] = p.sha256(obs)
    with pytest.raises(ValueError, match="Forbidden/missing"):
        p.audit_dataset(root, protocol, index)


def test_audit_rejects_action_order_and_hidden_archive_path(saved_episode):
    root, protocol, index, data = saved_episode
    obs = root/index[0]["observations"]
    changed = dict(data["observations"])
    changed["actions_absolute"] = changed["actions_absolute"][::-1].copy()
    changed["actions_normalized"] = p.normalize_actions(changed["actions_absolute"])
    np.savez_compressed(obs, **changed)
    index[0]["observations_sha256"] = p.sha256(obs)
    with pytest.raises(ValueError, match="protocol prefix"):
        p.audit_dataset(root, protocol, index)
    index[0]["observations"] = "evaluation_only/one.json"
    with pytest.raises(ValueError, match="remain in observations"):
        p.audit_dataset(root, protocol, index)


def test_protocol_written_before_first_simulator_and_existing_output_preserved(tmp_path):
    calls = []
    def factory(mask):
        assert (tmp_path/"protocol.json").is_file()
        receipt = json.loads((tmp_path/"protocol_frozen_before_collection.json").read_text())
        assert receipt["before_environment_creation"]
        assert receipt["protocol_sha256"] == p.sha256(tmp_path/"protocol.json")
        calls.append(mask.enabled)
        return p.PushTVisualAdapter(PixelEnv(steps=1), mask)
    result = p.run(tmp_path, factory=factory)
    assert len(calls) == 96 and result["frames_per_variant"] == 96
    assert not result["encoder_run"] and not result["baseline_model_planning_verified"]
    assert result["goal_image_status"] == "pending_not_rendered"
    assert not result["evaluation_quality_measured"]
    original = (tmp_path/"protocol.json").read_bytes()
    with pytest.raises(ValueError, match="new empty"):
        p.run(tmp_path, factory=factory)
    assert (tmp_path/"protocol.json").read_bytes() == original
