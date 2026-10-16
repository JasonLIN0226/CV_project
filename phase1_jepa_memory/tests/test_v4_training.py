"""Tiny paired smoke integration only; no research eligibility or HPC work."""
import copy
import shutil
import signal
from pathlib import Path

import pytest
import torch

from phase1.cache import encode, fit_normalization
from phase1.config import DEFAULT, SPLITS
from phase1.data import generate, validate_data
from phase1.training import train, trusted_checkpoint


@pytest.fixture(scope="module")
def cached(tmp_path_factory):
    c = copy.deepcopy(DEFAULT)
    c["output"] = str(tmp_path_factory.mktemp("v4 paired smoke"))
    c["data"]["pairs"] = dict.fromkeys(SPLITS, 2)
    c["model"].update(hidden=16, heads=4, rank=8, dropout=.05)
    c["training"].update(epochs=2, seeds=[17], batch_size=4, pair_weight=2., rank_weight=.5)
    generate(c)
    validate_data(c)
    encode(c)
    fit_normalization(c)
    return c


def arm(cached, root, role="candidate", legacy=False):
    c = copy.deepcopy(cached)
    c["output"] = str(root)
    c["experiment"] = {"stage": "development", "arm": "reference", "tasks": ["gate"]} if legacy else {
        "protocol_version": 4, "role": role, "stage": "development", "tasks": ["gate"],
        "variant": "reference" if role == "reference" else "gate_common_anchor_v1",
        "common_anchor_weight": 0 if role == "reference" else 1, "matched_teacher_forwards": True}
    for part in ("dataset", "features"):
        shutil.copytree(Path(cached["output"])/part, root/part)
    return c


def checkpoint(c, name="latest.pt"):
    return trusted_checkpoint(Path(c["output"])/"predictor/gate/seed_17"/name)


def test_matched_teacher_reference_exactly_preserves_legacy_trajectory_and_rng(cached, tmp_path):
    original, reference, candidate = [arm(cached, tmp_path/name, role=role, legacy=old)
        for name, role, old in (("old", "reference", True), ("reference", "reference", False), ("candidate", "candidate", False))]
    for c in (original, reference, candidate):
        train(c)
    old, ref, cand = map(checkpoint, (original, reference, candidate))
    assert all(torch.equal(old["model"][key], ref["model"][key]) for key in old["model"])
    assert torch.equal(old["rng"]["torch"], ref["rng"]["torch"])
    assert any(not torch.equal(ref["model"][key], cand["model"][key]) for key in ref["model"])
    assert old["step"] == ref["step"] == cand["step"] == 8
    assert ref["extra_teacher_forward_accounting"] == cand["extra_teacher_forward_accounting"] == {
        "extra_forward_calls": 6, "extra_forward_examples": 24}
    for ck in (ref, cand):
        assert ck["v4_training_authorization"]["research_authorized"] is False
        assert ck["discarded_teacher_forward_accounting"] == {"calls": 0, "examples": 0}
        assert len(ck["logs"]) == 2 and ck["checkpoint_selection"] == "validation_mse"
        assert all(row["selection_score"] == row["validation_mse"] for row in ck["logs"])
        assert all(torch.isfinite(value).all() for value in ck["model"].values())
        assert ck["best"] == min(row["validation_mse"] for row in ck["logs"])
    assert "extra_teacher_forward_accounting" not in old  # Legacy schema unaffected.
    from phase1.v4_pipeline import gate_all_models_completion, require_gate_evaluation_ready
    complete = gate_all_models_completion(reference, candidate)
    assert complete["complete"]
    assert require_gate_evaluation_ready(reference, complete) is complete
    incomplete = copy.deepcopy(complete)
    incomplete["roles"]["candidate"]["complete"] = False
    with pytest.raises(ValueError, match="Both actual Gate"):
        require_gate_evaluation_ready(reference, incomplete)


def test_candidate_exact_boundary_resume_preserves_accounting_optimizer_and_model(cached, tmp_path):
    full, resumed = arm(cached, tmp_path/"full"), arm(cached, tmp_path/"resume")
    train(full)
    train(resumed, stop_after_epoch=1)
    assert checkpoint(resumed)["extra_teacher_forward_accounting"]["extra_forward_calls"] == 3
    train(resumed)
    a, b = checkpoint(full), checkpoint(resumed)
    assert a["extra_teacher_forward_accounting"] == b["extra_teacher_forward_accounting"]
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    assert a["scheduler"] == b["scheduler"] and torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert all(torch.equal(value, b["optimizer"]["state"][key][name])
               for key, state in a["optimizer"]["state"].items() for name, value in state.items() if torch.is_tensor(value))


def test_partial_epoch_resume_records_discarded_actual_teacher_cost(cached, tmp_path, monkeypatch):
    import phase1.common_anchor as common
    full, resumed = arm(cached, tmp_path/"complete"), arm(cached, tmp_path/"interrupted")
    train(full)
    original, fired = common.recent_teacher_prediction, [False]
    def terminate_once(*args, **kwargs):
        result = original(*args, **kwargs)
        if not fired[0]:
            fired[0] = True
            signal.raise_signal(signal.SIGTERM)
        return result
    monkeypatch.setattr(common, "recent_teacher_prediction", terminate_once)
    result = train(resumed)
    assert result["interrupted"] and checkpoint(resumed)["epoch"] == 0
    assert checkpoint(resumed)["discarded_teacher_forward_accounting"] == {"calls": 1, "examples": 4}
    train(resumed)
    a, b = checkpoint(full), checkpoint(resumed)
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    assert b["step"] == 8 and b["extra_teacher_forward_accounting"] == {"extra_forward_calls": 7, "extra_forward_examples": 28}
    assert b["extra_teacher_forward_accounting"]["extra_forward_calls"]-b["discarded_teacher_forward_accounting"]["calls"] == 6


def test_research_authorization_missing_blocks_before_features_or_science(cached, monkeypatch):
    import phase1.training as training
    c = copy.deepcopy(cached)
    c.update(mode="confirm")
    c["encoder"]["backend"] = "ijepa"
    c["experiment"] = {"protocol_version": 4, "role": "candidate", "stage": "development", "tasks": ["gate"],
        "variant": "gate_common_anchor_v1", "common_anchor_weight": 1, "matched_teacher_forwards": True}
    monkeypatch.setattr(training, "FeatureStore", lambda *args: pytest.fail("Unauthorized research reached cached features"))
    with pytest.raises(ValueError, match="actual diagnostic/source/audit"):
        train(c)


@pytest.mark.parametrize("change", ["rank", "weight", "matched", "task", "stage"])
def test_invalid_v4_profile_rejected_before_science(cached, change):
    c = copy.deepcopy(cached)
    c["experiment"] = {"protocol_version": 4, "role": "candidate", "stage": "development", "tasks": ["gate"],
        "variant": "gate_common_anchor_v1", "common_anchor_weight": 1, "matched_teacher_forwards": True}
    if change == "rank": c["training"]["rank_weight"] = 0
    if change == "weight": c["experiment"]["common_anchor_weight"] = .5
    if change == "matched": c["experiment"]["matched_teacher_forwards"] = False
    if change == "task": c["experiment"]["tasks"] = ["motion"]
    if change == "stage": c["experiment"]["stage"] = "confirmation"
    with pytest.raises(ValueError):
        train(c)
