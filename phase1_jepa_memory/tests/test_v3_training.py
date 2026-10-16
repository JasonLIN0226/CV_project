"""Protocol checks use smoke pixels only; they are not research measurements."""
import ast
import copy
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from phase1.cache import encode, fit_normalization
from phase1.config import DEFAULT, SPLITS, scientific_hash
from phase1.data import generate, validate_data
from phase1.training import (TRAIN_CONDITIONS, experiment_tasks, rank_weight,
                             regression_loss, selection_metric, train,
                             trusted_checkpoint, validation_score)
from phase1.utils import read_json, save_torch


def test_development_reuses_exact_basis_and_preserves_source(tiny_features, tmp_path):
    from phase1.config import prepare
    from phase1.model import fit_basis
    from phase1.inputs import FeatureStore
    from phase1.experiment import prepare_shared, arm_config
    from phase1.utils import file_hash
    prepare(tiny_features)
    store = FeatureStore(tiny_features)
    for task in tiny_features['data']['tasks']:
        fit_basis(tiny_features, store, task)
    source = Path(tiny_features['output'])
    original_hashes = {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*') if p.is_file()}
    base = prepare_shared(source, tmp_path/'shared')
    arms = [arm_config(base, tmp_path/'development', 'gate', name) for name in ['reference', 'gate_no_rank']]
    for c in arms:
        assert Path(c['output'],'dataset').is_symlink()
        assert Path(c['output'],'features').is_symlink()
        assert file_hash(Path(c['output'])/'predictor/gate/basis.npz') == file_hash(source/'predictor/gate/basis.npz')
        arrays = fit_basis(c, FeatureStore(c), 'gate')
        assert len(arrays['train_ids']) == 4
        meta = read_json(Path(c['output'])/'predictor/gate/basis.json')
        assert meta['source_scientific_hash'] == scientific_hash(base)
    assert original_hashes == {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='outside the original'):
        prepare_shared(source, source/'would_overwrite')


def test_confirmation_cannot_launch_with_missing_or_rejected_development(tmp_path):
    from phase1.experiment import confirmation_config, development_report
    from phase1.utils import write_json
    c = copy.deepcopy(DEFAULT)
    c['output'] = str(tmp_path/'confirmation')
    c['mode'] = 'confirm'
    dev = tmp_path/'development'
    result = development_report(dev)
    assert not result['confirmation_ready']
    with pytest.raises(ValueError, match='Both actual development'):
        confirmation_config(c, dev)
    for task, status in [('gate','SUPPORTED'),('motion','REJECTED')]:
        write_json(dev/task/'comparison.json', {'status':status,'reasons':['fixture']})
    with pytest.raises(ValueError, match='Both actual development'):
        confirmation_config(c, dev)
    assert not Path(c['output']).exists()
    write_json(dev/'motion/comparison.json', {'status':'SUPPORTED','reasons':[]})
    # A JSON label is not evidence: recomputation finds missing actual arms.
    with pytest.raises((ValueError, FileNotFoundError)):
        confirmation_config(c, dev)


def profile(c, arm="reference", task="gate"):
    result = copy.deepcopy(c)
    result["experiment"] = {"arm": arm, "tasks": [task], "stage": "development"}
    return result


def test_ranking_ablation_changes_only_the_ranking_term():
    base = copy.deepcopy(DEFAULT)
    base["training"].update(pair_weight=2.0, rank_weight=0.5)
    reference, ablation = profile(base), profile(base, "gate_no_rank")
    target = torch.tensor([[[0., 1.], [1., 2.]], [[2., -1.], [3., 0.]]])
    prediction = torch.tensor([[[.4, .7], [1.4, 1.7]], [[1.7, -.6], [2.7, .4]]], requires_grad=True)
    original = regression_loss(prediction, target, 2, rank_weight(reference, "gate"))
    no_rank = regression_loss(prediction, target, 2, rank_weight(ablation, "gate"))
    ranking = regression_loss(prediction, target, 0, 1)[0] - original[1]
    assert torch.equal(original[1], no_rank[1]) and torch.equal(original[2], no_rank[2])
    assert torch.allclose(original[0]-no_rank[0], 0.5*ranking)
    assert torch.equal(no_rank[0], no_rank[1]+2*no_rank[2])
    assert rank_weight(profile(base, "motion_wide_span", "motion"), "motion") == 0.5
    confirm = copy.deepcopy(base)
    confirm["experiment"] = {"arm": "confirm_candidate", "tasks": ["gate", "motion"], "stage": "confirmation"}
    assert rank_weight(confirm, "gate") == 0 and rank_weight(confirm, "motion") == 0.5
    assert scientific_hash(reference) != scientific_hash(ablation)


def test_checkpoint_score_is_the_same_mse_for_every_arm():
    # Training-loss order disagrees with MSE order. New arms must all prefer B.
    a = {condition: {"mse": 0.4+i*.1, "loss": .01, "pair_mse": .1}
         for i, condition in enumerate(TRAIN_CONDITIONS)}
    b = {condition: {"mse": 0.2+i*.1, "loss": 2., "pair_mse": .1}
         for i, condition in enumerate(TRAIN_CONDITIONS)}
    configs = [profile(DEFAULT), profile(DEFAULT, "gate_no_rank"),
               profile(DEFAULT, "reference", "motion"), profile(DEFAULT, "motion_wide_span", "motion")]
    confirm = copy.deepcopy(DEFAULT)
    confirm["experiment"] = {"arm": "confirm_candidate", "tasks": ["gate", "motion"], "stage": "confirmation"}
    configs.append(confirm)
    for c in configs:
        assert selection_metric(c) == "validation_mse"
        assert validation_score(c, a) == pytest.approx(.55)
        assert validation_score(c, b) == pytest.approx(.35)
        assert validation_score(c, b) < validation_score(c, a)
    assert selection_metric(DEFAULT) == "validation_loss"
    assert validation_score(DEFAULT, a) < validation_score(DEFAULT, b)
    bad = copy.deepcopy(a)
    bad["full_history"]["mse"] = float("nan")
    with pytest.raises(RuntimeError, match="Non-finite validation"):
        validation_score(configs[0], bad)


@pytest.mark.parametrize("experiment", [
    {"arm": "unknown", "tasks": ["gate"], "stage": "development"},
    {"arm": "gate_no_rank", "tasks": ["motion"], "stage": "development"},
    {"arm": "motion_wide_span", "tasks": ["gate"], "stage": "development"},
    {"arm": "reference", "tasks": ["gate", "gate"], "stage": "development"},
    {"arm": "reference", "tasks": ["hidden_label"], "stage": "development"},
    {"arm": "confirm_candidate", "tasks": ["gate"], "stage": "confirmation"},
    {"arm": "gate_no_rank", "tasks": ["gate"], "stage": "confirmation"},
])
def test_invalid_arm_task_or_stage_rejected_before_training(experiment):
    c = copy.deepcopy(DEFAULT)
    c["experiment"] = experiment
    with pytest.raises(ValueError):
        train(c)


@pytest.fixture(scope="module")
def tiny_features(tmp_path_factory):
    c = copy.deepcopy(DEFAULT)
    c["output"] = str(tmp_path_factory.mktemp("v3 tiny cached smoke features"))
    c["data"]["pairs"] = dict.fromkeys(SPLITS, 2)
    c["model"].update(hidden=32, layers=1, rank=16, dropout=.05)
    c["training"].update(epochs=3, seeds=[17], batch_size=4, pair_weight=2., rank_weight=.5)
    generate(c)
    validate_data(c)
    encode(c)
    fit_normalization(c)
    return c


def copied_arm(ready, tmp_path, name, arm="gate_no_rank"):
    c = profile(ready, arm)
    c["output"] = str(tmp_path/name)
    for part in ["dataset", "features"]:
        shutil.copytree(Path(ready["output"])/part, Path(c["output"])/part)
    return c


def test_common_mse_best_earliest_tie_and_exact_boundary_resume(tiny_features, tmp_path, monkeypatch):
    import phase1.training as training
    calls = {}

    def fixed_validation(c, store, model, pairs, device, task=None):
        index = calls.get(c["output"], 0)
        calls[c["output"]] = index+1
        mse, composite = [(.4, .1), (.2, .9), (.2, .05)][index]
        return {condition: {"mse": mse, "loss": composite, "pair_mse": .1}
                for condition in TRAIN_CONDITIONS}

    monkeypatch.setattr(training, "validation_loss", fixed_validation)
    uninterrupted = copied_arm(tiny_features, tmp_path, "uninterrupted")
    resumed = copied_arm(tiny_features, tmp_path, "resumed")
    train(uninterrupted)
    train(resumed, stop_after_epoch=1)
    first = trusted_checkpoint(Path(resumed["output"])/"predictor/gate/seed_17/latest.pt")
    assert first["epoch"] == 1 and first["best"] == .4
    train(resumed)
    full_latest = trusted_checkpoint(Path(uninterrupted["output"])/"predictor/gate/seed_17/latest.pt")
    latest_path = Path(resumed["output"])/"predictor/gate/seed_17/latest.pt"
    latest = trusted_checkpoint(latest_path)
    best = trusted_checkpoint(latest_path.with_name("best.pt"))
    assert latest["epoch"] == 3 and best["epoch"] == 2 and best["best"] == .2
    assert latest["checkpoint_selection"] == best["checkpoint_selection"] == "validation_mse"
    assert latest["experiment"] == resumed["experiment"] == latest["signature"]["experiment"]
    assert latest["signature"]["checkpoint_selection"] == "validation_mse"
    assert latest["signature"]["scientific_hash"] == scientific_hash(resumed)
    assert latest["resume_granularity"] == "epoch_boundary"
    assert not (Path(resumed["output"])/"predictor/motion").exists()
    for row in latest["logs"]:
        assert row["selection_score"] == row["validation_mse"]
        assert row["selection_metric"] == "validation_mse"
        assert set(ast.literal_eval(row["mix_counts"]).values()) == {4}
    assert latest["step"] == full_latest["step"] == 12
    assert latest["scheduler"] == full_latest["scheduler"]
    assert all(torch.equal(latest["model"][k], full_latest["model"][k]) for k in latest["model"])
    log = read_json(latest_path.with_name("training.json"))
    assert log["experiment"] == resumed["experiment"] and log["checkpoint_selection"] == "validation_mse"
    # A manually corrupted selection tag cannot resume under the valid signature.
    latest["checkpoint_selection"] = "validation_loss"
    save_torch(latest_path, latest)
    with pytest.raises(ValueError, match="experiment/selection mismatch"):
        train(resumed)


def test_matched_arms_preserve_sample_order_and_initial_rng(tiny_features, tmp_path, monkeypatch):
    import phase1.training as training
    traces = {}
    original = training.FeatureStore.sample
    initial_models = []
    original_model = training.DynamicsPredictor

    def traced(store, rec, condition, rng=None, dropout=0):
        sample = original(store, rec, condition, rng, dropout)
        if rec["split"] == "train":
            traces.setdefault(store.c["output"], []).append((rec["id"], condition, tuple(sample[2]["indices"])))
        return sample

    def traced_model(*args, **kwargs):
        model = original_model(*args, **kwargs)
        initial_models.append({key: value.clone() for key, value in model.state_dict().items()})
        return model

    monkeypatch.setattr(training.FeatureStore, "sample", traced)
    monkeypatch.setattr(training, "DynamicsPredictor", traced_model)
    reference = copied_arm(tiny_features, tmp_path, "reference", "reference")
    candidate = copied_arm(tiny_features, tmp_path, "candidate", "gate_no_rank")
    train(reference, stop_after_epoch=1)
    train(candidate, stop_after_epoch=1)
    assert traces[reference["output"]] == traces[candidate["output"]]
    assert len(traces[reference["output"]]) == 4*4
    assert len(initial_models) == 2
    assert all(torch.equal(initial_models[0][k], initial_models[1][k]) for k in initial_models[0])
    checkpoints = []
    for c in [reference, candidate]:
        ck = trusted_checkpoint(Path(c["output"])/"predictor/gate/seed_17/latest.pt")
        checkpoints.append(ck)
        assert ck["epoch"] == 1 and ck["step"] == 4
        assert np.isfinite([row["selection_score"] for row in ck["logs"]]).all()
    assert torch.equal(checkpoints[0]["rng"]["torch"], checkpoints[1]["rng"]["torch"])
