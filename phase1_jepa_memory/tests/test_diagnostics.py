"""Boundary tests with synthetic arrays only; never research measurements."""
import copy

import numpy as np
import pytest
import torch

from phase1.config import CONDITIONS, DEFAULT, SPLITS
from phase1.diagnostics import (ARMS, POLICIES, _guard_output, _motion_pairs,
    _paired_rows_for_policies, _verify_source_proof, assert_exact,
    cross_policy_comparisons, evaluate_frozen_motion, policy_view, verify_native_metrics)
from phase1.inputs import FeatureStore
from phase1.utils import code_hash, file_hash, write_json


@pytest.fixture
def fixed_motion(monkeypatch):
    import phase1.inputs as inputs
    import phase1.diagnostics as diagnostics
    c = copy.deepcopy(DEFAULT)
    c["training"].update(batch_size=4, seeds=[17, 29])
    c["evaluation"]["bootstrap"] = 100
    c["data"]["pairs"] = {split: 2 for split in SPLITS}
    c["experiment"] = {"stage": "development", "arm": "reference", "tasks": ["motion"]}
    records, features, meta, observations = [], {}, {}, {}
    for split in SPLITS[1:]:
        for pair in range(2):
            q, end, n = 16, 7, 24
            for sibling in range(2):
                eid = f"{split}/{pair}/{sibling}"
                rec = {"id": eid, "sibling": f"{split}/{pair}/{1-sibling}", "task": "motion",
                    "pair": f"{split}/{pair}", "split": split, "query": q, "gap": 9}
                records.append(rec)
                z = np.column_stack([np.arange(n)+pair, np.arange(n)**2/20]).astype(np.float32)
                z[:q-c["data"]["recent"]+1] += sibling*np.array([2., 3.], np.float32)
                z[end] -= sibling*np.array([2., 3.], np.float32)
                z[q+1:] += sibling
                features[eid] = z
                meta[eid] = {"evidence": np.array([3, end]),
                    "states": np.column_stack([np.arange(n)/100, np.ones(n)/2]).astype(np.float32),
                    "classes": np.full(n, sibling)}
                observations[eid] = {"actions": np.zeros((n, 2), np.float32)}
    store = FeatureStore.__new__(FeatureStore)
    store.c, store.records, store.manifest = c, {r["id"]: r for r in records}, {"records": records}
    store.bases = {"motion": {"input_mean": np.zeros(2), "projection": np.eye(2)}}
    store.features = lambda r: features[r["id"]]
    store.projected_features = lambda r: features[r["id"]].copy()
    store.norm = {"mean": np.zeros(2, np.float32), "std": np.array([1., 2.], np.float32)}
    def load(config, rec, metadata=False, keys=None):
        source = meta[rec["id"]] if metadata else observations[rec["id"]]
        # Predictor path is allowed to read only evidence indices and actions.
        return {k: source[k].copy() for k in keys}
    monkeypatch.setattr(inputs, "load_episode", load)
    monkeypatch.setattr(diagnostics, "load_episode", load)
    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(.1))
            self.register_buffer("projection", torch.eye(2))
        def forward(self, batch):
            mask = batch["mask"] & (batch["type"] == 1)
            memory = (batch["projected_z"]*mask[..., None]).sum(1)
            pred = batch["current"]+self.gain*memory
            return torch.stack([pred, pred+1], 1)
    class Decoder:
        def predict(self, prediction):
            return np.column_stack([prediction[:, 0], prediction])
    return c, store, features, meta, Predictor(), Decoder()


def test_policy_view_isolated_and_wrong_uses_actual_selected_images(fixed_motion):
    c, store, features, _, _, _ = fixed_motion
    original = copy.deepcopy(c)
    rec = store.manifest["records"][0]
    for policy, indices in zip(POLICIES, ([3, 7], [0, 7])):
        view = policy_view(store, policy)
        sample, target, info = view.sample(rec, "wrong_evidence")
        assert info["indices"].tolist() == indices
        assert np.array_equal(sample["z"][c["data"]["recent"]:], features[rec["sibling"]][indices])
        assert np.array_equal(sample["projected_z"][c["data"]["recent"]:], features[rec["sibling"]][indices])
        own, own_target, own_info = view.sample(rec, "privileged_small_budget")
        for key in ("age", "type", "actions", "past_actions", "current", "projected_current"):
            assert np.array_equal(sample[key], own[key])
        assert np.array_equal(target, own_target)
        assert np.array_equal(info["indices"], own_info["indices"])
    assert c == original and store.c == original


def test_frozen_matrix_has_exact_recent_and_original_pair_sample_units(fixed_motion):
    c, store, _, _, model, decoder = fixed_motion
    original = copy.deepcopy(c)
    rows, pairs, invariants = evaluate_frozen_motion(c, store, model, decoder, 17, torch.device("cpu"))
    assert not model.training and not model.gain.requires_grad
    assert store.c == original
    assert len(rows) == 2*4*4*len(CONDITIONS)*2
    assert len(pairs) == len(rows)//2
    assert len(invariants) == 4*3
    assert all(v["byte_identical"] and len(set(v["prediction_sha256"].values())) == 1 for v in invariants)
    # The second fake seed perturbs one policy, exposing incorrect seed pooling.
    more = [{**r, "seed": 29, "normalized_mse": r["normalized_mse"]+(1 if r["policy"] == POLICIES[1] else 0)} for r in rows]
    comparisons = cross_policy_comparisons(c, rows+more)
    assert len(comparisons) == 4*2*len(CONDITIONS)*3
    assert all(r["metrics"]["normalized_mse"]["absolute_error_reduction"]["pairs"] == 2 for r in comparisons)
    item = lambda s: next(r for r in comparisons if r["seed"] == s and r["split"] == "validation"
                         and r["horizon"] == 3 and r["condition"] == "recent_only")
    assert item(17)["metrics"]["normalized_mse"]["absolute_error_reduction"]["mean"] == 0
    assert item(29)["metrics"]["normalized_mse"]["absolute_error_reduction"]["mean"] == -1
    assert item("all_seeds_mean")["metrics"]["normalized_mse"]["absolute_error_reduction"]["mean"] == -.5


def test_recent_prediction_drift_fails_closed(fixed_motion):
    c, store, _, _, model, decoder = fixed_motion
    forward, calls = model.forward, [0]
    def drifting(batch):
        calls[0] += 1
        return forward(batch)+calls[0]*.001
    model.forward = drifting
    with pytest.raises(ValueError, match="predictions"):
        evaluate_frozen_motion(c, store, model, decoder, 17, torch.device("cpu"))
    with pytest.raises(ValueError, match="invariant"):
        assert_exact(np.array([0.], np.float32), np.array([-0.], np.float32), "signed-zero bytes")


def test_pair_statistics_reject_missing_seed_policy_and_missing_sibling(fixed_motion):
    c, store, _, _, model, decoder = fixed_motion
    records = store.manifest["records"][:4]
    with pytest.raises(ValueError, match="sibling"):
        _motion_pairs(records[:3], 2)
    rows, _, _ = evaluate_frozen_motion(c, store, model, decoder, 17, torch.device("cpu"))
    subset = [r for r in rows if r["split"] == "validation" and r["horizon"] == 3 and r["condition"] == "recent_only"]
    with pytest.raises(ValueError, match="every matched seed"):
        list(_paired_rows_for_policies(subset, [17, 29]))
    only_one_sibling = [r for r in subset if r["episode"].endswith("/0")]
    with pytest.raises(ValueError, match="exactly two episodes"):
        list(_paired_rows_for_policies(only_one_sibling, [17]))


def test_output_guard_protects_runs_shared_symlink_and_existing_reports(tmp_path):
    roots = {arm: tmp_path/arm for arm in ARMS}
    shared = tmp_path/"shared_features"
    shared.mkdir()
    for root in roots.values():
        root.mkdir()
        (root/"features").symlink_to(shared, target_is_directory=True)
    for output in (roots[ARMS[0]]/"diagnostic", shared/"diagnostic", tmp_path):
        with pytest.raises(ValueError, match="separate"):
            _guard_output(output, roots)
    output = tmp_path/"new_diagnostic"
    assert _guard_output(output, roots) == output
    output.mkdir()
    (output/"motion_cross_policy.json").write_text("{}")
    with pytest.raises(ValueError, match="already exists"):
        _guard_output(output, roots)


def test_original_checkpoint_proof_cannot_be_replaced_silently(tmp_path, monkeypatch):
    import phase1.diagnostics as diagnostics
    from types import SimpleNamespace
    c = {"output": str(tmp_path)}
    basis = tmp_path/"predictor/motion"
    basis.mkdir(parents=True)
    (basis/"basis.npz").write_bytes(b"fixture only")
    (basis/"basis.json").write_text("{}")
    evaluation = tmp_path/"evaluation"
    evaluation.mkdir()
    (evaluation/"raw_records.csv").write_text("fixture only")
    checked = {"best_sha256": "original", "latest_sha256": "latest", "training_json_sha256": "log",
               "signature": {"fixture": True}, "best_epoch": 3}
    completion = {"complete": True, "models": {"motion/17": checked}}
    monkeypatch.setattr(diagnostics, "training_completion", lambda *args: completion)
    proof = {"cache_signature": "cache", "raw_sha256": file_hash(evaluation/"raw_records.csv"),
        "evaluation_code_hash": code_hash("evaluation.py", "stats.py"),
        "training_completion": copy.deepcopy(completion), "checkpoints": {"motion/17": {"best_sha256": "original"}}}
    write_json(evaluation/"metrics.json", proof)
    assert _verify_source_proof(c, SimpleNamespace(ix={"signature": "cache"})) == completion
    completion["models"]["motion/17"]["best_sha256"] = "replacement"
    with pytest.raises(ValueError, match="retained evaluation proof"):
        _verify_source_proof(c, SimpleNamespace(ix={"signature": "cache"}))


def test_native_policy_reproduces_all_original_metrics_and_rejects_drift(tmp_path):
    from phase1.evaluation import METRICS
    from phase1.utils import write_csv
    path = tmp_path/"original_raw.csv"
    row = {"seed": 17, "task": "motion", "split": "validation", "horizon": 3,
        "condition": "recent_only", "pair": "pair", "episode": "episode",
        **dict.fromkeys(METRICS, 1.), "state_true_y": 999.}
    write_csv(path, [row, {**row, "condition": "persistence"}])
    before = file_hash(path)
    raw = [{**row, "policy": p, "state_true_y": .5} for p in POLICIES]
    for arm, policy in zip(ARMS, POLICIES):
        proof = verify_native_metrics(raw, path, arm)
        assert proof["native_policy"] == policy and proof["matched_rows"] == 1
        assert proof["maximum_absolute_difference"] == 0
        assert proof["compared_metrics"] == METRICS
    changed = copy.deepcopy(raw)
    changed[0]["pair_margin"] += .01
    with pytest.raises(ValueError, match="pair_margin"):
        verify_native_metrics(changed, path, "reference")
    assert verify_native_metrics(changed, path, "motion_wide_span")["reproduced"]
    with pytest.raises(ValueError, match="coverage"):
        verify_native_metrics(raw[1:], path, "reference")
    with pytest.raises(ValueError, match="Duplicate"):
        verify_native_metrics(raw+[raw[0]], path, "reference")
    assert file_hash(path) == before


def test_gate_export_preserves_complete_earliest_selected_smoke_and_original_files(tmp_path, monkeypatch):
    """Tiny synthetic integration test; frozen export is never science evidence."""
    import shutil
    from pathlib import Path
    import phase1.training as training
    from phase1.cache import encode, fit_normalization
    from phase1.config import prepare
    from phase1.data import generate, validate_data
    from phase1.evaluation import evaluate
    from phase1.gate_diagnostics import export_gate_pairs
    from phase1.probes import probe

    base = copy.deepcopy(DEFAULT)
    base["output"] = str(tmp_path/"synthetic_cache")
    base["data"]["pairs"] = dict.fromkeys(SPLITS, 2)
    base["encoder"]["smoke_size"] = 6
    base["model"].update(hidden=16, rank=8, dropout=0.)
    base["training"].update(epochs=2, seeds=[17], batch_size=4, pair_weight=2., rank_weight=.5)
    base["evaluation"].update(bootstrap=100, latency_warmup=0, latency_repeats=1)
    base["probe"]["ridge"] = [.1]
    generate(base)
    validate_data(base)
    encode(base)
    fit_normalization(base)
    # Tied validation scores force the real checkpoint mechanism to preserve
    # epoch 1, making it possible to detect accidental latest-checkpoint export.
    monkeypatch.setattr(training, "validation_loss", lambda *args, **kwargs:
        {condition: {"mse": .25, "loss": .5, "pair_mse": .1} for condition in training.TRAIN_CONDITIONS})
    development = tmp_path/"synthetic_development"
    for arm in ("reference", "gate_no_rank"):
        c = copy.deepcopy(base)
        c["output"] = str(development/"gate"/arm)
        c["experiment"] = {"stage": "development", "arm": arm, "tasks": ["gate"]}
        for name in ("dataset", "features"):
            shutil.copytree(Path(base["output"])/name, Path(c["output"])/name)
        prepare(c)
        probe(c)
        training.train(c)
        evaluate(c)
    before = {str(p): file_hash(p) for p in development.rglob("*") if p.is_file()}
    result = export_gate_pairs(development, tmp_path/"synthetic_diagnostic")
    assert result["independent_phase1_result"] is False
    for arm in result["arms"].values():
        assert arm["training_completion"]["complete"]
        assert arm["checkpoints"]["gate/17"]["best_epoch"] == 1
        assert arm["training_completion"]["models"]["gate/17"]["completed_epochs"] == 2
        assert arm["original_calibration_max_mean_difference"] <= 1e-7
        recent = [r for r in arm["rows"] if r["condition"] == "recent_only"]
        assert recent and all(r["recent_prediction_equal"] and r["recent_prediction_max_abs_difference"] == 0 for r in recent)
    assert before == {str(p): file_hash(p) for p in development.rglob("*") if p.is_file()}
