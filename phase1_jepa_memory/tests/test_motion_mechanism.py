"""Synthetic boundary checks only; these tests are not research measurements."""
import copy
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phase1.config import DEFAULT, SPLITS
from phase1.inputs import collate
from phase1.model import DynamicsPredictor
from phase1.motion_mechanism import (BasisOracle, _rng_hash, basis_ceiling,
    frozen_forward_pair, model_state_hash, validation_decision, velocity_forward,
    velocity_scalars, _guard_destination, evaluate_mechanism_seed, frozen_runtime,
    verify_basis_provenance, registered_source_config)
from phase1.training import TRAIN_CONDITIONS


@pytest.fixture
def mechanism_model():
    # Fixed local synthetic model; no fit, encoder, cached research data or HPC.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(117)
        model = DynamicsPredictor(4, {"hidden": 8, "dropout": 0.}, [3, 6])
    return model.eval().requires_grad_(False)


def sample(history=True, small_span=False):
    z = np.array([[1., 2., 1., 0.], [1., 2., 1., 0.],
                  [0., 0., 1., 1.], [1., 1., 2., 0.]], np.float32)
    ages = np.array([1., 0., 8., 7. if small_span else 4.], np.float32)
    past = np.array([[.1, .2], [0., 0.], [1., -1.], [.2, .4]], np.float32)
    n = 4 if history else 2
    return {"z": z[:n].copy(), "projected_z": z[:n].copy(), "current": z[1].copy(),
        "projected_current": z[1].copy(), "age": ages[:n], "type": np.array([0, 0, 1, 1], np.int64)[:n],
        "past_actions": past[:n], "actions": np.arange(12, dtype=np.float32).reshape(6, 2)/10}


def test_clone_and_native_math_are_exact_and_rng_weights_recent_are_unchanged(mechanism_model):
    model = mechanism_model
    clone = copy.deepcopy(model)
    state, rng = model_state_hash(model), _rng_hash()
    batch = collate([sample(False), sample(True), sample(True, True)], torch.device("cpu"))
    predictions, trace = frozen_forward_pair(model, clone, batch)
    assert np.array_equal(predictions["native"][0], predictions["pure_p_velocity"][0])
    assert not np.array_equal(predictions["native"][1], predictions["pure_p_velocity"][1])
    assert model_state_hash(model) == state == model_state_hash(clone)
    assert _rng_hash() == rng
    assert np.allclose(trace["native_velocity"], trace["pure_velocity"]+trace["action_velocity"], atol=2e-5)
    # A clone with changed weights is detected through exact native replay.
    with torch.no_grad():
        clone.output[-1].bias.add_(.1)
    with pytest.raises(ValueError, match="native mechanism clone"):
        frozen_forward_pair(model, clone, batch)


def test_b2_actual_velocity_decomposes_and_keeps_variance_floor(mechanism_model):
    batch = collate([sample(True), sample(True, True)], torch.device("cpu"))
    with torch.inference_mode():
        _, trace = velocity_forward(mechanism_model, batch)
    ordinary, narrow = [velocity_scalars(trace, i, require_two=True) for i in range(2)]
    assert ordinary["span_steps"] == 4
    assert narrow["span_steps"] == 1
    assert 0 < narrow["kappa"] < 1 and narrow["variance_floor_active"]
    assert ordinary["raw_action_secant_mean_square"] > 0
    assert ordinary["action_to_visual_energy_ratio"] is not None
    assert max(ordinary["two_frame_formula_max_abs_error"], narrow["two_frame_formula_max_abs_error"]) < 2e-5
    bad = {k: v.clone() for k, v in trace.items()}
    bad["native_velocity"][0] += 1
    with pytest.raises(ValueError, match="decomposition"):
        velocity_scalars(bad, 0, require_two=True)


def test_predictor_uses_no_future_target_and_respects_horizon_action_prefix(mechanism_model):
    source = sample(True)
    # collate is the original whitelist: even supplied evaluation metadata is
    # not passed to either forward path.
    source.update(target=np.full((2, 4), 999.), states=np.full((2, 2), 999.), label=999)
    batch = collate([source], torch.device("cpu"))
    assert not {"target", "states", "label"}.intersection(batch)
    with torch.inference_mode():
        native, _ = velocity_forward(mechanism_model, batch)
        pure, _ = velocity_forward(mechanism_model, batch, True)
    source["target"][:] = np.nan
    source["states"][:] = np.nan
    source["label"] = -1
    same = collate([source], torch.device("cpu"))
    with torch.inference_mode():
        a, _ = velocity_forward(mechanism_model, same)
        b, _ = velocity_forward(mechanism_model, same, True)
    assert torch.equal(native, a) and torch.equal(pure, b)
    changed = {k: v.clone() for k, v in same.items()}
    changed["actions"][:, 3:] += 100
    with torch.inference_mode():
        later_native, _ = velocity_forward(mechanism_model, changed)
        later_pure, _ = velocity_forward(mechanism_model, changed, True)
    assert torch.equal(native[:, 0], later_native[:, 0])
    assert torch.equal(pure[:, 0], later_pure[:, 0])
    assert not torch.equal(pure[:, 1], later_pure[:, 1])


def test_basis_oracle_uses_pseudoinverse_for_nonorthogonal_rank_deficient_basis():
    # Nonorthogonal, dependent columns: B B^T is NOT the projection.
    basis = np.array([[2., 4.], [1., 2.], [0., 0.]])
    mean = np.array([.3, .2, .1])
    current = np.array([1., 2., 3.])
    residual = np.array([[2., 1., 3.], [1., -2., 0.]])
    oracle = BasisOracle(basis, mean)
    result = oracle.error(current, residual+current+mean)
    assert oracle.rank == 1
    assert result.tolist() == pytest.approx([3., 5/3])
    coeff = np.linalg.lstsq(basis, residual.T, rcond=1e-12)[0]
    assert result == pytest.approx(np.mean((residual-coeff.T@basis.T)**2, axis=1))
    # All actual vectors in this affine space have error >= the oracle.
    prediction = current+mean+np.array([[.4, -.1], [.1, .1]])@basis.T
    assert np.all(np.mean((prediction-(residual+current+mean))**2, axis=1) >= result-1e-12)


def decision_fixture():
    c = copy.deepcopy(DEFAULT)
    c["training"]["seeds"] = [17, 29, 43]
    c["evaluation"]["bootstrap"] = 100
    c["data"]["pairs"] = dict.fromkeys(SPLITS, 2)
    rows, oracle = [], []
    for split in SPLITS[1:]:
        for pair in range(2):
            for sibling in range(2):
                for h in c["data"]["horizons"]:
                    key = {"split": split, "horizon": h, "pair": f"{split}/{pair}",
                           "episode": f"{split}/{pair}/{sibling}"}
                    oracle.append({**key, "oracle_normalized_mse": .2})
                    for seed in c["training"]["seeds"]:
                        for condition in TRAIN_CONDITIONS:
                            for variant in ("native", "pure_p_velocity") if split == "validation" else ("native",):
                                mse = .8 if variant == "pure_p_velocity" and condition == "privileged_small_budget" else 1.
                                rows.append({**key, "seed": seed, "variant": variant, "condition": condition,
                                    "normalized_mse": mse, "state_position_mse": mse/10, "pair_margin": 1.})
    return c, rows, oracle


def test_primary_gate_needs_every_seed_horizon_and_equal_condition_mean():
    c, rows, _ = decision_fixture()
    assert validation_decision(c, rows)["passed"]
    bad = copy.deepcopy(rows)
    for r in bad:
        if r["seed"] == 29 and r["horizon"] == 3 and r["variant"] == "pure_p_velocity" and r["condition"] == "privileged_small_budget":
            r["normalized_mse"] = 1.
    assert not validation_decision(c, bad)["passed"]  # zero effect has CI low == 0
    worse_full = copy.deepcopy(rows)
    for r in worse_full:
        if r["variant"] == "pure_p_velocity" and r["condition"] == "full_history":
            r["normalized_mse"] = 2.
    checks = validation_decision(c, worse_full)
    assert not checks["passed"]
    assert all(r["passed"] for r in checks["checks"] if r["kind"] == "privileged_validation_future_mse")
    assert all(not r["passed"] for r in checks["checks"] if r["kind"] == "four_condition_all_horizon_validation")
    with pytest.raises(ValueError, match="Missing matched"):
        validation_decision(c, rows[1:])


def test_old_development_oracle_ceiling_veto_keeps_original_ten_percent():
    c, rows, oracle = decision_fixture()
    native = [r for r in rows if r["variant"] == "native"]
    passed = basis_ceiling(c, native, oracle)
    assert not passed["velocity_training_veto"]
    assert all(r["maximum_possible_improvement"]["pairs"] == 2 for r in passed["checks"])
    weak_basis = copy.deepcopy(oracle)
    for r in weak_basis:
        if r["split"] == "test_distractor_shift" and r["horizon"] == 6:
            r["oracle_normalized_mse"] = .95
    blocked = basis_ceiling(c, native, weak_basis)
    assert blocked["velocity_training_veto"]
    assert len(blocked["vetoes"]) == 4  # three seeds and measured-seed mean
    assert all(r["required_original_threshold"] == .1 for r in blocked["vetoes"])
    assert all(r["maximum_possible_improvement"]["mean"] == pytest.approx(.05) for r in blocked["vetoes"])


def test_mechanism_destination_protects_sources_and_coexists_with_first_diagnostic(tmp_path):
    roots = {arm: tmp_path/arm for arm in ("reference", "motion_wide_span")}
    for root in roots.values():
        root.mkdir()
    with pytest.raises(ValueError, match="outside"):
        _guard_destination(roots["reference"]/"new", roots)
    out = tmp_path/"diagnostic"
    out.mkdir()
    (out/"motion_cross_policy.json").write_text("{}")
    assert _guard_destination(out, roots) == out
    (out/"motion_mechanism.json").write_text("{}")
    with pytest.raises(ValueError, match="already exists"):
        _guard_destination(out, roots)


def test_frozen_seed_exports_native_all_slices_intervention_validation_only(monkeypatch, mechanism_model, tmp_path):
    import phase1.inputs as inputs
    import phase1.diagnostics as diagnostics
    import phase1.motion_mechanism as mechanism
    from phase1.inputs import FeatureStore
    from phase1.utils import write_csv
    c = copy.deepcopy(DEFAULT)
    c["data"]["pairs"] = dict.fromkeys(SPLITS, 2)
    c["training"].update(batch_size=4, seeds=[17])
    c["experiment"] = {"stage": "development", "arm": "reference", "tasks": ["motion"]}
    records, features, metadata, observations = [], {}, {}, {}
    for split in SPLITS[1:]:
        for pair in range(2):
            for sibling in range(2):
                eid = f"{split}/{pair}/{sibling}"
                q, n = 16, 24
                rec = {"id": eid, "sibling": f"{split}/{pair}/{1-sibling}", "pair": f"{split}/{pair}",
                    "task": "motion", "split": split, "query": q, "gap": 9}
                records.append(rec)
                t = np.arange(n, dtype=np.float32)
                z = np.column_stack([t/10, t**2/200, np.ones(n), np.full(n, pair)]).astype(np.float32)
                z[:13] += sibling/10
                z[7] -= sibling/10
                z[q+1:] += sibling/10
                features[eid] = z
                observations[eid] = {"actions": np.column_stack([t/20, -t/30]).astype(np.float32)}
                metadata[eid] = {"evidence": np.array([3, 7]), "classes": np.full(n, sibling),
                    "states": z[:, :2].copy()}
    store = FeatureStore.__new__(FeatureStore)
    store.c, store.manifest = c, {"records": records}
    store.records = {r["id"]: r for r in records}
    store.features = lambda r: features[r["id"]]
    store.projected_features = lambda r: features[r["id"]].copy()
    store.bases = {"motion": {}}
    store.norm = {"mean": np.zeros(4, np.float32), "std": np.ones(4, np.float32)}
    def load(config, rec, meta=False, keys=None):
        arrays = metadata[rec["id"]] if meta else observations[rec["id"]]
        return {k: arrays[k].copy() for k in keys}
    monkeypatch.setattr(inputs, "load_episode", load)
    monkeypatch.setattr(diagnostics, "load_episode", load)
    monkeypatch.setattr(mechanism, "load_episode", load)
    class Decoder:
        def predict(self, value):
            return value[:, :3]
    rows, trace, oracle, proof = evaluate_mechanism_seed(c, store, mechanism_model, Decoder(), 17,
                                                       "reference", collect_oracle=True)
    assert len(rows) == 200 and len(trace) == 20 and len(oracle) == 32
    assert {r["split"] for r in rows if r["variant"] == "pure_p_velocity"} == {"validation"}
    assert proof["clone_exact"] and proof["weights_unchanged"] and proof["forward_rng_unchanged"]
    assert proof["recent"][0]["byte_identical"]
    assert len(set(proof["recent"][0]["prediction_sha256"].values())) == 1
    assert proof["basis_oracle"]["target_aware_not_deployable"]
    assert max(r["oracle_normalized_mse"] for r in oracle) < 1e-20
    native = [r for r in rows if r["variant"] == "native"]
    path = tmp_path/"synthetic_native.csv"
    write_csv(path, native)
    assert diagnostics.verify_native_metrics(native, path, "reference")["maximum_absolute_difference"] == 0


def test_basis_checks_train_membership_and_actual_checkpoint_buffers(tmp_path, mechanism_model):
    from phase1.utils import file_hash
    c = copy.deepcopy(DEFAULT)
    c["output"] = str(tmp_path)
    c["data"]["pairs"]["train"] = 1
    folder = tmp_path/"predictor/motion"
    folder.mkdir(parents=True)
    arrays = {name: getattr(mechanism_model, name).numpy() for name in
              ("input_mean", "projection", "target_mean", "output_basis")}
    arrays["train_ids"] = np.array(["train/a", "train/b"])
    store = SimpleNamespace(manifest={"records": [
        {"id": x, "task": "motion", "split": "train"} for x in ["train/a", "train/b"]] +
        [{"id": "validation/a", "task": "motion", "split": "validation"}]})
    path = folder/"basis.npz"
    def save():
        np.savez(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps({"fit_split": "train", "task": "motion",
            "episodes": 2, "input_rank": 4, "output_rank": 4, "sha256": file_hash(path)}))
    save()
    assert verify_basis_provenance(c, store, mechanism_model)["checkpoint_buffers_equal_persisted_basis"]
    changed = copy.deepcopy(mechanism_model)
    changed.target_mean[0] += .1
    with pytest.raises(ValueError, match="checkpoint basis buffer"):
        verify_basis_provenance(c, store, changed)
    arrays["train_ids"][1] = "validation/a"
    save()
    with pytest.raises(ValueError, match="training IDs"):
        verify_basis_provenance(c, store)


def test_archived_science_really_executes_separately_without_hash_substitution(tmp_path):
    import phase1.motion_mechanism as current
    from phase1 import training
    package = tmp_path/"archived/src/phase1"
    package.mkdir(parents=True)
    source = Path(current.__file__).parent
    for path in source.glob("*.py"):
        if path.name not in {"diagnostics.py", "motion_mechanism.py", "gate_diagnostics.py"}:
            shutil.copy2(path, package/path.name)
    # Harmless source marker demonstrates that hashing really uses the archive.
    old_training = package/"training.py"
    old_training.write_text(old_training.read_text()+"\nARCHIVE_SENTINEL = 'original science'\n")
    modules_before, rng_before = set(sys.modules), _rng_hash()
    with frozen_runtime(tmp_path/"archived") as archived:
        assert archived is not current
        assert archived.load_predictor is not current.load_predictor
        assert Path(archived.scientific_utils.__file__).parent == package
        assert archived.load_predictor.__globals__["ARCHIVE_SENTINEL"] == "original science"
        assert archived.code_hash("training.py", "model.py", "inputs.py") != training.code_hash("training.py", "model.py", "inputs.py")
        # Archived source also supplies the signature function; it was not just
        # spliced into current code's expected hash.
        assert archived.load_predictor.__globals__["checkpoint_signature"].__module__.startswith("_phase1_mechanism_frozen_")
        # The added marker is not registered v3 source. A claimed old code hash
        # cannot make this altered archive pass the independent actual-file check.
        with pytest.raises(ValueError, match="Actual loaded checkpoint"):
            archived.loaded_code_proof(package)
        actual = {"checkpoint": archived.code_hash("training.py", "model.py", "inputs.py"),
                  "evaluation": archived.code_hash("evaluation.py", "stats.py")}
        archived.REGISTERED_CODE_HASHES = actual  # synthetic test registration only
        proof = archived.loaded_code_proof(package)
        assert proof["actual_code_hashes"] == actual
        assert proof["file_sha256"]["training.py"] == archived.file_hash(old_training)
    assert not [key for key in set(sys.modules)-modules_before if key.startswith("_phase1_mechanism_frozen_")]
    assert not list(package.rglob("__pycache__"))
    assert _rng_hash() == rng_before
    with pytest.raises(ValueError, match="source changed"):
        with frozen_runtime(package):
            old_training.write_text(old_training.read_text()+"\nCHANGED = True\n")


def test_registered_configs_lock_original_coefficients_and_frame_policy():
    from phase1.config import load_config
    path = Path(__file__).parents[1]/"configs/phase1.yaml"
    for arm in ("reference", "motion_wide_span"):
        c = load_config(path)
        c["experiment"] = {"stage": "development", "arm": arm, "tasks": ["motion"]}
        registered_source_config(c, arm)
        for category, key, value in [("training", "rank_weight", 0), ("training", "pair_weight", 1),
                                     ("model", "rank", 256), ("decision", "latent_relative_improvement", .05),
                                     ("data", "horizons", [1, 3])]:
            modified = copy.deepcopy(c)
            modified[category][key] = value
            with pytest.raises(ValueError, match="scientific config/coefficients"):
                registered_source_config(modified, arm)


def test_preregistration_matches_executable_primary_gates_and_original_hashes():
    from phase1.motion_mechanism import REGISTERED_CODE_HASHES, REGISTERED_SCIENTIFIC_HASHES
    p = Path(__file__).parents[1]/"configs/phase1_v4_motion_mechanism_protocol.json"
    protocol = json.loads(p.read_text())
    assert protocol["primary"]["checkpoint_arm"] == "reference"
    assert protocol["primary"]["input_policy"] == "old_start_end"
    assert protocol["replication"]["can_rescue_failed_primary"] is False
    for section in ("primary", "replication"):
        row = protocol[section]
        assert row["scientific_hash"] == REGISTERED_SCIENTIFIC_HASHES[row["checkpoint_arm"]]
    assert protocol["source_proof"]["training_code_hash"] == REGISTERED_CODE_HASHES["checkpoint"]
    assert protocol["source_proof"]["evaluation_code_hash"] == REGISTERED_CODE_HASHES["evaluation"]
    assert protocol["fixed_science"]["pair_ranking_coefficient"] == .5
    assert protocol["fixed_science"]["pair_difference_mse_coefficient"] == 2
    assert protocol["basis_feasibility_diagnostic"]["original_relative_threshold"] == .1
    assert protocol["statistics_and_eligibility"]["bootstrap_replicates"] == 2000
    assert protocol["statistics_and_eligibility"]["bootstrap_seed"] == 0
    assert protocol["statistics_and_eligibility"]["eligibility_not_submission"]
