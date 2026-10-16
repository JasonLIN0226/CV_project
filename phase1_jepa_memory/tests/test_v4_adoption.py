"""Compact synthetic boundary fixtures; no training or research measurement."""
import copy
from pathlib import Path
import pytest

from phase1.config import CONDITIONS, DEFAULT, SPLITS, scientific_hash
from phase1.evaluation import METRICS, summarize
from phase1.utils import code_hash, digest, file_hash, read_json, write_csv, write_json
from phase1.v4_adoption import comparison, validate_confirmation_lock, validate_gate_diagnostic


def _config(root, role, stage="development"):
    c = copy.deepcopy(DEFAULT)
    c.update(mode="confirm", output=str(root))
    c["encoder"].update(backend="ijepa", revision="f157467ea509bc356ff9f61fd3c0d840eec5e04e")
    c["data"].update(seed=6301 if stage == "development" else 9137, pairs=dict.fromkeys(SPLITS, 32))
    c["training"].update(epochs=120, seeds=[17, 29, 43], batch_size=64, pair_weight=2., rank_weight=.5)
    c["model"]["dropout"] = 0
    c["evaluation"]["bootstrap"] = 100
    c["experiment"] = {"protocol_version": 4, "role": role, "stage": stage, "tasks": ["gate"],
                       "variant": "reference" if role == "reference" else "gate_common_anchor_v1",
                       "common_anchor_weight": 0 if role == "reference" else 1, "gate_diagnostic_sha256": "diagnostic-sha"}
    c["experiment"]["matched_teacher_forwards"] = True
    if stage == "confirmation":
        c["experiment"].update(tasks=["gate", "motion"], variant={
            "gate": "reference" if role == "reference" else "gate_common_anchor_v1",
            "motion": "reference" if role == "reference" else "synthetic_explicit_unselected_motion"})
    return c


def _completion(root, config):
    cache, norm = "same-cache", "same-normalization"
    write_json(root/"features/normalization.json", {"fit_split": "train", "cache_signature": cache, "sha256": norm,
                                                    "std_floor": .05, "episodes": 128})
    write_json(root/"features/index.json", {"signature": cache, "dataset_signature": "same-dataset"})
    write_json(root/"dataset/manifest.json", {"signature": "same-dataset", "config_hash": digest(config["data"])})
    models = {}
    for task in config["experiment"]["tasks"]:
        write_json(root/"predictor"/task/"basis.json", {"fit_split": "train", "task": task, "episodes": 64,
            "sha256": f"basis-{task}", "fit_cache_signature": cache, "fit_dataset_signature": "same-dataset"})
        for seed in config["training"]["seeds"]:
            signature = {"scientific_hash": scientific_hash(config), "task": task, "experiment": config["experiment"],
                "checkpoint_selection": "validation_mse", "cache_signature": cache, "normalization_sha256": norm,
                "basis_sha256": f"basis-{task}", "code_hash": "same-training-code"}
            logs = [{"epoch": epoch, "step": 4*epoch, "validation_mse": 1/epoch,
                     "selection_metric": "validation_mse", "selection_score": 1/epoch,
                     "mix_counts": str(dict.fromkeys(CONDITIONS[:4], 64))} for epoch in range(1, 121)]
            logpath = root/"predictor"/task/f"seed_{seed}"/"training.json"
            write_json(logpath, {"signature": signature, "logs": logs,
                                "checkpoint_selection": "validation_mse", "experiment": config["experiment"],
                                "extra_teacher_forward_accounting": {"extra_forward_calls": 360, "extra_forward_examples": 23040},
                                "discarded_teacher_forward_accounting": {"calls": 0, "examples": 0}})
            models[f"{task}/{seed}"] = {"complete": True, "task": task, "seed": seed, "completed_epochs": 120,
                "final_step": 480, "selection_metric": "validation_mse", "best_epoch": 120,
                "best_sha256": digest([config["experiment"], task, seed, "best"]),
                "latest_sha256": digest([config["experiment"], task, seed, "latest"]),
                "training_json_sha256": file_hash(logpath), "signature": signature,
                "expected_samples_per_condition_epoch": 64, "expected_steps_per_epoch": 4, "issues": []}
    complete = {"complete": True, "required_epochs": 120, "models": models}
    write_json(root/"training_completion.json", complete)
    return complete


def _arm(root, role, *, recent=1., privileged=None, failed_seed=None):
    c = _config(root, role)
    privileged = (.6 if role == "reference" else .4) if privileged is None else privileged
    write_json(root/"resolved_config.json", c)
    write_json(root/"dataset/split_audit.json", {"passed": True, "signature": "same-dataset"})
    probes = {"results": []}
    for split in SPLITS[2:]:
        for variant, accuracy in (("gate_cue", 1.), ("recent_only", .5)):
            probes["results"].append({"task": "gate", "split": split, "variant": variant,
                "accuracy": {"mean": accuracy}, "wilson_effective_pairs": [.8, 1.]})
        for h in c["data"]["horizons"]:
            probes["results"].append({"task": "gate", "split": split, "horizon": h,
                "variant": "state_decoder_on_true_target", "accuracy": {"mean": 1.}, "position_mse": {"mean": .0001}})
    write_json(root/"probes/metrics.json", probes)
    raw = []
    for seed in c["training"]["seeds"]:
        for split in SPLITS[1:]:
            for h in c["data"]["horizons"]:
                for condition in CONDITIONS+["persistence"]:
                    for pair in range(32):
                        for sibling in (0, 1):
                            mse = {"recent_only": recent, "full_history": .4, "privileged_small_budget": privileged,
                                "random_small_budget": .8, "wrong_evidence": 1.2, "persistence": 1.}[condition]
                            if seed == failed_seed and split == "test_distractor_shift" and condition == "privileged_small_budget":
                                mse = .8
                            row = {"seed": seed, "task": "gate", "split": split, "horizon": h, "condition": condition,
                                "episode": f"{split}-{pair}-{sibling}", "pair": f"{split}-{pair}", "query": 10, "gap": 5}
                            row.update(dict.fromkeys(METRICS, 0.))
                            row.update(normalized_mse=mse, latent_mse=mse,
                                state_accuracy=.5 if condition in {"recent_only", "persistence"} else .9,
                                state_position_mse=.01 if condition in {"recent_only", "persistence"} else .004,
                                pair_accuracy=.5 if condition == "recent_only" else .9,
                                pair_margin=-1. if condition == "wrong_evidence" else 1., target_pair_distance=.2,
                                history_count=0)
                            raw.append(row)
    write_csv(root/"evaluation/raw_records.csv", raw)
    complete = _completion(root, c)
    checkpoints = {key: {"task": p["task"], "seed": p["seed"], "best_epoch": p["best_epoch"], "best_sha256": p["best_sha256"]}
                   for key, p in complete["models"].items()}
    metrics = {"mode": "confirm", "cache_signature": "same-cache", "raw_sha256": file_hash(root/"evaluation/raw_records.csv"),
        "evaluation_code_hash": code_hash("evaluation.py", "stats.py"), "training_completion": complete,
        "checkpoints": checkpoints, **summarize(c, raw)}
    write_json(root/"evaluation/metrics.json", metrics)
    return c


@pytest.fixture
def arms(tmp_path, monkeypatch):
    import phase1.v4_adoption as adoption
    reference, candidate = tmp_path/"reference", tmp_path/"candidate"
    source = _arm(reference, "reference")
    _arm(candidate, "candidate")
    source["experiment"] = {"arm": "reference", "stage": "development", "tasks": ["gate"]}
    proofs = read_json(reference/"evaluation/metrics.json")["training_completion"]["models"]
    monkeypatch.setattr(adoption, "validate_gate_diagnostic", lambda *args, **kwargs: {
        "status": "SUPPORTED", "diagnostic_sha256": "diagnostic-sha", "cache_signature": "same-cache", "reasons": [],
        "dataset_signature": "same-dataset", "source_audit_sha256": file_hash(reference/"dataset/split_audit.json"),
        "source_reference_raw_path": str(reference/"evaluation/raw_records.csv"),
        "source_reference_probes_path": str(reference/"probes/metrics.json"),
        "source_reference_config": source, "source_checkpoint_signatures": {key: p["signature"] for key, p in proofs.items()}})
    return reference, candidate


def compare(arms, **kwargs):
    return comparison(*arms, "gate", gate_diagnostic="synthetic-not-read.json", frozen_v3_root="immutable-fixture",
                      frozen_feature_root="fixture", frozen_source_directory="fixture", **kwargs)


def test_v4_adoption_uses_original_rules_and_never_emits_go(arms):
    result = compare(arms)
    assert result["status"] == "SUPPORTED", result["reasons"]
    assert not result["phase1_go"] and all(check["passed"] for check in result["checks"])
    assert len([c for c in result["checks"] if c["kind"] == "critical_absolute_mse_reduction"]) == 8
    assert len([c for c in result["checks"] if c["kind"] == "candidate_vs_reference_recent"]) == 24


def test_critical_per_seed_absolute_failure_not_hidden_by_pooled_gain(arms):
    _arm(arms[1], "candidate", failed_seed=43)
    result = compare(arms)
    assert result["status"] == "REJECTED"
    assert any("seed43" in r and "absolute" in r for r in result["reasons"])
    assert any(c["passed"] for c in result["checks"] if c.get("slice", "").endswith("seedall_seeds_mean") and c["kind"] == "critical_absolute_mse_reduction")


def test_candidate_weakened_recent_does_not_improve_fixed_reference_anchor(arms):
    _arm(arms[0], "reference", recent=.4, privileged=.6)
    _arm(arms[1], "candidate", recent=1., privileged=.4)
    result = compare(arms)
    assert result["status"] == "REJECTED"
    assert not any(c["passed"] for c in result["checks"] if c["kind"] == "candidate_vs_reference_recent")


@pytest.mark.parametrize("change", ["ranking", "weight", "threshold", "version", "normalization", "log", "probe", "summary"])
def test_scientific_or_incomplete_evidence_changes_cannot_pass(arms, change):
    root = arms[1]
    if change in {"ranking", "weight", "threshold", "version"}:
        c = read_json(root/"resolved_config.json")
        if change == "ranking": c["training"]["rank_weight"] = 0
        if change == "weight": c["experiment"]["common_anchor_weight"] = .5
        if change == "threshold": c["decision"]["latent_relative_improvement"] = .05
        if change == "version": c["experiment"]["protocol_version"] = 3
        write_json(root/"resolved_config.json", c)
    elif change == "normalization":
        value = read_json(root/"features/normalization.json")
        value["fit_split"] = "validation"
        write_json(root/"features/normalization.json", value)
    elif change == "log":
        path = root/"predictor/gate/seed_43/training.json"
        value = read_json(path)
        value["logs"][-1]["mix_counts"] = str(dict.fromkeys(CONDITIONS[:4], 63))
        write_json(path, value)
    elif change == "probe":
        path = root/"probes/metrics.json"
        value = read_json(path)
        value["results"].pop()
        write_json(path, value)
    else:
        path = root/"evaluation/metrics.json"
        value = read_json(path)
        value["per_seed"][0]["metrics"]["normalized_mse"]["mean"] += 1
        write_json(path, value)
    assert compare(arms)["status"] == "INCOMPLETE"


def test_motion_is_not_selected_by_a_support_label(arms):
    result = comparison(*arms, "motion", gate_diagnostic="unused", frozen_v3_root="unused",
                        frozen_feature_root="unused", frozen_source_directory="unused")
    assert result["status"] == "INCOMPLETE" and "No v4 motion" in result["reasons"][0]


def test_supported_diagnostic_label_without_actual_pair_export_cannot_authorize(tmp_path):
    path = tmp_path/"gate_common.json"
    write_json(path, {"schema_version": 1, "primary_arm": "reference", "status": "SUPPORTED",
                      "arms": {"reference": {"status": "SUPPORTED"}}})
    result = validate_gate_diagnostic(path, tmp_path/"source", tmp_path/"features", tmp_path/"code")
    assert result["status"] == "INCOMPLETE" and not result["phase1_go"]


def test_actual_pair_export_checksum_is_mandatory_before_provenance_recomputation(tmp_path, monkeypatch):
    import phase1.v4_adoption as adoption
    path, pair = tmp_path/"gate_common.json", tmp_path/"gate_pair_calibration.json"
    write_json(pair, {"fixture": True})
    write_json(path, {"schema_version": 1, "primary_arm": "reference", "status": "SUPPORTED",
                      "inputs": {"pair_calibration": {"sha256": "wrong"}}})
    def should_not_recompute(*args):
        raise AssertionError("Checksum mismatch must block source/pair recomputation")
    monkeypatch.setattr(adoption, "_recompute_gate_diagnostic", should_not_recompute)
    result = validate_gate_diagnostic(path, tmp_path, tmp_path, tmp_path)
    assert result["status"] == "INCOMPLETE" and "checksum" in result["reasons"][0]


def _locked_roots(tmp_path):
    roots, lock = {}, {"schema_version": 1, "protocol_version": 4, "stage": "confirmation", "data_seed": 9137,
                       "models": {}, "resolved_config_sha256": {}, "training_completion_sha256": {}}
    for role in ("reference", "candidate"):
        root = tmp_path/role
        c = _config(root, role, "confirmation")
        write_json(root/"resolved_config.json", c)
        complete = _completion(root, c)
        lock["models"].update({f"{role}/{key}": value for key, value in complete["models"].items()})
        lock["resolved_config_sha256"][role] = file_hash(root/"resolved_config.json")
        lock["training_completion_sha256"][role] = file_hash(root/"training_completion.json")
        roots[role] = root
    path = tmp_path/"confirmation_lock.json"
    write_json(path, lock)
    return roots["reference"], roots["candidate"], path


def test_twelve_model_lock_is_only_provenance_never_confirmation_go(tmp_path):
    result = validate_confirmation_lock(*_locked_roots(tmp_path))
    assert result["status"] == "LOCK_VALID", result["reasons"]
    assert result["models"] == 12 and not result["phase1_go"] and not result["can_evaluate"]


@pytest.mark.parametrize("change", ["missing_model", "old_seed", "partial", "old_basis", "log_after_lock"])
def test_confirmation_lock_fails_closed_for_incomplete_or_stale_evidence(tmp_path, change):
    reference, candidate, path = _locked_roots(tmp_path)
    lock = read_json(path)
    if change == "missing_model":
        lock["models"].pop("candidate/motion/43")
        write_json(path, lock)
    elif change == "old_seed":
        c = read_json(candidate/"resolved_config.json")
        c["data"]["seed"] = 6301
        write_json(candidate/"resolved_config.json", c)
    elif change == "partial":
        proof = read_json(candidate/"training_completion.json")
        proof["models"]["motion/43"]["completed_epochs"] = 119
        write_json(candidate/"training_completion.json", proof)
    elif change == "old_basis":
        basis = read_json(candidate/"predictor/motion/basis.json")
        basis["fit_dataset_signature"] = "old-6301-dataset"
        write_json(candidate/"predictor/motion/basis.json", basis)
    else:
        log = candidate/"predictor/motion/seed_43/training.json"
        value = read_json(log)
        value["logs"][-1]["validation_mse"] += 1
        write_json(log, value)
    assert validate_confirmation_lock(reference, candidate, path)["status"] == "INCOMPLETE"


def _small_matched_rows(task, recent=1., candidate=.4):
    reference, alternative = [], []
    for seed in (17, 29, 43):
        for split in SPLITS[2:]:
            for horizon in (3, 6):
                for condition in ("recent_only", "privileged_small_budget"):
                    for sibling in (0, 1):
                        row = {"seed": seed, "task": task, "split": split, "horizon": horizon, "condition": condition,
                               "pair": split, "episode": f"{split}-{sibling}"}
                        reference.append({**row, "normalized_mse": recent if condition == "recent_only" else .6})
                        alternative.append({**row, "normalized_mse": 1. if condition == "recent_only" else candidate})
    return reference, alternative


def test_paired_comparison_preserves_low_baseline_absolute_rule():
    from phase1.v4_adoption import _matched_checks
    c = _config("synthetic", "candidate")
    checks = _matched_checks(c, *_small_matched_rows("gate", recent=.01, candidate=.005), "gate")
    fixed = [check for check in checks if check["kind"] == "candidate_vs_reference_recent"]
    assert all(check["improvement"]["scale"] == "absolute_error_reduction" and check["point_threshold"] == .01
               and not check["passed"] for check in fixed)


def test_generic_statistics_keep_motion_long_gap_critical_without_selecting_candidate():
    from phase1.v4_adoption import _matched_checks
    checks = _matched_checks(_config("synthetic", "candidate"), *_small_matched_rows("motion"), "motion")
    absolute = [check for check in checks if check["kind"] == "critical_absolute_mse_reduction"]
    assert len(absolute) == 8 and all("test_long_gap" in check["slice"] for check in absolute)


@pytest.mark.parametrize("change", [None, "label", "summary", "source_checksum", "recent_equality"])
def test_diagnostic_declaration_must_match_actual_recomputed_source_pairs(tmp_path, monkeypatch, change):
    import phase1.v4_adoption as adoption
    path, pair = tmp_path/"gate_common.json", tmp_path/"gate_pair_calibration.json"
    write_json(pair, {"synthetic_pair_fixture": True})
    verify = {"scientific_source_modified_during_run": False,
              "current_own_best_latest_match_complete_evaluation_proofs": True, "source_code_hash": "synthetic-source-code"}
    probe_path = tmp_path/"gate/reference/probes/metrics.json"
    write_json(probe_path, {"synthetic": True})
    verify["actual_hpc_compact_sha256"] = {"gate/reference/probes/metrics.json": file_hash(probe_path)}
    write_json(tmp_path/"verification.json", verify)
    inputs = {"pair_calibration": {"sha256": file_hash(pair)}, "data_audit": {"sha256": "actual-audit"},
              "reference": {"raw": {"sha256": "actual-raw"}}}
    summary = {"status": "SUPPORTED", "core_slice_count": 6, "core_passed_count": 6,
               "id_regressions_vs_original_history": [], "slices": [{"synthetic_pairs": 32, "hybrid_vs_recent": {"mean": .2, "low": .1, "high": .3}}]}
    declared = {"schema_version": 1, "primary_arm": "reference", "status": "SUPPORTED", "inputs": copy.deepcopy(inputs),
                "arms": {"reference": {**copy.deepcopy(summary), "pair_export_verified": True,
                                       "direct_recent_prediction_equality_verified": True}}}
    if change == "label": declared["status"] = "NOT_SUPPORTED"
    if change == "summary": declared["arms"]["reference"]["core_passed_count"] = 5
    if change == "source_checksum": declared["inputs"]["reference"]["raw"]["sha256"] = "changed"
    if change == "recent_equality": declared["arms"]["reference"]["direct_recent_prediction_equality_verified"] = False
    write_json(path, declared)
    c = _config(tmp_path, "reference")
    metrics = {"checkpoints": {"gate/17": {"best_sha256": "checked-original-best"}}, "cache_signature": "cache",
               "training_completion": {"models": {"gate/17": {"signature": {"fixture": True}}}}}
    # Shared dataset provenance is returned separately by the actual script;
    # this mock only substitutes its independently verified computation boundary.
    def recomputed(*args):
        return summary, inputs, c, metrics, {**verify, "dataset_signature": "dataset"}
    monkeypatch.setattr(adoption, "_recompute_gate_diagnostic", recomputed)
    result = validate_gate_diagnostic(path, tmp_path, tmp_path, tmp_path)
    assert result["status"] == ("SUPPORTED" if change is None else "INCOMPLETE"), result["reasons"]


def test_actual_frozen_source_rejects_substituted_current_training_bytes(tmp_path):
    from phase1.v4_adoption import verify_frozen_source
    source = tmp_path/"src/phase1"
    source.mkdir(parents=True)
    for name in ("training.py", "model.py", "inputs.py"):
        (source/name).write_text("# synthetic frozen original " + name)
    actual = digest({name: file_hash(source/name) for name in ("training.py", "model.py", "inputs.py")})
    verification = {"source_code_hash": actual}
    models = {"gate/17": {"signature": {"code_hash": actual}}}
    assert verify_frozen_source(tmp_path, verification, models) == actual
    current = Path(__file__).resolve().parents[1]/"src/phase1/training.py"
    (source/"training.py").write_bytes(current.read_bytes())
    with pytest.raises(ValueError, match="Actual frozen"):
        verify_frozen_source(tmp_path, verification, models)


@pytest.mark.parametrize("change", ["extra_seed", "query", "pair"])
def test_raw_keys_and_timing_bind_exactly_to_frozen_source(change):
    from phase1.v4_adoption import bind_raw_to_source
    source, _ = _small_matched_rows("gate")
    for row in source:
        row.update(query=10, gap=5)
    changed = copy.deepcopy(source)
    if change == "extra_seed": changed.append({**changed[0], "seed": 99})
    if change == "query": changed[0]["query"] += 1
    if change == "pair": changed[0]["pair"] = "mislabeled-pair"
    with pytest.raises(ValueError):
        bind_raw_to_source(changed, source)


def test_asymmetric_interruption_cost_is_reported_but_retained_budgets_match(arms):
    root = arms[1]
    path = root/"predictor/gate/seed_17/training.json"
    value = read_json(path)
    value["extra_teacher_forward_accounting"]["extra_forward_calls"] += 1
    value["extra_teacher_forward_accounting"]["extra_forward_examples"] += 64
    value["discarded_teacher_forward_accounting"] = {"calls": 1, "examples": 64}
    write_json(path, value)
    metrics = read_json(root/"evaluation/metrics.json")
    metrics["training_completion"]["models"]["gate/17"]["training_json_sha256"] = file_hash(path)
    write_json(root/"evaluation/metrics.json", metrics)
    result = compare(arms)
    assert result["status"] == "SUPPORTED", result["reasons"]
    counters = result["teacher_compute"]["17"]
    assert counters["reference"]["retained"] == counters["candidate"]["retained"]
    assert counters["candidate"]["actual"]["extra_forward_calls"] == counters["reference"]["actual"]["extra_forward_calls"]+1
