"""V4 Gate adoption and confirmation-lock checks over compact own artifacts.

No training, checkpoint loading, test evaluation, or motion candidate selection
occurs here.  Development SUPPORT is not GO.  A valid twelve-model lock is only
a provenance check: independent confirmation still needs both task adoption
gates, the original decision rules, and evaluator enforcement of this lock.
"""
import ast
import copy
import importlib.util
from pathlib import Path

import numpy as np

from .config import DEFAULT, SPLITS, scientific_hash
from .evaluation import summarize
from .reporting import _development_artifacts, _raw_records, decision
from .stats import clustered_interval, paired_improvement
from .utils import digest, file_hash, read_json, write_json


SEEDS = [17, 29, 43]
GATE_VARIANT = "gate_common_anchor_v1"
_FAILURES = (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, SyntaxError, ImportError)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _science(config):
    value = copy.deepcopy(config)
    for key in ("output", "device", "threads", "experiment"):
        value.pop(key, None)
    return value


def _paired_config(config):
    value = copy.deepcopy(config)
    for key in ("output", "device", "threads"):
        value.pop(key, None)
    for key in ("role", "variant", "common_anchor_weight"):
        value["experiment"].pop(key, None)
    return value


def _profile(config, role, stage):
    exp = config.get("experiment", {})
    _require(exp.get("protocol_version") == 4 and exp.get("role") == role and exp.get("stage") == stage,
             "V4 requires explicit protocol_version=4, role and stage")
    _require(config["mode"] == "confirm" and config["encoder"]["backend"] == "ijepa"
             and config["training"]["seeds"] == SEEDS and config["training"]["epochs"] == 120,
             "V4 research requires official I-JEPA and all three fixed 120-epoch models")
    _require(config["encoder"]["repository"] == "facebook/ijepa_vith14_1k"
             and config["encoder"]["revision"] == "f157467ea509bc356ff9f61fd3c0d840eec5e04e"
             and config["encoder"]["pooling"] == "grid2" and config["encoder"]["dtype"] == "float32"
             and config["encoder"]["batch_size"] == 2 and config["normalization"]["std_floor"] == .05,
             "Original frozen encoder or normalization recipe changed")
    _require(config["decision"] == DEFAULT["decision"], "Original Phase 1 decision thresholds changed")
    _require(config["training"]["pair_weight"] == 2 and config["training"]["rank_weight"] == .5
             and config["model"]["dropout"] == 0 and config["data"]["horizons"] == [3, 6],
             "Original pair/ranking loss, deterministic model or horizons changed")
    _require(exp.get("common_anchor_weight") == (0 if role == "reference" else 1),
             "Gate common anchor permits only matched weight zero versus fixed weight one")
    _require(exp.get("matched_teacher_forwards") is True, "V4 requires matched teacher forward budgets")
    if stage == "development":
        _require(exp.get("tasks") == ["gate"] and config["data"]["seed"] == 6301,
                 "Only Gate development on revealed seed 6301 is currently preregistered")
        _require(exp.get("variant") == ("reference" if role == "reference" else GATE_VARIANT),
                 "Unexpected v4 Gate reference/candidate variant")
    else:
        _require(exp.get("tasks") == ["gate", "motion"] and config["data"]["seed"] == 9137,
                 "Independent confirmation requires both tasks and fresh seed 9137")
        variants = exp.get("variant")
        _require(isinstance(variants, dict) and set(variants) == {"gate", "motion"},
                 "Confirmation variant must explicitly name each task")
        _require(variants["gate"] == ("reference" if role == "reference" else GATE_VARIANT),
                 "Unexpected confirmation Gate variant")
        _require(variants["motion"] == "reference" if role == "reference" else
                 isinstance(variants["motion"], str) and bool(variants["motion"]),
                 "A confirmation role cannot have an empty motion variant")
    return exp


def _same_tree(actual, expected):
    """Compare independently recomputed summaries, allowing only float roundoff."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(_same_tree(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(_same_tree(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-7, atol=1e-10)
    return actual == expected


def _recompute_gate_diagnostic(path, frozen_v3_root, frozen_feature_root, frozen_source_directory, pair_export):
    # Reuse the original compact diagnostic's exact provenance and pair-CI
    # checks, but read evaluator/statistics bytes from the frozen v3 checkout.
    script = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_gate_common.py"
    spec = importlib.util.spec_from_file_location("v4_frozen_gate_diagnostic_checks", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = Path(frozen_source_directory).resolve()
    actual_inputs = {}
    shared = module.validate_shared(Path(frozen_v3_root), Path(frozen_feature_root), actual_inputs)
    data = module.load_arm(Path(frozen_v3_root), "reference", shared, actual_inputs)
    verify_frozen_source(frozen_source_directory, shared["verification"], data["metrics"]["training_completion"]["models"])
    export = read_json(pair_export)
    stats = module.load_stats()
    pairs = module.load_pair_rows(export, "reference", data, stats)
    summary = module.summarize_arm(data, pairs, stats)
    actual_inputs["pair_calibration"] = {"path": str(pair_export), "sha256": file_hash(pair_export)}
    verified = {**shared["verification"], "dataset_signature": shared["reuse_manifest"]["dataset_signature"]}
    return summary, actual_inputs, data["config"], data["metrics"], verified


def verify_frozen_source(source_directory, verification, models):
    """Bind actual old trainer/model/input bytes, never a copied hash label."""
    names = ("training.py", "model.py", "inputs.py")
    actual = digest({name: file_hash(Path(source_directory)/"src/phase1"/name) for name in names})
    _require(actual == verification["source_code_hash"] and bool(models)
             and all(value["signature"]["code_hash"] == actual for value in models.values()),
             "Actual frozen training/model/input source differs from all verified original checkpoints")
    return actual


def validate_gate_diagnostic(path, frozen_v3_root, frozen_feature_root, frozen_source_directory, *, pair_export=None):
    """Recompute reference SUPPORT from actual pair export and immutable v3 files.

    Diagnostic status labels alone never authorize training.  The caller must
    supply the old source checkout, old compact run and old feature metadata
    explicitly; paths recorded in JSON are not treated as trusted locations.
    """
    result = {"status": "INCOMPLETE", "reasons": [], "scope": "development mechanism diagnostic only", "phase1_go": False}
    try:
        path = Path(path)
        declared = read_json(path)
        pair_export = Path(pair_export) if pair_export is not None else path.with_name("gate_pair_calibration.json")
        _require(declared.get("schema_version") == 1 and declared.get("primary_arm") == "reference",
                 "Gate diagnostic schema/primary arm mismatch")
        _require(file_hash(pair_export) == declared["inputs"]["pair_calibration"]["sha256"],
                 "Gate diagnostic lacks the actual matching pair-export checksum")
        summary, inputs, source_config, source_metrics, verification = _recompute_gate_diagnostic(
            path, frozen_v3_root, frozen_feature_root, frozen_source_directory, pair_export)
        _require(verification.get("scientific_source_modified_during_run") is False
                 and verification.get("current_own_best_latest_match_complete_evaluation_proofs") is True,
                 "Frozen v3 own-checkpoint verification is missing")
        _require(source_config["data"]["seed"] == 6301 and source_config["training"]["rank_weight"] == .5,
                 "Gate diagnostic is not bound to the original ranked v3 reference")
        for name, actual in inputs.items():
            if name == "reference":
                for part, record in actual.items():
                    _require(declared["inputs"][name][part]["sha256"] == record["sha256"], "Gate diagnostic old reference artifact changed")
            else:
                _require(declared["inputs"][name]["sha256"] == actual["sha256"], "Gate diagnostic source artifact checksum changed")
        arm = declared["arms"]["reference"]
        _require(arm.get("pair_export_verified") is True and arm.get("direct_recent_prediction_equality_verified") is True,
                 "Gate diagnostic lacks direct paired recent equality and pair-export proof")
        for key, value in summary.items():
            _require(_same_tree(arm.get(key), value), "Gate diagnostic summary does not match independently recomputed pair statistics")
        _require(declared["status"] == summary["status"], "Gate diagnostic label differs from actual reference result")
        probe_path = Path(frozen_v3_root)/"gate/reference/probes/metrics.json"
        _require(file_hash(probe_path) == verification["actual_hpc_compact_sha256"]["gate/reference/probes/metrics.json"],
                 "Frozen original evidence/readout probe differs from verified compact source")
        result.update(status="SUPPORTED" if summary["status"] == "SUPPORTED" else "REJECTED",
                      diagnostic_sha256=file_hash(path), pair_export_sha256=file_hash(pair_export),
                      source_verification_sha256=file_hash(Path(frozen_v3_root)/"verification.json"),
                      source_code_hash=verification["source_code_hash"], source_reference_config=source_config,
                      source_reference_checkpoints=source_metrics["checkpoints"],
                      source_checkpoint_signatures={key: value["signature"] for key, value in source_metrics["training_completion"]["models"].items()},
                      cache_signature=source_metrics["cache_signature"],
                      dataset_signature=verification["dataset_signature"],
                      source_audit_sha256=inputs["data_audit"]["sha256"], recomputed=summary)
        result.update(source_reference_raw_path=str(Path(frozen_v3_root)/"gate/reference/evaluation/raw_records.csv"),
                      source_reference_probes_path=str(probe_path), source_reference_probes_sha256=file_hash(probe_path))
        if result["status"] != "SUPPORTED":
            result["reasons"].append("The predeclared original reference paired Gate diagnostic did not support this candidate")
    except _FAILURES as exc:
        result["reasons"].append(str(exc))
    return result


def _logs_and_shared(root, config, completion, tasks):
    expected = {f"{task}/{seed}" for task in tasks for seed in SEEDS}
    _require(completion.get("complete") and completion.get("required_epochs") == 120
             and set(completion.get("models", {})) == expected, "Incomplete scheduled model completion proof")
    normalization = read_json(Path(root)/"features/normalization.json")
    _require(normalization.get("fit_split") == "train" and normalization.get("std_floor") == config["normalization"]["std_floor"]
             and normalization.get("episodes") == 2*len(config["data"]["tasks"])*config["data"]["pairs"]["train"],
             "Normalization must use only all declared training episodes and the fixed std floor")
    if config["experiment"]["stage"] == "confirmation":
        index = read_json(Path(root)/"features/index.json")
        dataset = read_json(Path(root)/"dataset/manifest.json")
        _require(dataset["config_hash"] == digest(config["data"]) and index["dataset_signature"] == dataset["signature"],
                 "Confirmation cache does not originate from fresh seed 9137")
        _require(normalization.get("cache_signature") == index["signature"], "Fresh normalization/cache provenance mismatch")
    steps = 4 * int(np.ceil(config["data"]["pairs"]["train"] / (config["training"]["batch_size"]//2)))
    count = 2 * config["data"]["pairs"]["train"]
    history_conditions = {"recent_only", "full_history", "privileged_small_budget", "random_small_budget"}
    for task in tasks:
        basis = read_json(Path(root)/"predictor"/task/"basis.json")
        _require(basis.get("fit_split") == "train" and basis.get("task") == task
                 and basis.get("episodes") == 2*config["data"]["pairs"]["train"], "Basis must use only the declared task training episodes")
        if config["experiment"]["stage"] == "confirmation":
            _require(basis.get("fit_cache_signature") == index["signature"]
                     and basis.get("fit_dataset_signature") == dataset["signature"],
                     "Confirmation basis lacks explicit fresh dataset/cache fit provenance")
        for seed in SEEDS:
            proof = completion["models"][f"{task}/{seed}"]
            signature = proof["signature"]
            _require(proof.get("complete") and proof.get("task") == task and proof.get("seed") == seed
                     and proof.get("completed_epochs") == 120 and proof.get("final_step") == 120*steps
                     and proof.get("selection_metric") == "validation_mse", "Incomplete exact epochs/steps/task/seed/selection proof")
            _require(proof.get("issues") == [] and proof.get("expected_samples_per_condition_epoch") == count
                     and proof.get("expected_steps_per_epoch") == steps and bool(proof.get("best_sha256"))
                     and bool(proof.get("latest_sha256")), "Actual model proof lacks balanced counts or own best/latest hashes")
            _require(signature["scientific_hash"] == scientific_hash(config) and signature.get("task") == task
                     and signature.get("experiment") == config["experiment"]
                     and signature.get("checkpoint_selection") == "validation_mse", "V4 checkpoint protocol provenance mismatch")
            _require(signature["normalization_sha256"] == normalization["sha256"]
                     and signature["basis_sha256"] == basis["sha256"]
                     and signature["cache_signature"] == normalization["cache_signature"], "Cache/normalization/basis provenance mismatch")
            logpath = Path(root)/"predictor"/task/f"seed_{seed}"/"training.json"
            _require(file_hash(logpath) == proof["training_json_sha256"], "Actual training-log checksum mismatch")
            training = read_json(logpath)
            _require(training["signature"] == signature and len(training["logs"]) == 120
                     and training.get("checkpoint_selection") == "validation_mse"
                     and training.get("experiment") == config["experiment"], "Actual training logs/signature/selection incomplete")
            for epoch, row in enumerate(training["logs"], 1):
                counts = ast.literal_eval(row["mix_counts"]) if isinstance(row["mix_counts"], str) else row["mix_counts"]
                _require(row["epoch"] == epoch and row["step"] == epoch*steps and set(counts) == history_conditions
                         and set(counts.values()) == {count}, "Unbalanced or noncontiguous actual training logs")
                _require(row.get("selection_metric") == "validation_mse" and row.get("selection_score") == row["validation_mse"],
                         "Actual epoch selection was not the common validation MSE")
                _require(all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float))), "Non-finite actual training log")
            _require(proof["best_epoch"] == min(training["logs"], key=lambda r: r["validation_mse"])["epoch"],
                     "Selected checkpoint is not the earliest minimum common validation MSE")
            if task == "gate":
                counters = training["extra_teacher_forward_accounting"]
                discarded = training["discarded_teacher_forward_accounting"]
                batches = int(np.ceil(config["data"]["pairs"]["train"]/(config["training"]["batch_size"]//2)))
                _require(counters["extra_forward_calls"]-discarded["calls"] == 120*3*batches
                         and counters["extra_forward_examples"]-discarded["examples"] == 120*3*count,
                         "Gate teacher forwards do not match the complete prescribed retained budget")
    return normalization


def _probe_coverage(probes, task, config):
    def finite_tree(value):
        if isinstance(value, dict):
            return all(finite_tree(item) for item in value.values())
        if isinstance(value, list):
            return all(finite_tree(item) for item in value)
        return bool(np.isfinite(value)) if isinstance(value, (int, float)) else True
    _require(finite_tree(probes), "Non-finite evidence probe/readout")
    present = {(row["task"], row["split"], row["variant"], row.get("horizon")) for row in probes["results"]}
    for split in SPLITS[2:]:
        for variant in ("gate_cue" if task == "gate" else "motion_two_frame", "recent_only"):
            _require((task, split, variant, None) in present, "Candidate original evidence probe is incomplete")
        for horizon in config["data"]["horizons"]:
            _require((task, split, "state_decoder_on_true_target", horizon) in present, "Candidate original true-target readout is incomplete")


def _matched_checks(config, reference_rows, candidate_rows, task):
    keys = ("seed", "task", "split", "horizon", "condition", "pair", "episode")
    reference = {tuple(row[k] for k in keys): row for row in reference_rows}
    candidate = {tuple(row[k] for k in keys): row for row in candidate_rows}
    _require(set(reference) == set(candidate), "Reference/candidate raw pair and sibling keys differ")
    checks = []
    critical = "test_distractor_shift" if task == "gate" else "test_long_gap"
    rule, repeats = config["decision"], config["evaluation"]["bootstrap"]
    for split in SPLITS[2:]:
        for horizon in config["data"]["horizons"]:
            for seed in SEEDS + ["all_seeds_mean"]:
                grouped = {}
                for row in candidate_rows:
                    if (row["split"] != split or row["horizon"] != horizon or row["condition"] != "privileged_small_budget"
                            or (seed != "all_seeds_mean" and row["seed"] != seed)):
                        continue
                    key = tuple(row[k] for k in keys)
                    recent_key = (*key[:4], "recent_only", *key[5:])
                    grouped.setdefault((row["pair"], row["episode"]), []).append([
                        reference[key]["normalized_mse"], row["normalized_mse"], reference[recent_key]["normalized_mse"]])
                pairkeys = sorted(grouped)
                _require(bool(pairkeys), "Missing matched comparison slice")
                values = np.array([np.mean(grouped[key], axis=0) for key in pairkeys])
                groups = [key[0] for key in pairkeys]
                name = f"{task}/{split}/h{horizon}/seed{seed}"
                if split == critical:
                    interval = clustered_interval(values[:, 0]-values[:, 1], groups, repeats)
                    checks.append({"kind": "critical_absolute_mse_reduction", "slice": name,
                                   "passed": interval["mean"] > 0 and interval["low"] > 0,
                                   "absolute_mse_reduction": interval})
                interval = paired_improvement(values[:, 2], values[:, 1], groups, repeats,
                                              rule["baseline_floor"], rule["absolute_latent_gain"])
                threshold = rule["latent_relative_improvement"] if interval["scale"].startswith("relative") else rule["absolute_latent_gain"]
                checks.append({"kind": "candidate_vs_reference_recent", "slice": name,
                               "passed": interval["mean"] >= threshold and interval["low"] > 0,
                               "improvement": interval, "point_threshold": threshold})
    return checks


def bind_raw_to_source(raw, source):
    fields = ("seed", "task", "split", "horizon", "condition", "pair", "episode")
    a = {tuple(row[key] for key in fields): row for row in raw}
    b = {tuple(row[key] for key in fields): row for row in source}
    _require(len(a) == len(raw) and len(b) == len(source) and set(a) == set(b),
             "V4 raw full keys differ from the actual frozen 6301 reference source")
    _require(all(a[key][field] == b[key][field] for key in b for field in ("query", "gap")),
             "V4 query/gap does not match the audited original source episode")


def comparison(reference_root, candidate_root, task, *, gate_diagnostic, frozen_v3_root,
               frozen_feature_root, frozen_source_directory, pair_export=None, output=None):
    """Fresh matched v4 Gate development adoption; motion remains unselected."""
    result = {"protocol_version": 4, "task": task, "status": "INCOMPLETE", "phase1_go": False,
              "scope": "development adoption only; revealed 6301 tests are not confirmation", "checks": [], "reasons": []}
    try:
        _require(task == "gate", "No v4 motion candidate is preregistered; frozen diagnosis alone cannot create SUPPORT")
        diagnostic = validate_gate_diagnostic(gate_diagnostic, frozen_v3_root, frozen_feature_root,
                                             frozen_source_directory, pair_export=pair_export)
        result["diagnostic"] = diagnostic
        _require(diagnostic["status"] != "INCOMPLETE", "Incomplete actual paired diagnostic provenance")
        if diagnostic["status"] != "SUPPORTED":
            result.update(status="REJECTED", reasons=diagnostic["reasons"])
        else:
            ref, rm, rr, ra, _ = _development_artifacts(reference_root, task)
            cand, cm, cr, ca, cp = _development_artifacts(candidate_root, task)
            for c, role in ((ref, "reference"), (cand, "candidate")):
                exp = _profile(c, role, "development")
                _require(exp.get("gate_diagnostic_sha256") == diagnostic["diagnostic_sha256"], "Training protocol is not bound to the authorized actual diagnostic")
                _require(_science(c) == _science(diagnostic["source_reference_config"]), "Scientific settings changed from the frozen original v3 reference")
            _require(_paired_config(ref) == _paired_config(cand), "Matched v4 arms differ beyond Gate common weight zero versus one")
            source_raw = _raw_records(diagnostic["source_reference_raw_path"])
            source_probes = read_json(diagnostic["source_reference_probes_path"])
            for root, c, metrics, raw in ((reference_root, ref, rm, rr), (candidate_root, cand, cm, cr)):
                _logs_and_shared(root, c, metrics["training_completion"], [task])
                bind_raw_to_source(raw, source_raw)
                probes = read_json(Path(root)/"probes/metrics.json")
                _require(_same_tree(probes, source_probes), "Gate evidence/readout probes differ from the verified original same-input probes")
                _require(all(np.isfinite(value) for row in raw for value in row.values() if isinstance(value, (int, float))),
                         "Non-finite raw evaluation record")
                _require(all(row[key] >= 0 for row in raw for key in ("normalized_mse", "latent_mse", "state_position_mse", "target_pair_distance"))
                         and all(0 <= row[key] <= 1 for row in raw for key in ("state_accuracy", "pair_accuracy")),
                         "Impossible raw error/accuracy value")
                actual = summarize(c, raw)
                _require(all(_same_tree(metrics[key], actual[key]) for key in actual), "Evaluation summaries differ from actual paired raw records")
            _require(ra["passed"] and ca["passed"] and ra.get("signature") == ca.get("signature") == diagnostic["dataset_signature"],
                     "Matched source data audits do not pass or differ")
            _require(all(file_hash(Path(root)/"dataset/split_audit.json") == diagnostic["source_audit_sha256"]
                         for root in (reference_root, candidate_root)), "Development source audit changed from the diagnosed dataset")
            for seed in SEEDS:
                rs = rm["training_completion"]["models"][f"{task}/{seed}"]["signature"]
                cs = cm["training_completion"]["models"][f"{task}/{seed}"]["signature"]
                for key in ("cache_signature", "normalization_sha256", "basis_sha256", "code_hash"):
                    _require(rs[key] == cs[key], f"Matched arms differ in {key}")
                _require(rs["cache_signature"] == diagnostic["cache_signature"], "Development changed the diagnosed feature cache")
                for key in ("cache_signature", "normalization_sha256", "basis_sha256"):
                    _require(rs[key] == diagnostic["source_checkpoint_signatures"][f"{task}/{seed}"][key],
                             f"Development changed the diagnosed original {key}")
                ref_log = read_json(Path(reference_root)/"predictor/gate"/f"seed_{seed}"/"training.json")
                cand_log = read_json(Path(candidate_root)/"predictor/gate"/f"seed_{seed}"/"training.json")
                def effective(log):
                    total, lost = log["extra_teacher_forward_accounting"], log["discarded_teacher_forward_accounting"]
                    return (total["extra_forward_calls"]-lost["calls"], total["extra_forward_examples"]-lost["examples"])
                _require(effective(ref_log) == effective(cand_log), "Matched retained teacher compute budgets differ")
                result.setdefault("teacher_compute", {})[str(seed)] = {
                    role: {"actual": log["extra_teacher_forward_accounting"],
                           "discarded": log["discarded_teacher_forward_accounting"], "retained": list(effective(log))}
                    for role, log in (("reference", ref_log), ("candidate", cand_log))}
            _probe_coverage(cp, task, cand)
            status, reasons, original = decision(cand, ca, cp, cm)
            _require(status != "DEVELOPMENT_INCOMPLETE", "Candidate original task checks are incomplete")
            result["checks"].append({"kind": "candidate_original_task_rules", "passed": status == "DEVELOPMENT_PASS",
                                     "status": status, "checks": original, "reasons": reasons})
            result["checks"].extend(_matched_checks(cand, rr, cr, task))
            result["reasons"].extend(reasons)
            result["reasons"].extend(check["slice"]+" "+check["kind"]+" did not pass"
                                     for check in result["checks"] if not check["passed"] and "slice" in check)
            result.update(status="SUPPORTED" if all(check["passed"] for check in result["checks"]) else "REJECTED",
                          candidate_original_status=status, raw_sha256={"reference": rm["raw_sha256"], "candidate": cm["raw_sha256"]},
                          experiment={"reference": ref["experiment"], "candidate": cand["experiment"]}, preregistered_rules=cand["decision"])
    except _FAILURES as exc:
        result["status"] = "INCOMPLETE"
        result["reasons"].append(str(exc))
    if output is not None:
        out = Path(output).resolve()
        frozen = Path(frozen_v3_root).resolve()
        _require(out != frozen and frozen not in out.parents, "Cannot write comparison into immutable v3 evidence")
        write_json(out/"comparison.json", result)
    return result


def validate_confirmation_lock(reference_root, candidate_root, lock_path):
    """Check the fresh shared provenance and all twelve locked model proofs.

    Role roots must contain actual training_completion.json generated by the
    checkpoint verifier before evaluation.  The lock binds that proof, both
    resolved configs and every training log.  This function never emits GO or
    authorizes an unselected motion candidate/test evaluation by itself.
    """
    result = {"status": "INCOMPLETE", "phase1_go": False, "can_evaluate": False, "reasons": []}
    try:
        lock = read_json(lock_path)
        _require(lock.get("schema_version") == 1 and lock.get("protocol_version") == 4
                 and lock.get("stage") == "confirmation" and lock.get("data_seed") == 9137,
                 "Wrong independent confirmation lock schema/stage/data seed")
        expected = {f"{role}/{task}/{seed}" for role in ("reference", "candidate") for task in ("gate", "motion") for seed in SEEDS}
        _require(set(lock.get("models", {})) == expected, "Confirmation lock must bind all twelve scheduled models")
        configs, proofs = {}, {}
        for role, root in (("reference", reference_root), ("candidate", candidate_root)):
            root = Path(root)
            config = read_json(root/"resolved_config.json")
            _profile(config, role, "confirmation")
            _require(file_hash(root/"resolved_config.json") == lock["resolved_config_sha256"][role], "Locked resolved configuration changed")
            completion = read_json(root/"training_completion.json")
            _require(file_hash(root/"training_completion.json") == lock["training_completion_sha256"][role], "Locked actual completion proof changed")
            _logs_and_shared(root, config, completion, ["gate", "motion"])
            for key, proof in completion["models"].items():
                _require(lock["models"][f"{role}/{key}"] == proof, "Locked best/latest model provenance changed")
            configs[role], proofs[role] = config, completion
        _require(_paired_config(configs["reference"]) == _paired_config(configs["candidate"]), "Confirmation roles changed non-variant scientific settings")
        for task in ("gate", "motion"):
            for seed in SEEDS:
                rs, cs = (proofs[role]["models"][f"{task}/{seed}"]["signature"] for role in ("reference", "candidate"))
                for key in ("cache_signature", "normalization_sha256", "basis_sha256", "code_hash"):
                    _require(rs[key] == cs[key], f"Confirmation roles do not share fresh {key}")
        result.update(status="LOCK_VALID", models=12, lock_sha256=file_hash(lock_path),
                      required_next_step="Both actual development adoption gates and evaluator lock enforcement remain mandatory; no v4 motion candidate is selected here.")
    except _FAILURES as exc:
        result["reasons"].append(str(exc))
    return result
