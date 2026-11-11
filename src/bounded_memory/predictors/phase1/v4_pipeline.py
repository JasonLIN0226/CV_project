"""Explicit, conditional Gate-only v4 training boundary; no job orchestration."""
import copy
from dataclasses import dataclass
from pathlib import Path

from .utils import file_hash, read_json


GATE_VARIANT = "gate_common_anchor_v1"


@dataclass(frozen=True)
class GateTrainingAuthorization:
    diagnostic_path: str
    frozen_v3_root: str
    frozen_feature_root: str
    frozen_source_directory: str
    audit_path: str
    pair_export: str | None = None


def v4_tasks(config):
    exp = config["experiment"]
    if exp.get("protocol_version") != 4 or exp.get("role") not in {"reference", "candidate"}:
        raise ValueError("V4 requires explicit protocol_version=4 and reference/candidate role")
    if exp.get("stage") != "development" or exp.get("tasks") != ["gate"]:
        raise ValueError("V4 currently supports Gate development only; independent confirmation awaits a declared Motion candidate")
    role = exp["role"]
    if exp.get("variant") != ("reference" if role == "reference" else GATE_VARIANT):
        raise ValueError("Unknown v4 Gate variant")
    if exp.get("common_anchor_weight") != (0 if role == "reference" else 1):
        raise ValueError("V4 common anchor is fixed at reference zero versus candidate one")
    if exp.get("matched_teacher_forwards") is not True:
        raise ValueError("V4 fixes matched_teacher_forwards=true for both roles")
    if config["training"]["pair_weight"] != 2 or config["training"]["rank_weight"] != .5:
        raise ValueError("V4 preserves pair weight two and original ranking weight one half")
    return ["gate"]


def _actual_binding(authorization):
    if not isinstance(authorization, GateTrainingAuthorization):
        raise ValueError("Research v4 training requires explicit actual diagnostic/source/audit paths")
    from .v4_adoption import validate_gate_diagnostic
    result = validate_gate_diagnostic(authorization.diagnostic_path, authorization.frozen_v3_root,
        authorization.frozen_feature_root, authorization.frozen_source_directory, pair_export=authorization.pair_export)
    if result["status"] != "SUPPORTED":
        raise ValueError("Actual original-reference Gate diagnostic is not SUPPORTED: " + "; ".join(result["reasons"]))
    audit = read_json(Path(authorization.audit_path))
    if (audit.get("status") != "APPROVED" or audit.get("protocol_version") != 4
            or audit.get("diagnostic_sha256") != result["diagnostic_sha256"]
            or audit.get("pair_export_sha256") != result["pair_export_sha256"]
            or audit.get("source_verification_sha256") != result["source_verification_sha256"]):
        raise ValueError("Independent Gate audit does not approve the exact actual diagnostic/source pairs")
    binding = {"gate_diagnostic_sha256": result["diagnostic_sha256"],
        "gate_pair_export_sha256": result["pair_export_sha256"],
        "gate_source_verification_sha256": result["source_verification_sha256"],
        "gate_source_code_hash": result["source_code_hash"], "gate_cache_signature": result["cache_signature"],
        "gate_audit_sha256": file_hash(authorization.audit_path)}
    return binding, result


def bind_gate_authorization(config, authorization):
    """Return a config copy with verified content hashes, before prepare/hash.

    These are provenance hashes, not cryptographic signatures or new access
    credentials.  Runtime paths are kept in authorization, outside the frozen
    scientific config.  Actual files are verified again by train on resume.
    """
    v4_tasks(config)
    binding, _ = _actual_binding(authorization)
    result = copy.deepcopy(config)
    result["experiment"].update(binding)
    return result


def verify_gate_authorization(config, authorization):
    v4_tasks(config)
    if config["mode"] == "smoke" and config["encoder"]["backend"].startswith("smoke"):
        return {"scope": "SMOKE_ONLY_NOT_RESEARCH_AUTHORIZED", "research_authorized": False}
    binding, diagnostic = _actual_binding(authorization)
    if any(config["experiment"].get(key) != value for key, value in binding.items()):
        raise ValueError("Frozen v4 configuration differs from actual diagnostic/audit authorization")
    from .v4_adoption import _profile, _science
    _profile(config, config["experiment"]["role"], "development")
    if _science(config) != _science(diagnostic["source_reference_config"]):
        raise ValueError("V4 changed scientific settings beyond its common-anchor term")
    return {"scope": "ACTUAL_PAIRED_GATE_DEVELOPMENT_ONLY", "research_authorized": True, **binding}


def gate_all_models_completion(reference_config, candidate_config):
    """Actual two-role completion proof; orchestrator calls before either eval."""
    from .config import scientific_hash
    from .evaluation import training_completion
    configs = {"reference": reference_config, "candidate": candidate_config}
    results = {}
    for role, config in configs.items():
        v4_tasks(config)
        if config["experiment"]["role"] != role:
            raise ValueError("Gate completion requires both distinct declared roles")
        results[role] = training_completion(config)
    return {"protocol_version": 4, "scope": "BOTH_GATE_ROLES_TRAINING_COMPLETION_NOT_GO",
            "complete": all(proof["complete"] for proof in results.values()), "roles": results,
            "scientific_hashes": {role: scientific_hash(config) for role, config in configs.items()}}


def require_gate_evaluation_ready(config, all_models_completion):
    from .config import scientific_hash
    v4_tasks(config)
    proof = all_models_completion
    role = config["experiment"]["role"]
    if (proof.get("protocol_version") != 4 or not proof.get("complete")
            or set(proof.get("roles", {})) != {"reference", "candidate"}
            or not all(part.get("complete") for part in proof["roles"].values())
            or proof.get("scientific_hashes", {}).get(role) != scientific_hash(config)):
        raise ValueError("Both actual Gate arm models must finish and lock before either evaluation")
    expected = {f"gate/{seed}" for seed in config["training"]["seeds"]}
    if any(set(part.get("models", {})) != expected or not all(p.get("complete") for p in part["models"].values())
           for part in proof["roles"].values()):
        raise ValueError("Gate evaluation lacks all predeclared role/seed models")
    return proof
