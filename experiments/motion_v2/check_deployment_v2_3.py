"""Read-only environment/source preflight for the registered v2.3 deployment.

Run with the existing venv's absolute Python path and ``-I -B``. This checks
metadata and imports only: no model construction, downloads, CUDA calls,
dataset array decoding, fitting, installation, or Slurm submission occurs.
All new receipts, temporary paths and caches are confined to a fresh output
directory in deployment2. The former deployment is protected against writes.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback


sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
NEW_ROOT = Path("/scratch/xy3257/motion-v2-20261031-deployment2")
OLD_ROOT = Path("/scratch/xy3257/motion-v2-20261031-deployment1")
MANIFEST_SHA256 = "9ac43b81eb75456e17f3542e64e8a0ab58282207752ae3712836f6d2ea20032a"
REQUIREMENTS = "experiments/motion_v2/requirements-gpu.txt"
READER = "experiments/motion_v2/reader_v2.py"
ENVIRONMENT = {
    "deployment_root": str(NEW_ROOT),
    "read_only_root": str(OLD_ROOT),
    "venv": str(OLD_ROOT / "venv"),
    "python_major_minor": [3, 12],
    "platform_system": "Linux",
    "platform_machine": "x86_64",
    "requirements": REQUIREMENTS,
    "cuda_build": "12.1",
    "cuda_local_versions": {"torch": ["cu121"], "torchvision": ["cu121"]},
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, ensure_ascii=False, allow_nan=False)
        target.write("\n")


def within(path, parent):
    return Path(path).resolve().is_relative_to(Path(parent).resolve())


def no_symlink_components(path):
    path = Path(path).absolute()
    for component in (path, *path.parents):
        require(not component.is_symlink(), "Symlink path is not permitted: " + str(component))


def repo_file(value):
    path = Path(value)
    path = path if path.is_absolute() else REPO / path
    no_symlink_components(path)
    require(within(path, REPO), "Registered file is outside the verified checkout")
    require(path.is_file(), "Registered file is missing: " + str(path))
    return path.resolve()


def git(*arguments):
    return subprocess.run(["git", *arguments], cwd=REPO, capture_output=True,
                          check=True, timeout=30).stdout


def committed_file(path):
    path = repo_file(path)
    relative = path.relative_to(REPO).as_posix()
    require(git("show", "HEAD:" + relative) == path.read_bytes(),
            "File differs from committed HEAD: " + relative)
    return relative


def check_hashes(mapping):
    require(isinstance(mapping, dict) and mapping, "Missing source SHA256 mapping")
    checked = {}
    for relative, expected in sorted(mapping.items()):
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                "Malformed registered SHA256: " + relative)
        path = repo_file(relative)
        actual = sha256(path)
        require(actual == expected, "Registered SHA256 mismatch: " + relative)
        committed_file(path)
        checked[relative] = actual
    return checked


def validate_registration(args, deployment):
    require(re.fullmatch(r"[0-9a-f]{40}", args.expected_head) is not None,
            "--expected-head must be the exact pushed 40-character commit")
    head = git("rev-parse", "HEAD").decode().strip()
    require(head == args.expected_head, "Checkout HEAD differs from the registered deployment commit")
    require(git("symbolic-ref", "--short", "HEAD").decode().strip() ==
            "experiments/motion-v2-20261030", "Unexpected checkout branch")
    committed_file(args.protocol)
    committed_file(args.science_protocol)
    locked = deployment["frozen_scientific_protocol"]
    require(repo_file(locked["path"]) == repo_file(args.science_protocol),
            "Scientific protocol path differs from deployment registration")
    require(sha256(repo_file(args.science_protocol)) == locked["sha256"],
            "Scientific protocol SHA256 mismatch")
    deployment_sources = check_hashes(deployment["source_sha256"])
    require(Path(__file__).resolve().relative_to(REPO).as_posix() in deployment_sources,
            "Deployment registration does not freeze this checker")
    require(REQUIREMENTS in deployment_sources, "Requirements file is not frozen")
    science = read_json(repo_file(args.science_protocol))
    require(science.get("status") == "FROZEN_STEP2_PROTOCOL" and
            science.get("registered_before_run") is True,
            "Scientific protocol is not registered before execution")
    science_sources = check_hashes(science["source_sha256"])
    require(READER in science_sources, "Scientific protocol does not freeze the reader")
    require(science["dataset_manifest_sha256"] == MANIFEST_SHA256,
            "Scientific protocol manifest SHA256 mismatch")
    transfer = deployment["dataset_transfer"]
    require(transfer["manifest_sha256"] == MANIFEST_SHA256,
            "Deployment protocol manifest SHA256 mismatch")
    manifest_path = repo_file(transfer["manifest_path"])
    require((args.dataset / "manifest.json").resolve() == manifest_path,
            "Dataset argument is not the registered checkout manifest directory")
    committed_file(manifest_path)
    require(sha256(manifest_path) == MANIFEST_SHA256, "Dataset manifest SHA256 mismatch")
    manifest = read_json(manifest_path)
    records = manifest["records"]
    require(manifest.get("generator_version") == "motion_v2.1", "Unexpected generator version")
    require(Counter((row["task"], row["split"]) for row in records) ==
            {("motion", "train"): 1024, ("motion", "validation"): 192},
            "Manifest contains a missing, extra, or forbidden split")
    require(len({row["id"] for row in records}) == 1216, "Duplicate or missing manifest IDs")
    pairs = {}
    for row in records:
        pairs.setdefault((row["split"], row["pair"]), []).append(row)
    require(len(pairs) == 608 and all(
        len(value) == 2 and value[0]["sibling"] == value[1]["id"]
        and value[1]["sibling"] == value[0]["id"] for value in pairs.values()),
            "Manifest pair membership is incomplete")
    return {
        "head_commit": head,
        "deployment_protocol_sha256": sha256(repo_file(args.protocol)),
        "scientific_protocol_sha256": locked["sha256"],
        "deployment_source_sha256": deployment_sources,
        "scientific_source_sha256": science_sources,
        "manifest_sha256": MANIFEST_SHA256,
        "manifest_episodes": {"train": 1024, "validation": 192},
        "manifest_pairs": 608,
        "dataset_npz_opened": False,
    }


def protect_old_deployment():
    """Reject Python-audited writes into the preserved deployment, including pyc."""
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

    def guarded_path(value):
        if isinstance(value, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(value)).absolute()
            if path.is_relative_to(OLD_ROOT) or within(path, OLD_ROOT):
                raise PermissionError("Attempted write to preserved deployment1: " + str(path))

    def audit(event, values):
        if event == "open":
            path, mode, flags = values
            if (isinstance(mode, str) and any(char in mode for char in "wax+")) or (
                    isinstance(flags, int) and flags & write_flags):
                guarded_path(path)
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.utime"}:
            guarded_path(values[0])
        elif event in {"os.rename", "os.link", "os.symlink"}:
            guarded_path(values[0])
            guarded_path(values[1])

    sys.addaudithook(audit)


def configure_caches(output):
    suffixes = {
        "TMPDIR": "temporary", "TMP": "temporary", "TEMP": "temporary",
        "XDG_CACHE_HOME": "cache/xdg", "HF_HOME": "cache/huggingface",
        "HF_HUB_CACHE": "cache/huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": "cache/huggingface/hub",
        "HF_ASSETS_CACHE": "cache/huggingface/assets",
        "TORCH_HOME": "cache/torch", "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
        "TRITON_CACHE_DIR": "cache/triton", "NUMBA_CACHE_DIR": "cache/numba",
        "MPLCONFIGDIR": "cache/matplotlib", "CUDA_CACHE_PATH": "cache/cuda",
        "PIP_CACHE_DIR": "cache/pip", "UV_CACHE_DIR": "cache/uv",
    }
    result = {}
    for key, suffix in suffixes.items():
        path = output / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
        result[key] = str(path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return result


def check_environment():
    require(sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1,
            "Invoke the registered absolute venv Python with both -I and -B")
    require(Path(sys.prefix).absolute() == OLD_ROOT / "venv", "The registered venv is not active")
    require(Path(sys.executable).absolute().parent == OLD_ROOT / "venv/bin",
            "Use the registered venv's absolute Python executable")
    require(list(sys.version_info[:2]) == [3, 12], "Expected Python 3.12")
    require(platform.system() == "Linux" and platform.machine() == "x86_64",
            "Expected the registered Linux x86_64 environment")
    from packaging.requirements import Requirement
    from packaging.markers import default_environment
    from packaging.utils import canonicalize_name
    from packaging.version import Version

    distributions = {}
    for distribution in importlib.metadata.distributions():
        name = canonicalize_name(distribution.metadata["Name"])
        require(name not in distributions, "Duplicate installed distribution: " + name)
        require(within(distribution.locate_file(""), OLD_ROOT / "venv"),
                "Installed distribution resolves outside the registered venv: " + name)
        distributions[name] = distribution
    versions = {}
    for raw in repo_file(REQUIREMENTS).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        pins = list(requirement.specifier)
        require(len(pins) == 1 and pins[0].operator == "==" and requirement.marker is None
                and not requirement.extras and requirement.url is None,
                "Only unconditional exact package pins are registered")
        name = canonicalize_name(requirement.name)
        require(name in distributions, "Missing pinned distribution: " + name)
        version = Version(distributions[name].version)
        require(version.public == pins[0].version, "Pinned version mismatch: " + name)
        permitted_local = ENVIRONMENT["cuda_local_versions"].get(name, [])
        require(version.local is None or version.local in permitted_local,
                "Unregistered package local version: " + name)
        versions[name] = str(version)
    dependency_checks = []
    skipped_extra_requirements = 0
    marker_environment = default_environment()
    marker_environment["extra"] = ""
    for name, distribution in sorted(distributions.items()):
        for raw in distribution.requires or []:
            dependency = Requirement(raw)
            if dependency.marker is not None and not dependency.marker.evaluate(marker_environment):
                skipped_extra_requirements += 1
                continue
            dependency_name = canonicalize_name(dependency.name)
            require(dependency.url is None, "Cannot verify a direct-URL dependency without network: " + raw)
            require(dependency_name in distributions, "Missing installed dependency: " + name + " -> " + raw)
            installed = distributions[dependency_name].version
            require(dependency.specifier.contains(installed, prereleases=True),
                    "Unsatisfied installed dependency: " + name + " -> " + raw + " (" + installed + ")")
            dependency_checks.append({"package": name, "requirement": raw, "installed_version": installed})
    modules = {
        "numpy": (), "torch": (), "torchvision": (), "transformers":
        ("IJepaModel", "AutoConfig", "AutoImageProcessor"), "huggingface_hub": ("HfApi",),
        "safetensors": (), "yaml": ("safe_load",), "PIL.Image": ("fromarray",),
        "sklearn.ensemble": ("HistGradientBoostingClassifier",),
        "sklearn.model_selection": ("GroupKFold",), "scipy": (), "pytest": (),
    }
    imported = {}
    for name, attributes in modules.items():
        module = importlib.import_module(name)
        require(within(module.__file__, OLD_ROOT / "venv"), "Import resolves outside registered venv: " + name)
        for attribute in attributes:
            require(callable(getattr(module, attribute)), "Missing required import symbol: " + name + "." + attribute)
        imported[name] = {"file": str(module.__file__), "symbols": list(attributes)}
    torch = sys.modules["torch"]
    require(torch.version.cuda == ENVIRONMENT["cuda_build"], "Registered CUDA wheel build differs from 12.1")
    spec = importlib.util.spec_from_file_location("motion_v2_deployment_reader", repo_file(READER))
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    frozen_class = reader.load_reference("encoders").FrozenIJepa
    require(isinstance(frozen_class, type), "Frozen I-JEPA class import failed")
    return {
        "python": sys.version, "python_executable": sys.executable, "sys_prefix": sys.prefix,
        "platform": platform.platform(), "machine": platform.machine(),
        "isolated": bool(sys.flags.isolated), "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
        "pinned_versions": versions, "installed_versions":
        {name: item.version for name, item in sorted(distributions.items())},
        "active_dependency_checks": dependency_checks,
        "inactive_marker_requirements_skipped": skipped_extra_requirements,
        "dependency_check_method": "installed Requires-Dist metadata; environment markers with extra=''",
        "imports": imported, "frozen_encoder_class_imported": True,
        "torch_cuda_build": torch.version.cuda, "cuda_availability_probed": False,
        "models_constructed": 0, "weights_loaded": 0, "gpu_seconds": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--science-protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(REPO == NEW_ROOT / "repo", "Run only in the registered new deployment checkout")
    no_symlink_components(NEW_ROOT)
    require(NEW_ROOT.stat().st_uid == os.getuid(), "New deployment is not owned by the current user")
    deployment = read_json(repo_file(args.protocol))
    registered_environment = deployment["environment_check"]
    for key, expected in ENVIRONMENT.items():
        require(registered_environment.get(key) == expected, "Unregistered environment check setting: " + key)
    output = args.output.absolute()
    no_symlink_components(output)
    require(within(output, NEW_ROOT) and output != NEW_ROOT, "Output must be a new directory within deployment2")
    require(not output.exists(), "Preflight output already exists; no retry or overwrite is allowed")
    output.mkdir(parents=False, exist_ok=False)
    receipt = {"started_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "CPU environment/source imports and manifest JSON only",
               "deployment1_read_only": str(OLD_ROOT), "output": str(output),
               "command": [sys.executable, "-I", "-B", *sys.argv]}
    status = {"status": "stopped", "stage": "registration", "gpu_seconds": 0,
              "scientific_entrypoint_run": False, "test_data_accessed": False,
              "dataset_arrays_opened": False, "installs_performed": False}
    exit_code = 1
    try:
        receipt["registration"] = validate_registration(args, deployment)
        protect_old_deployment()
        receipt["cache_environment"] = configure_caches(output)
        status["stage"] = "environment_and_imports"
        receipt["environment"] = check_environment()
        status.update(status="passed", stage="complete")
        exit_code = 0
    except Exception as error:
        status["error_type"] = type(error).__name__
        status["error"] = str(error)
        receipt["traceback"] = traceback.format_exc()
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "receipt.json", receipt)
    status["receipt_sha256"] = sha256(output / "receipt.json")
    write_json(output / "status.json", status)
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as target:
        for name in ("receipt.json", "status.json"):
            target.write(sha256(output / name) + "  " + name + "\n")
    print(json.dumps(status, ensure_ascii=False, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
