"""Motion v2 online validation, run again on the frozen study assets outside the original protocol.

The original step-3/4 script (pipeline/run_step34_recovery1.py) only runs against registrations
committed under the old experiments/ paths, inside an A100-seconds budget and a one-A100 allocation.
This module calls the same unchanged code (step34_core.FrozenAssets and evaluate_online) and replaces
the registration with an explicit checksum check of every input:

    step2            three frozen readers, I-JEPA features, normalisation, basis, position decoder and
                     resolved config, checked against pipeline/step34_assets.json. hf_cache/ is skipped:
                     nothing here encodes frames, so the cached encoder weights are never read.
    dataset          the Motion v2 train/validation dataset; its manifest SHA256 must equal the one the
                     frozen features/index.json was built from.
    evidence         the tracked step-3/4 output (results/ in this repository): the 54 student
                     checkpoints and the raw online losses a replay is compared with.
    longgap_receipt  optional: the CPU long-gap validation dataset, checked against the receipt in
                     pipeline/protocol_step34_recovery1_execution.json.
    longgap_features optional: its I-JEPA features, checked against the tracked step-3/4 inventory.

Nothing is written to any input folder.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASSET_MANIFEST = HERE / "pipeline" / "step34_assets.json"
EXECUTION = HERE / "pipeline" / "protocol_step34_recovery1_execution.json"
SKIPPED_PREFIXES = ("hf_cache/",)
INTERVALS = ("ordinary", "long_gap")


class FrozenInputError(ValueError):
    """An input is missing or differs from its recorded checksum."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_tree(root, expected, skip=()):
    """Hash every expected file under root. Returns counts and the paths that are missing or differ."""
    root = Path(root)
    report = {"root": str(root), "checked": 0, "skipped": 0, "missing": [], "mismatched": []}
    for relative, checksum in sorted(expected.items()):
        if relative.startswith(tuple(skip)):
            report["skipped"] += 1
            continue
        path = root / relative
        if not path.is_file():
            report["missing"].append(relative)
        elif sha256(path) != checksum:
            report["mismatched"].append(relative)
        else:
            report["checked"] += 1
    report["ok"] = not report["missing"] and not report["mismatched"]
    return report


def frozen_step2_checksums(skip=SKIPPED_PREFIXES):
    """The step-2 inventory that step 3/4 reused (job 16614), without the skipped folders."""
    manifest = json.loads(ASSET_MANIFEST.read_text())
    if manifest.get("status") != "FROZEN_ORIGINAL_16614_ASSETS":
        raise FrozenInputError("pipeline/step34_assets.json is not the frozen 16614 inventory")
    return {k: v for k, v in manifest["artifact_sha256"].items() if not k.startswith(tuple(skip))}


@dataclass(frozen=True)
class Paths:
    step2: Path
    dataset: Path
    evidence: Path
    longgap_receipt: Path | None = None
    longgap_features: Path | None = None

    @classmethod
    def from_config(cls, assets):
        def path(key):
            return Path(assets[key]).resolve() if assets.get(key) else None
        missing = [k for k in ("step2", "dataset", "evidence") if not assets.get(k)]
        if missing:
            raise FrozenInputError(f"config assets lack {missing}")
        return cls(path("step2"), path("dataset"), path("evidence"), path("longgap_receipt"), path("longgap_features"))


def _evidence_inventory(evidence):
    return json.loads((Path(evidence) / "artifact_sha256.json").read_text())


def verify(paths, intervals=("ordinary",)):
    """Check every input the requested intervals read. Raises FrozenInputError listing every problem."""
    unknown = set(intervals) - set(INTERVALS)
    if unknown:
        raise FrozenInputError(f"unknown intervals {sorted(unknown)}")
    report, problems = {}, []

    report["step2"] = check_tree(paths.step2, frozen_step2_checksums())
    index = paths.step2 / "features" / "index.json"
    manifest = paths.dataset / "manifest.json"
    dataset = {"root": str(paths.dataset), "manifest_present": manifest.is_file()}
    if manifest.is_file() and index.is_file():
        dataset["manifest_sha256"] = sha256(manifest)
        dataset["expected_sha256"] = json.loads(index.read_text())["manifest_sha256"]
    dataset["ok"] = bool(dataset.get("manifest_sha256")) and dataset["manifest_sha256"] == dataset["expected_sha256"]
    report["dataset"] = dataset

    inventory = _evidence_inventory(paths.evidence)
    tracked = {k: v for k, v in inventory.items()
               if k.startswith("students/") or any(k.startswith(f"online/{i}/raw_losses/") for i in intervals)}
    report["evidence"] = check_tree(paths.evidence, tracked)

    if "long_gap" in intervals:
        if paths.longgap_receipt is None or paths.longgap_features is None:
            raise FrozenInputError("long_gap needs assets.longgap_receipt and assets.longgap_features")
        receipt = json.loads(EXECUTION.read_text())["cpu_receipt"]["artifact_sha256"]
        report["longgap_receipt"] = check_tree(paths.longgap_receipt, receipt)
        features = {k[len("longgap_features/"):]: v for k, v in inventory.items() if k.startswith("longgap_features/")}
        report["longgap_features"] = check_tree(paths.longgap_features, features)

    for name, part in report.items():
        if not part["ok"]:
            detail = {k: part[k][:5] for k in ("missing", "mismatched") if part.get(k)}
            problems.append(f"{name}: {detail or part}")
    if problems:
        raise FrozenInputError("frozen inputs do not match their checksums:\n  " + "\n  ".join(problems))
    return report


def _no_budget(*args, **kwargs):
    """The original protocol stopped on an A100-seconds budget; a replay has none."""


def core():
    """The unchanged step-3/4 module, loaded the way the original scripts load their siblings."""
    from . import step34_core
    return step34_core


def load(paths, device):
    """FrozenAssets on the three frozen readers. Hashes the step-2 files again, as the original did."""
    return core().FrozenAssets(paths.step2, paths.dataset, frozen_step2_checksums(), device, _no_budget)


def interval_inputs(assets, paths, interval):
    """(records, external) that evaluate_online takes for one interval."""
    if interval == "ordinary":
        return assets.store.manifest["records"], None
    root = paths.longgap_receipt / "data" / "dataset"
    manifest = json.loads((root / "manifest.json").read_text())
    if len(manifest["records"]) != 192 or any(r["split"] != "validation" for r in manifest["records"]):
        raise FrozenInputError("the long-gap dataset must hold exactly the 192 validation episodes")
    index = json.loads((paths.longgap_features / "index.json").read_text())
    if index["manifest_sha256"] != sha256(root / "manifest.json"):
        raise FrozenInputError("long-gap features were not encoded from this long-gap dataset")
    return manifest["records"], (root, paths.longgap_features, index)


def replay(assets, paths, interval, output):
    """Every baseline, the 54 tracked students and the three references, on one interval."""
    if Path(output).exists():
        raise FileExistsError(f"{output} exists; the replay writes each file once")
    records, external = interval_inputs(assets, paths, interval)
    summary = json.loads((paths.evidence / "students" / "summary.json").read_text())
    return core().evaluate_online(assets, records, summary, paths.evidence / "students", output, interval, external)


def compare(new_dir, tracked_dir):
    """Per method: identical, close (rtol 1e-5) or differs, from the raw online loss arrays."""
    new_dir, tracked_dir = Path(new_dir), Path(tracked_dir)
    rows = {}
    names = sorted({p.name for p in tracked_dir.glob("*.npz")} | {p.name for p in new_dir.glob("*.npz")})
    for name in names:
        method = name[:-4]
        old, new = tracked_dir / name, new_dir / name
        if not new.exists() or not old.exists():
            rows[method] = {"verdict": "missing in replay" if not new.exists() else "not in the tracked evidence"}
            continue
        with np.load(old, allow_pickle=False) as a, np.load(new, allow_pickle=False) as b:
            row = {"verdict": "identical"}
            for key in ("position", "feature", "success"):
                x, y = a[key].astype(np.float64), b[key].astype(np.float64)
                if x.shape != y.shape:
                    row.update(verdict="differs", shape=[list(x.shape), list(y.shape)])
                    break
                row[f"{key}_max_abs_diff"] = float(np.abs(x - y).max())
                if not np.array_equal(x, y):
                    row["verdict"] = "close" if row["verdict"] != "differs" and np.allclose(x, y, rtol=1e-5, atol=1e-10) else "differs"
            if "shape" not in row:
                row["position_mse_+3"] = {"tracked": float(a["position"][..., 0].mean()), "replay": float(b["position"][..., 0].mean())}
        rows[method] = row
    counts = {}
    for row in rows.values():
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    return {"counts": counts, "methods": rows}
