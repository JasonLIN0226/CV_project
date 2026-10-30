"""Read-only byte-hash intake check; never download or deserialize model files.

This checks only the explicit inventory in required_phase1_assets.json. Matching
those old checkpoint/transform hashes neither verifies the omitted latest repair
source/maps nor establishes complete stores, causal validity or model behavior.
No historical /scratch path is read unless the caller explicitly maps it there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys


DEFAULT_MANIFEST = Path(__file__).with_name("required_phase1_assets.json")
DEFAULT_SOURCE_PREFIX = "/scratch/xl4358"


def _source_path(value, label):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty absolute POSIX path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be an absolute POSIX path without '..'")
    return path


def _read_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("unsupported asset manifest schema")
    required = data.get("required_hashed_assets")
    optional = data.get("optional_original_diagnostic_assets", [])
    if not isinstance(required, list) or not required or not isinstance(optional, list):
        raise ValueError("manifest needs a nonempty required_hashed_assets list")
    seen = set()
    for record in required + optional:
        if not isinstance(record, dict):
            raise ValueError("asset records must be objects")
        path = str(_source_path(record.get("path"), "asset path"))
        if path != record["path"] or path in seen:
            raise ValueError("asset paths must be canonical and unique")
        seen.add(path)
        if not isinstance(record.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ValueError(f"invalid SHA-256 for {path}")
    return data


def _hash_regular_file(path):
    """Read raw bytes only, refuse non-regular files and detect ordinary edits."""
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("not_a_regular_file")
    # NONBLOCK also prevents a replaced file from blocking on a FIFO open.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("not_a_regular_file")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(stream.fileno())
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if signature(info) != signature(before) or signature(before) != signature(after) or size != after.st_size:
        raise ValueError("file_changed_during_hashing")
    if signature(after) != signature(path.stat()):
        raise ValueError("file_replaced_during_hashing")
    return digest.hexdigest(), size


def _resolve_asset(source, *, root, prefix, mapping):
    """Exact map wins; root mapping is component-aware and cannot escape root."""
    if source in mapping:
        local = Path(mapping[source]).expanduser()
        bounded = not local.is_absolute()
        if bounded:
            if root is None:
                raise ValueError("relative_mapping_requires_root")
            local = root / local
    else:
        if root is None:
            return None, "unmapped"
        try:
            relative = PurePosixPath(source).relative_to(prefix)
        except ValueError:
            return None, "outside_source_prefix"
        local = root.joinpath(*relative.parts)
        bounded = True
    local = local.resolve()
    if bounded:
        try:
            local.relative_to(root)
        except ValueError:
            raise ValueError("mapped_path_escapes_root")
    return local, None


def verify_assets(manifest_path=DEFAULT_MANIFEST, *, root=None,
                  source_prefix=DEFAULT_SOURCE_PREFIX, path_mapping=None,
                  include_optional=False):
    """Return a JSON-ready report without writing any asset or loading a model.

    root maps the suffix after source_prefix into an intake directory. Exact
    path_mapping overrides root mapping for individual paths. Absolute explicit
    destinations may be anywhere; relative destinations and root-derived paths
    must stay inside root, including after resolving symbolic links.
    """
    manifest = _read_manifest(manifest_path)
    prefix = _source_path(source_prefix, "source_prefix")
    mapping = {} if path_mapping is None else path_mapping
    if not isinstance(mapping, dict):
        raise ValueError("path_mapping must be a JSON object of source:destination strings")
    all_records = manifest["required_hashed_assets"] + manifest.get("optional_original_diagnostic_assets", [])
    known = {r["path"] for r in all_records}
    for source, destination in mapping.items():
        if source not in known:
            raise ValueError(f"mapping contains an unknown manifest path: {source}")
        if not isinstance(destination, str) or not destination or "\x00" in destination:
            raise ValueError("mapped destinations must be nonempty path strings")
    if root is None and not mapping:
        raise ValueError("provide --root or a nonempty explicit path mapping; original remote paths are never implicit")
    root = None if root is None else Path(root).expanduser().resolve()
    selected = [(record, True) for record in manifest["required_hashed_assets"]]
    if include_optional:
        selected += [(record, False) for record in manifest.get("optional_original_diagnostic_assets", [])]
    rows = []
    for record, required in selected:
        row = {"source_path": record["path"], "kind": record.get("kind"),
               "required": required, "expected_sha256": record["sha256"],
               "local_path": None, "actual_sha256": None, "size_bytes": None}
        try:
            path, absent_reason = _resolve_asset(record["path"], root=root, prefix=prefix, mapping=mapping)
            if path is None:
                row.update(status="missing", reason=absent_reason)
            else:
                row["local_path"] = str(path)
                actual, size = _hash_regular_file(path)
                row.update(actual_sha256=actual, size_bytes=size,
                           status="matched" if actual == record["sha256"] else "incompatible",
                           reason=None if actual == record["sha256"] else "sha256_mismatch")
        except FileNotFoundError:
            row.update(status="missing", reason="file_not_found")
        except ValueError as error:
            row.update(status="incompatible", reason=str(error))
        except (OSError, RuntimeError) as error:
            # Includes unreadable files and unresolvable symlink loops.
            row.update(status="unreadable", reason=type(error).__name__)
        rows.append(row)
    counts = {}
    for group, required in [("required", True), ("optional", False)]:
        subset = [r for r in rows if r["required"] == required]
        counts[group] = {status: sum(r["status"] == status for r in subset)
                         for status in ("matched", "missing", "incompatible", "unreadable")}
        counts[group]["total"] = len(subset)
    required = counts["required"]
    status = ("INCOMPATIBLE" if required["incompatible"] else "UNREADABLE" if required["unreadable"]
              else "INCOMPLETE" if required["missing"] else "MATCHED")
    return {"schema_version": 1, "status": status,
            "required_hashed_assets_match": status == "MATCHED", "counts": counts,
            "audited_source_commit": manifest.get("audited_commit"), "assets": rows,
            "scope": "Only the selected historical asset hashes; not latest-repair completeness or scientific verification.",
            "latest_repair_verified": False, "models_deserialized": False,
            "downloads_performed": False, "assets_modified": False,
            "unverified_requirements": ["Latest repair source, protocols, maps and their provenance",
                                        "Complete linked data/feature store and original split identities",
                                        "Frozen-model behavior, input causality and scientific results"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, help="Intake directory receiving paths after --source-prefix")
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument("--mapping", type=Path, help="JSON object mapping exact original asset paths to local paths")
    parser.add_argument("--include-optional", action="store_true", help="Also inspect optional historical state readouts")
    args = parser.parse_args(argv)
    try:
        mapping = None if args.mapping is None else json.loads(args.mapping.read_text(encoding="utf-8"))
        report = verify_assets(args.manifest, root=args.root, source_prefix=args.source_prefix,
                               path_mapping=mapping, include_optional=args.include_optional)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    return 0 if report["required_hashed_assets_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
