#!/usr/bin/env python3
"""Portable source-only upload archive. Never include experiments, environments, weights or caches."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (args.output or root.parent/"phase1_jepa_memory_upload.zip").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    root_files = {".gitignore", "README.md", "requirements.txt", "pyproject.toml"}
    suffixes = {"configs": {".yaml"}, "src": {".py"}, "scripts": {".py", ".sh", ".sbatch"},
                "tests": {".py"}}
    blocked = {"__pycache__", ".pytest_cache", ".venv", "node_modules", ".git", "runs", "logs", "huggingface"}
    files = []
    for file in sorted(root.rglob("*")):
        if not file.is_file() or file.is_symlink():
            continue
        rel = file.relative_to(root)
        if any(part in blocked or part.startswith(".partial-") for part in rel.parts):
            continue
        if len(rel.parts) == 1:
            allowed = file.name in root_files
        else:
            allowed = file.suffix in suffixes.get(rel.parts[0], set())
        if not allowed or file.stat().st_size > 2*1024*1024:
            continue
        files.append(file)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            info = zipfile.ZipInfo((Path(root.name)/file.relative_to(root)).as_posix())
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, file.read_bytes())
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".sha256").write_text(f"{sha}  {output.name}\n")
    print(json.dumps({"zip": str(output), "files": len(files), "bytes": output.stat().st_size, "sha256": sha}, indent=2))


if __name__ == "__main__":
    main()
