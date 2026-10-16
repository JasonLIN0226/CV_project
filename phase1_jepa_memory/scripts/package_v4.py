#!/usr/bin/env python3
"""Package source for an isolated v4 checkout; excludes all experiment payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="phase1_jepa_memory_v4", help="Fresh isolated checkout name")
    args = parser.parse_args()
    if Path(args.prefix).name != args.prefix or not args.prefix.startswith("phase1_jepa_memory_v4"):
        parser.error("A simple isolated Phase 1 v4 checkout name is required")
    root = Path(__file__).resolve().parents[1]
    top_files = {".gitignore", "README.md", "requirements.txt", "pyproject.toml"}
    suffixes = {"configs": {".yaml", ".json"}, "src": {".py"},
                "scripts": {".py", ".sh", ".sbatch"}, "tests": {".py"}}
    blocked = {"__pycache__", ".pytest_cache", ".venv", "node_modules", ".git", "runs", "logs", "huggingface"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root)
        if any(part in blocked or part.startswith(".partial-") for part in rel.parts):
            continue
        allowed = path.name in top_files if len(rel.parts) == 1 else path.suffix in suffixes.get(rel.parts[0], set())
        if allowed and path.stat().st_size <= 2 * 1024 * 1024:
            files.append(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = (Path(args.prefix) / path.relative_to(root)).as_posix()
            info = zipfile.ZipInfo(name)
            info.create_system, info.external_attr = 3, 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".sha256").write_text(f"{sha}  {args.output.name}\n")
    print(json.dumps({"zip": str(args.output.resolve()), "files": len(files), "bytes": args.output.stat().st_size, "sha256": sha}))


if __name__ == "__main__":
    main()
