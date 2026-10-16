#!/usr/bin/env python3
"""Verify frozen source and decoder bytes before any third-party/model import."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()


def verify(control, expected_lock):
    lock = control / "source_lock.json"
    if sha(lock) != expected_lock:
        raise ValueError("source lock bytes changed")
    for name, expected in json.loads(lock.read_text())["files"].items():
        path = control / name
        if path.is_symlink() or sha(path) != expected:
            raise ValueError("source/config/input lock mismatch")
    protocol = json.loads((control/"protocol.json").read_text())
    wheel, vendor = Path(protocol["decoder"]["wheel"]), Path(protocol["decoder"]["vendor"])
    if sha(wheel) != protocol["decoder"]["wheel_sha256"]:
        raise ValueError("official decoder wheel changed")
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():
            relative = Path(entry.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("invalid wheel path")
            if not entry.is_dir():
                p = vendor / relative
                if p.is_symlink() or sha(p) != hashlib.sha256(archive.read(entry)).hexdigest():
                    raise ValueError("data-only decoder bytes changed")
    return protocol, vendor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--source-lock-sha", required=True)
    p.add_argument("--rgb-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol, vendor = verify(args.control, args.source_lock_sha)
    print("ALL_SOURCE_INPUT_DECODER_BYTES_VERIFIED_BEFORE_MODEL_IMPORT",flush=True)
    sys.path[:0] = [str(vendor),str(args.control/"src"),str(args.control/"scripts")]
    import av
    if av.__version__ != "18.0.0":raise ValueError("decoder version changed")
    from phase1.v5_encoding import main as encode
    return encode(args.control, args.rgb_root, args.output, av)


if __name__ == "__main__":
    raise SystemExit(main())
