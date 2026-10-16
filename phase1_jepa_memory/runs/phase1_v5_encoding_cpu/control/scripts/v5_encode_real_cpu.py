#!/usr/bin/env python3
"""Locked, offline CPU-only v5 encoding on allocated course Cloud."""
import argparse
from pathlib import Path
import sys
import time
from v5_encode_real import verify


def main():
    launch_started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--source-lock-sha", required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, vendor = verify(args.control, args.source_lock_sha)
    print("ALL_SOURCE_INPUT_DECODER_BYTES_VERIFIED_BEFORE_MODEL_IMPORT", flush=True)
    sys.path[:0] = [str(vendor), str(args.control/"src"), str(args.control/"scripts")]
    import av
    if av.__version__ != "18.0.0":
        raise ValueError("decoder version changed")
    from phase1.v5_encoding_cpu import main as encode
    return encode(args.control, args.rgb_root, args.output, av, launch_started)


if __name__ == "__main__":
    raise SystemExit(main())

