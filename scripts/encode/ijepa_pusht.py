"""Encode a hidden-CoG PushT corpus with the frozen I-JEPA and run its readability probes.

    python scripts/encode/ijepa_pusht.py --version v5c --data-root <dir> --device cuda [--cache-dir <hf cache>]

Encodes train and validation of <data-root>/<corpus> into <data-root>/<corpus>-ijepa (pinned
facebook/ijepa_vith14_1k, grid2 pooling, 5120 features per frame; the test split is not read), then
writes <corpus>-ijepa/probe/probe_results.json. A finished encoding (its encoding_manifest.json) is
not redone; a partial one is removed first. CUDA only. Weights are read offline from --cache-dir
(default $HF_HOME/hub, the cache the Gate encoder uses).
"""
import argparse
import os
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUSHT = ROOT / "src/bounded_memory/environments/pusht"
sys.path.insert(0, str(PUSHT))
from corpora import VERSIONS  # noqa: E402


def run(argv):
    script = PUSHT / "probe_pusht_cog_features.py"
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, choices=sorted(VERSIONS))
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda"])
    ap.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    data = args.data_root / VERSIONS[args.version]["corpus"]
    features = Path(f"{data}-ijepa")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not (features / "encoding_manifest.json").exists():
        if features.exists():
            shutil.rmtree(features)          # partial output from an interrupted run
        run(["encode", "--data", str(data), "--features", str(features), "--cache-dir", str(args.cache_dir),
             "--batch-size", str(args.batch_size)])
    run(["probe", "--data", str(data), "--features", str(features)])


if __name__ == "__main__":
    main()
