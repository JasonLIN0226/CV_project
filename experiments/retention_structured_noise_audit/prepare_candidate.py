"""Create an independent candidate output root over the same audited dataset."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_config = args.source_config.expanduser().resolve()
    source = source_config.parent
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing overwrite: {output}")
    output.mkdir(parents=True)
    for name in ("dataset", "features", "probes"):
        (output / name).symlink_to(source / name, target_is_directory=True)
    (output / "predictor/gate").mkdir(parents=True)
    for name in ("basis.npz", "basis.json"):
        shutil.copyfile(
            source / "predictor/gate" / name,
            output / "predictor/gate" / name,
        )
    config = json.loads(source_config.read_text())
    config["output"] = str(output)
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    print(output / "resolved_config.json")


if __name__ == "__main__":
    main()
