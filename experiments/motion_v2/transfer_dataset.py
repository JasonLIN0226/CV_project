"""Copy the audited train/validation bytes; never generate or decode episodes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

MANIFEST_SHA256 = "9ac43b81eb75456e17f3542e64e8a0ab58282207752ae3712836f6d2ea20032a"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def entries(dataset):
    raw = (dataset / "manifest.json").read_bytes()
    if digest(raw) != MANIFEST_SHA256:
        raise ValueError("Dataset manifest differs from the frozen passing audit")
    manifest = json.loads(raw)
    records = manifest["records"]
    if Counter(r["split"] for r in records) != {"train": 1024, "validation": 192}:
        raise ValueError("Only the complete audited train/validation dataset may be transferred")
    expected = {}
    for record in records:
        for key, prefix in (("obs", "observations"), ("eval", "evaluation_only")):
            name = record[key]
            if name != f"{prefix}/{record['split']}/{record['id']}.npz" or name in expected:
                raise ValueError("Unsafe or duplicate artifact path")
            expected[name] = record[key + "_sha256"]
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pack", "unpack"))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--archive-sha256")
    args = parser.parse_args()
    expected = entries(args.dataset)
    if args.receipt.exists() or args.receipt.is_symlink():
        raise FileExistsError("Receipt must be new")
    if args.mode == "pack":
        if args.archive.exists() or args.archive.is_symlink():
            raise FileExistsError("Archive must be new")
        with zipfile.ZipFile(args.archive, "x", compression=zipfile.ZIP_STORED) as archive:
            for name, checksum in sorted(expected.items()):
                data = (args.dataset / name).read_bytes()
                if digest(data) != checksum:
                    raise ValueError("Artifact SHA256 mismatch: " + name)
                archive.writestr(name, data)
    else:
        if not args.archive_sha256 or digest(args.archive.read_bytes()) != args.archive_sha256:
            raise ValueError("Archive SHA256 mismatch")
        with zipfile.ZipFile(args.archive) as archive:
            if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
                raise ValueError("Archive is not exactly the registered train/validation artifact set")
            # Validate every byte before writing any destination file.
            for name, checksum in sorted(expected.items()):
                destination = args.dataset / name
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError("Refusing to replace an artifact: " + name)
                if digest(archive.read(name)) != checksum:
                    raise ValueError("Artifact SHA256 mismatch: " + name)
            for name in sorted(expected):
                destination = args.dataset / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as handle:
                    handle.write(archive.read(name))
    receipt = {"status": "passed", "operation": args.mode,
               "manifest_sha256": MANIFEST_SHA256, "artifact_files": len(expected),
               "archive_sha256": digest(args.archive.read_bytes()),
               "archive_bytes": args.archive.stat().st_size,
               "splits": ["train", "validation"], "generation_or_decoding": False,
               "test_access": False, "file_deletion": False}
    with args.receipt.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
