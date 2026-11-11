"""Fetch and unpack the public release assets of the Synergy (terminal R/D/S) study.

    python scripts/prepare_data/synergy_release.py --work <dir>

Downloads three GitHub release archives of the team repository (terminal-study results,
confirmation release, confirmation features), checks the published checksum where one exists,
and unpacks them into the layout configs/synergy/release.yaml expects:

    <work>/terminal/run      records, basis, action banks, TRAIN normalisation
    <work>/conf_full         readers, released stage-A arrays, predictions, student archives
    <work>/conf_xfer         records and projected features of the locked confirmation set
    <work>/students_released the released student archives, unzipped

Existing files are kept. The encoded TRAIN/VAL/DEV features are not part of any release; see
scripts/prepare_data/synergy_trainvaldev.py.
"""
import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

RELEASES = "https://github.com/JasonLIN0226/CV_project/releases/download"
ARCHIVES = {   # local name -> (url, sha256 or None, folder, marker inside the folder)
    "terminal_results.zip": (f"{RELEASES}/terminal-structures-20261025/terminal-structures-results.zip", None, "terminal", "run"),
    "conf_full.zip": (f"{RELEASES}/qualified-reader-retention-confirmation-20261026/qualified-reader-retention-confirmation-20261026.zip",
                      None, "conf_full", "build"),
    "conf_transfer.zip": (f"{RELEASES}/qualified-reader-retention-confirmation-20261026/conf_transfer.zip",
                          "ffe0b67e43e090ab32419046c62435b92495eb39e6dbee75b936f88d8a090b1e", "conf_xfer", "records.json"),
}
STUDENTS = "conf_full/results/literature_diagnosis/qualified-reader-retention-20261025/students"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    downloads = args.work / "dl"
    downloads.mkdir(parents=True, exist_ok=True)
    for name, (url, expected, folder, marker) in ARCHIVES.items():
        archive = downloads / name
        if not archive.exists():
            partial = archive.with_suffix(".part")
            urllib.request.urlretrieve(url, partial)
            partial.rename(archive)
        if expected and sha256(archive) != expected:
            raise ValueError(f"checksum mismatch for {name}")
        if not (args.work / folder / marker).exists():
            zipfile.ZipFile(archive).extractall(args.work / folder)
        print(f"{folder}: ready")
    for archive in sorted((args.work / STUDENTS).glob("*.zip")):
        target = args.work / "students_released" / archive.stem
        if not (target / "manifest.json").exists():
            zipfile.ZipFile(archive).extractall(target)
        print(f"students_released/{archive.stem}: ready")


if __name__ == "__main__":
    main()
