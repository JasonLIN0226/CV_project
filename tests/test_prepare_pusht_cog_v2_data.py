"""Hidden-CoG corpus boundaries: frozen protocol, label isolation, audit."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("gym_pusht")
spec = importlib.util.spec_from_file_location(
    "prepare_pusht_cog_v2_data", Path(__file__).resolve().parents[1]
    / "experiments/phase2/prepare_pusht_cog_v2_data.py")
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)


def test_protocol_is_frozen_disjoint_and_label_free():
    protocol, assignment = p.make_protocol()
    assert p.make_protocol()[0] == protocol
    seeds = [f["environment_seed"] for f in protocol["families"]]
    assert len(set(seeds)) == len(seeds)
    lo, hi = p.DEV_INSPECTED_SEEDS
    assert not any(lo <= s <= hi for s in seeds)
    text = json.dumps(protocol)
    assert "sibling_classes" in text and protocol["class_assignment_sha256"] == p.json_sha(assignment)
    for family in protocol["families"]:
        assert all(c not in sib for sib in family["siblings"] for c in p.cue.COG_CLASSES)
    # The shuffle actually varies the class order across families.
    assert len({tuple(v) for v in assignment.values()}) > 1


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("cog") / "data"
    counts = p.SPLIT_COUNTS
    p.SPLIT_COUNTS = {"train": 2, "validation": 1, "evaluation": 1}
    try:
        receipt = p.run(out)
    finally:
        p.SPLIT_COUNTS = counts
    return out, receipt


def test_small_run_layout_and_isolation(small_run):
    out, receipt = small_run
    assert receipt["structural_audit_passed"] and receipt["counts"]["train"]["episodes"] == 6
    index = json.loads((out / "observation_index.json").read_text())
    assert "cog_class" not in json.dumps(index)
    for record in index["records"]:
        with np.load(out / record["observations"]) as archive:
            assert set(archive.files) == p.FIELDS
    classes = json.loads((out / "evaluation_only/sibling_classes.json").read_text())
    diag = json.loads((out / "evaluation_only/cog_train_000_s0.json").read_text())
    assert diag["cog_class"] == classes["cog_train_000"][0]
    with pytest.raises(ValueError):
        p.run(out)  # existing output is never overwritten


def test_audit_rejects_tampered_archive(small_run, tmp_path):
    out, _ = small_run
    protocol = json.loads((out / "protocol.json").read_text())
    records = json.loads((out / "observation_index.json").read_text())["records"]
    target = tmp_path / "data"
    (target / "observations").mkdir(parents=True)
    for r in records:
        (target / r["observations"]).write_bytes((out / r["observations"]).read_bytes())
    assert p.audit_dataset(target, protocol, records)["passed"]

    victim = records[0]
    with np.load(target / victim["observations"]) as a:
        data = {k: a[k] for k in a.files}
    data["rgb"][victim["query_frame"]] ^= 1  # recent window now differs between siblings
    np.savez_compressed(target / victim["observations"], **data)
    tampered = [dict(r) for r in records]
    tampered[0]["observations_sha256"] = p.sha256(target / victim["observations"])
    with pytest.raises(ValueError, match="Sibling pixels differ"):
        p.audit_dataset(target, protocol, tampered)

    data["cog_class"] = np.array(0)
    np.savez_compressed(target / victim["observations"], **data)
    tampered[0]["observations_sha256"] = p.sha256(target / victim["observations"])
    with pytest.raises(ValueError, match="Forbidden"):
        p.audit_dataset(target, protocol, tampered)
