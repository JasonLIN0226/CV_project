"""Migration check for Synergy: the new package against the code it was extracted from.

Both sides read the same release assets and frozen reader. Compared exactly: the store's arrays,
reader weights, stage-A quantities, the metric, student predictions from the released archives
against the released predictions, and the three literature baselines with identical settings.

    BM_SYN=<folder with terminal/run (or terminal_min/run), conf_xfer, conf_full> \
    BM_SYN_STUDENTS=<unzipped released student archives> python -m pytest -q tests/test_migration_synergy.py

The old code is the base commit's experiments/, taken from BM_LEGACY_TREE or extracted from git
(tests/legacy_tree.py). test_dataset_and_reader_training_* need no assets.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
import legacy_tree  # noqa: E402

BASE, STUDENTS = os.environ.get("BM_SYN"), os.environ.get("BM_SYN_STUDENTS")
LEGACY = legacy_tree.path() if BASE else None
pytestmark = pytest.mark.skipif(not (BASE and STUDENTS and LEGACY), reason="set BM_SYN and BM_SYN_STUDENTS (and BM_LEGACY_TREE outside a git checkout)")
if LEGACY:
    legacy_tree.use(LEGACY)
READERS = os.environ.get("BM_MIGRATION_READERS", "multihead").split(",")
N = int(os.environ.get("BM_MIGRATION_RECORDS", "192"))


@pytest.fixture(scope="module", params=READERS)
def sides(request):
    from bounded_memory.environments.synergy import assets as new_assets
    from experiments.memory_baselines import synergy_data as old_data

    base = Path(BASE)
    run = base / "terminal/run" if (base / "terminal/run").is_dir() else base / "terminal_min/run"
    conf = base / "conf_xfer"
    agg = base / "conf_full/build/reader-aggregation-20261025/hpc/extracted/run"
    released = base / "conf_full/results/literature_diagnosis/qualified-reader-retention-20261025"
    old = old_data.Store(conf / "records.json", conf / "actions.npz", conf, run)
    new = new_assets.Store(conf / "records.json", conf / "actions.npz", conf, run)
    reader = request.param
    return old, old_data.load_reader(reader, old, agg), new, new_assets.load_reader(reader, new, agg), released, reader


def test_store_and_reader_are_identical(sides):
    old, old_model, new, new_model, _, _ = sides
    assert old.records == new.records
    for field in ("P", "FB", "tmB", "G"):
        assert np.array_equal(getattr(old, field), getattr(new, field)), field
    a, b = old_model.state_dict(), new_model.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    r = old.records[5]
    for key, value in old.capsule(r).items():
        assert np.array_equal(value, new.capsule(r)[key]), key
    assert all(np.array_equal(x, y) for x, y in zip(old.student_features(r), new.student_features(r)))


def test_stage_a_matches_the_old_pipeline_and_the_release(sides):
    from bounded_memory.methods.deletion_cost import terminal
    from experiments.memory_baselines import synergy_data as old_data

    old, old_model, new, new_model, released, READER = sides
    records = new.records[:N]
    arrays = terminal.stage_a(new, new_model, records, train=False)
    pipe = old_data.pipe
    evaluation = old.banks["evaluation"]
    for a in range(0, N, 64):                      # the old stage A also ran 64 records per reader call
        rr = old.records[a:a + 64]
        caps = [old.capsule(r) for r in rr]
        assert np.array_equal(arrays["attention"][a:a + 64], pipe.pooled_attention(old_model, caps))
        rep = [c for c in caps for _ in evaluation]
        plans = np.tile(evaluation, (len(caps), 1, 1))
        resid = np.stack([old.target_residual(r, "evaluation") for r in rr])
        for d in range(3):
            coeff = pipe.coefficients(old_model, rep, pipe.KEEPS[d], plans).reshape(len(caps), len(evaluation), 2, 128)
            assert np.array_equal(arrays["eval_losses"][a:a + 64, :, d], pipe.qform(coeff - resid, old.G))
    theirs = np.load(released / f"conf/stageA/{READER}_confirmation.npz")
    assert np.abs(arrays["eval_cost"] - theirs["eval_cost"][:N]).max() < 1e-6
    assert (arrays["eval_cost"].argmin(1) == theirs["eval_cost"][:N].argmin(1)).all()


def test_metric_is_identical(sides):
    from bounded_memory.evaluation.metrics import synergy as new_metrics
    from experiments.memory_baselines import synergy_data as old_data

    old, _, new, _, released, READER = sides
    cost = np.load(released / f"conf/stageA/{READER}_confirmation.npz")["eval_cost"]
    rng = np.random.default_rng(0)
    drops = [rng.integers(0, 3, len(new.records)) for _ in range(3)]
    assert old_data.score(old.records, cost, drops=drops) == new_metrics.score(new.records, cost, drops=drops)
    losses = [rng.random(len(new.records))]
    assert old_data.score(old.records, cost, losses=losses) == new_metrics.score(new.records, cost, losses=losses)


def test_released_students_reproduce_the_released_predictions(sides):
    """Every released student arm must choose the same card as the released predictions.

    One exception is measured rather than required: the appearance-only control scores each card
    from its colour statistics alone, so cards that look identical get equal logits and the choice
    between them is decided by floating-point rounding, which differs between CPUs.
    """
    import json

    from bounded_memory.methods.deletion_cost import terminal, terminal_students
    from experiments.phase2 import fit_terminal_students as old_students

    _, _, new, _, released, READER = sides
    appearance = np.load(released / "conf/appearance.npz", allow_pickle=False)
    inputs = terminal.student_inputs(new, new.records)
    app = {"items": appearance["items"], "context": np.empty((len(new.records), 0), np.float32)}
    agreement = {}
    for mode in ("standard", "distilled"):
        folder = Path(STUDENTS) / f"{READER}_{mode}"
        prediction = terminal_students.predict_students(folder, inputs, app if mode == "standard" else None)
        old = old_students.predict_students(folder, inputs, app if mode == "standard" else None)
        theirs = np.load(released / f"conf/predictions/{READER}_{mode}/confirmation_predictions.npz")
        for key, value in prediction.items():
            assert np.array_equal(value["drop"], old[key]["drop"]), (mode, key)      # new code == old code, same machine
            if key + "__drop" in theirs.files:
                agreement[f"{mode}/{key}"] = float((value["drop"] == theirs[key + "__drop"]).mean())
    report = os.environ.get("BM_MIGRATION_REPORT")
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        report = report.replace("READER", READER)
        Path(report).write_text(json.dumps({"reader": READER, "agreement_with_released_predictions": agreement}, indent=1) + "\n")
    exact = {k: v for k, v in agreement.items() if "appearance" not in k}
    assert len(exact) >= 8 and all(v == 1.0 for v in exact.values()), {k: v for k, v in exact.items() if v != 1.0}
    assert all(v > 0.9 for k, v in agreement.items() if "appearance" in k), agreement


def test_literature_baselines_are_identical(sides):
    from bounded_memory.methods.framemorrow import synergy as new_fm
    from bounded_memory.methods.pacc import synergy as new_pacc
    from bounded_memory.methods.pafu_kv import synergy as new_pafu
    from experiments.memory_baselines import synergy_run as old_run

    old, old_model, new, new_model, _, _ = sides
    a, b = old.records[:N], new.records[:N]
    assert np.array_equal(old_run.pafu_teacher(old_model, old, a, "evaluation"), new_pafu.pafu_teacher(new_model, new, b, "evaluation"))
    assert np.array_equal(old_run.pafu(old_model, old, a, old, a, 17, steps=40), new_pafu.pafu(new_model, new, b, new, b, 17, steps=40))
    assert np.array_equal(old_run.pacc(old_model, old, a, old, a, 17, steps=6), new_pacc.pacc(new_model, new, b, new, b, 17, steps=6))
    kw = dict(epochs=1, hidden=32, layers=1)
    assert np.array_equal(old_run.framemorrow(old, a, old, a, 17, **kw), new_fm.framemorrow(new, b, new, b, 17, **kw))
    assert np.array_equal(old_run.framemorrow_teacher(old, a), new_fm.framemorrow_teacher(new, b))


def _terminal_run():
    base = Path(BASE)
    return base / "terminal/run" if (base / "terminal/run").is_dir() else base / "terminal_min/run"


def test_dataset_generation_matches_the_old_code(tmp_path):
    """Two families per split, rendered by the old generator and by the package: every file identical."""
    import copy
    import hashlib
    import json

    from bounded_memory.environments.synergy import dataset as new_dataset
    from experiments.phase2 import run_terminal_structures as old_dataset

    run = _terminal_run()
    config = copy.deepcopy(json.loads((run / "protocol.json").read_text())["config"])
    splits = ["train", "validation", "development"]
    for split in splits:
        config["splits"][split]["families"] = 2
    banks = dict(np.load(run / "data/action_banks.npz", allow_pickle=False))
    old_dataset.generate(config, tmp_path / "old", splits, existing_banks=banks)
    new_dataset.generate(config, tmp_path / "new", splits, existing_banks=banks)

    def tree(folder):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}
    assert tree(tmp_path / "old") == tree(tmp_path / "new") and len(tree(tmp_path / "new")) == 8


@pytest.mark.parametrize("variant", ["multihead", "item_attention"])
def test_reader_training_matches_the_old_code(tmp_path, variant):
    """Six updates on made-up features: same schedule, same validation losses, same weights, same file."""
    import copy
    import json
    import shutil

    from bounded_memory.environments.synergy import dataset as new_dataset, train_readers as new_trainer
    from bounded_memory.predictors.readers import build_reader
    from experiments.phase2 import run_reader_aggregation as old_trainer, run_terminal_structures as old_dataset

    run = _terminal_run()
    base_config = json.loads((run / "protocol.json").read_text())["config"]
    small = copy.deepcopy(base_config)
    for split in ("train", "validation"):
        small["splits"][split]["families"] = 2
    banks = dict(np.load(run / "data/action_banks.npz", allow_pickle=False))
    basis = dict(np.load(run / "basis.npz", allow_pickle=False))
    config = json.loads((ROOT / "configs/synergy/reader_aggregation_20261025.json").read_text())
    config.update(steps=6, validate_every=3, groups_per_task=1)
    torch.manual_seed(config["seed"])
    common = build_reader(len(basis["input_mean"]), base_config["reader"], new_dataset.H, basis, "original", config["heads"]).state_dict()
    results = {}
    for side, generate, trainer in (("old", old_dataset.generate, old_trainer), ("new", new_dataset.generate, new_trainer)):
        source = tmp_path / side
        generate(small, source / "data", ["train", "validation"], existing_banks=banks)
        frames = json.loads((source / "data/generation.json").read_text())["unique_rgb"]
        np.save(source / "data/features.npy", np.random.default_rng(0).standard_normal((frames, 5120)).astype(np.float32) * 0.1)
        shutil.copy(run / "basis.npz", source / "basis.npz")
        results[side] = trainer.train_one(config, base_config, source, source / "out", variant, "cpu", common)
        results[side]["weights"] = torch.load(source / "out/readers" / variant / "reader_selected.pt", weights_only=False)["model"]
    old, new = results["old"], results["new"]
    for key in ("training_schedule_sha256", "checkpoint_sha256", "selected_step", "validation"):
        assert old[key] == new[key], key
    assert all(torch.equal(old["weights"][k], new["weights"][k]) for k in old["weights"])
