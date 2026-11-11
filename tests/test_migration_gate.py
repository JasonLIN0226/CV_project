"""Migration check for Gate: the new package against the code it was extracted from.

Both sides read the same asset tree and the same frozen predictor, and every stage must agree
exactly: episode arrays, predictor outputs, retained-frame sequences, metric rows, teacher labels,
trained scorer weights and the three literature baselines.

Needs a Gate asset tree. The old code is the base commit's experiments/, taken from BM_LEGACY_TREE
or extracted from git (tests/legacy_tree.py):

    BM_GATE_ROOT=<tree with resolved_config.json> python -m pytest -q tests/test_migration_gate.py

Optional: BM_GATE_PREDICTOR_SHA256='{"17": "..."}' pins the checkpoints (the team bundle's pins are
used when the tree is the team bundle); BM_MIGRATION_SEED picks the predictor seed (default 17).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
import legacy_tree  # noqa: E402

GATE_ROOT = os.environ.get("BM_GATE_ROOT")
LEGACY = legacy_tree.path() if GATE_ROOT else None
pytestmark = pytest.mark.skipif(not (GATE_ROOT and LEGACY), reason="set BM_GATE_ROOT (and BM_LEGACY_TREE outside a git checkout)")
SEED = int(os.environ.get("BM_MIGRATION_SEED", "17"))
if LEGACY:
    legacy_tree.use(LEGACY)
    SNAPSHOT = legacy_tree.predictor_snapshot(LEGACY)


def same_state(a, b):
    a, b = a.state_dict() if hasattr(a, "state_dict") else a, b.state_dict() if hasattr(b, "state_dict") else b
    assert list(a) == list(b)
    for key in a:
        assert torch.equal(a[key].cpu(), b[key].cpu()), key


@pytest.fixture(scope="module")
def sides():
    from bounded_memory.data.io import file_hash
    from bounded_memory.environments.gate import assets as new_assets, bundle
    from experiments.memory_baselines import gate_data as old_data

    old_data.add_paths(SNAPSHOT)
    old = old_data.load_gate(Path(GATE_ROOT) / "resolved_config.json")
    pins = os.environ.get("BM_GATE_PREDICTOR_SHA256")
    if pins:
        pins = {int(k): v for k, v in json.loads(pins).items()}
    else:       # the old loader has already accepted this checkpoint through its own signature check
        path = Path(GATE_ROOT) / f"predictor/gate/seed_{SEED}/best.pt"
        pins = dict(bundle.TEAM_PREDICTOR_SHA256) if file_hash(path) == bundle.TEAM_PREDICTOR_SHA256.get(SEED) \
            else {SEED: file_hash(path)}
    new = new_assets.load_gate(GATE_ROOT, predictor_sha256=pins)
    return old, old.predictor(SEED), new, new.predictor(SEED)


def test_episode_arrays_are_identical(sides):
    old, _, new, _ = sides
    assert [e.id for e in old.episodes] == [e.id for e in new.episodes]
    for a, b in zip(old.episodes, new.episodes):
        assert (a.pair, a.sibling, a.split, a.q, a.cue) == (b.pair, b.sibling, b.split, b.q, b.cue)
        for field in ("evidence", "z", "p", "actions", "classes", "prefix"):
            assert np.array_equal(getattr(a, field), getattr(b, field)), (a.id, field)


def test_predictor_weights_and_outputs_are_identical(sides):
    from bounded_memory.environments.gate import protocol
    from bounded_memory.evaluation.oracles import gate as oracles
    from experiments.memory_baselines import gate_data as old_data, gate_policies as old_policies

    old, old_model, new, new_model = sides
    same_state(old_model, new_model)
    val_old, val_new = old.split("validation"), new.split("validation")
    for name in ("recent_only", "oracle_keyframe"):
        a, b = getattr(old_policies, name)(val_old), getattr(oracles, name)(val_new)
        assert a == b
        pa = old_data.predict_selection(old_model, val_old, a, "cpu")
        pb = protocol.predict_selection(new_model, val_new, b, "cpu")
        assert np.array_equal(pa, pb)
        from bounded_memory.evaluation.metrics import gate as metrics
        assert old_data.metric_rows(old, val_old, pa, a, name, SEED) == metrics.metric_rows(new, val_new, pb, b, name, SEED)
    full = [list(range(e.candidates)) for e in val_old]
    assert oracles.full_history(val_new) == full


def test_heuristics_keep_the_same_frames(sides):
    from bounded_memory.evaluation.heuristics import gate as heuristics
    from experiments.memory_baselines import gate_policies as old_policies

    old, old_model, new, new_model = sides
    n = int(os.environ.get("BM_MIGRATION_EPISODES", "24"))
    assert old_policies.heuristics(old_model, old.split("validation")[:n], SEED, "cpu") == \
        heuristics.heuristics(new_model, new.split("validation")[:n], SEED, "cpu")


def test_teacher_labels_scorer_weights_and_selections_are_identical(sides):
    from bounded_memory.methods.deletion_cost import retention, training
    from experiments.memory_baselines import gate_policies as old_policies

    old, old_model, new, new_model = sides
    train_old, train_new = old.split("train"), new.split("train")
    old_states = [(e.id, policy, state.candidates) for e, policy, state in old_policies.scorer_states(train_old[:16], SEED)]
    new_states = [(e.id, policy, state.candidates) for e, policy, state in retention.scorer_states(train_new[:16], SEED)]
    assert old_states == new_states
    old_scorers, old_info = old_policies.train_scorers(old_model, train_old, SEED, "cpu")
    x, ya, yd, _ = training.teacher_labels(new_model, train_new, SEED, "cpu")
    new_scorers = training.as_models(training.fit_scorers(x, ya, yd, SEED, "cpu"), "cpu")
    assert old_info["training_states"] == len(x)
    for name in training.ARMS:
        same_state(old_scorers[name], new_scorers[name])
        assert old_policies.scorer_selections(old_scorers[name], old.split("validation")) == \
            retention.scorer_selections(new_scorers[name], new.split("validation"))


def test_pafu_kv_is_identical(sides):
    from bounded_memory.methods.pafu_kv import gate as new_pafu, repo_adaptation as new_repo
    from experiments.memory_baselines import pafu as old_pafu, pafu_repo as old_repo

    old, old_model, new, new_model = sides
    val_old, val_new = old.split("validation")[:24], new.split("validation")[:24]
    a, info_a = old_pafu.train(old_model, old.split("train"), SEED, "cpu", steps=60)
    b, info_b = new_pafu.train(new_model, new.split("train"), SEED, "cpu", steps=60)
    same_state(a, b)
    assert info_a == info_b
    assert old_pafu.select(a, val_old, "cpu") == new_pafu.select(b, val_new, "cpu")
    assert old_pafu.teacher_select(old_model, val_old, "cpu") == new_pafu.teacher_select(new_model, val_new, "cpu")
    a, info_a = old_repo.train(old_model, old.split("train")[:48], SEED, "cpu", max_states=300)
    b, info_b = new_repo.train(new_model, new.split("train")[:48], SEED, "cpu", max_states=300)
    same_state(a, b)
    assert info_a == info_b and old_repo.select(a, val_old) == new_repo.select(b, val_new)


@pytest.mark.parametrize("variant", ["full", "recurrent"])
def test_compress_to_remember_is_identical(sides, variant):
    from bounded_memory.methods.pacc import gate as new_pacc
    from experiments.memory_baselines import pacc as old_pacc

    old, old_model, new, new_model = sides
    a, info_a = old_pacc.train(old_model, old.split("train"), SEED, "cpu", variant=variant, steps=6, batch=8)
    b, info_b = new_pacc.train(new_model, new.split("train"), SEED, "cpu", variant=variant, steps=6, batch=8)
    same_state(a, b)
    assert info_a == info_b
    pa, gap_a = old_pacc.predict(old_model, a, old.split("validation")[:16], "cpu", variant=variant)
    pb, gap_b = new_pacc.predict(new_model, b, new.split("validation")[:16], "cpu", variant=variant)
    assert np.array_equal(pa, pb) and gap_a == gap_b


@pytest.mark.parametrize("variant", ["published", "online"])
def test_framemorrow_is_identical(sides, variant):
    from bounded_memory.methods.framemorrow import gate as new_fm
    from experiments.memory_baselines import fm as old_fm

    old, _, new, _ = sides
    train_old, train_new = old.split("train")[:40], new.split("train")[:40]
    val_old, val_new = old.split("validation")[:12], new.split("validation")[:12]
    visual_old = {e.id: e.z for e in old.episodes}
    visual_new = {e.id: e.z for e in new.episodes}
    kw = dict(variant=variant, epochs=1, hidden=32, layers=1, heads=2)
    a, info_a = old_fm.train(train_old, visual_old, SEED, "cpu", **kw)
    b, info_b = new_fm.train(train_new, visual_new, SEED, "cpu", **kw)
    same_state(a, b)
    assert info_a == info_b
    select_old = old_fm.select_published if variant == "published" else old_fm.select_online
    select_new = new_fm.select_published if variant == "published" else new_fm.select_online
    assert select_old(a, val_old, visual_old, "cpu") == select_new(b, val_new, visual_new, "cpu")
    assert old_fm.select_teacher(val_old, visual_old, "cpu") == new_fm.select_teacher(val_new, visual_new, "cpu")


def test_rebuild_chain_matches_the_old_code(tmp_path):
    """Dataset, features, bases, probes and three predictors, built by the old sources and by the package.

    A small configuration (32 + 8 pairs, the pixel stand-in encoder, 8 epochs): every dataset and
    feature file must be byte-identical and every predictor tensor equal.
    """
    import hashlib
    import subprocess

    config = json.loads((ROOT / "configs/gate/dataset_config_template.json").read_text())
    config["mode"], config["encoder"]["backend"], config["training"]["epochs"] = "smoke", "smoke_pixels", 8
    config["data"]["pairs"].update(train=32, validation=8)
    (tmp_path / "source.json").write_text(json.dumps(config))
    old_sources = [LEGACY / "experiments/gate_clean_future/materialize.py", LEGACY / "experiments/gate_clean_future/fit_features.py",
                   LEGACY / "experiments/retention_structured_noise_audit/train_reference.py"]
    old_env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (
        SNAPSHOT, LEGACY / "experiments/retention_position_noise_audit", LEGACY / "experiments/retention_structured_noise_audit",
        LEGACY / "experiments/gate_clean_future", LEGACY))))
    new_env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    for side, env, commands in (
            ("old", old_env, [[str(LEGACY / "experiments/memory_baselines/run_with_snapshot.py"), str(SNAPSHOT), str(s)] for s in old_sources]),
            ("new", new_env, [["-m", f"bounded_memory.environments.gate.{m}"] for m in ("generate", "features", "train_predictor")])):
        tree = tmp_path / side / "data"
        for command, args in zip(commands, (["--source-config", tmp_path / "source.json", "--output", tree],
                                            ["--config", tree / "resolved_config.json", "--output", tmp_path / side / "features_report"],
                                            ["--config", tree / "resolved_config.json", "--output", tmp_path / side / "training_report"])):
            subprocess.run([sys.executable, *command, *map(str, args)], check=True, env=env, capture_output=True)

    def files(tree):
        return {p.relative_to(tree).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tree.rglob("*")) if p.is_file()}
    old, new = files(tmp_path / "old/data"), files(tmp_path / "new/data")
    assert set(old) == set(new)
    different = sorted(name for name in old if old[name] != new[name])
    # what may differ: wall-clock logs, the code checksum recorded as metadata, the output path, and the
    # checkpoint files that embed them (their tensors are compared below)
    allowed = ("features/index.json", "features/pair_feature_audit.json", "probes/metrics.json", "resolved_config.json", "predictor/gate/seed_")
    assert all(name.startswith(allowed) for name in different), different
    assert sum(name.startswith(("dataset/", "features/episodes/")) or name.endswith(".npz") for name in old) > 100
    for seed in (17, 29, 43):
        a = torch.load(tmp_path / f"old/data/predictor/gate/seed_{seed}/best.pt", map_location="cpu", weights_only=False)
        b = torch.load(tmp_path / f"new/data/predictor/gate/seed_{seed}/best.pt", map_location="cpu", weights_only=False)
        same_state(a.get("model", a), b.get("model", b))
