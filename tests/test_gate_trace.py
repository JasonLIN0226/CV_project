"""Gate trace software checks. Constructed fixtures are not scientific results.

The separate released-asset test is skipped unless both asset paths are supplied.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bounded_memory.data.io import file_hash, save_torch, write_json, write_npz
from bounded_memory.environments.gate.assets import Episode
from bounded_memory.environments.gate.experiment import open_assets
from bounded_memory.environments.gate.protocol import HORIZONS, predict_selection
from bounded_memory.evaluation.gate_trace import POLICY, format_trace, trace_episode
from bounded_memory.evaluation.metrics.gate import LinearReadout, metric_rows
from bounded_memory.methods.deletion_cost.retention import rollout, scorer_selections
from bounded_memory.methods.deletion_cost.scorer import DIM, EvictionScorer, score_update
from bounded_memory.methods.deletion_cost.training import frozen_scorers, save_students
from bounded_memory.predictors.dynamics import DynamicsPredictor

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gate_trace_cli", ROOT / "scripts/evaluate/gate_trace.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def episode(q=9, name="validation-z", split="validation"):
    rng = np.random.default_rng(11)
    z = rng.normal(size=(q + 7, DIM)).astype(np.float32)
    actions = rng.normal(size=(q + 6, 2)).astype(np.float32)
    return Episode(
        id=name, pair="pair-0", sibling="", split=split, q=q, cue=(1, 2),
        evidence=np.array([1], np.int64), z=z, p=z.copy(), actions=actions,
        classes=np.arange(q + 7) % 2,
        prefix=np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(actions, axis=0)]),
    )


@pytest.fixture
def models():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        scorer = EvictionScorer().eval()
        predictor = DynamicsPredictor(DIM, {"hidden": 8, "dropout": 0.0}, HORIZONS).eval()
    decoder = LinearReadout()
    decoder.mean, decoder.std = np.zeros(DIM), np.ones(DIM)
    # The real decoder predicts route AND state; only column zero defines route.
    decoder.weight = np.ones((DIM, 3)) / DIM
    decoder.bias, decoder.ridge = np.zeros(3), np.array(0.1)
    return SimpleNamespace(decoder=decoder, device=torch.device("cpu")), scorer, predictor


class FixedScorer(torch.nn.Module):
    """Known outputs for testing selection rules, never used by the CLI."""

    def __init__(self, scores):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("scores", torch.tensor(scores, dtype=torch.float32))
        self.calls = 0

    def forward(self, features):
        self.calls += 1
        return self.scores.unsqueeze(0).expand(len(features), -1) + self.anchor * 0


def assert_matches_evaluation(assets, e, scorer, predictor):
    expected_kept, expected_trace = rollout(
        e.p[:e.q + 1].copy(), e.actions[:e.q].copy(), e.q,
        lambda state: score_update(scorer, state),
    )
    expected_prediction = predict_selection(predictor, [e], [expected_kept], assets.device)
    expected_rows = metric_rows(assets, [e], expected_prediction, [expected_kept], POLICY, 17)
    hooks_before = dict(scorer._forward_hooks)
    weights_before = {k: v.clone() for k, v in predictor.state_dict().items()}
    result = trace_episode(assets, e, scorer, predictor, 17)
    assert scorer._forward_hooks == hooks_before
    assert all(torch.equal(v, predictor.state_dict()[k]) for k, v in weights_before.items())
    assert result["kept"] == expected_kept == scorer_selections(scorer, [e])[0]
    assert result["rows"] == expected_rows
    assert np.array_equal(np.asarray(result["prediction"], np.float32), expected_prediction[0])
    assert [{k: row[k] for k in original} for row, original in zip(result["trace"], expected_trace)] == expected_trace
    for route, row in zip(result["routes"], result["rows"]):
        assert float(route["predicted_route_positive"] == route["actual_route_positive"]) == row["route_accuracy"]
    json.dumps(result, allow_nan=False)
    return result


def test_trace_matches_existing_replay_prediction_and_metrics(models):
    assets, scorer, predictor = models
    result = assert_matches_evaluation(assets, episode(), scorer, predictor)
    text = format_trace(result)
    assert "not scored (filling slots)" in text
    assert "normalized_mse=" in text and "+3 (frame 12)" in text and "+6 (frame 15)" in text


@pytest.mark.parametrize("q", [4, 5, 9])
@pytest.mark.parametrize("scores", [[0.1, 0.8, 0.2], [1.0, 1.0, 1.0]])
def test_slots_scores_ties_and_no_resurrection(models, q, scores):
    assets, _, predictor = models
    scorer = FixedScorer(scores)
    result = trace_episode(assets, episode(q), scorer, predictor, 17)
    assert scorer.calls == max(0, q - 5)  # recording never adds another forward pass
    removed_so_far = set()
    for row in result["trace"]:
        assert not removed_so_far.intersection(row["candidates"])
        assert len(row["selected"]) <= 2
        assert not set(row["recent"]).intersection(row["candidates"])
        assert row["candidates"] == row["previous"] + [row["arrival"]]
        if row["clock"] < 6:
            assert row["deletion_logits"] is None and row["removed"] is None
        else:
            assert np.allclose(row["deletion_logits"], scores)
            if scores[0] == scores[1] == scores[2]:
                assert row["removed"] == min(row["candidates"])
            else:
                assert row["removed"] == row["candidates"][1]
            removed_so_far.add(row["removed"])


@pytest.mark.parametrize("change", ["future_observations", "future_actions", "annotations"])
def test_future_and_privileged_metadata_cannot_change_selection(models, change):
    assets, scorer, predictor = models
    original = episode()
    changed = copy.deepcopy(original)
    if change == "future_observations":
        changed.p[changed.q + 1:] += 200
        changed.z[changed.q + 1:] -= 200
    elif change == "future_actions":
        changed.actions[changed.q:] *= -12
        changed.prefix = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(changed.actions, axis=0)])
    else:
        changed.cue = (0, 0)
        changed.classes = 1 - changed.classes
        changed.evidence = np.array([0])
        changed.id = "different-evaluation-id"
    a = trace_episode(assets, original, scorer, predictor, 17)
    b = trace_episode(assets, changed, scorer, predictor, 17)
    assert a["kept"] == b["kept"]
    keys = ("clock", "candidates", "deletion_logits", "removed", "selected")
    assert [[row[k] for k in keys] for row in a["trace"]] == [[row[k] for k in keys] for row in b["trace"]]


def test_hook_is_removed_when_scoring_fails(models):
    assets, _, predictor = models
    scorer = FixedScorer([float("nan"), 1, 2])
    existing = scorer.register_forward_hook(lambda *args: None)
    before = dict(scorer._forward_hooks)
    try:
        with pytest.raises(ValueError, match="finite deletion logits"):
            trace_episode(assets, episode(), scorer, predictor, 17)
        assert scorer._forward_hooks == before
    finally:
        existing.remove()


def test_train_episode_is_rejected(models):
    assets, scorer, predictor = models
    with pytest.raises(ValueError, match="validation episodes only"):
        trace_episode(assets, episode(split="train"), scorer, predictor, 17)


@pytest.fixture
def asset_tree(tmp_path, models):
    """A tiny, checksum-valid software fixture exercises the real loaders, without training."""
    assets, scorer, predictor = models
    root, students = tmp_path / "software-fixture", tmp_path / "students"
    records, index = [], {"signature": "software-fixture", "encoder_hash": "not-ijepa", "episodes": {}}
    for e in [episode(name="train", split="train"), episode(), episode(name="validation-a")]:
        obs, meta = f"observations/{e.id}.npz", f"evaluation_only/{e.id}.npz"
        write_npz(root / "dataset" / obs, actions=e.actions)
        write_npz(root / "dataset" / meta, classes=e.classes, evidence=e.evidence)
        records.append(dict(id=e.id, task="gate", pair=e.pair, sibling=e.sibling, split=e.split,
                            query=e.q, cue_start=e.cue[0], cue_end=e.cue[1], obs=obs, eval=meta,
                            obs_sha256=file_hash(root / "dataset" / obs), eval_sha256=file_hash(root / "dataset" / meta)))
        feature = root / "features" / e.split / f"{e.id}.npz"
        write_npz(feature, z=e.z, signature=np.array(index["signature"]), encoder_hash=np.array(index["encoder_hash"]))
        index["episodes"][e.id] = dict(path=f"{e.split}/{e.id}.npz", shape=list(e.z.shape), sha256=file_hash(feature))
    write_json(root / "resolved_config.json", {"data": {"recent": 4, "budget": 2, "horizons": [3, 6]},
                                               "normalization": {"std_floor": 0.05}})
    write_json(root / "dataset/manifest.json", {"signature": "fixture", "records": records})
    write_json(root / "dataset/split_audit.json", {"passed": True, "signature": "fixture"})
    write_json(root / "features/index.json", index)
    write_json(root / "features/pair_feature_audit.json", {"passed": True, "signature": index["signature"],
                                                          "index_sha256": file_hash(root / "features/index.json")})
    write_npz(root / "features/normalization.npz", mean=np.zeros(DIM, np.float32), std=np.ones(DIM, np.float32),
              signature=np.array(index["signature"]), floor=np.array(0.05), train_ids=np.array(["train"]))
    write_json(root / "features/normalization.json", {"sha256": file_hash(root / "features/normalization.npz")})
    write_npz(root / "predictor/gate/basis.npz", **{k: getattr(predictor, k).numpy() for k in
                                                  ("input_mean", "projection", "target_mean", "output_basis")})
    write_json(root / "predictor/gate/basis.json", {"sha256": file_hash(root / "predictor/gate/basis.npz")})
    assets.decoder.save(root / "probes/gate_state_decoder.npz")
    write_json(root / "probes/metrics.json", {"cache_signature": index["signature"], "checkpoints": {
        "gate_state_decoder.npz": file_hash(root / "probes/gate_state_decoder.npz")}})
    checkpoint = root / "predictor/gate/seed_17/best.pt"
    save_torch(checkpoint, {"seed": 17, "task": "gate", "horizons": HORIZONS, "feature_dim": DIM,
                            "model_config": {"hidden": 8, "dropout": 0.0}, "model": predictor.state_dict()})
    save_students(students / "student_17.pt", {"attention_proxy_student": scorer.state_dict(), POLICY: scorer.state_dict()}, 17)
    config = {"environment": "gate", "assets": {"root": str(root), "students": str(students),
                                                 "predictor_sha256": {17: file_hash(checkpoint)}}}
    config_path = tmp_path / "config.json"  # JSON is also valid YAML
    write_json(config_path, config)
    return SimpleNamespace(root=root, students=students, config=config, config_path=config_path,
                           out=tmp_path / "output", checkpoint=checkpoint)


def arguments(tree, *extra):
    return ["--config", str(tree.config_path), "--seed", "17", "--out", str(tree.out), "--threads", "1", *extra]


def test_cli_real_loaders_default_episode_and_output_provenance(asset_tree, capsys):
    tree = asset_tree
    assert cli.main(arguments(tree)) == 0
    result = json.loads((tree.out / "gate_trace_seed17.json").read_text())
    assert result["episode"]["id"] == "validation-z"  # manifest order, not alphabetical
    assert result["seed"] == 17 and result["device"] == "cpu"
    assert result["info"]["predictor"]["sha256"] == file_hash(tree.checkpoint)
    assert result["info"]["students"]["sha256"] == file_hash(tree.students / "student_17.pt")
    assert result["info"]["config_sha256"] == file_hash(tree.config_path)
    assert len(result["source_sha256"]) == 64 and len(result["info"]["entrypoint_sha256"]) == 64
    loaded = open_assets(tree.config, "cpu")
    scorer = frozen_scorers(tree.students, 17, "cpu")[0][POLICY]
    expected = assert_matches_evaluation(loaded, loaded.split("validation")[0], scorer, loaded.predictor(17))
    for key, value in expected.items():
        assert result[key] == value
    assert "deletion logits" in capsys.readouterr().out


def test_cli_requires_force_to_replace_and_supports_students_override(asset_tree, capsys):
    tree = asset_tree
    tree.out.mkdir()
    path = tree.out / "gate_trace_seed17.json"
    path.write_bytes(b"existing-result")
    assert cli.main(arguments(tree)) == 1
    assert path.read_bytes() == b"existing-result"
    assert "--force" in capsys.readouterr().err
    config = copy.deepcopy(tree.config)
    config["assets"].pop("students")
    write_json(tree.config_path, config)
    assert cli.main(arguments(tree, "--force", "--episode-id", "validation-a", "--students", str(tree.students))) == 0
    assert json.loads(path.read_text())["episode"]["id"] == "validation-a"


@pytest.mark.parametrize("episode_id", ["unknown", "train"])
def test_cli_unknown_or_nonvalidation_episode_writes_nothing(asset_tree, capsys, episode_id):
    assert cli.main(arguments(asset_tree, "--episode-id", episode_id)) == 1
    assert "validation episode" in capsys.readouterr().err
    assert not asset_tree.out.exists()


@pytest.mark.parametrize("corruption", ["predictor_checksum", "feature_checksum", "missing_feature", "missing_student", "student_metadata"])
def test_cli_bad_assets_write_no_success_record(asset_tree, capsys, corruption):
    tree = asset_tree
    feature = tree.root / "features/validation/validation-z.npz"
    if corruption == "predictor_checksum":
        tree.checkpoint.write_bytes(b"not-a-checkpoint")
    elif corruption == "feature_checksum":
        feature.write_bytes(b"changed-cache")
    elif corruption == "missing_feature":
        feature.unlink()
    elif corruption == "missing_student":
        (tree.students / "student_17.pt").unlink()
    else:
        save_torch(tree.students / "student_17.pt", {"seed": 29})
    assert cli.main(arguments(tree)) == 1
    error = capsys.readouterr().err
    assert "Gate trace failed:" in error
    if "checksum" in corruption:
        assert "checksum mismatch" in error
    assert not tree.out.exists()


def test_cli_unset_environment_and_missing_root_are_actionable(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("BM_GATE_TRACE_TEST_MISSING", raising=False)
    config = tmp_path / "config.yaml"
    config.write_text('environment: gate\nassets:\n  root: ${BM_GATE_TRACE_TEST_MISSING}/root\n', encoding="utf-8")
    out = tmp_path / "out"
    args = ["--config", str(config), "--seed", "17", "--out", str(out), "--threads", "1"]
    assert cli.main(args) == 1
    assert "environment variable not set" in capsys.readouterr().err
    write_json(config, {"environment": "gate", "assets": {"root": str(tmp_path / "absent"), "students": str(tmp_path / "students")}})
    assert cli.main(args) == 1
    error = capsys.readouterr().err
    assert "resolved_config.json" in error and "student_17.pt" in error
    assert not out.exists()


def test_released_assets_trace_matches_existing_evaluation():
    root, students = os.environ.get("BM_GATE_ROOT"), os.environ.get("BM_GATE_STUDENTS")
    if not root or not students:
        pytest.skip("real Gate acceptance pending: set BM_GATE_ROOT and BM_GATE_STUDENTS to released assets")
    assets = open_assets({"assets": {"root": root, "predictor_sha256": "team"}}, "cpu")
    scorer = frozen_scorers(students, 17, "cpu")[0][POLICY]
    assert_matches_evaluation(assets, assets.split("validation")[0], scorer, assets.predictor(17))
