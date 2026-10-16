import copy
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from phase1.config import DEFAULT, CONDITIONS, SPLITS, validate, scientific_hash
from phase1.data import generate, validate_data, load_episode
from phase1.envs import scene, episode, render, motion_step
from phase1.cache import encode, fit_normalization, load_index, load_feature, normalize
from phase1.inputs import FeatureStore, memory_indices, collate
from phase1.encoders import SmokePixels, FrozenIJepa, pool_tokens
from phase1.model import DynamicsPredictor, fit_basis, project
from phase1.probes import probe
from phase1.training import train, trusted_checkpoint, paired_rows, regression_loss
from phase1.evaluation import evaluate, pair_diagnostic
from phase1.reporting import report, decision
from phase1.stats import clustered_interval
from phase1.utils import seed_all, write_npz, read_json


@pytest.fixture(scope="module")
def ready(tmp_path_factory):
    c = copy.deepcopy(DEFAULT)
    c["output"] = str(tmp_path_factory.mktemp("phase1 data with spaces"))
    c["data"]["pairs"] = {s: 4 if s == "train" else 2 for s in SPLITS}
    c["model"].update(hidden=32, layers=1, dropout=0)
    c["training"].update(epochs=3, seeds=[17], batch_size=8)
    c["evaluation"].update(bootstrap=100, latency_warmup=1, latency_repeats=2)
    c["probe"]["ridge"] = [.1, 1.]
    validate(c)
    seed_all(17)
    generate(c)
    validate_data(c)
    encode(c)
    fit_normalization(c)
    probe(c)
    return c


def test_deterministic_paired_splits(ready):
    audit = validate_data(ready)
    assert audit["passed"]
    assert all(r["labels"]["0"] == r["labels"]["1"] for r in audit["splits"])
    ix = read_json(Path(ready["output"])/"dataset/manifest.json")
    by = {}
    for r in ix["records"]:
        by.setdefault(r["pair"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by.values())


@pytest.mark.parametrize("task", ["gate", "motion"])
def test_exact_pair_recent_history_future_and_actions(ready, task):
    s = scene(ready, task, "test_id", 1)
    a, b = episode(s, 0), episode(s, 1)
    q, R = s["query"], ready["data"]["recent"]
    assert np.array_equal(a["rgb"][q-R+1:q+1], b["rgb"][q-R+1:q+1])
    assert np.array_equal(a["actions"], b["actions"])
    assert np.all(np.array(a["evidence"]) < q-R+1)
    for h in ready["data"]["horizons"]:
        assert not np.array_equal(a["rgb"][q+h], b["rgb"][q+h])
        assert not np.array_equal(a["states"][q+h], b["states"][q+h])
    assert not np.array_equal(a["rgb"][a["evidence"]], b["rgb"][b["evidence"]])


def test_occlusion_in_rgb_no_hidden_position_leak(ready):
    s = scene(ready, "motion", "train", 0)
    q = s["query"]
    # Different hidden object positions, same opaque rendered RGB and then same smoke features.
    rgb = np.stack([render(s, q, label, x, .50, False) for label, x in [(0, .1), (1, .9)]])
    assert np.array_equal(rgb[0], rgb[1])
    z = SmokePixels(ready["encoder"], torch.device("cpu")).encode(rgb)
    assert np.array_equal(z[0], z[1])
    assert np.array_equal(episode(s, 0)["rgb"][s["end"]], episode(s, 1)["rgb"][s["end"]])


@pytest.mark.parametrize("task", ["gate", "motion"])
def test_actions_change_future_and_are_not_label(ready, task):
    s = scene(ready, task, "train", 0)
    alt = copy.deepcopy(s)
    q = s["query"]
    if task == "gate":
        alt["actions"][q, 0] = -s["actions"][q, 0]
    else:
        alt["actions"][q, 0] += 1.
    a, b = episode(s, 0), episode(alt, 0)
    assert np.array_equal(a["states"][:q+1], b["states"][:q+1])
    assert not np.array_equal(a["states"][q+3], b["states"][q+3])
    assert np.array_equal(episode(s, 0)["actions"], episode(s, 1)["actions"])


def test_motion_update_equation():
    x, v = motion_step(.4, .01, np.array([1., 0.]))
    assert x == pytest.approx(.4115) and v == .01
    x, v = motion_step(.919, .01, np.zeros(2))
    assert x == pytest.approx(.911) and v == -.01


def test_memory_boundaries_budget_uniform_no_replacement():
    q, R, B, evidence = 15, 4, 2, [2, 5]
    hits = 0
    rng = np.random.default_rng(8)
    for _ in range(2000):
        indices = memory_indices(q, R, B, evidence, "random_small_budget", rng)
        assert len(indices) == B and len(set(indices)) == B
        assert np.all(indices < q-R+1)
        hits += set(evidence).issubset(indices)
    assert 10 < hits < 60  # P=1/C(12,2), not deliberately excluded evidence.
    assert set(memory_indices(q, R, B, evidence, "privileged_small_budget", rng)) == set(evidence)
    assert len(memory_indices(q, R, B, evidence, "full_history", rng)) == q-R+1
    with pytest.raises(ValueError, match="Insufficient history"):
        memory_indices(4, 4, 2, [0], "random_small_budget", rng)


def test_wrong_evidence_replaces_content_only(ready):
    store = FeatureStore(ready)
    rec = next(r for r in store.manifest["records"] if r["task"] == "motion")
    priv, target, pi = store.sample(rec, "privileged_small_budget")
    wrong, wt, wi = store.sample(rec, "wrong_evidence")
    R = ready["data"]["recent"]
    assert np.array_equal(priv["z"][:R], wrong["z"][:R])
    assert not np.array_equal(priv["z"][R:], wrong["z"][R:])
    for key in ["age", "type", "past_actions", "actions", "current"]:
        assert np.array_equal(priv[key], wrong[key])
    assert np.array_equal(pi["indices"], wi["indices"])
    assert np.array_equal(target, wt)


def test_metadata_hidden_values_cannot_enter_inputs_or_predictions(ready, monkeypatch):
    import phase1.inputs as inputs
    store = FeatureStore(ready)
    rec = store.manifest["records"][0]
    expected = store.sample(rec, "privileged_small_budget")[0]
    original = inputs.load_episode
    def changed(c, rec, metadata=False, keys=None):
        result = original(c, rec, metadata, keys)
        if metadata:
            result["label"] = np.int64(12345)
            result["states"] = np.full((1, 2), 9999)
            result["classes"] = np.array([-999])
        return result
    monkeypatch.setattr(inputs, "load_episode", changed)
    actual = store.sample(rec, "privileged_small_budget")[0]
    assert set(actual) == {"z", "age", "type", "past_actions", "actions", "current"}
    for key in actual:
        assert np.array_equal(actual[key], expected[key])
    seed_all(5)
    model = DynamicsPredictor(len(store.norm["mean"]), ready["model"], ready["data"]["horizons"]).eval()
    with torch.inference_mode():
        assert torch.equal(model(collate([actual], "cpu")), model(collate([expected], "cpu")))


def test_padding_and_horizon_action_causality(ready):
    store = FeatureStore(ready)
    rows = store.manifest["records"][:2]
    samples = [store.sample(rows[0], "recent_only")[0], store.sample(rows[1], "full_history")[0]]
    batch = collate(samples, "cpu")
    model = DynamicsPredictor(len(store.norm["mean"]), ready["model"], [3, 6]).eval()
    with torch.inference_mode():
        baseline = model(batch)
        changed = {k: v.clone() for k, v in batch.items()}
        changed["z"][~batch["mask"]] = float("nan")
        changed["age"][~batch["mask"]] = float("nan")
        changed["type"][~batch["mask"]] = 999
        changed["past_actions"][~batch["mask"]] = float("nan")
        assert torch.equal(baseline, model(changed))
        changed = {k: v.clone() for k, v in batch.items()}
        changed["actions"][:, 3:] += 5  # a[t+3] is NOT legal for h=3.
        pred = model(changed)
        assert torch.equal(baseline[:, 0], pred[:, 0])
        assert not torch.allclose(baseline[:, 1], pred[:, 1])
        changed["actions"][:, 2] += 5  # a[t+2] DOES affect h=3.
        assert not torch.allclose(baseline[:, 0], model(changed)[:, 0])


def test_train_only_normalization(ready):
    store = FeatureStore(ready)
    train = np.concatenate([load_feature(ready, r, store.ix) for r in store.manifest["records"] if r["split"] == "train"])
    assert np.allclose(store.norm["mean"], train.astype(np.float64).mean(0), atol=1e-6)
    assert np.min(store.norm["std"]) >= ready["normalization"]["std_floor"]-1e-7
    assert set(store.norm["train_ids"]) == {r["id"] for r in store.manifest["records"] if r["split"] == "train"}


def test_changed_config_rejected(ready):
    other = copy.deepcopy(ready)
    other["encoder"]["pooling"] = "mean"
    with pytest.raises(ValueError, match="mismatch"):
        load_index(other)
    other = copy.deepcopy(ready)
    other["data"]["horizons"] = [3, 7]
    with pytest.raises(ValueError, match="mismatch"):
        load_index(other)


def test_past_features_independent_of_future_rgb(ready):
    s = scene(ready, "gate", "train", 0)
    a = episode(s, 0)["rgb"]
    b = a.copy()
    b[s["query"]+1:] = 255-b[s["query"]+1:]
    encoder = SmokePixels(ready["encoder"], torch.device("cpu"))
    za, zb = encoder.encode(a), encoder.encode(b)
    assert np.array_equal(za[:s["query"]+1], zb[:s["query"]+1])


def test_tiny_ijepa_adapter_freeze_pooling_no_off_by_one(ready):
    from transformers import IJepaConfig, IJepaModel, ViTImageProcessor
    seed_all(3)
    config = IJepaConfig(image_size=32, patch_size=8, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4)
    model = IJepaModel(config)
    processor = ViTImageProcessor(size={"height": 32, "width": 32})
    ec = copy.deepcopy(ready["encoder"])
    ec.update(backend="ijepa", pooling="grid2")
    adapter = FrozenIJepa(ec, torch.device("cpu"), model, processor)
    s = scene(ready, "gate", "train", 0)
    rgb = episode(s, 0)["rgb"][:2]
    z = adapter.encode(rgb)
    assert z.shape == (2, 128)
    assert not adapter.model.training
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert "UNTRAINED" in adapter.metadata["resolved_revision"]
    tokens = torch.arange(16.).reshape(1, 16, 1)
    assert torch.equal(pool_tokens(tokens, 32, 32, 8, "grid2"), torch.tensor([[2.5, 4.5, 10.5, 12.5]]))
    with pytest.raises(ValueError, match="do not guess"):
        pool_tokens(torch.zeros(1, 17, 32), 32, 32, 8, "grid2")


def test_on_disk_ijepa_load_offline_preflight_and_missing_weights(tmp_path, monkeypatch):
    from transformers import IJepaConfig, IJepaModel, ViTImageProcessor
    from safetensors.torch import load_file, save_file
    from phase1.cli import preflight
    import huggingface_hub
    folder = tmp_path/"tiny checkpoint with spaces"
    model = IJepaModel(IJepaConfig(image_size=32, patch_size=8, hidden_size=32,
                       intermediate_size=64, num_hidden_layers=1, num_attention_heads=4))
    model.save_pretrained(folder)
    ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(folder)
    c = copy.deepcopy(DEFAULT)
    c.update(mode="pilot", output=str(tmp_path/"preflight"))
    c["encoder"].update(backend="ijepa", repository=str(folder))
    def forbidden_head(*args, **kwargs):
        raise AssertionError("Local checkpoint must not make Hub HEAD requests")
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", forbidden_head)
    info = preflight(c, check_checkpoint=True)
    assert info["passed"] and info["resolved_revision"].startswith("local-sha256:")
    adapter = FrozenIJepa(c["encoder"], torch.device("cpu"))
    z = adapter.encode(np.zeros((2, 96, 96, 3), dtype=np.uint8))
    assert z.shape == (2, 128) and np.array_equal(z[0], z[1])
    assert all(not p.requires_grad for p in adapter.model.parameters())
    weights = folder/"model.safetensors"
    state = load_file(weights)
    del state[next(iter(state))]
    save_file(state, weights)
    with pytest.raises(ValueError, match="refusing randomly initialized"):
        FrozenIJepa(c["encoder"], torch.device("cpu"))


def test_encoder_resumable_episode_commit(tmp_path):
    c = copy.deepcopy(DEFAULT)
    c["output"] = str(tmp_path/"resumable")
    c["data"]["pairs"] = dict.fromkeys(SPLITS, 2)
    generate(c)
    validate_data(c)
    partial = encode(c, max_episodes=1)
    first = next(iter(partial["episodes"].values()))
    p = Path(c["output"])/"features"/first["path"]
    before = p.stat().st_mtime_ns
    full = encode(c)
    assert len(full["episodes"]) == 40 and p.stat().st_mtime_ns == before
    assert not list((Path(c["output"])/"features").rglob(".partial-*"))


def test_probe_recent_chance_and_motion_single_end_chance(ready):
    result = read_json(Path(ready["output"])/"probes/metrics.json")
    for p in result["results"]:
        if p["variant"] in {"recent_only", "motion_single_end"}:
            assert p["accuracy"]["mean"] == .5


def test_overfit_tiny_actual_targets(ready):
    seed_all(31)
    store = FeatureStore(ready)
    rows = [r for r in store.manifest["records"] if r["split"] == "train"]
    rows = [r for r in rows if r["scene_index"] == 0]  # Both tasks, both siblings.
    samples = [store.sample(r, "privileged_small_budget") for r in rows]
    batch = collate([s[0] for s in samples], "cpu")
    target = torch.from_numpy(np.stack([s[1] for s in samples]))
    model = DynamicsPredictor(len(store.norm["mean"]), ready["model"], [3, 6])
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    initial = float(((model(batch)-target)**2).mean().detach())
    for _ in range(500):
        optimizer.zero_grad()
        loss = ((model(batch)-target)**2).mean()
        loss.backward()
        optimizer.step()
    final = float(((model(batch)-target)**2).mean().detach())
    print(f"overfit actual targets: initial={initial:.8f}, final={final:.8f}, steps=500")
    assert final < min(.05, initial*.1), (initial, final)


def test_checkpoint_resume_epoch_boundary_exact(ready, tmp_path):
    full, resumed = copy.deepcopy(ready), copy.deepcopy(ready)
    for c, name in [(full, "uninterrupted"), (resumed, "resumed")]:
        c["output"] = str(tmp_path/name)
        shutil.copytree(Path(ready["output"])/"dataset", Path(c["output"])/"dataset")
        shutil.copytree(Path(ready["output"])/"features", Path(c["output"])/"features")
        c["model"]["dropout"] = .05
    train(full)
    train(resumed, stop_after_epoch=1)
    ck = trusted_checkpoint(Path(resumed["output"])/"predictor/gate/seed_17/latest.pt")
    assert ck["epoch"] == 1 and ck["resume_granularity"] == "epoch_boundary"
    train(resumed)
    a = trusted_checkpoint(Path(full["output"])/"predictor/gate/seed_17/latest.pt")
    b = trusted_checkpoint(Path(resumed["output"])/"predictor/gate/seed_17/latest.pt")
    assert a["epoch"] == b["epoch"] == 3 and a["step"] == b["step"]
    assert a["scheduler"] == b["scheduler"]
    import ast
    for row in a["logs"]:
        counts = ast.literal_eval(row["mix_counts"])
        assert len(counts) == 4 and set(counts.values()) == {len([r for r in FeatureStore(full).manifest["records"] if r["split"] == "train" and r["task"] == "gate"])}
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert all(x["validation_mse"] == y["validation_mse"] for x, y in zip(a["logs"], b["logs"]))
    wrong = copy.deepcopy(resumed)
    wrong["training"]["lr"] *= 2
    with pytest.raises(ValueError, match="mismatch"):
        train(wrong)


def test_pair_loss_and_balanced_history_training(ready):
    # Ignoring a differing future branch is penalized even at the paired midpoint.
    target = torch.tensor([[[0.]], [[2.]]])
    midpoint = torch.ones_like(target, requires_grad=True)
    loss, mse, difference = regression_loss(midpoint, target, .5)
    assert float(mse.detach()) == 1 and float(difference.detach()) == 4 and float(loss.detach()) == 3
    loss.backward()
    assert midpoint.grad[0] > 0 and midpoint.grad[1] < 0
    assert float(regression_loss(target, target, .5)[0]) == 0
    store = FeatureStore(ready)
    rows = [r for r in store.manifest["records"] if r["split"] == "train"]
    assert len(paired_rows(rows))*2 == len(rows)
    with pytest.raises(ValueError, match="complete sibling"):
        paired_rows(rows[:-1])


def test_controls_since_each_observation_use_only_past(ready):
    store = FeatureStore(ready)
    rec = next(r for r in store.manifest["records"] if r["task"] == "motion")
    actions = load_episode(ready, rec, keys=("actions",))["actions"]
    sample = store.sample(rec, "privileged_small_budget")[0]
    times = rec["query"]-sample["age"].astype(int)
    expected = np.stack([actions[t:rec["query"]].sum(0) for t in times])
    assert np.allclose(sample["past_actions"], expected, atol=1e-6)
    assert np.array_equal(sample["past_actions"][ready["data"]["recent"]-1], np.zeros(2))



def test_pair_ties_and_clustered_independence():
    own, other = np.array([0., 0.]), np.array([1., 1.])
    assert pair_diagnostic(np.array([.5, .5]), own, other)[0] == .5
    assert pair_diagnostic(own, own, other)[0] == 1
    a = clustered_interval([0, 1], ["a", "b"], 100, 5)
    b = clustered_interval([0]*20+[1]*20, ["a"]*20+["b"]*20, 100, 5)
    assert a == b and b["pairs"] == 2


def test_cpu_smoke_complete_and_report_cannot_go(ready):
    train(ready)
    evaluate(ready)
    result = report(ready)
    assert result["status"] == "SMOKE_ONLY — NOT A RESEARCH CONCLUSION"
    root = Path(ready["output"])
    assert (root/"report.md").exists() and len(list((root/"figures").glob("*.png"))) >= 15
    met = read_json(root/"evaluation/metrics.json")
    for row in met["per_seed"]:
        if row["condition"] == "recent_only":
            assert row["metrics"]["pair_accuracy"]["mean"] == .5


def test_pilot_and_confirm_decision_rules(ready):
    # Synthetic positive metrics exercise decision logic ONLY; not a research measurement.
    probes = {"results": []}
    metrics = {"comparisons": [], "per_seed": []}
    for task in ["gate", "motion"]:
        for split in SPLITS[2:]:
            probes["results"].append({"task": task, "split": split, "variant": "gate_cue" if task == "gate" else "motion_two_frame",
                "accuracy": {"mean": .9}, "wilson_effective_pairs": [.8, 1.]})
            probes["results"].append({"task": task, "split": split, "variant": "recent_only", "accuracy": {"mean": .5}})
            for h in [3, 6]:
                probes["results"].append({"task": task, "split": split, "horizon": h,
                    "variant": "state_decoder_on_true_target", "accuracy": {"mean": .9}, "position_mse": {"mean": .0001}})
                metrics["comparisons"].append({"task": task, "split": split, "horizon": h, "seed": "all_seeds_mean",
                    "alternative": "privileged_small_budget", "improvement": {k: {"mean": .2, "low": .1, "high": .3,
                    "scale": "relative_error_reduction"} for k in ["normalized_mse", "state_accuracy", "state_position_mse"]}})
                metrics["comparisons"].append({"task": task, "split": split, "horizon": h, "seed": "all_seeds_mean",
                    "alternative": "wrong_evidence", "toward_wrong_future": {"mean": .2, "low": .1, "high": .3}})
    c = copy.deepcopy(ready)
    c["mode"] = "pilot"
    c["encoder"]["backend"] = "ijepa"
    assert decision(c, {"passed": True}, probes, metrics)[0] == "INCONCLUSIVE"
    c["mode"] = "confirm"
    c["training"]["seeds"] = [17, 29, 43]
    c["data"]["pairs"] = dict.fromkeys(SPLITS, 64)
    for task in ["gate", "motion"]:
        for split in SPLITS[2:]:
            for h in [3, 6]:
                for seed in c["training"]["seeds"]:
                    for cond in CONDITIONS+["persistence"]:
                        metrics["per_seed"].append({"seed": seed, "task": task, "split": split, "horizon": h,
                            "condition": cond, "metrics": {"pair_accuracy": {"mean": .5 if cond == "recent_only" else .9}},
                            "almost_indistinguishable_fraction": 0.})
                    for cond in ["privileged_small_budget", "full_history", "random_small_budget", "wrong_evidence"]:
                        cmp = {"task": task, "split": split, "horizon": h, "seed": seed, "alternative": cond,
                            "improvement": {k: {"mean": .2, "low": .1, "high": .3, "scale": "relative_error_reduction"}
                                            for k in ["normalized_mse", "state_accuracy", "state_position_mse"]}}
                        if cond == "wrong_evidence":
                            cmp["toward_wrong_future"] = {"mean": .2, "low": .1, "high": .3}
                        metrics["comparisons"].append(cmp)
    assert decision(c, {"passed": True}, probes, metrics)[0] == "GO"
    incomplete = copy.deepcopy(metrics)
    incomplete["per_seed"].pop()
    assert decision(c, {"passed": True}, probes, incomplete)[0] == "INCONCLUSIVE"
    metrics["comparisons"][0]["improvement"]["normalized_mse"].update(mean=-.1, low=-.2, high=0.)
    assert decision(c, {"passed": True}, probes, metrics)[0] == "REVISE"


def test_source_mutation_invalidates_audit(ready, tmp_path):
    from phase1.data import require_audit
    c = copy.deepcopy(ready)
    c["output"] = str(tmp_path/"mutated")
    shutil.copytree(Path(ready["output"])/"dataset", Path(c["output"])/"dataset")
    rec = read_json(Path(c["output"])/"dataset/manifest.json")["records"][0]
    p = Path(c["output"])/"dataset"/rec["obs"]
    o = load_episode(c, rec)
    o["rgb"][rec["query"], 0, 0, 0] ^= 1
    write_npz(p, **o)
    with pytest.raises(ValueError, match="checksum"):
        require_audit(c)


def test_mistagged_smoke_backend_never_go(ready):
    c = copy.deepcopy(ready)
    c["mode"] = "confirm"
    assert decision(c, {"passed": True}, {"results": []}, {"comparisons": [], "per_seed": []})[0].startswith("SMOKE_ONLY")
    with pytest.raises(ValueError, match="Smoke backend"):
        validate(c)


def test_sigterm_rollback_then_resumes(ready, tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time
    import yaml
    c = copy.deepcopy(ready)
    c["output"] = str(tmp_path/"signal restart")
    c["training"]["epochs"] = 50
    for name in ["dataset", "features"]:
        shutil.copytree(Path(ready["output"])/name, Path(c["output"])/name)
    conf = tmp_path/"signal.yaml"
    conf.write_text(yaml.safe_dump(c))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]/"src")
    child = subprocess.Popen([sys.executable, "-m", "phase1.cli", "train", "--config", str(conf)],
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    latest = Path(c["output"])/"predictor/gate/seed_17/latest.pt"
    try:
        deadline = time.monotonic()+15
        while not latest.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert latest.exists(), "Training process failed before signal checkpoint"
        child.send_signal(signal.SIGTERM)
        out, _ = child.communicate(timeout=15)
        assert child.returncode == 143 and "Termination signal" in out, out
        interrupted = trusted_checkpoint(latest)
        assert interrupted["epoch"] < 50
        # Continue at least one epoch from the saved boundary; optimizer/RNG signature stays valid.
        train(c, stop_after_epoch=interrupted["epoch"]+1)
        restored = trusted_checkpoint(latest)
        assert restored["epoch"] == interrupted["epoch"]+1
        assert restored["step"] > interrupted["step"]
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()


def test_shell_templates_and_cli_help():
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    for script in (root/"scripts").iterdir():
        if script.suffix in {".sh", ".sbatch"}:
            subprocess.run(["bash", "-n", str(script)], check=True)
        if script.suffix == ".sbatch":
            s = script.read_text()
            assert "#SBATCH --requeue" in s and "#SBATCH --account=csci_ga_2271-2026fa" in s
            assert "#SBATCH --gres=gpu:1" in s
    from phase1.cli import parser
    for cmd in ["preflight", "generate", "validate-data", "encode", "probe", "train", "evaluate", "report", "run-all", "encoder-check"]:
        with pytest.raises(SystemExit) as ex:
            parser().parse_args([cmd, "--help"])
        assert ex.value.code == 0


def test_batch_launch_chain_and_exports_without_submitting(tmp_path):
    import json
    import os
    import subprocess
    import sys
    root = tmp_path/"uploaded repo with spaces"
    shutil.copytree(Path(__file__).resolve().parents[1]/"scripts", root/"scripts")
    fake_bin = tmp_path/"fake bin"
    fake_bin.mkdir()
    # Only these selected arguments/paths are recorded; never dump environment credentials.
    stub = (f"#!{sys.executable}\nimport json,os,sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['TRACE']).write_text(json.dumps({'args':sys.argv[1:],"
            "'project':os.environ.get('PROJECT_DIR'),'output':os.environ['OUTPUT_DIR'],"
            "'cublas':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}))\n")
    launcher = fake_bin/"verified python launcher"
    launcher.write_text(stub)
    launcher.chmod(0o755)
    sbatch = fake_bin/"sbatch"
    sbatch.write_text(stub)
    sbatch.chmod(0o755)
    out, trace = tmp_path/"scratch output with spaces", tmp_path/"trace.json"
    env = dict(os.environ, PATH=str(fake_bin)+os.pathsep+os.environ["PATH"],
               PYTHON_BIN=str(launcher), OUTPUT_DIR=str(out), TRACE=str(trace),
               CONFIG_PATH="configs/phase1.yaml")
    for gpu in ["l4", "a100"]:
        subprocess.run(["bash", str(root/"scripts/submit.sh"), gpu], env=env, check=True)
        info = json.loads(trace.read_text())
        assert "--export=ALL" in info["args"] and "--account=csci_ga_2271-2026fa" in info["args"]
        assert info["args"][-2:] == [str(root/"scripts/submit.sh"), "--run"]
        assert info["project"] == str(root) and (root/"logs").is_dir() and out.is_dir()
        launch_env = dict(env, PROJECT_DIR=str(root))
        subprocess.run(["bash", str(root/"scripts/submit.sh"), "--run"], env=launch_env, check=True)
        info = json.loads(trace.read_text())
        assert info["args"] == ["-m", "phase1.cli", "run-all", "--config", "configs/phase1.yaml", "--output", str(out), "--device", "cuda"]
        assert info["cublas"] == ":4096:8" and info["output"] == str(out)
    cpu_env = dict(env, AFTER_JOB='123', TASK_DEVICE='cpu', TASK_COMMAND='finish')
    subprocess.run(['bash', str(root/'scripts/submit.sh'), 'cpu'], env=cpu_env, check=True)
    info = json.loads(trace.read_text())
    assert '--dependency=afterok:123' in info['args'] and '--partition=n2c48m24' in info['args']
    assert not any(a.startswith('--gres') for a in info['args'])
    subprocess.run(['bash', str(root/'scripts/submit.sh'), '--run'], env=dict(cpu_env, PROJECT_DIR=str(root)), check=True)
    assert json.loads(trace.read_text())['args'] == ['-m','phase1.cli','finish','--config','configs/phase1.yaml','--output',str(out),'--device','cpu']


def test_train_only_visual_basis_and_task_boundary(ready):
    store = FeatureStore(ready)
    for task in ready['data']['tasks']:
        basis = fit_basis(ready, store, task)
        expected = {r['id'] for r in store.manifest['records'] if r['split'] == 'train' and r['task'] == task}
        assert set(basis['train_ids']) == expected
        assert len(basis['train_ids']) == 2*ready['data']['pairs']['train']
        rec = next(r for r in store.manifest['records'] if r['task'] == task)
        sample = store.sample(rec, 'privileged_small_budget')[0]
        model = DynamicsPredictor(len(store.norm['mean']), ready['model'], [3,6], basis).eval()
        with torch.inference_mode():
            pred = model(collate([sample], 'cpu'))
        assert pred.shape == (1,2,len(store.norm['mean'])) and torch.isfinite(pred).all()
        assert np.isfinite(project(store.features(rec), basis)).all()
        assert not any(k in sample for k in ['label','task','states','evidence','condition'])

def test_pair_ranking_requires_correct_future_and_static_nuisance(ready):
    target = torch.tensor([[[0.,1.]], [[2.,-1.]]])
    correct = regression_loss(target, target, 0, 1)[0]
    reversed_future = regression_loss(target.flip(0), target, 0, 1)[0]
    assert correct < reversed_future and torch.isfinite(reversed_future)
    for task in ready['data']['tasks']:
        s = scene(ready, task, 'test_long_gap', 3)
        assert np.array_equal(s['distractors'], np.broadcast_to(s['distractors'][0], s['distractors'].shape))
        if task == 'motion':
            for label in [0,1]:
                x = episode(s, label)['states'][:s['end']+1,0]
                assert np.allclose(np.diff(x), (2*label-1)*s['speed'], atol=1e-7)


def test_cached_projection_preserves_wrong_content_intervention(ready):
    store = FeatureStore(ready)
    rec = next(r for r in store.manifest['records'] if r['task'] == 'motion')
    basis = fit_basis(ready, store, 'motion')
    model = DynamicsPredictor(len(store.norm['mean']), ready['model'], [3,6], basis).eval()
    for condition in ['privileged_small_budget','wrong_evidence','recent_only']:
        store.bases = {}
        raw = collate([store.sample(rec, condition)[0]], 'cpu')
        store.bases = {'motion':basis}
        cached = collate([store.sample(rec, condition)[0]], 'cpu')
        with torch.inference_mode():
            assert torch.allclose(model(raw), model(cached), atol=1e-5, rtol=1e-5)
