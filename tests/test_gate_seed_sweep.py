"""Gate seed sweep: a training seed of its own (initial weights and minibatch order) for our scorer and for
the literature baselines, on a fixed predictor, and the sweep summary. Synthetic inputs only; no scientific result."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bounded_memory.methods.deletion_cost import training  # noqa: E402
from test_memory_baselines import TinyReader, episode  # noqa: E402


def label_pack(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3, 517)).astype(np.float32)
    ya = rng.dirichlet(np.ones(3), size=n).astype(np.float32)
    yd = rng.dirichlet(np.ones(3), size=n).astype(np.float32)
    return x, ya, yd


def load_script(relative):
    spec = importlib.util.spec_from_file_location("script_" + relative.replace("/", "_")[:-3], ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- our scorer

def test_scorer_fit_with_the_predictor_seed_as_training_seed_is_the_published_fit():
    x, ya, yd = label_pack(4096)
    published = training.fit_pair(x, ya, yd, seed=17, split="train")
    mine = training.fit_scorers_with_training_seed(x, ya, yd, train_seed=17, device="cpu")
    for arm, key in (("attention_proxy_student", "attention"), ("deletion_cost_student", "deletion")):
        assert mine[arm].keys() == published[key].keys()
        for name, weight in published[key].items():
            assert torch.equal(mine[arm][name], weight), (arm, name)


def test_scorer_fit_follows_the_training_seed_and_accepts_seeds_outside_the_predictor_seeds():
    x, ya, yd = label_pack(256)
    first = training.fit_scorers_with_training_seed(x, ya, yd, train_seed=1, device="cpu")["deletion_cost_student"]
    again = training.fit_scorers_with_training_seed(x, ya, yd, train_seed=1, device="cpu")["deletion_cost_student"]
    other = training.fit_scorers_with_training_seed(x, ya, yd, train_seed=2, device="cpu")["deletion_cost_student"]
    assert all(torch.equal(first[k], again[k]) for k in first)
    assert any(not torch.equal(first[k], other[k]) for k in first)


def test_scorer_script_trains_each_training_seed_into_its_own_files(tmp_path):
    x, ya, yd = label_pack(128)
    labels, students, out = tmp_path / "labels", tmp_path / "students", tmp_path / "out"
    labels.mkdir()
    np.savez(labels / "teacher_labels_seed17.npz", features=x, attention_target=ya, deletion_target=yd)
    config = tmp_path / "gate.yaml"
    config.write_text("environment: gate\n")
    common = ["--config", config, "--seed", 17, "--out", out, "--labels", labels, "--checkpoints", students, "--threads", 1]
    for n in (3, 4):
        subprocess.run([sys.executable, str(ROOT / "scripts/train_scorer/gate.py"), *map(str, common), "--train-seed", str(n)],
                       check=True, capture_output=True, text=True)
    for n in (3, 4):
        assert json.loads((out / f"train_scorer_t{n}_seed17.json").read_text())["info"]["train_seed"] == n
        loaded, info = training.frozen_scorers(students, 17, "cpu", filename=f"student_{{seed}}_t{n}.pt")
        assert info["file"] == f"student_17_t{n}.pt"
        expected = training.fit_scorers_with_training_seed(x, ya, yd, train_seed=n, device="cpu")["deletion_cost_student"]
        state = loaded["deletion_cost_student"].state_dict()
        assert all(torch.equal(state[k], v) for k, v in expected.items())
    assert not (students / "student_17.pt").exists() and not (out / "train_scorer_seed17.json").exists()


def test_scorer_checkpoint_name_of_a_training_seed():
    assert training.student_filename() == "student_{seed}.pt"
    assert training.student_filename(3).format(seed=17) == "student_17_t3.pt"


# ---------------------------------------------------------------- literature baselines

class FakeScorer:
    """Stands in for environments.gate.experiment.Scorer: the frozen predictor and episodes, no metric."""
    episodes = [episode(i) for i in range(4)]

    def __init__(self, assets, seed, task):
        self.assets, self.model, self.train, self.val, self.rows = assets, TinyReader().eval(), self.episodes, self.episodes, []
        self.model.requires_grad_(False)

    def add(self, policy, selected=None, prediction=None):
        self.rows.append({"policy": policy, "seed": None, "route_accuracy": 1.0})

    def payload(self):
        return {"environment": "gate", "rows": self.rows}


def test_baseline_script_trains_with_the_training_seed_and_keeps_runs_apart(tmp_path, monkeypatch):
    module = load_script("scripts/train_baselines/gate.py")
    monkeypatch.setattr(module, "Scorer", FakeScorer)
    monkeypatch.setattr(module, "open_assets", lambda config, device: SimpleNamespace(device=torch.device("cpu")))
    config, out, checkpoints = tmp_path / "gate.yaml", tmp_path / "out", tmp_path / "checkpoints"
    config.write_text("assets: {root: synthetic}\nbaselines: {pacc: {steps: 2}}\n")

    def run(predictor_seed, *extra):
        monkeypatch.setattr(sys, "argv", ["gate.py", "--config", str(config), "--seed", str(predictor_seed), "--out", str(out),
                                          "--threads", "1", "--method", "pacc_recurrent", "--checkpoints", str(checkpoints), *extra])
        module.main()

    run(17, "--train-seed", "3")
    run(17, "--train-seed", "4")
    run(29, "--train-seed", "3")
    run(17)
    for predictor, n in ((17, 3), (17, 4), (29, 3)):
        record = json.loads((out / f"pacc_recurrent_t{n}_seed{predictor}.json").read_text())
        assert record["seed"] == predictor and record["info"]["train_seed"] == n
        state = torch.load(checkpoints / f"pacc_recurrent_seed{predictor}_t{n}_cpu.pt", weights_only=True)
        assert json.loads(state["identity"])["seed"] == n and state["step"] == 2
    record = json.loads((out / "pacc_recurrent_seed17.json").read_text())
    assert record["info"]["train_seed"] == 17
    assert json.loads(torch.load(checkpoints / "pacc_recurrent_seed17_cpu.pt", weights_only=True)["identity"])["seed"] == 17


# ---------------------------------------------------------------- summary

def sweep_file(folder, task, predictor, n, policy, accuracies, students=None):
    info = {"train_seed": n}
    if students:
        info["students"] = {"file": students}
    rows = [{"policy": policy, "seed": predictor, "route_accuracy": a} for a in accuracies]
    (folder / f"{task}_t{n}_seed{predictor}.json").write_text(json.dumps(
        {"task": f"{task}_t{n}", "seed": predictor, "device": "cpu", "threads": 8, "info": info, "rows": rows}))


def test_sweep_summary_counts_collapses_and_pools_each_training_seed(tmp_path):
    folder = tmp_path / "run/gate/sweep"
    folder.mkdir(parents=True)
    for predictor, n, acc in ((17, 1, [1, 1, 1, 0.8]), (17, 2, [0.5, 0.5, 0.5, 0.5]), (29, 1, [1, 1, 0.75, 1]), (29, 2, [1, 0.75, 1, 1])):
        sweep_file(folder, "pacc_recurrent", predictor, n, "pacc_recurrent", acc)
    for predictor, n, acc in ((17, 1, [1, 1, 0.75, 1]), (17, 2, [1, 1, 1, 0.75]), (29, 1, [1, 1, 1, 1]), (29, 2, [1, 0.75, 1, 1])):
        sweep_file(folder, "deletion_cost", predictor, n, "deletion_cost_student", acc, students=f"student_{predictor}_t{n}.pt")
    subprocess.run([sys.executable, str(ROOT / "scripts/aggregate_results/gate_seed_sweep.py"), "--run", str(tmp_path / "run"),
                    "--collapse-below", "0.75"], check=True, capture_output=True, text=True)
    summary = json.loads((tmp_path / "run/aggregate/seed_sweep.json").read_text())
    c2r, ours = summary["methods"]["pacc_recurrent"], summary["methods"]["deletion_cost"]
    assert (c2r["runs"], c2r["collapsed"], ours["runs"], ours["collapsed"]) == (4, 1, 4, 0)
    assert c2r["route_accuracy_mean"] == pytest.approx((0.95 + 0.5 + 0.9375 + 0.9375) / 4)
    assert c2r["route_accuracy_mean_trained"] == pytest.approx((0.95 + 0.9375 + 0.9375) / 3)
    assert c2r["pooled_by_training_seed"] == pytest.approx({"1": (0.95 + 0.9375) / 2, "2": (0.5 + 0.9375) / 2})
    assert ours["route_accuracy_mean"] == pytest.approx((0.9375 * 3 + 1.0) / 4)
    assert summary["head_to_head"] == {"cells": 4, "pacc_recurrent_higher": 1, "deletion_cost_higher": 2, "tied": 1}
    assert (tmp_path / "run/aggregate/seed_sweep.md").read_text().startswith("# Gate seed sweep")


def test_sweep_summary_refuses_a_score_of_the_wrong_scorer_checkpoint(tmp_path):
    folder = tmp_path / "run/gate/sweep"
    folder.mkdir(parents=True)
    sweep_file(folder, "deletion_cost", 17, 1, "deletion_cost_student", [1.0], students="student_17.pt")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/aggregate_results/gate_seed_sweep.py"), "--run", str(tmp_path / "run")],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "student_17_t1.pt" in result.stderr
