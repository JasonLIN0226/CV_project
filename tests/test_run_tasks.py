"""scripts/run_tasks.py runs many manifest tasks inside one allocation.

Small manifests of sleeping scripts check the scheduling rules: dependency order, the limit on
units running at once, the memory budget, what a failure blocks, and that the same command
continues an unfinished run. The last tests check the real manifest selection that
burst/README.md tells people to run.
"""
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_tasks.py"
WORK = '''
import argparse, pathlib, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--name"); ap.add_argument("--sleep", type=float, default=0.0); ap.add_argument("--fail", action="store_true")
ap.add_argument("--touch"); ap.add_argument("--seed", default=""); ap.add_argument("--device"); ap.add_argument("--out"); ap.add_argument("--marks")
a = ap.parse_args()
marks = pathlib.Path(a.marks); marks.mkdir(parents=True, exist_ok=True)
name = a.name + a.seed
(marks / (name + ".start")).write_text(repr(time.time()))
(marks / (name + ".threads")).write_text(__import__("os").environ["OMP_NUM_THREADS"] + " " + __import__("os").environ["MKL_NUM_THREADS"])
time.sleep(a.sleep)
if a.fail:
    sys.exit(3)
if a.touch:
    pathlib.Path(a.touch).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.touch).write_text("done")
(marks / (name + ".end")).write_text(repr(time.time()))
'''


def task(name, args="", needs=(), mem="1G", cpus=1, **extra):
    return {"script": "work.py", "args": f"--name {name} --marks ${{BM_RUN}}/marks {args}".strip(), "needs": list(needs),
            "resources": {"cpus": cpus, "mem": mem, "time": "00:01:00"}, **extra}


class World:
    def __init__(self, tmp):
        self.tmp, self.repo = tmp, tmp / "repo"
        self.repo.mkdir()
        (self.repo / "work.py").write_text(WORK)
        (tmp / "bin").mkdir()
        (tmp / "bin/python").symlink_to(sys.executable)          # the runner starts every unit with `python`
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "BM_"))}
        self.env.update(PATH=f"{tmp / 'bin'}:{os.environ['PATH']}", BM_SCRATCH=str(tmp / "scratch"), BM_REPO=str(self.repo),
                        HF_HOME=str(tmp / "hf"), BM_POLL_SECONDS="0.05")
        self.marks = tmp / "scratch/reproduction/t/marks"

    def command(self, tasks, *args):
        (self.tmp / "manifest.yaml").write_text(yaml.safe_dump({"tasks": tasks}, sort_keys=False))
        return [sys.executable, str(RUNNER), "--run-id", "t", "--manifest", str(self.tmp / "manifest.yaml"), *args]

    def run(self, tasks, *args):
        done = subprocess.run(self.command(tasks, *args), env=self.env, capture_output=True, text=True, timeout=120)
        return done.returncode, self.status(), done.stdout + done.stderr

    def status(self):
        units = json.loads((self.repo / "runs/t/pack_status.json").read_text())["units"]
        return {f"{u['task']}[{u['index']}]": u["state"] for u in units}

    def span(self, name):
        return tuple(float((self.marks / f"{name}.{end}").read_text()) for end in ("start", "end"))


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def overlapping(spans):
    """Largest number of spans open at the same moment."""
    events = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans], key=lambda x: (x[0], x[1]))
    most = now = 0
    for _, step in events:
        now += step
        most = max(most, now)
    return most


def test_a_task_starts_after_every_seed_of_the_tasks_it_needs(world):
    tasks = {"x.a": task("a", "--sleep 0.3", seeds=[1, 2], out="x"), "x.b": task("b", needs=["x.a"])}
    code, states, out = world.run(tasks, "--select", "x")
    assert code == 0, out
    assert states == {"x.a[0]": "done", "x.a[1]": "done", "x.b[0]": "done"}
    assert world.span("b")[0] >= max(world.span("a1")[1], world.span("a2")[1])
    assert overlapping([world.span("a1"), world.span("a2")]) == 2          # the two seeds ran side by side


def test_no_more_units_run_at_once_than_asked(world):
    tasks = {f"x.{n}": task(n, "--sleep 0.4") for n in "abcd"}
    code, states, out = world.run(tasks, "--select", "x", "--parallel", "2")
    assert code == 0 and set(states.values()) == {"done"}, out
    assert overlapping([world.span(n) for n in "abcd"]) == 2


def test_units_that_do_not_fit_the_memory_budget_together_take_turns(world):
    tasks = {"x.a": task("a", "--sleep 0.3", mem="10G"), "x.b": task("b", "--sleep 0.3", mem="10G"),
             "x.big": task("big", mem="64G")}                               # larger than the budget: runs alone
    code, states, out = world.run(tasks, "--select", "x", "--parallel", "3", "--mem-budget", "15")
    assert code == 0 and set(states.values()) == {"done"}, out
    assert overlapping([world.span(n) for n in ("a", "b", "big")]) == 1


def test_a_unit_gets_the_thread_count_of_its_own_job(world):
    """The per-task jobs set the thread count to the task's cpus; a different count can change rounding."""
    world.env["SLURM_CPUS_PER_TASK"] = "4"
    tasks = {"x.a": task("a", cpus=2), "x.b": task("b", cpus=4), "x.c": task("c", cpus=16)}      # 16 does not fit the allocation
    code, _, out = world.run(tasks, "--select", "x", "--parallel", "3")
    assert code == 0, out
    assert [(world.marks / f"{n}.threads").read_text() for n in "abc"] == ["2 2", "4 4", "4 4"]


def test_a_failure_blocks_only_the_tasks_that_need_it(world):
    tasks = {"x.a": task("a", "--fail"), "x.b": task("b", needs=["x.a"]), "x.c": task("c", needs=["x.b"]), "x.d": task("d")}
    code, states, out = world.run(tasks, "--select", "x")
    assert code == 1
    assert states == {"x.a[0]": "failed", "x.b[0]": "blocked", "x.c[0]": "blocked", "x.d[0]": "done"}
    text = (world.repo / "runs/t/pack_status.txt").read_text()
    log = re.search(r"failed .* x\.a\[0\]\s+log: (\S+)", text).group(1)
    assert "work.py --name a" in Path(log).read_text()


def test_the_same_command_continues_an_unfinished_run(world):
    flag = "${BM_SCRATCH}/a_ready"
    tasks = {"x.a": task("a", f"--touch {flag}", done_when=flag), "x.b": task("b", "--fail", needs=["x.a"])}
    code, states, _ = world.run(tasks, "--select", "x")
    assert code == 1 and states == {"x.a[0]": "done", "x.b[0]": "failed"}
    first = world.span("a")
    tasks["x.b"] = task("b", needs=["x.a"])                                 # the cause of the failure is gone
    code, states, out = world.run(tasks, "--select", "x")
    assert code == 0 and states == {"x.a[0]": "already done", "x.b[0]": "done"}, out
    assert world.span("a") == first                                         # a was not run again


def test_a_stop_signal_ends_the_running_units_and_reports_them(world):
    tasks = {"x.a": task("a", "--sleep 60"), "x.b": task("b", needs=["x.a"])}
    process = subprocess.Popen(world.command(tasks, "--select", "x"), env=world.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 30
    while not (world.marks / "a.start").exists() and time.time() < deadline:
        time.sleep(0.05)
    status = world.repo / "runs/t/pack_status.json"
    while not (status.exists() and '"running"' in status.read_text()) and time.time() < deadline:
        time.sleep(0.05)
    before = status.stat().st_mtime_ns
    time.sleep(0.5)                                                          # ten polls with nothing starting or ending
    assert status.stat().st_mtime_ns == before                               # the status files are written only on a change
    process.send_signal(signal.SIGTERM)
    out, _ = process.communicate(timeout=90)
    assert process.returncode == 1, out
    assert world.status() == {"x.a[0]": "interrupted", "x.b[0]": "pending"}
    assert not (world.marks / "a.end").exists()


def runner():
    spec = importlib.util.spec_from_file_location("bm_run_tasks", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_and_skip_take_task_prefixes():
    tasks = {"x.a": task("a"), "x.b.one": task("b1", needs=["x.a", "w.z"]), "x.b.two": task("b2"), "xy.c": task("c"), "w.z": task("z")}
    chosen, units = runner().plan({"tasks": tasks}, ["x"], ["x.b.two"])
    assert chosen == ["x.a", "x.b.one"]                                     # "xy.c" does not start with "x."
    assert units[1]["needs"] == ["x.a"] and units[1]["assumed_done"] == ["w.z"]
    with pytest.raises(SystemExit):
        runner().plan({"tasks": tasks}, ["nothing"], [])


def readme_commands():
    """Every `burst/submit.sh --run-id ...` command of burst/README.md, as (select, skip), in order."""
    text = (ROOT / "burst/README.md").read_text().replace("\\\n", " ")
    found = []
    for line in re.findall(r"burst/submit\.sh (--run-id [^`\n]*)", text):
        options, flag = {"--select": [], "--skip": []}, None
        for word in line.split():
            if word.startswith("--"):
                flag = word
            elif flag in options:
                options[flag].append(word)
        if (options["--select"], options["--skip"]) not in found:
            found.append((options["--select"], options["--skip"]))
    return found


def test_the_commands_of_the_burst_guide_need_nothing_that_they_do_not_run():
    """A new machine has no finished task, so what the guide says to run must be complete."""
    manifest = yaml.safe_load((ROOT / "configs/reproduction.yaml").read_text())
    commands = readme_commands()
    assert len(commands) >= 2                                               # a first small job, then the full run
    ran_before = set()
    for select, skip in commands:
        chosen, units = runner().plan(manifest, select, skip)
        missing = {u["task"]: u["assumed_done"] for u in units if not set(u["assumed_done"]) <= ran_before}
        assert not missing, missing
        ran_before |= set(chosen)
    units = runner().plan(manifest, *commands[-1])[1]
    assert not any(u["assumed_done"] for u in units)                        # the full run stands alone
    assert sum(u["gpu"] for u in units) >= 10 and all(("--device cuda" in u["command"]) == u["gpu"] for u in units)
