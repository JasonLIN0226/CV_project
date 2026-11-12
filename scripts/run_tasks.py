"""Run manifest tasks inside one allocation, sharing one GPU or scheduling across several.

    python scripts/run_tasks.py --run-id <id> --select <task prefixes> [--skip <prefixes>] [--parallel 3] [--dry-run]

Where a GPU is scarce (one A100 on the course cluster, a long queue on Torch), one job should do
all the work instead of one Slurm job per seed. This expands the selected tasks of
configs/reproduction.yaml exactly as torch/make_jobs.py does (every seed and variant is one unit),
starts a task when the tasks it needs have finished, and keeps up to --parallel units running at
once. With BM_GPU_COUNT greater than one, each CUDA unit gets an available GPU to itself.
A unit starts only if the memory the manifest lists for
it still fits in --mem-budget. Every unit gets the thread count the manifest gives its task
(`cpus`, at most the cores of the allocation), the same as when the task runs as its own job, so
both ways of running a task compute with the same number of threads.

Everything is resumable: finished units are skipped (each script skips its own finished output,
and a task with `done_when` is skipped as a whole), so after a time limit or a failure the same
command continues where it stopped. A unit that fails blocks only the tasks that need it.

Needs BM_SCRATCH, BM_REPO and HF_HOME in the environment (torch/cluster.sh or burst/cluster.sh).
Logs: $BM_SCRATCH/reproduction/<run_id>/logs/<task>_<index>.log
Status: runs/<run_id>/pack_status.txt (and .json), rewritten whenever a unit starts or ends.
Exit code 0 when every selected unit is done, 1 otherwise.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
STATES = ("done", "already done", "running", "pending", "failed", "blocked", "interrupted")


def expander():
    """commands(task_id, task) of torch/make_jobs.py: one command line per (variant, seed)."""
    spec = importlib.util.spec_from_file_location("bm_make_jobs", REPO / "torch/make_jobs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.commands


def gigabytes(text):
    text = str(text).strip().upper()
    return float(text[:-1]) / (1024 if text.endswith("M") else 1) if text[-1] in "GM" else float(text) / 1024


def allocated_devices(count):
    """Return child-process CUDA masks in the allocation's visible order."""
    import torch

    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible < count:
        raise SystemExit(f"BM_GPU_COUNT={count}, but PyTorch sees {visible} CUDA devices in this allocation")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        if visible > count:
            raise SystemExit(f"PyTorch sees {visible} devices and CUDA_VISIBLE_DEVICES is unset; "
                             f"set it to the {count} devices assigned to this run")
        return [str(index) for index in range(count)]
    return [device.strip() for device in mask.split(",")[:count]]


def plan(manifest, select, skip):
    """Selected tasks in manifest order, each expanded into its units."""
    tasks = manifest["tasks"]
    def matches(name, prefixes):
        return any(name == p or name.startswith(p + ".") for p in prefixes)
    chosen = [t for t in tasks if matches(t, select) and not matches(t, skip)]
    if not chosen:
        raise SystemExit(f"no task matches {select}; known: {sorted(tasks)}")
    commands = expander()
    units = []
    for name in chosen:
        task = tasks[name]
        if task["script"].startswith("-m pytest"):
            raise SystemExit(f"{name} is a test task; run it with pytest")
        for index, line in enumerate(commands(name, task)):
            units.append({"task": name, "index": index, "command": line, "state": "pending",
                          "needs": [n for n in task.get("needs", []) if n in chosen],
                          "assumed_done": [n for n in task.get("needs", []) if n not in chosen],
                          "done_when": task.get("done_when"), "env": task.get("env") or {},
                          "memory_gb": gigabytes(task["resources"]["mem"]), "cpus": int(task["resources"]["cpus"]),
                          "gpu": bool(task["resources"].get("gpu"))})
    return chosen, units


def write_status(folder, run_id, units, started, note=""):
    counts = {state: sum(u["state"] == state for u in units) for state in STATES}
    lines = [f"run {run_id}   {time.strftime('%Y-%m-%d %H:%M:%S')}   " + "   ".join(f"{k}: {v}" for k, v in counts.items() if v) + (f"   {note}" if note else ""), ""]
    for u in units:
        took = f"{u['seconds']:7.0f}s" if u.get("seconds") is not None else " " * 8
        device = f"   GPU {u['gpu_device']}" if "gpu_device" in u else ""
        lines.append(f"{u['state']:13s} {took}  {u['task']}[{u['index']}]{device}" + (f"   log: {u['log']}" if u["state"] in ("failed", "running") else ""))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pack_status.txt").write_text("\n".join(lines) + "\n")
    (folder / "pack_status.json").write_text(json.dumps({"run_id": run_id, "started": started, "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                         "counts": counts, "units": units}, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--select", nargs="+", required=True, help="task-id prefixes from the manifest")
    ap.add_argument("--skip", nargs="*", default=[], help="task-id prefixes to leave out")
    ap.add_argument("--parallel", type=int, default=int(os.environ.get("BM_PARALLEL", "3")), help="units running at once")
    ap.add_argument("--mem-budget", type=float, default=float(os.environ.get("BM_MEM_BUDGET_GB", "0")),
                    help="GB the running units may claim together (0: the Slurm allocation, else no limit)")
    ap.add_argument("--manifest", type=Path, default=REPO / "configs/reproduction.yaml")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    args = ap.parse_args()

    count = os.environ.get("BM_GPU_COUNT", "1")
    if not count.isascii() or not count.isdecimal() or int(count) < 1:
        ap.error("BM_GPU_COUNT must be a positive integer")
    gpu_count = int(count)
    chosen, units = plan(yaml.safe_load(args.manifest.read_text()), args.select, args.skip)
    if args.dry_run:
        for u in units:
            print(f"{u['task']}[{u['index']}]  needs {u['needs'] or '-'}" + (f"  (assumes done: {u['assumed_done']})" if u["assumed_done"] else "")
                  + ("  GPU" if u["gpu"] else "") + f"\n    python {u['command']}")
        print(f"{len(chosen)} tasks, {len(units)} units, up to {args.parallel} at once; configured GPUs: {gpu_count}")
        return 0

    for name in ("BM_SCRATCH", "BM_REPO", "HF_HOME"):
        if not os.environ.get(name):
            raise SystemExit(f"{name} is not set: source torch/cluster.sh or burst/cluster.sh first")
    env = dict(os.environ, BM_RUN=f"{os.environ['BM_SCRATCH']}/reproduction/{args.run_id}",
               BM_OUT=f"{os.environ['BM_REPO']}/runs/{args.run_id}", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    cores = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    budget = args.mem_budget or (float(os.environ["SLURM_MEM_PER_NODE"]) / 1024 if os.environ.get("SLURM_MEM_PER_NODE") else float("inf"))
    logs, out = Path(env["BM_RUN"]) / "logs", Path(env["BM_OUT"])
    logs.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    workdir = Path(os.environ.get("BM_CODE") or os.environ["BM_REPO"])       # a code snapshot when the submitter made one

    def say(text):
        print(f"{time.strftime('%H:%M:%S')} {text}", flush=True)

    for u in units:
        u["log"] = str(logs / f"{u['task']}_{u['index']}.log")
        if u["done_when"] and Path(os.path.expandvars(u["done_when"].replace("${BM_RUN}", env["BM_RUN"]).replace("${BM_OUT}", env["BM_OUT"]))).exists():
            u["state"] = "already done"
    devices = []
    if gpu_count > 1 and any(u["gpu"] and u["state"] == "pending" for u in units):
        devices = allocated_devices(gpu_count)
        say("CUDA worker devices: " + ", ".join(devices))
    # Prefer ready GPU work before filling the remaining slots with CPU work.
    dispatch_order = sorted(units, key=lambda u: not u["gpu"]) if devices else units
    say(f"{len(chosen)} tasks, {len(units)} units, {args.parallel} at once, {cores} cores, memory budget {budget:.0f} GB, code {workdir}")

    running = {}                                    # process -> unit
    written = None                                  # the states last written to the status files
    stop = {"signal": None}
    def on_signal(number, frame):
        stop["signal"] = number
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def finished(task):
        return all(u["state"] in ("done", "already done") for u in units if u["task"] == task)

    def broken(task):
        return any(u["state"] in ("failed", "blocked") for u in units if u["task"] == task)

    while True:
        for process, u in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            del running[process]
            u["seconds"] = time.time() - u.pop("began")
            u["state"] = "done" if code == 0 else "failed"
            say(f"{u['state']:6s} {u['task']}[{u['index']}] after {u['seconds']:.0f}s" + ("" if code == 0 else f" (exit {code}); see {u['log']}"))
        for u in units:
            if u["state"] == "pending" and any(broken(n) for n in u["needs"]):
                u["state"] = "blocked"
                say(f"blocked {u['task']}[{u['index']}]: a task it needs failed")
        if stop["signal"]:
            break
        busy = {u.get("gpu_device") for u in running.values()}
        free = [device for device in devices if device not in busy]
        for u in dispatch_order:
            if len(running) >= args.parallel:
                break
            if u["state"] != "pending" or not all(finished(n) for n in u["needs"]):
                continue
            claimed = sum(r["memory_gb"] for r in running.values())
            if running and claimed + u["memory_gb"] > budget:
                continue
            threads = str(max(1, min(u["cpus"], cores)))
            worker_env = dict(env, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
            task_env = u["env"]
            if devices and u["gpu"]:
                if not free:
                    continue
                u["gpu_device"] = free.pop(0)
                worker_env["CUDA_VISIBLE_DEVICES"] = u["gpu_device"]
                task_env = {k: v for k, v in task_env.items() if k != "CUDA_VISIBLE_DEVICES"}
            exports = "".join(f'export {k}="{v}"\n' for k, v in task_env.items())
            script = f'set -euo pipefail\n{exports}cd "{workdir}"\nexec python {u["command"]}\n'
            with open(u["log"], "a") as log:
                log.write(f"== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {u['task']}[{u['index']}]: {u['command']}\n")
                log.flush()
                process = subprocess.Popen(["bash", "-c", script], env=worker_env,
                                           stdout=log, stderr=subprocess.STDOUT)
            u["state"], u["began"] = "running", time.time()
            running[process] = u
            device = f" on GPU {u['gpu_device']}" if "gpu_device" in u else ""
            say(f"start  {u['task']}[{u['index']}]{device}")
        if written != [u["state"] for u in units]:
            write_status(out, args.run_id, units, started)
            written = [u["state"] for u in units]
        if not running and not any(u["state"] == "pending" and all(finished(n) for n in u["needs"]) for u in units):
            break
        time.sleep(float(os.environ.get("BM_POLL_SECONDS", "5")))

    if stop["signal"]:
        say(f"signal {stop['signal']}: stopping {len(running)} running units; run the same command again to continue")
        for process, u in running.items():
            process.terminate()
        for process, u in running.items():
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
            u["state"], u["seconds"] = "interrupted", time.time() - u.pop("began")
    write_status(out, args.run_id, units, started, note="stopped by a signal" if stop["signal"] else "")
    left = [u for u in units if u["state"] not in ("done", "already done")]
    say(f"finished: {len(units) - len(left)} of {len(units)} units done" + (f"; not done: {sorted({u['task'] for u in left})}" if left else ""))
    say(f"status: {out / 'pack_status.txt'}")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
