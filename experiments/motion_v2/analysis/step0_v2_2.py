"""Apply registered v2.2 rules to saved results, then run Phase 1 headroom once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

PROTOCOL = Path("experiments/motion_v2/protocol_step0_v2_2.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify(protocol):
    for name, expected in {**protocol["source_sha256"], str(PROTOCOL): sha(PROTOCOL)}.items():
        if sha(name) != expected:
            raise RuntimeError(f"Hash mismatch: {name}")
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"])
        if hashlib.sha256(committed).hexdigest() != expected:
            raise RuntimeError(f"Must commit protocol/source before running: {name}")


def rejudge(protocol, out):
    ref = json.loads(Path(protocol["reference_json"]).read_text())
    actual = json.loads(Path(protocol["saved_json"]).read_text())
    checks = []
    def check(name, expected, observed, passed, **extra):
        checks.append(dict(name=name, reference=expected, observed=observed, passed=bool(passed), **extra))
    for key in ("pair_audit", "online_policies"):
        check(key, ref[key], actual.get(key), ref[key] == actual.get(key))
    for version in ("v1", "v2"):
        for subset in ("end_only", "stale_plus_end", "one_post_non_end", "two_post"):
            expected = ref["leakage_long_gap"][version][subset]
            observed = actual["leakage_long_gap"][version][subset]
            name = f"leakage_long_gap.{version}.{subset}"
            if subset in ("end_only", "stale_plus_end"):
                check(name, expected, observed, expected == observed, rule="exact")
            else:
                threshold = 0.58 if subset == "one_post_non_end" else 0.95
                side = (lambda x: x <= threshold) if subset == "one_post_non_end" else (lambda x: x >= threshold)
                same_side = side(expected) == side(observed)
                tolerance = abs(expected - observed) <= protocol["classifier_absolute_tolerance"]
                check(name, expected, observed, tolerance and same_side and (version == "v1" or side(observed)),
                      rule="absolute_difference_and_same_threshold_side", absolute_difference=abs(expected - observed),
                      threshold=threshold, same_threshold_side=same_side)
    save(out / "decision.json", {"passed": all(x["passed"] for x in checks), "checks": checks,
                                "recomputed_measurements": False, "external_recalculations_executed": False})
    return all(x["passed"] for x in checks)


def phase1(protocol, out):
    previous = Path(protocol["output_root"]) / "rejudge/status.json"
    if json.loads(previous.read_text())["status"] != "passed":
        raise RuntimeError("v2.2 rejudgment must pass first")
    command = [sys.executable, "-I", "-B", *protocol["phase1_arguments"]]
    with (out / "stdout.json").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
        run = subprocess.run(command, env=dict(os.environ, **protocol["thread_environment"]), stdout=stdout,
                             stderr=stderr, timeout=3600)
    if run.returncode:
        save(out / "comparison.json", {"passed": False, "exit_code": run.returncode, "command": command})
        return False
    reference = json.loads(Path(protocol["phase1_reference_json"]).read_text())
    observed = json.loads((out / "stdout.json").read_text())
    passed = reference == observed
    save(out / "comparison.json", {"passed": passed, "exact_json_match": passed, "command": command,
                                    "manifest_pairwise_verified": False,
                                    "interpretation": "按公开代码重建，未与原 manifest 逐对核对"})
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["rejudge", "phase1_headroom"])
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    verify(protocol)
    out = Path(protocol["output_root"]) / args.step
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    status = dict(status="running", step=args.step, protocol_sha256=sha(PROTOCOL),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  python=sys.version, platform=platform.platform(), gpu_seconds=0,
                  raw_test_data_opened_or_generated=False, measurements_rerun=args.step == "phase1_headroom")
    save(out / "status.json", status)
    try:
        passed = rejudge(protocol, out) if args.step == "rejudge" else phase1(protocol, out)
        status["status"] = "passed" if passed else "stopped_check_failed"
    except Exception as exc:
        status.update(status="stopped_runtime_failure", error=f"{type(exc).__name__}: {exc}")
    status["wall_seconds"] = time.perf_counter() - began
    save(out / "status.json", status)
    (out / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.as_posix()}\n" for p in sorted(out.iterdir()) if p.is_file()))
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
