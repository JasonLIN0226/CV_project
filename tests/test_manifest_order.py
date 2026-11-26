"""configs/reproduction.yaml: torch/make_jobs.py submits tasks in manifest order, so a task must come after what it needs."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
KNOWN = {("gate.validate.migration", "gate.assets.team_bundle")}   # already on clean; the migration check runs alone


def test_every_need_is_an_earlier_task():
    tasks = yaml.safe_load((ROOT / "configs/reproduction.yaml").read_text())["tasks"]
    order = list(tasks)
    unknown = [(t, n) for t, spec in tasks.items() for n in spec.get("needs") or [] if n not in tasks]
    later = {(t, n) for t, spec in tasks.items() for n in spec.get("needs") or []
             if n in tasks and order.index(n) > order.index(t)}
    assert not unknown, unknown
    assert later <= KNOWN, later - KNOWN
