"""Verify the published snapshot and compact fresh-confirmation decision.

This checks hashes and decision consistency, not fresh raw-row recomputation.
The original raw-row audit was performed on HPC; its arrays remain external.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / "results/publication_manifest.json").read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.is_file(), f"Missing published file: {entry['path']}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], str(path)
    result = json.loads((ROOT / "results/phase1_independent_confirmation/"
                         "local_verified_results_20261020.json").read_text())
    totals = {}
    for task in ("gate", "motion"):
        groups = [result[key][task] for key in
                  ("original_prediction_checks", "fixed_reference_checks", "deletion_checks")]
        totals[task] = [sum(g['passed'] for g in groups), sum(g['total'] for g in groups)]
    assert totals == {"gate": [155, 155], "motion": [147, 155]}
    failures = result["deletion_checks"]["motion_long_gap_failures"]
    assert len(failures) == 8
    assert all(r['task_task_gain_low'] < 0 < r['task_task_gain_high'] for r in failures)
    d = result['data']
    assert (d['fresh_models'], d['epochs_per_model'], d['steps_per_model']) == (12, 120, 7680)
    assert d['test_opened_only_after_all_models_locked']
    assert result['audit']['decision_checks_replayed'] == 310
    assert result['independent_audit'] == 'PASS_INDEPENDENT_FRESH_PROTOCOL_AUDIT'
    assert result['status'] == 'PHASE1_ORIGINAL_STANDARD_NOT_MET'
    assert result['phase1_go'] is False
    assert result['budget']['GPU_settled_seconds'] <= result['budget']['GPU_cap_seconds']
    return dict(snapshot_integrity="PASS", files_verified=len(manifest['files']),
                registered_checks=totals, Phase1_GO=False,
                local_check_scope="Hashes and compact-decision consistency; not raw HPC replay")


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
