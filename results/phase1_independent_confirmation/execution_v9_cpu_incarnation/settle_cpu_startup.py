"""Read-only native settlement of the V8 CPU readiness stop; no model calls."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys

R = Path('/scratch/xl4358/phase1_independent_confirmation_execution_v8_efficiency')
sys.path.insert(0, str(R))
from budget_guard import native_allocations
from watchdog import native_job

me = native_job(int(os.environ.get('SLURM_JOB_ID', '8945')))
assert me['JobState'] == 'RUNNING' and me['Account'] == 'csci_ga_2271-2026fa'
assert 'gres/gpu=' not in me['AllocTRES']
rows = native_allocations([8898, 8908, 8927, 8928])
assert all(x['state'] in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT') for x in rows)
assert all('gres/gpu' not in x['tres'] for x in rows)
lock = json.loads((R/'source_lock.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(R/'source_lock.json') == '5150b1e4645a4dcebcb9d1c463b2842e73b7eeb1b670491278cb3bce0b0e8ec2'
for name, h in lock['files'].items():
    assert sha(R/name) == h, name
receipts = [str(p.relative_to(R)) for p in (R/'receipts').glob('*submission*.json')
            if not p.name.startswith('resume_cpu_')]
assert receipts == []
state = json.loads((R/'fresh/sequence_budget.json').read_text())
assert state['GPU_job_ids'] == [] and state['epochs'] == {} and state['encoded_frames'] == 812
failure = json.loads((R/'stage_failures/resume_cpu.json').read_text())
assert 'Independent peer not ready; no GPU submitted' in json.dumps(failure)
assert not (R/'fresh/all_models_lock.json').exists()
proof = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             actual_CPU_receiver=me, native_jobs=rows, GPU_jobs=[], GPU_receipts=receipts,
             all106source_SHA_verified=True, prior_GPU_seconds=40499,
             cumulative_cap_seconds=69299, remaining_GPU_seconds=28800,
             reused_frames=812, new_frames=0, models=0, maps=0,
             fresh_test_unopened=True, CPU_timeout_failure=failure,
             science_or_budget_failure=False, no_model_forward=True,
             pre_batch_peer_requeue_observed=dict(job=8928, initial_node='b-10-48',
                 subsequent_node='b-10-146', Restarts=1, Requeue=1))
dest = R/'native_CPU_startup_stop_evidence.json'
assert not dest.exists()
dest.write_text(json.dumps(proof, indent=2)+'\n')
print(json.dumps(dict(file=str(dest), sha256=sha(dest), GPU_seconds_added=0,
                     remaining_GPU_seconds=28800, native_jobs=rows)))
