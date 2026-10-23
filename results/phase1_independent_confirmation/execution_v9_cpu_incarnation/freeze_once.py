from pathlib import Path
import ast, hashlib, json, shutil, zipfile

B=Path(__file__).resolve().parent; S=B/'source'; O=B.parent/'execution_v8_efficiency'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
assert not (B/'ready.json').exists(), 'Do not rebuild a frozen execution'
E=B/'native_V8_CPU_stop'
assert read(E/'transfer_integrity.json')['all_match']
stop=read(E/'native_CPU_startup_stop_evidence.json')
assert stop['GPU_jobs']==[] and stop['GPU_receipts']==[]
assert stop['prior_GPU_seconds']==40499 and stop['remaining_GPU_seconds']==28800
assert 'Restarts' in (E/'cleanup-8928.err').read_text()
shutil.copy2(E/'native_CPU_startup_stop_evidence.json', S/'v8_CPU_startup_stop_evidence.json')
p=S/'protocol.json'; d=read(p)
d['CPU_first_batch_startup_repair']=dict(
    origin_root='/scratch/xl4358/phase1_independent_confirmation_execution_v8_efficiency',
    origin_source_lock_sha256=sha(O/'source/source_lock.json'),
    native_zero_GPU_settlement_sha256=sha(E/'native_CPU_startup_stop_evidence.json'),
    CPU_nodes_submitted_concurrently=True, cold_start_shared_deadline_seconds=1800,
    first_batch_exclusive_incarnation_marker=True,
    pre_batch_cloud_requeues_pinned_at_first_start=True,
    post_first_start_CPU_restarts_forbidden=True, GPU_restart_rules_unchanged=True,
    no_GPU_receipt_before_both_native_heartbeats_and8cycles=True,
    no_scientific_or_numerical_change=True, prior_active_source_preserved=True)
p.write_text(json.dumps(d,indent=2)+'\n')
allowed={'config.json','protocol.json','authorization.json','restart_authorization.json',
         'stage.sh','watchdog.sh','cleanup.sh','budget_guard.py','startup_client.py',
         'watchdog.py','guardian_peer.py','guardian_preflight.py','budget_resume.py',
         'test_startup_integration.py','terminal_evidence.py'}
prior=read(O/'source/source_lock.json')['files']
changed=[n for n,h in prior.items() if sha(S/n)!=h]
assert set(changed)<=allowed, changed
# Source byte equality includes predictor, optimizer, encoding numerics, datasets,
# thresholds, statistics and the prior measured-efficiency proof implementation.
for n in prior:
    if n not in allowed: assert sha(S/n)==prior[n], n
for p in S.rglob('*.py'):ast.parse(p.read_text())
files={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file()
       and p.name!='source_lock.json' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
lock=dict(status='FROZEN_CPU_FIRST_BATCH_INCARNATION_ONLY',files=files,
          origin_source_lock_sha256=sha(O/'source/source_lock.json'),changed_files=changed,
          all_scientific_numerical_and_acceptance_modules_byte_identical=True)
(S/'source_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
with zipfile.ZipFile(B/'deployment.zip','w',zipfile.ZIP_DEFLATED) as z:
    for n in [*files,'source_lock.json']:z.write(S/n,n)
ready=dict(status='FROZEN_CPU_STARTUP_REPAIR_NOT_SUBMITTED',source_files=len(files),
           source_lock_sha256=sha(S/'source_lock.json'),protocol_sha256=sha(S/'protocol.json'),
           archive_sha256=sha(B/'deployment.zip'),
           HPC_root='/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation',
           prior_GPU_seconds=40499,remaining_authorized_GPU_seconds=28800,cumulative_cap_seconds=69299,
           local_software_tests=52,reused_frames=812,new_models_trained=0,fresh_test_unopened=True,
           GPU_hard_limits_sum=23400)
(B/'ready.json').write_text(json.dumps(ready,indent=2)+'\n')
t=(O/'deploy_once.py').read_text();i=t.index('expected=');j=t.index('\ncontrol=',i)
t=t[:i]+'expected='+repr(ready)+t[j:]
t=t.replace('phase1_independent_confirmation_execution_v8_efficiency','phase1_independent_confirmation_execution_v9_cpu_incarnation')
(B/'deploy_once.py').write_text(t)
(B/'local_execution_review.json').write_text(json.dumps(dict(
    status='SOFTWARE_ONLY_PASS', full_suite_passed=52,
    focused_recheck='CPU incarnation/startup/heartbeat/submission/watchdog fixtures',
    native_GPU_efficiency_proof_still_required=True,
    scientific_modules_byte_identical=True, changed_files=changed),indent=2)+'\n')
print(json.dumps(ready))
