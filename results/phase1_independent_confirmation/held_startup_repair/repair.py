"""One execution-only replacement after native cancellation, never a science retry."""
import datetime, hashlib, json, os, pathlib, subprocess, sys

R = pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v10_checkpoint_resume')
O = R / 'execution_repair/held_startup_repair'
O.mkdir(exist_ok=True)
os.environ['SLURM_JOB_ID'] = '9266'
sys.path.insert(0, str(R / 'execution_repair/final_continuation'))
import runner
runner.self_live()
runner.w.source_check()

def native(job):
    raw = subprocess.check_output(['/opt/slurm/bin/scontrol', 'show', 'job', '-o', str(job)], text=True)
    return dict(t.split('=', 1) for t in raw.split() if '=' in t)

old = native(9255)
assert old['JobState'] == 'CANCELLED' and old['Account'] == 'csci_ga_2271-2026fa'
assert old['Reason'] == 'user_env_retrieval_failed_requeued_held'
assert all(native(j)['JobState'] == 'PENDING' for j in (9256, 9257, 9258))
assert not (R / 'stage_completed/encode_test.json').exists()
assert not (O / 'submission_attempt.json').exists(), 'Unknown attempt must not be repeated'
D = pathlib.Path(runner.w.config()['output'])
assert (D / 'all_models_lock.json').exists()
lock_hash = runner.w.file_hash(D / 'all_models_lock.json')
assert lock_hash == '72b5426e96d5e909c1e2d02547c8db774a6b0ddee23b775b25f4fd5247b9ecde'
assert json.loads((D / 'confirmation_test_generation_complete.json').read_text())['passed']
archive = O / 'before'
archive.mkdir(exist_ok=False)
paths = ['execution_repair/additional_charge.py', 'execution_repair/final_continuation/submitted.json',
         'receipts/encode_test_submission.json', 'receipts/evaluate_cpu_submission.json']
for name in paths:
    target = archive / name.replace('/', '__')
    target.write_bytes((R / name).read_bytes())
# The requeued job has an allocated attempt followed by an unallocated cancellation.
# Preserve the earlier allocation charge and handle only that terminal empty attempt.
# 3853 s is a conservative submission-to-cancellation upper bound, NOT actual GPU wall.
patch = '''import budget_guard, startup_client
_original_charge = budget_guard.charged_seconds
_original_guarded = startup_client.guarded_allocation_rows
def charged(rows, now):
    normalized=[]
    for r in rows:
        q=dict(r)
        if q['job']==9255 and not q['tres'].get('gres/gpu'):
            assert q['state']=='CANCELLED' and q['start'] in ('None','Unknown','')
            assert q.get('current_attempt_elapsed',0)==0
            q['elapsed']=0
        normalized.append(q)
    total,active=_original_charge(normalized,now)
    return total+40,active
def guarded(root,rows):
    return [dict(r,guarded_elapsed=max(r['guarded_elapsed'],3853 if r['job']==9255 else 0))
            for r in _original_guarded(root,rows)]
budget_guard.charged_seconds=charged
startup_client.guarded_allocation_rows=guarded
'''
(R / 'execution_repair/additional_charge.py').write_text(patch)
compile(patch, 'additional_charge.py', 'exec')
# Install the same patch in this already imported process.
import budget_guard, additional_charge
budget_guard.charged_seconds = additional_charge._original_charge
exec(compile(patch, 'additional_charge.py', 'exec'), {})
report = runner.w.SequenceGuard(D).check()
assert report['remaining_cap_seconds'] >= 7320
assert not any(r['state'] in ('RUNNING','CONFIGURING') for r in report['native'])
(O / 'preflight.json').write_text(json.dumps(dict(old_job=old,budget=report,lock_sha256=lock_hash,
    charge_9255_upper_bound_seconds=3853,scientific_pass=False), indent=2))
attempt = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_job=9255,
               unknown_do_not_retry=True,source_lock_sha256=runner.w.file_hash(R/'source_lock.json'))
(O / 'submission_attempt.json').write_text(json.dumps(attempt, indent=2))
env = {k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_','SLURM_'))}
env.update(SLURM_EXPORT_ENV='ALL')
args = ['/opt/slurm/bin/sbatch','--parsable','--hold','--account=csci_ga_2271-2026fa',
        '--partition=c12m85-a100-1','--gres=gpu:1','--cpus-per-task=4','--mem=60G',
        '--time=120','--no-requeue','--export=ALL','--job-name=p1-final-encode-test-env',
        '--chdir='+str(R),'--output='+str(R/'encode_test-env-%j.out'),
        '--error='+str(R/'encode_test-env-%j.err'),
        '--wrap=/scratch/xl4358/phase1_env/bin/python -u '+str(R/'execution_repair/final_continuation/runner.py')+' encode_test']
raw = subprocess.check_output(args, env=env, text=True, timeout=30).strip()
job = int(raw.split(';')[0])
(O / 'actual_receipt.json').write_text(json.dumps(dict(job=job,raw=raw),indent=2))
manifest_path=R/'execution_repair/final_continuation/submitted.json'
manifest=json.loads(manifest_path.read_text())
for row in manifest['receipts']:
    if row['stage']=='encode_test':
        row.update(job=job,replaces_terminal_job=9255,environment_repair=True)
        (R/'receipts/encode_test_submission.json').write_text(json.dumps(row,indent=2))
    if row['stage']=='evaluate_cpu':
        row['parent']=job
        (R/'receipts/evaluate_cpu_submission.json').write_text(json.dumps(row,indent=2))
manifest['held_environment_repair_receipt']=str(O/'actual_receipt.json')
manifest_path.write_text(json.dumps(manifest,indent=2))
guard=runner.w.SequenceGuard(D)
guard.state['GPU_job_ids'].append(job)
guard.save()
subprocess.check_call(['/opt/slurm/bin/scontrol','update','JobId=9256','Dependency=afterok:'+str(job)])
assert str(job) in native(9256)['Dependency']
runner.w.source_check()
assert runner.w.file_hash(D/'all_models_lock.json')==lock_hash
(O/'repair_manifest.json').write_text(json.dumps(dict(new_job=job,old_job=9255,
    pending_evaluation_job=9256,unchanged_followups=[9257,9258],
    encoding_training_replayed=False,source_lock_sha256=runner.w.file_hash(R/'source_lock.json'),
    additional_charge_sha256=runner.w.file_hash(R/'execution_repair/additional_charge.py')),
    indent=2))
subprocess.check_call(['/opt/slurm/bin/scontrol','release',str(job)])
print(json.dumps(dict(replacement_job=job,native=native(job),budget_before=report['remaining_cap_seconds'])))
