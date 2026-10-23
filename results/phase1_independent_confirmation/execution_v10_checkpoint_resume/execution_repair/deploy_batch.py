"""One reviewed execution recovery. No science or cache mutations."""
import hashlib,json,os,pathlib,subprocess,sys,datetime
R=pathlib.Path(__file__).resolve().parent.parent
sys.path[:0]=[str(R),str(R/'original_src'),str(R/'execution_repair')]
from watchdog import native_job,write
from budget_guard import native_allocations

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    me=native_job(int(os.environ['SLURM_JOB_ID']))
    assert me['JobId']=='9151' and me['JobState']=='RUNNING' and 'gres/gpu=' not in me['AllocTRES']
    import orchestrator as w
    w.source_check(); audit=json.loads((R/'execution_repair/audit_9120.json').read_text())
    assert audit['cumulative_GPU_seconds']==45114 and audit['tests_unopened'] and audit['source_exact']
    active=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-u','xl4358','-o','%i'],text=True).split()
    assert active==['9151'],active
    assert native_allocations([9120])[0]['state']=='CANCELLED'
    assert not (R/'execution_repair/batch_deploy_attempt.json').exists()
    write(R/'execution_repair/batch_deploy_attempt.json',dict(unknown_do_not_retry=True,audit_sha256=sha(R/'execution_repair/audit_9120.json')))
    archive=R/'execution_repair/retained_after9120';archive.mkdir(exist_ok=False)
    names=['watchdog_submission.json','watchdog_cleanup_submission.json','watchdog_status.json','guardian_peer_status.json',
           'watchdog_terminal.json','guardian_peer_terminal.json','cpu_incarnation_primary.json','cpu_incarnation_peer.json',
           'guardian_native_preflight.json','watchdog_done.json','watchdog_cleanup_attempts.json','peer_cleanup_attempts.json',
           'stage_attempts/train_gpu.json','stage_failures/train_gpu.json']
    names += [str(p.relative_to(R)) for p in (R/'receipts').glob('*.json') if not 'attempt' in p.name]
    for name in names:
        p=R/name
        if p.exists():
            target=archive/name;target.parent.mkdir(parents=True,exist_ok=True);p.rename(target)
    for p in (R/'receipts').glob('*_submission_attempt.json'):
        target=archive/'receipts'/p.name;target.parent.mkdir(parents=True,exist_ok=True);p.rename(target)
    source=sha(R/'source_lock.json')
    write(R/'watchdog_request.json',dict(source_lock_sha256=source,GPU_cap_seconds=69299,prior_GPU_charge_seconds=45114,
          startup_limit_seconds=300,stage_allocation_limits=w.GPU_LIMITS,
          GPU_job_names={s:'p1-'+s+'-'+source[:8] for s in w.GPU_LIMITS}))
    files=['batch_runtime.py','batch_primary.py','batch_stage.sh','deploy_batch.py','fresh_training_resume.py','audit_9120.json']
    write(R/'execution_repair/batch_manifest.json',dict(files={str(pathlib.Path('execution_repair')/n):sha(R/'execution_repair'/n) for n in files},
          scientific_source_lock_sha256=source,remaining_GPU_seconds=24185,original_thresholds_unchanged=True))
    pair=[]
    for role in ['primary','peer']:
        args=['/opt/slurm/bin/sbatch','--parsable','--hold','--account=csci_ga_2271-2026fa','--partition=interactive',
              '--cpus-per-task=1','--mem=1500M','--time=12:00:00','--no-requeue','--export=NIL',
              '--job-name=p1-standalone-'+role,f'--output={R}/standalone-{role}-%j.out',f'--error={R}/standalone-{role}-%j.err']
        command=f'/scratch/xl4358/phase1_env/bin/python -u {R}/execution_repair/batch_primary.py' if role=='primary' else f'/bin/bash {R}/cleanup.sh'
        with (R/'execution_repair'/f'{role}_submission_attempt.json').open('x') as f:json.dump(dict(unknown_do_not_retry=True),f)
        raw=subprocess.check_output(args+['--wrap='+command],text=True,timeout=30).strip();job=int(raw.split(';')[0]);pair.append(job)
        write(R/('watchdog_submission.json' if role=='primary' else 'watchdog_cleanup_submission.json'),dict(job=job,source_lock_sha256=source,CPU_only=True,standalone_batch=True))
    for job in pair:subprocess.run(['/opt/slurm/bin/scontrol','release',str(job)],check=True,timeout=20)
    write(R/'execution_repair/batch_pair_submitted.json',dict(primary=pair[0],peer=pair[1],no_OOD_lifecycle_dependency=True))
    print(json.dumps(dict(primary=pair[0],peer=pair[1])))

if __name__=='__main__':main()
