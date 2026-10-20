"""Standalone Slurm primary; survives closing the browser, Jupyter, or local SSH."""
import hashlib,json,math,os,pathlib,subprocess,sys,time,traceback
R=pathlib.Path(__file__).resolve().parent.parent
sys.path[:0]=[str(R),str(R/'original_src'),str(R/'execution_repair')]
from batch_runtime import install,STAGES
from watchdog import write,native_job

def main():
    w=install(); source=hashlib.sha256((R/'source_lock.json').read_bytes()).hexdigest()
    me=native_job(int(os.environ['SLURM_JOB_ID']))
    assert me['JobName']=='p1-standalone-primary' and me['Partition']=='interactive'
    process=subprocess.Popen(['/bin/bash',str(R/'watchdog.sh')],stdin=subprocess.DEVNULL)
    deadline=time.monotonic()+1800
    while True:
        if process.poll() is not None:raise RuntimeError('Primary watchdog exited before launch')
        try:w.startup_require_live(R,source);break
        except (FileNotFoundError,ValueError):
            if time.monotonic()>deadline:raise TimeoutError('Paired CPU startup expired; no GPU')
            time.sleep(3)
    from guardian_preflight import verify
    verify(R)
    D=pathlib.Path(w.config()['output']); guard=w.SequenceGuard(D); report=guard.check()
    assert report['remaining_cap_seconds']>=10800+7200+240
    receipts=[];parent=None
    # The train wrapper waits for the durable full-chain marker before importing models.
    for stage in STAGES:
        attempt=R/'receipts'/f'{stage}_submission_attempt.json'
        with attempt.open('x') as f:json.dump(dict(stage=stage,parent=parent,unknown_do_not_retry=True,prequeued_batch_chain=True),f)
        gpu=stage in w.GPU_LIMITS
        limit=w.GPU_LIMITS[stage] if gpu else 10800 if stage=='evaluate_cpu' else 900 if stage=='evidence_cpu' else 3600
        cpus=1 if stage in ('audit_cpu','evidence_cpu') else 4
        mem='60G' if gpu else '1500M' if stage=='evidence_cpu' else '16G'
        wrap=f'/bin/bash {R}/execution_repair/batch_stage.sh {stage}'
        args=['/opt/slurm/bin/sbatch','--parsable','--account=csci_ga_2271-2026fa',
              '--partition='+('c12m85-a100-1' if gpu else 'n2c48m24'),f'--cpus-per-task={cpus}',
              '--mem='+mem,'--time='+str(math.ceil(limit/60)),'--no-requeue','--export=NIL',
              '--job-name=p1-'+stage+'-'+source[:8],f'--output={R}/{stage}-batch-%j.out',f'--error={R}/{stage}-batch-%j.err']
        if gpu:args.append('--gres=gpu:1')
        if parent is not None:args.append('--dependency=afterok:'+str(parent))
        raw=subprocess.check_output(args+['--wrap='+wrap],text=True,timeout=30).strip();job=int(raw.split(';')[0])
        rec=dict(stage=stage,job=job,parent=parent,source_lock_sha256=source,prequeued_batch_chain=True,stdout=raw)
        write(R/'receipts'/f'{stage}_submission.json',rec);receipts.append(rec)
        if gpu:
            guard=w.SequenceGuard(D);guard.state.setdefault('GPU_job_ids',[]).append(job);guard.save()
        parent=job
    write(R/'execution_repair/batch_chain_submitted.json',dict(receipts=receipts,budget_before=report,
           independent_of_OOD_and_local_SSH=True,source_unchanged=True,starting_epoch=114,reencoded_frames=0))
    write(R/'execution_repair/batch_chain_released.json',dict(jobs=[r['job'] for r in receipts]))
    print(json.dumps(dict(released=receipts)),flush=True)
    return process.wait()

if __name__=='__main__':
    try:sys.exit(main())
    except BaseException:
        write(R/'execution_repair/batch_primary_failure.json',dict(error=traceback.format_exc(),no_automatic_retry=True))
        raise
