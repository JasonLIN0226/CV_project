"""Submit final frozen stages once; archive only the verified failed lock attempt."""
import os,sys,pathlib,json,subprocess,datetime,hashlib
R=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v10_checkpoint_resume')
sys.path[:0]=[str(R),str(R/'original_src'),str(R/'execution_repair'),str(R/'execution_repair/final_continuation')]
import runner
w=runner.w; w.source_check(); runner.self_live()
D=pathlib.Path(w.config()['output'])
assert w.terminal(9242)['exitcode']=='0:0'
proof=json.loads((R/'stage_completed/train_gpu.json').read_text());assert int(proof['job'])==9242 and proof['execution_completed']
state=json.loads((D/'sequence_budget.json').read_text())
assert len(state['epochs'])==12 and all(e==120 for e in state['epochs'].values()) and state['maps_locked']
# Failed lock stopped before test generation; inspect its actual error before archiving.
f=R/'stage_failures/lock_cpu.json'; failure=json.loads(f.read_text())
assert str(failure['job'])=='9245' and 'Current frozen CPU guardian heartbeat' in failure['error']
rows=w.native_allocations([9245]);assert rows[0]['state']=='FAILED'
report=w.SequenceGuard(D).check();assert report['remaining_cap_seconds']>=7320
out=R/'execution_repair/final_continuation';marker=out/'submitted.json';assert not marker.exists()
archive=out/'failed9245';archive.mkdir(exist_ok=False)
for name in ['stage_failures/lock_cpu.json','stage_attempts/lock_cpu.json']:
    p=R/name
    if p.exists():p.rename(archive/p.name)
for stage in runner.STAGES[1:]:
    for name in [f'receipts/{stage}_submission.json',f'receipts/{stage}_submission_attempt.json']:
        p=R/name
        if p.exists():
            r=json.loads(p.read_text());job=r.get('job')
            if job:
                n=w.native_allocations([int(job)])[0]
                assert n['state'] in ['FAILED','CANCELLED','COMPLETED','TIMEOUT'],(stage,job,n['state'])
            p.rename(archive/(stage+'_'+p.name))
receipts=[];parent=9242
for stage in runner.STAGES[1:]:
    marker_attempt=R/'receipts'/f'{stage}_submission_attempt.json'
    with marker_attempt.open('x') as fh:json.dump(dict(stage=stage,parent=parent,unknown_do_not_retry=True,final_continuation=True),fh)
    gpu=stage=='encode_test';seconds=7200 if gpu else 10800 if stage=='evaluate_cpu' else 900 if stage=='evidence_cpu' else 3600
    args=['/opt/slurm/bin/sbatch','--parsable','--account=csci_ga_2271-2026fa','--partition='+('c12m85-a100-1' if gpu else 'n2c48m24'),'--cpus-per-task='+('1' if stage in ['audit_cpu','evidence_cpu'] else '4'),'--mem='+('60G' if gpu else '1500M' if stage=='evidence_cpu' else '16G'),'--time='+str(seconds//60),'--no-requeue','--dependency=afterok:'+str(parent),'--job-name=p1-final-'+stage,'--output='+str(R/(stage+'-final-%j.out')),'--error='+str(R/(stage+'-final-%j.err')),'--export=NIL']
    if gpu:args+=['--gres=gpu:1']
    args+=['--wrap=/scratch/xl4358/phase1_env/bin/python -u '+str(out/'runner.py')+' '+stage]
    raw=subprocess.check_output(args,text=True,timeout=30).strip();job=int(raw.split(';')[0])
    receipt=dict(stage=stage,job=job,parent=parent,source_lock_sha256=w.file_hash(R/'source_lock.json'),final_continuation=True,actual_submission_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (R/'receipts'/f'{stage}_submission.json').write_text(json.dumps(receipt,indent=2))
    if gpu:
        g=w.SequenceGuard(D);g.state['GPU_job_ids'].append(job);g.save()
    receipts.append(receipt);parent=job
marker.write_text(json.dumps(dict(receipts=receipts,source_lock_sha256=w.file_hash(R/'source_lock.json'),budget_before={k:report[k] for k in ['actual_GPU_seconds','remaining_cap_seconds']},runner_sha256=w.file_hash(out/'runner.py')),indent=2))
print(marker.read_text())
