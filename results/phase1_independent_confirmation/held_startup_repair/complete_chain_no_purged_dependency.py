"""Cloud Slurm rejects release/update; replace only terminal or cancelled pending jobs."""
import json,os,pathlib,subprocess,sys,datetime
R=pathlib.Path('/scratch/xl4358/phase1_independent_confirmation_execution_v10_checkpoint_resume')
O=R/'execution_repair/held_startup_repair'
os.environ['SLURM_JOB_ID']='9266'
sys.path.insert(0,str(R/'execution_repair/final_continuation'))
import runner
runner.self_live();runner.w.source_check()
def native(j):
    raw=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(j)],text=True)
    return dict(s.split('=',1) for s in raw.split() if '=' in s)
assert not (O/'no_purged_dependency_attempt.json').exists()
assert not (O/'encode_test_receipt.json').exists()
assert runner.w.terminal(9254)['state']=='COMPLETED'
assert json.loads((O/'actual_receipt.json').read_text())['job']==9268
terminal={j:native(j) for j in (9256,9257,9258,9268)}
assert all(n['JobState']=='CANCELLED' for n in terminal.values())
D=pathlib.Path(runner.w.config()['output'])
budget=runner.w.SequenceGuard(D).check()
assert budget['remaining_cap_seconds']>=7320
assert not any(n['state'] in ('RUNNING','CONFIGURING') for n in budget['native'])
manifest_path=R/'execution_repair/final_continuation/submitted.json'
manifest=json.loads(manifest_path.read_text())
(O/'manifest_before_no_purged_dependency.json').write_text(json.dumps(manifest,indent=2))
(O/'no_purged_dependency_attempt.json').write_text(json.dumps(dict(unknown_do_not_retry=True,
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
env={k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_','SLURM_'))}
env['SLURM_EXPORT_ENV']='ALL'
receipts=[];parent=9254
for stage in ('encode_test','evaluate_cpu','audit_cpu','evidence_cpu'):
    assert not (O/(stage+'_nopurged_attempt.json')).exists()
    (O/(stage+'_nopurged_attempt.json')).write_text(json.dumps(dict(parent=parent,unknown_do_not_retry=True)))
    gpu=stage=='encode_test'
    cpus=1 if stage in ('audit_cpu','evidence_cpu') else 4
    mem='60G' if gpu else '1500M' if stage=='evidence_cpu' else '16G'
    minutes=120 if gpu else 180 if stage=='evaluate_cpu' else 60 if stage=='audit_cpu' else 15
    args=['/opt/slurm/bin/sbatch','--parsable','--account=csci_ga_2271-2026fa',
          '--partition='+('c12m85-a100-1' if gpu else 'n2c48m24'),
          '--cpus-per-task='+str(cpus),'--mem='+mem,'--time='+str(minutes),
          '--no-requeue','--export=ALL',
          '--job-name=p1-env-'+stage,'--chdir='+str(R),
          '--output='+str(R/(stage+'-env-final-%j.out')),'--error='+str(R/(stage+'-env-final-%j.err'))]
    if gpu:args+=['--gres=gpu:1']
    else:args+=['--dependency=afterok:'+str(parent)]
    args+=['--wrap=/scratch/xl4358/phase1_env/bin/python -u '+str(R/'execution_repair/final_continuation/runner.py')+' '+stage]
    raw=subprocess.check_output(args,text=True,env=env,timeout=30).strip();job=int(raw.split(';')[0])
    receipt=dict(stage=stage,job=job,parent=parent,final_continuation=True,environment_repair=True,
        source_lock_sha256=runner.w.file_hash(R/'source_lock.json'),actual_submission_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (O/(stage+'_receipt.json')).write_text(json.dumps(receipt,indent=2))
    p=R/'receipts'/f'{stage}_submission.json'
    (O/(stage+'_prior_receipt.json')).write_bytes(p.read_bytes())
    p.write_text(json.dumps(receipt,indent=2))
    manifest['receipts']=[receipt if row['stage']==stage else row for row in manifest['receipts']]
    manifest_path.write_text(json.dumps(manifest,indent=2))
    if gpu:
        g=runner.w.SequenceGuard(D);g.state['GPU_job_ids'].append(job);g.save()
    receipts.append(receipt);parent=job
runner.w.source_check()
result=dict(receipts=receipts,budget_before=budget,
            reason='Cloud scontrol rejected dependency update and release; all replaced pending jobs native CANCELLED with zero allocation',
            no_training_or_train_encoding_replay=True)
(O/'no_purged_dependency_complete.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'receipts':receipts,'remaining_cap_upper_bound':budget['remaining_cap_seconds']}))
