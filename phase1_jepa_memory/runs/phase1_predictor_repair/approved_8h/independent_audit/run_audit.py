import pathlib,json,subprocess,os,hashlib,datetime
from verify_scalar_results import audit
r=pathlib.Path('/scratch/xl4358/phase1_predictor_repair/approved_8h')
receipt=json.loads((r/'analysis_submission.json').read_text());train=json.loads((r/'training_submission.json').read_text());job=receipt['job_id'];tjob=train['job_id']
proof={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase1_go':False,'confirmation_opened':False,'parent_training_job':tjob,'parent_evaluation_job':job}
for parent in (tjob,job):
 state=subprocess.check_output(['/opt/slurm/bin/sacct','-n','-X','-j',str(parent),'--format=State,ExitCode','-P'],text=True)
 proof[str(parent)]=state
 if not state.startswith('COMPLETED|0:0'):
  proof['status']='PARENT_INCOMPLETE_NO_OVERALL_RESULT';(r/'independent_partial.json').write_text(json.dumps(proof,indent=2)+'\n');raise SystemExit(3)
lock=json.loads((r/'independent_audit/source_lock.json').read_text())
for name,expected in lock['files'].items():
 if hashlib.sha256((r/'independent_audit'/name).read_bytes()).hexdigest()!=expected:raise ValueError('Independent source changed')
science=json.loads((r/'control/source_lock.json').read_text())
for name,expected in science['files'].items():
 if hashlib.sha256((r/'control'/name).read_bytes()).hexdigest()!=expected:raise ValueError('Current scientific source changed')
result=audit(r/'evaluation'/f'job_{job}');result.update(native_parent_and_source_verification=proof,science_source_files=len(science['files']),independent_source_files=len(lock['files']))
(r/'evaluation'/f'job_{job}'/'independent_scalar_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['status'])
