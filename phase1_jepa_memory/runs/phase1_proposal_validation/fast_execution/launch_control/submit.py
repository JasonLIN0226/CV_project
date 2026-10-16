import pathlib,json,datetime,subprocess,hashlib,os
r=pathlib.Path('/scratch/xl4358/phase1_proposal_validation/fast_execution')
pre=json.loads((r/'final_preflight.json').read_text());assert pre['contract_exit']==pre['software_exit']==pre['all_inputs_exit']==0
fast=json.loads((r/'native_fast_input_audit.json').read_text());assert fast['all_exact'] and all(x['exact_samples_targets_info_rng']==4864 and x['original_validation_batches_exact']==12 for x in fast['tasks'].values())
auth=json.loads((r/'budget_authorization.json').read_text());assert auth['approved'] is True and auth['cumulative_GPU_wall_cap_seconds']==18000 and auth['already_used_GPU_seconds']==226
assert fast['source_lock_sha256']==hashlib.sha256((r/'control/source_lock.json').read_bytes()).hexdigest()
state=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-u','xl4358','-o','%i|%j|%P|%T|%a'],text=True);assert all(line.split('|')[0] in {'7482',os.environ.get('SLURM_JOB_ID')} for line in state.splitlines()),state
assert not (r/'training_submission_attempt.json').exists(),'Unknown or prior attempt; refuse retry'
(r/'training_submission_attempt.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'ATTEMPT_STARTED_UNKNOWN','approved_cumulative_GPU_seconds':18000,'prior_GPU_seconds':226,'new_wall_seconds':17760},indent=2)+'\n')
def submit(name,args,fields):
 if name!='training':
  assert not (r/(name+'_submission_attempt.json')).exists()
  (r/(name+'_submission_attempt.json')).write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'ATTEMPT_STARTED_UNKNOWN','fields':fields},indent=2)+'\n')
 s=subprocess.run(['/opt/slurm/bin/sbatch','--parsable']+args,capture_output=True,text=True)
 assert s.returncode==0,(s.returncode,s.stdout,s.stderr)
 job=s.stdout.strip().split(';')[0];assert job.isdigit(),s.stdout
 receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':int(job),'stdout':s.stdout,'stderr':s.stderr,**fields}
 (r/(name+'_submission.json')).write_text(json.dumps(receipt,indent=2)+'\n');return job,receipt
job,train=submit('training',[str(r/'control/train.sbatch')],{'account':'csci_ga_2271-2026fa','source_root':str(r/'control'),'source_lock_sha256':fast['source_lock_sha256'],'GPU':1,'threads':4,'memory_gb':60,'wall_seconds':17760,'prior_GPU_seconds':226,'cumulative_GPU_cap_seconds':18000,'soft_seconds':17174,'actual_all_inputs_bitwise_proof':True})
ajob,analysis=submit('analysis',['--dependency=afterany:'+job,'--export=ALL,TRAINING_JOB='+job,str(r/'control/evaluate.sbatch')],{'parent':int(job),'GPU':0,'threads':4,'memory_gb':16,'wall_seconds':3600,'no_training':True,'requires_all_six_complete_before_heldout':True})
ijob,independent=submit('independent',['--dependency=afterany:'+ajob,str(r/'independent_audit/audit.sbatch')],{'parent_evaluation':int(ajob),'parent_training':int(job),'GPU':0,'threads':1,'memory_mib':1500,'wall_seconds':900,'no_model_forward':True,'source_root':str(r/'independent_audit'),'source_lock_sha256':hashlib.sha256((r/'independent_audit/source_lock.json').read_bytes()).hexdigest()})
print(json.dumps({'training':train,'analysis':analysis,'independent':independent}))
