"""Bind six actually completed predictors before one downstream artifact job."""
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess
import torch
from completed_predictor_proof import completed_proof,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit():raise RuntimeError('actual course metadata allocation required')
    native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',job],text=True)
    fields=dict(x.split('=',1) for x in native.split() if '=' in x)
    if fields.get('Account')!='csci_ga_2271-2026fa' or fields.get('JobState')!='RUNNING' or 'gres/gpu' in fields.get('AllocTRES',''):
        raise RuntimeError('course zero-GPU metadata proof allocation required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    artifact=Path('/scratch/xl4358/phase1_space_paper_artifacts')
    training=Path('/scratch/xl4358/phase1_space_paper_training')
    for root,filename in ((args.root,'source_lock.json'),(artifact,'software_lock.json')):
        lock=json.loads((root/filename).read_text())
        for name,value in lock['files'].items():
            if (root/name).is_symlink() or sha(root/name)!=value:raise RuntimeError('prepared downstream source differs')
    proof=completed_proof(training/'training_submission.json',training)
    proof_path=args.root/'completed_predictor_proof.json'
    if proof_path.exists():raise RuntimeError('immutable proof already written; no duplicate attempt')
    proof_path.write_text(json.dumps(proof,sort_keys=True,indent=2)+'\n')
    pp=artifact/'artifact_protocol.json';marker=artifact/'submission_attempt.json'
    if pp.exists() or marker.exists() or (artifact/'source_lock.json').exists():raise RuntimeError('do not repeat artifact binding/submission')
    protocol={'paper':'SPACE','upstream_commit':'5a073bcc303f006834bfb6ae0010c11747ded615',
        'training_submission':str(training/'training_submission.json'),'training_root':str(training),
        'completed_predictor_proof':str(proof_path),'completed_predictor_proof_sha256':sha(proof_path),
        'all_six_models':proof['models'],'training_report_sha256':proof['training_report_sha256'],
        'official_data_report_sha256':proof['data_report_sha256'],'manifest_sha256':proof['manifest_sha256'],
        'seeds':[0,1,2],'slow_basis':{'rank':32,'short_lag':1,'long_lag':32,'regularization':.001,'participant_weight':'equal'},
        'bank':{'states_per_training_participant':64,'participants':29,'policies':['fifo','reservoir'],
                'deletions':17,'rollout_offsets':list(range(32)),'selection_seed':20260813,
                'short_rollout_weight':.5,'common_continuation':'true counter Reservoir; streams100/101'},
        'predeclared_corrections':['invalid zero-padding excluded from online bank features','uniform continuation replaced with paper-specified Reservoir'],
        'resources':{'gpus':1,'type':'A100','science_threads':4,'memory_gb':60,'allocation_seconds':14400,'soft_budget_seconds':13800},
        'test_data_used':False,'strict_exact_released_code_reproduction_claim':False,'source_changes_after_observing_this_stage_results':False}
    pp.write_text(json.dumps(protocol,sort_keys=True,indent=2)+'\n')
    software=json.loads((artifact/'software_lock.json').read_text())
    lock={'files':dict(software['files'],**{'artifact_protocol.json':sha(pp)}),'protocol_sha256':sha(pp)}
    (artifact/'source_lock.json').write_text(json.dumps(lock,sort_keys=True,indent=2)+'\n')
    active=subprocess.check_output(['/opt/slurm/bin/squeue','-u','xl4358','-h','-o','%i|%j'],text=True).splitlines()
    if any(row.split('|')[0]!=job for row in active):raise RuntimeError('unexpected active allocation at artifact handoff')
    marker.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       'protocol_sha256':sha(pp),'source_lock_sha256':sha(artifact/'source_lock.json')},indent=2)+'\n')
    result=subprocess.run(['/opt/slurm/bin/sbatch','--parsable',str(artifact/'scripts/submit_space_artifacts.sh')],capture_output=True,text=True)
    submission={'returncode':result.returncode,'stdout':result.stdout.strip(),'stderr':result.stderr.strip(),
                'handoff_job_id':int(job),'training_job_id':proof['training_job_id'],'protocol_sha256':sha(pp),
                'source_lock_sha256':sha(artifact/'source_lock.json'),'completed_model_proof_sha256':sha(proof_path)}
    if result.returncode==0:
        jid=result.stdout.strip().split(';')[0]
        if not jid.isdigit():raise RuntimeError('unknown submission outcome; do not retry')
        submission['job_id']=int(jid);submission['output']=f'/scratch/xl4358/phase1_space_artifact_results/job_{jid}'
    (artifact/'artifact_submission.json').write_text(json.dumps(submission,sort_keys=True,indent=2)+'\n')
    print(json.dumps(submission),flush=True)
    if result.returncode:raise RuntimeError('artifact submission rejected; inspect actual reason')


if __name__=='__main__':main()
