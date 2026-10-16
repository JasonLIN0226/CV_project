"""Bind actual complete official TSM data before one GPU training submission."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import datetime

COURSE='csci_ga_2271-2026fa'
DATA_JOB=7305
ROOT=Path('/scratch/xl4358/phase1_space_postdata_handoff')
TRAIN=Path('/scratch/xl4358/phase1_space_paper_training')


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit():raise RuntimeError('actual metadata batch required')
    native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',job],text=True)
    fields=dict(x.split('=',1) for x in native.split() if '=' in x)
    if fields.get('Account')!=COURSE or fields.get('JobState')!='RUNNING' or 'gres/gpu' in fields.get('AllocTRES',''):
        raise RuntimeError('unexpected metadata course/resource')
    lock=json.loads((ROOT/'source_lock.json').read_text())
    for name,value in lock['files'].items():
        if (ROOT/name).is_symlink() or digest(ROOT/name)!=value:raise RuntimeError('handoff source changed')
    software=json.loads((TRAIN/'software_lock.json').read_text())
    for name,value in software['files'].items():
        if (TRAIN/name).is_symlink() or digest(TRAIN/name)!=value:raise RuntimeError('training software changed')
    actual=subprocess.check_output(['/opt/slurm/bin/sacct','-j',str(DATA_JOB),'-X','-n','-P',
                                    '--format=JobIDRaw,State,ExitCode,Account'],text=True)
    if not any(x.split('|')[:4]==[str(DATA_JOB),'COMPLETED','0:0',COURSE] for x in actual.splitlines()):
        raise RuntimeError('official data parent not COMPLETED0:0')
    report_path=Path(f'/scratch/xl4358/phase1_space_official_data/job_{DATA_JOB}/data_report.json')
    report=json.loads(report_path.read_text())
    if report['job_id']!=DATA_JOB or report['status']!='OFFICIAL_TSM_TRAIN_VAL_PREPARED_AND_AUDITED' or report['test_payload_deserialized']:
        raise RuntimeError('complete test-isolated official audit required')
    manifest=Path(report['manifest'])
    if digest(manifest)!=report['manifest_sha256']:raise RuntimeError('manifest changed')
    for row in report['features']:
        path=manifest.parent/(row['video_id']+'.pt')
        if path.is_symlink() or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
            raise RuntimeError('current feature differs from complete audit')
    marker=TRAIN/'training_submission_attempt.json'
    if marker.exists():raise RuntimeError('do not repeat an attempted GPU submission')
    active=subprocess.check_output(['/opt/slurm/bin/squeue','-u','xl4358','-h','-o','%i|%j'],text=True).splitlines()
    if any(r.split('|')[0] not in {job,'7304'} for r in active):
        raise RuntimeError('unexpected active allocation; retain one science at a time')
    protocol={'paper':'SPACE','upstream_commit':'5a073bcc303f006834bfb6ae0010c11747ded615',
              'data_job_id':DATA_JOB,'data_report':str(report_path),'data_report_sha256':digest(report_path),
              'manifest_sha256':digest(manifest),'seeds':[0,1,2],
              'predictor':{'hidden':512,'layers':4,'heads':8,'ffn':2048,'input_dimension':2048,
                           'recent':8,'capacity':16,'horizons':[1,4,16,64],'dropout':.1},
              'optimization':{'epochs':30,'batch':128,'precision':'bf16','loss':'cosine',
                              'learning_rate':.0003,'weight_decay':.05,'warmup_fraction':.05,'schedule':'cosine'},
              'sampling_reconstruction':'all complete-target anchors; uniform historical subset; train-only .15 memory dropout; deterministic validation',
              'resource_override':'zero data-loader workers, four science threads; examples/architecture/objective unchanged',
              'author_environment_torch':'2.5.1+cu121','python_minor_deviation':'3.11 existing approved interpreter versus author reference 3.10',
              'budget_seconds':13800,'allocation_hours':4,'gpus':1,'gpu_type':'A100',
              'strict_exact_reproduction_claim':False,'full_benchmark_evaluation_complete':False,
              'old_acceptance_thresholds_changed':False,'test_confirmation_opened':False}
    pp=TRAIN/'paper_training_protocol.json'
    if pp.exists() or (TRAIN/'source_lock.json').exists():raise RuntimeError('formal binding already exists')
    pp.write_text(json.dumps(protocol,sort_keys=True,indent=2)+'\n')
    files={**software['files'],'paper_training_protocol.json':digest(pp)}
    (TRAIN/'source_lock.json').write_text(json.dumps({'files':files,'protocol_sha256':digest(pp),'data_parent':DATA_JOB},sort_keys=True,indent=2)+'\n')
    marker.write_text(json.dumps({'attempted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'protocol_sha256':digest(pp),'source_lock_sha256':digest(TRAIN/'source_lock.json')},indent=2)+'\n')
    result=subprocess.run(['/opt/slurm/bin/sbatch','--parsable',str(TRAIN/'scripts/submit_predictor.sh')],capture_output=True,text=True)
    submission={'returncode':result.returncode,'stdout':result.stdout.strip(),'stderr':result.stderr.strip(),
                'protocol_sha256':digest(pp),'source_lock_sha256':digest(TRAIN/'source_lock.json'),
                'data_report_sha256':digest(report_path),'handoff_job_id':int(job),'data_job_id':DATA_JOB}
    if result.returncode==0:
        jid=result.stdout.strip().split(';')[0]
        if not jid.isdigit():raise RuntimeError('unknown submission outcome; do not retry')
        submission['job_id']=int(jid)
        submission['output']=f'/scratch/xl4358/phase1_space_predictor_results/job_{jid}'
    (TRAIN/'training_submission.json').write_text(json.dumps(submission,sort_keys=True,indent=2)+'\n')
    print(json.dumps(submission),flush=True)
    if result.returncode:raise RuntimeError('GPU submission rejected; record actual reason')


if __name__=='__main__':main()
