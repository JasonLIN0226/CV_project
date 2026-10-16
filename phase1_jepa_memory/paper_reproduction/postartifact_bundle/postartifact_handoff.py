"""Bind complete basis/banks and indexed labels before one benchmark GPU job."""
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess
import torch

from completed_predictor_proof import completed_proof,sha
from completed_artifact_proof import completed_artifact_proof


def write(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();jid=os.environ.get('SLURM_JOB_ID','')
    if not jid.isdigit():raise RuntimeError('actual course metadata allocation required')
    raw=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',jid],text=True)
    fields=dict(x.split('=',1) for x in raw.split() if '=' in x)
    if fields.get('Account')!='csci_ga_2271-2026fa' or fields.get('JobState')!='RUNNING' or 'gres/gpu' in fields.get('AllocTRES',''):
        raise RuntimeError('course zero-GPU completion-proof allocation required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    benchmark=Path('/scratch/xl4358/phase1_space_paper_benchmark')
    for root,lock_name in ((args.root,'source_lock.json'),(benchmark,'software_lock.json')):
        lock=json.loads((root/lock_name).read_text())
        for name,value in lock['files'].items():
            f=root/name
            if f.is_symlink() or sha(f)!=value:raise RuntimeError('prepared source changed')
    template=json.loads((benchmark/'benchmark_binding_template.json').read_text())
    if template['actual_binding_complete'] is not False:raise RuntimeError('unknown formal binding already exists')
    artifact_root=Path('/scratch/xl4358/phase1_space_paper_artifacts')
    sub=json.loads((artifact_root/'artifact_submission.json').read_text());parent=int(sub['job_id'])
    state=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j',str(parent),'-n','-P','--format=JobIDRaw,State,ExitCode,Account'],text=True)
    if not any(r.split('|')[:4]==[str(parent),'COMPLETED','0:0','csci_ga_2271-2026fa'] for r in state.splitlines()):
        raise RuntimeError('actual artifact parent must complete successfully')
    output=Path(sub['output']);report_path=output/'artifact_report.json';report=json.loads(report_path.read_text())
    if report['job_id']!=parent or report['status']!='PAPER_CONSISTENT_BASIS_AND_BANK_COMPLETE' or report['elapsed_seconds']>13800 or report['test_data_used']:
        raise RuntimeError('actual full bounded artifact report required')
    if report['counts']['bank_states']!=11136 or report['resources']['gpus']!=1 or 'A100' not in report['resources']['gpu']:
        raise RuntimeError('actual fixed counts/resources required')
    for row in report['outputs']:
        f=Path(row['path'])
        if f.is_symlink() or f.stat().st_size!=row['bytes'] or sha(f)!=row['sha256']:raise RuntimeError('current basis/bank changed')
    artifact_lock=json.loads((artifact_root/'source_lock.json').read_text())
    for name,value in artifact_lock['files'].items():
        if sha(artifact_root/name)!=value:raise RuntimeError('actual artifact source changed')
    if report['source_lock_sha256']!=sha(artifact_root/'source_lock.json'):raise RuntimeError('artifact executed source binding differs')
    proof=completed_proof(template['training_submission'],template['training_root'])
    artifact_protocol=json.loads((artifact_root/'artifact_protocol.json').read_text())
    bound=Path(artifact_protocol['completed_predictor_proof'])
    if proof!=json.loads(bound.read_text()) or sha(bound)!=report['completed_predictor_proof_sha256']:
        raise RuntimeError('current six predictor models differ from executed artifact proof')
    numeric=completed_artifact_proof(report,proof,output)
    numeric_path=args.root/'completed_artifact_numeric_proof.json'
    if numeric_path.exists():raise RuntimeError('numerical proof already written; no repeat handoff')
    write(numeric_path,numeric)
    for key in ('label_report','label_independent_audit'):
        f=Path(template[key])
        if sha(f)!=template[key+'_sha256']:raise RuntimeError('frozen label proof changed')
    labels=json.loads(Path(template['label_report']).read_text())
    if labels['validation_missing_labels']!=0 or labels['sealed_numeric_labels_interpreted']:raise RuntimeError('complete validation label coverage required')
    for row in labels['label_artifacts']:
        f=Path(row['path'])
        if f.is_symlink() or f.stat().st_size!=row['bytes'] or sha(f)!=row['sha256']:raise RuntimeError('current label file changed')
    pp=benchmark/'benchmark_protocol.json';marker=benchmark/'submission_attempt.json'
    if pp.exists() or marker.exists() or (benchmark/'source_lock.json').exists():raise RuntimeError('do not repeat benchmark binding or submission')
    protocol=dict(template,actual_binding_complete=True,artifact_job_id=parent,artifact_report=str(report_path),
        artifact_root=str(artifact_root),artifact_source_lock_sha256=sha(artifact_root/'source_lock.json'),
        artifact_protocol_sha256=sha(artifact_root/'artifact_protocol.json'),
        artifact_report_sha256=sha(report_path),completed_predictor_proof=str(bound),completed_predictor_proof_sha256=sha(bound),
        completed_artifact_numeric_proof=str(numeric_path),completed_artifact_numeric_proof_sha256=sha(numeric_path),
        all_six_predictors=proof['models'],actual_artifact_outputs=report['outputs'],source_bound_before_any_decoder_fit_or_validation_inference=True)
    write(pp,protocol);software=json.loads((benchmark/'software_lock.json').read_text())
    write(benchmark/'source_lock.json',{'files':dict(software['files'],**{'benchmark_protocol.json':sha(pp)}),'protocol_sha256':sha(pp)})
    active=subprocess.check_output(['/opt/slurm/bin/squeue','-u','xl4358','-h','-o','%i|%j'],text=True).splitlines()
    if any(r.split('|')[0]!=jid for r in active):raise RuntimeError('unexpected concurrent allocation before benchmark')
    write(marker,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_artifact_job_id':parent,
        'protocol_sha256':sha(pp),'source_lock_sha256':sha(benchmark/'source_lock.json')})
    result=subprocess.run(['/opt/slurm/bin/sbatch','--parsable',str(benchmark/'scripts/submit_paper_benchmark.sh')],capture_output=True,text=True)
    submission={'returncode':result.returncode,'stdout':result.stdout.strip(),'stderr':result.stderr.strip(),
        'handoff_job_id':int(jid),'artifact_job_id':parent,'protocol_sha256':sha(pp),'source_lock_sha256':sha(benchmark/'source_lock.json')}
    if result.returncode==0:
        text=result.stdout.strip().split(';')[0]
        if not text.isdigit():raise RuntimeError('unknown submission outcome; do not repeat')
        submission['job_id']=int(text);submission['output']=f'/scratch/xl4358/phase1_space_paper_benchmark_results/job_{text}'
    write(benchmark/'benchmark_submission.json',submission);print(json.dumps(submission),flush=True)
    if result.returncode:raise RuntimeError('benchmark submission rejected; read actual reason')


if __name__=='__main__':main()
