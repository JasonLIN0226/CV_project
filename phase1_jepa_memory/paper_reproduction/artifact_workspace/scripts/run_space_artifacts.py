"""Guarded author slow-basis and paper-consistent train-only bank execution."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import torch
from completed_predictor_proof import completed_proof,sha


def save(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def load_script(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();started=time.monotonic();job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit():raise RuntimeError('actual course GPU allocation required')
    native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',job],text=True)
    fields=dict(x.split('=',1) for x in native.split() if '=' in x)
    if fields.get('Account')!='csci_ga_2271-2026fa' or fields.get('Partition')!='c12m85-a100-1' or fields.get('JobState')!='RUNNING':
        raise RuntimeError('unexpected course GPU allocation')
    if torch.__version__!='2.5.1+cu121' or not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('one A100 and isolated reference libraries required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    lock=json.loads((args.root/'source_lock.json').read_text())
    for name,value in lock['files'].items():
        if (args.root/name).is_symlink() or sha(args.root/name)!=value:raise RuntimeError('artifact source changed')
    protocol=json.loads(args.protocol.read_text());args.output.mkdir(parents=True,exist_ok=False)
    proof=completed_proof(protocol['training_submission'],protocol['training_root'])
    if proof!=json.loads(Path(protocol['completed_predictor_proof']).read_text()):raise RuntimeError('current six-model proof differs from pre-submission binding')
    save(args.output/'executed_model_proof.json',proof)
    manifest=Path(proof['manifest']);rows=json.loads(manifest.read_text())['videos'];train=[r for r in rows if r['split']=='train']
    expected_basis_states=3*sum(int(r['valid_length'])-87 for r in train)
    expected_bank_states=3*2*29*64
    progress={'basis_states':0,'bank_states':0,'basis_seconds':0.,'bank_seconds':0.}
    basis=load_script('author_basis',args.root/'scripts/fit_assembly101_slow_basis.py')
    bank=load_script('paper_bank',args.root/'scripts/build_assembly101_utility_banks.py')
    original_extract=basis.extract_record;original_rollout=bank.proposal_rollout_losses
    def guard():
        if time.monotonic()-started>13800:raise RuntimeError('fixed artifact four-hour soft budget reached')
    def extract(*a,**kw):
        guard();tick=time.monotonic();result=original_extract(*a,**kw);elapsed=time.monotonic()-tick
        progress['basis_states']+=len(result);progress['basis_seconds']+=elapsed
        if progress['basis_states']==len(result):
            projection=time.monotonic()-started+elapsed/len(result)*(expected_basis_states-len(result))
            save(args.output/'first_basis_record_budget.json',{'first_complete_record_seconds':elapsed,'states':len(result),
                 'all_three_basis_projection_seconds':projection,'budget_seconds':13800,'bank_not_yet_included':True})
            if projection>13800:raise RuntimeError('full three-basis projection exceeds fixed budget')
        save(args.output/'progress.json',dict(progress,elapsed_seconds=time.monotonic()-started))
        return result
    def rollout(*a,**kw):
        guard();tick=time.monotonic();result=original_rollout(*a,**kw);elapsed=time.monotonic()-tick
        progress['bank_states']+=1;progress['bank_seconds']+=elapsed
        if progress['bank_states']==8:
            remaining_basis=(expected_basis_states-progress['basis_states'])*progress['basis_seconds']/max(1,progress['basis_states'])
            forecast=time.monotonic()-started+remaining_basis+(expected_bank_states-8)*progress['bank_seconds']/8
            save(args.output/'first_eight_bank_budget.json',dict(progress,full_stage_projection_seconds=forecast,budget_seconds=13800,
                passed=forecast<=13800,forecast_not_completion_guarantee=True))
            if forecast>13800:raise RuntimeError('full fixed bank/basis projection exceeds artifact budget')
        if progress['bank_states']%8==0:save(args.output/'progress.json',dict(progress,elapsed_seconds=time.monotonic()-started))
        return result
    basis.extract_record=extract;bank.proposal_rollout_losses=rollout
    outputs=[]
    try:
        for seed in (0,1,2):
            seed_dir=args.output/f'seed_{seed}';seed_dir.mkdir()
            model=next(row for row in proof['models'] if row['seed']==seed and row['tag']=='best')
            sys.argv=['fit_assembly101_slow_basis.py','--protocol',str(args.root/'configs/assembly101_from_scratch.json'),
                '--manifest',str(manifest),'--predictor',model['path'],'--predictor-seed',str(seed),'--device','cuda:0',
                '--precision','bf16','--output',str(seed_dir/'slow_basis.pt'),'--summary-output',str(seed_dir/'slow_basis_summary.json')]
            basis.main();guard();torch.cuda.empty_cache()
            sys.argv=['build_assembly101_utility_banks.py','--protocol',str(args.root/'configs/assembly101_from_scratch.json'),
                '--manifest',str(manifest),'--predictor',model['path'],'--device','cuda:0','--output-dir',str(seed_dir/'banks')]
            bank.main();guard();torch.cuda.empty_cache()
            if sha(model['path'])!=model['sha256']:raise RuntimeError('frozen predictor changed during artifact fit')
            for path in seed_dir.rglob('*'):
                if path.is_file():outputs.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
        if progress['basis_states']!=expected_basis_states or progress['bank_states']!=expected_bank_states:
            raise RuntimeError('full paper artifact counts differ')
        save(args.output/'artifact_report.json',{'status':'PAPER_CONSISTENT_BASIS_AND_BANK_COMPLETE','job_id':int(job),
            'resources':{'account':fields['Account'],'allocated_tres':fields.get('AllocTRES'),'gpu':torch.cuda.get_device_name(0),'gpus':1,'science_threads':4},
            'protocol_sha256':sha(args.protocol),'source_lock_sha256':sha(args.root/'source_lock.json'),
            'completed_predictor_proof_sha256':sha(Path(protocol['completed_predictor_proof'])),'counts':progress,'outputs':outputs,
            'elapsed_seconds':time.monotonic()-started,'test_data_used':False,'exact_released_implementation_claim':False,'full_paper_evaluation_complete':False})
    except BaseException as exc:
        save(args.output/'failure_boundary.json',{'exception':type(exc).__name__,'message':str(exc),'counts':progress,
                'elapsed_seconds':time.monotonic()-started,'frozen_models_retrained':False,'no_budget_extension':True})
        raise


if __name__=='__main__':main()
