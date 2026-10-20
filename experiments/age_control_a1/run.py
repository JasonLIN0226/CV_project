"""One zero-GPU native frozen-model inference; never fit or submit another job."""
import csv,datetime,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from analysis import aggregate,summarize
from counterbalanced_pools import gate_age_counterbalance,clock_only_static_pools
from provenance_guard import validate_parent_provenance

ROOT=Path(__file__).resolve().parent
OLD=Path('/scratch/xl4358/phase1_predictor_repair_v2/execution')
PREVIOUS=Path('/scratch/xl4358/phase1_deletion_assay')
sys.path.insert(0,str(OLD/'control/src'));sys.path.insert(0,str(PREVIOUS))
from run_assay import sample
from phase1.proposal_trial import verify_control,reference_model,verify_completed_model
from phase1.centered_transport import CenteredTransportPredictor
from phase1.inputs import FeatureStore,collate
from phase1.data import load_episode
from phase1.probes import LinearReadout
from phase1.training import paired_rows
from phase1.utils import stable_seed


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def locked_sources():
    lock=read(ROOT/'source_lock.json')
    assert all(sha(ROOT/p)==h for p,h in lock['files'].items())
    return lock

def main():
    begin=time.monotonic();protocol=read(ROOT/'protocol.json');lock=locked_sources()
    job=int(os.environ['SLURM_JOB_ID'])
    allocation=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True)
    fields=dict(t.split('=',1) for t in allocation.split() if '=' in t)
    assert fields['JobState']=='RUNNING' and fields['Account']=='csci_ga_2271-2026fa'
    assert fields['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in fields['AllocTRES']
    assert fields['Partition']=='n2c48m24' and os.environ.get('CUDA_VISIBLE_DEVICES','') in ('','-1')
    assert protocol['expected_raw_rows']==124416 and protocol['new_GPU_seconds']==0 and protocol['training'] is False
    out=ROOT/'output'/f'job_{job}';out.mkdir(parents=True,exist_ok=False)
    write(out/'attempt.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=job,allocation=allocation,
        protocol_sha256=sha(ROOT/'protocol.json'),source_lock_sha256=sha(ROOT/'source_lock.json'),new_GPU_seconds=0))
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    parent_states=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-n','-P','-j',','.join(map(str,protocol['parent_jobs'])),'--format=JobID,State,ExitCode'],text=True)
    interactive_state=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-n','-P','-j','7590','--format=JobID,State,ExitCode'],text=True)
    assert sha(PREVIOUS/'execution_receipt.json')==protocol['old_receipt_sha256']
    assert sha(PREVIOUS/'output/job_7590/independent_audit.json')==protocol['old_audit_sha256']
    validate_parent_provenance(parent_states,interactive_state,read(PREVIOUS/'execution_receipt.json'),read(PREVIOUS/'output/job_7590/independent_audit.json'),protocol['old_deletion_summary_sha256'])
    assert sha(OLD/'final_native_verification.json')==protocol['old_final_native_sha256']
    assert sha(PREVIOUS/'output/job_7590/summary.json')==protocol['old_deletion_summary_sha256']
    assert sha(PREVIOUS/'output/job_7590/asset_ledger.json')==protocol['old_asset_ledger_sha256']
    previous_lock=read(PREVIOUS/'source_lock.json')
    assert all(sha(PREVIOUS/k)==v for k,v in previous_lock['files'].items())
    p,old_lock,configs=verify_control(OLD/'control')
    models=read(OLD/'training/job_7574/all_models_completion.json')['models'];assert len(models)==6
    ledgertext=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-n','-P','-j','7477,7480,7485,7524,7535,7574,7582','--format=JobID,State,ExitCode,ElapsedRaw,Account,AllocTRES'],text=True)
    ledger={int(v[0]):int(v[3]) for line in ledgertext.splitlines() if (v:=line.split('|')) and v[0].isdigit()}
    assert len(ledger)==7 and sum(ledger.values())==12723 and sum(ledger.values())<=protocol['GPU_cap_seconds']
    assets={str(OLD/'final_native_verification.json'):sha(OLD/'final_native_verification.json')}
    proofs={};software=[];allrows=[]
    partial=out/'raw_deletions.partial.csv';raw_stream=partial.open('x',newline='');raw_writer=None
    for task in protocol['tasks']:
        c=configs[task];store=FeatureStore(c,max_cached=4)
        assert c['data']['seed']==6301
        basispath=Path(c['output'])/'predictor'/task/'basis.npz'
        assert sha(basispath)==p['immutable_inputs'][task]['basis_sha256']
        with np.load(basispath,allow_pickle=False) as f:basis={k:f[k] for k in f.files}
        store.bases[task]=basis
        dp=Path(c['output'])/'probes'/f'{task}_state_decoder.npz'
        assert sha(dp)==read(Path(c['output'])/'probes/metrics.json')['checkpoints'][dp.name]
        decoder=LinearReadout.load(dp)
        for path in (basispath,dp,Path(c['output'])/'features/normalization.npz',Path(c['output'])/'features/index.json'):
            assets[str(path)]=sha(path)
        for seed in protocol['seeds']:
            reference,ref_sha=reference_model(c,p,task,seed,basis)
            reference_path=Path(c['output'])/'predictor'/task/f'seed_{seed}'/'best.pt';assets[str(reference_path)]=ref_sha
            folder=OLD/'training/job_7574/predictor'/task/f'seed_{seed}'
            proof=verify_completed_model(folder,models[f'{task}/{seed}']['signature'],reference.state_dict())
            assert proof==models[f'{task}/{seed}'];cp=folder/'best.pt';assets[str(cp)]=sha(cp)
            ck=torch.load(cp,map_location='cpu',weights_only=False)
            model=CenteredTransportPredictor(reference,basis);model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False)
            proofs[f'{task}/{seed}']=proof
            for split in protocol['splits']:
                pairs=paired_rows([r for r in store.manifest['records'] if r['task']==task and r['split']==split]);assert len(pairs)==128
                records=[r for pair in pairs for r in pair]
                for offset in range(0,len(records),8):
                    batch_start=len(allrows);inputs=[];descriptions=[]
                    for rec in records[offset:offset+8]:
                        z=store.features(rec);projected=store.projected_features(rec);q=int(rec['query'])
                        evidence=tuple(map(int,load_episode(c,rec,True,keys=('evidence',))['evidence']))
                        actions=load_episode(c,rec,keys=('actions',))['actions'];end=q-int(rec['gap'])-4
                        cache=Path(c['output'])/'features'/store.ix['episodes'][rec['id']]['path']
                        if str(cache) not in assets:assets[str(cache)]=sha(cache)
                        if task=='gate':
                            pools=list(gate_age_counterbalance(evidence[0],end,q).items())
                        else:
                            seed_value=stable_seed(protocol['pool_seed'],'clock_only',rec['pair'])
                            pools=[(f'clock_{i}',t) for i,t in enumerate(clock_only_static_pools(q,seed_value,count=6))]
                        for name,parent in pools:
                            for deleted in parent:
                                kept=tuple(t for t in parent if t!=deleted)
                                inputs.append(sample(z,projected,actions,q,kept))
                                descriptions.append((rec,name,parent,deleted,kept,evidence))
                    batch=collate(inputs,'cpu')
                    with torch.inference_mode():pred=model(batch)
                    assert pred.shape==(len(inputs),2,5120) and torch.isfinite(pred).all()
                    if offset==0 and split=='test_id':
                        with torch.inference_mode():
                            repeat=model(batch)
                            perm={k:(v[:,[0,1,2,3,5,4]] if k in ('z','projected_z','age','type','past_actions','mask') else v) for k,v in batch.items()}
                            shuffled=model(perm);changed={k:v.clone() for k,v in batch.items()};changed['actions'][:,3:]+=10
                            short=model(changed)[:,0]
                        errors=dict(repeat=float((pred-repeat).abs().max()),permutation=float((pred-shuffled).abs().max()),action_suffix=float((pred[:,0]-short).abs().max()))
                        assert errors['repeat']<1e-6 and errors['permutation']<3e-5 and errors['action_suffix']<1e-6
                        software.append(dict(task=task,seed=seed,errors=errors,passed=True))
                    predictions=pred.numpy();decoded=decoder.predict(predictions.reshape(-1,5120)).reshape(len(predictions),2,-1)
                    for i,(rec,name,parent,deleted,kept,evidence) in enumerate(descriptions):
                        # Target latents and simulator scoring truth are accessed only after forward.
                        z=store.features(rec);q=int(rec['query']);meta=load_episode(c,rec,True,keys=('states','classes'))
                        role=('cue' if deleted==evidence[0] else 'other') if task=='gate' else ('start' if deleted==evidence[0] else ('end' if deleted==evidence[1] else 'other'))
                        for hi,h in enumerate((3,6)):
                            error=float((decoded[i,hi,0]>0)!=(meta['classes'][q+h]>0)) if task=='gate' else float(np.mean((decoded[i,hi,1:]-meta['states'][q+h])**2))
                            allrows.append(dict(task=task,seed=seed,split=split,pair=rec['pair'],episode=rec['id'],pool=name,
                                pool_group=name if task=='gate' else 'clock_all',query=q,evidence_end=q-int(rec['gap'])-4,evidence=';'.join(map(str,evidence)),
                                parent=';'.join(map(str,parent)),deleted=deleted,deleted_role=role,kept=';'.join(map(str,kept)),history_count=2,
                                horizon=h,feature_dim=5120,feature_mse=float(np.mean((predictions[i,hi]-z[q+h])**2)),task_error=error))
                    if raw_writer is None:
                        raw_writer=csv.DictWriter(raw_stream,fieldnames=list(allrows[0]),lineterminator='\n');raw_writer.writeheader()
                    raw_writer.writerows(allrows[batch_start:]);raw_stream.flush()
                    if time.monotonic()-begin>1500:raise RuntimeError('CPU soft budget reached; preserve partial, no resubmit')
                write(out/'progress.json',dict(task=task,seed=seed,split_completed=split,raw_rows=len(allrows),expected_rows=124416,new_GPU_seconds=0,elapsed=time.monotonic()-begin))
                print(f'{task} seed{seed} {split}: {len(allrows)}/124416',flush=True)
            assert sha(cp)==proof['best_sha256'];del model,reference,ck
    assert len(allrows)==124416 and len(software)==6
    raw_stream.close();raw=out/'raw_deletions.csv';partial.rename(raw)
    compact,coverage=aggregate(allrows)
    compact_path=out/'paired_policy_losses.csv'
    with compact_path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(compact[0]),lineterminator='\n');writer.writeheader();writer.writerows(compact)
    result=summarize(compact)
    assert result['paired_rows']==9216 and result['independent_scalar_statistics']==1296
    assert all(sha(path)==h for path,h in assets.items())
    verify_control(OLD/'control');locked_sources()
    assert all(sha(PREVIOUS/k)==v for k,v in previous_lock['files'].items())
    write(out/'asset_ledger.json',assets);write(out/'pool_coverage.json',coverage)
    result.update(job=job,allocation=allocation,parent_states=parent_states,models=proofs,software_checks=software,
        raw_rows=len(allrows),raw_sha256=sha(raw),paired_sha256=sha(compact_path),asset_ledger_sha256=sha(out/'asset_ledger.json'),
        source_lock_sha256=sha(ROOT/'source_lock.json'),protocol_sha256=sha(ROOT/'protocol.json'),pool_coverage_sha256=sha(out/'pool_coverage.json'),
        GPU_native_ledger=ledger,GPU_native_sacct=ledgertext,prior_GPU_seconds=sum(ledger.values()),remaining_GPU_seconds=28800-sum(ledger.values()),
        elapsed_seconds=time.monotonic()-begin,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(out/'summary.json',result)
    print(json.dumps(dict(status=result['status'],raw_rows=len(allrows),seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        folder=ROOT/'output'/f"job_{os.environ.get('SLURM_JOB_ID','unknown')}";folder.mkdir(parents=True,exist_ok=True)
        write(folder/'failure.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=traceback.format_exc(),refuse_automatic_retry=True))
        raise
