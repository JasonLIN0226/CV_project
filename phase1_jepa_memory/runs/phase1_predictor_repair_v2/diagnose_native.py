"""Read-only completed-model diagnosis; no optimizer or fresh data access."""
from pathlib import Path
import sys,os,json,hashlib,subprocess,datetime,socket
import numpy as np
import torch

OLD=Path('/scratch/xl4358/phase1_predictor_repair/approved_8h')
ROOT=Path('/scratch/xl4358/phase1_predictor_repair_v2')
sys.path.insert(0,str(OLD/'control/src'))
from phase1.proposal_trial import verify_control,reference_model
from phase1.relational_memory import RelationalMemoryPredictor
from phase1.inputs import FeatureStore,collate
from phase1.model import fit_basis
from phase1.training import paired_rows
from phase1.common_anchor import recent_only_teacher_batch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rms(x):return float(x.square().mean().sqrt())

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    job=sys.argv[1]
    allocation=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',job],text=True)
    f=dict(v.split('=',1) for v in allocation.split() if '=' in v)
    assert f['JobState']=='RUNNING' and f['Account']=='csci_ga_2271-2026fa'
    assert f['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in f['AllocTRES']
    dest=ROOT/'diagnostic_results.json'
    if dest.exists() or (ROOT/'diagnostic_attempt.json').exists():raise ValueError('Existing diagnostic outcome; do not retry')
    (ROOT/'diagnostic_attempt.json').write_text(json.dumps({'job':job,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'diagnostic_protocol.json')})+'\n')
    states=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-n','-P','-j','7535,7536,7537','--format=JobID,State,ExitCode'],text=True)
    assert set(states.splitlines())=={str(j)+'|COMPLETED|0:0' for j in (7535,7536,7537)}
    p,lock,configs=verify_control(OLD/'control')
    assert sha(OLD/'control/source_lock.json')=='69f577e71ae44cc02a45b89e75facfdc53854085ba36acba11ee45f285beaa17'
    completed=json.loads((OLD/'training/job_7535/all_models_completion.json').read_text())
    results=[];checkpoints={}
    for task in ('gate','motion'):
        cfg=configs[task];store=FeatureStore(cfg,max_cached=64);basis=fit_basis(cfg,store,task)
        for seed in (17,29,43):
            ref,_=reference_model(cfg,p,task,seed,basis)
            path=OLD/'training/job_7535/predictor'/task/f'seed_{seed}/best.pt'
            expected=completed['models'][f'{task}/{seed}']['best_sha256'];assert sha(path)==expected
            ck=torch.load(path,map_location='cpu',weights_only=False)
            model=RelationalMemoryPredictor(ref,basis);model.load_state_dict(ck['model'],strict=True);model.eval()
            checkpoints[f'{task}/{seed}']={'SHA256':expected,'epoch':ck['epoch']}
            for split in ('train','validation','test_long_gap','test_distractor_shift'):
                pairs=paired_rows([r for r in store.manifest['records'] if r['task']==task and r['split']==split])[:32]
                rows=[r for pair in pairs for r in pair]
                for condition in ('privileged_small_budget','full_history'):
                    vals=[]
                    for start in range(0,len(rows),32):
                        samples=[store.sample(r,condition) for r in rows[start:start+32]]
                        b=collate([s[0] for s in samples],'cpu');target=torch.from_numpy(np.stack([s[1] for s in samples]))
                        with torch.inference_mode():
                            pred=model(b);anchor=model.reference(recent_only_teacher_batch(b));corr=pred-anchor
                            common=lambda z:(z[::2]+z[1::2])/2
                            diff=lambda z:z[::2]-z[1::2]
                            mse=float((pred-target).square().mean());cm=float((common(pred)-common(target)).square().mean());dm=float((diff(pred)-diff(target)).square().mean())/4
                            assert np.isclose(mse,cm+dm,atol=1e-6,rtol=1e-5)
                            hist=b['mask']&(b['type']==1);age=b['age'];z=(b['z']-b['current'][:,None])@model.projection
                            order=age.masked_fill(~hist,-1).argsort(dim=1,descending=True,stable=True)
                            ordered=z.gather(1,order[...,None].expand_as(z));a=age.gather(1,order);valid=hist.gather(1,order)
                            dt=a[:,:-1]-a[:,1:];rv=valid[:,:-1]&valid[:,1:]&(dt>0)
                            vel=(ordered[:,1:]-ordered[:,:-1])/dt.clamp_min(1)[...,None]
                            transport=ordered[:,1:]+vel*(a[:,1:]+6)[...,None]
                            shifted={k:v.clone() for k,v in b.items()}
                            g=torch.Generator().manual_seed(0);offset=torch.randn(b['current'].shape[-1],generator=g);offset=.1*offset/offset.square().mean().sqrt()
                            shifted['z']+=offset;shifted['current']+=offset
                            if 'projected_z' in shifted:shifted['projected_z']+=offset@model.projection
                            if 'projected_current' in shifted:shifted['projected_current']+=offset@model.projection
                            changed=model(shifted)-model.reference(recent_only_teacher_batch(shifted))
                            values={'normalized_mse':mse,'pair_common_mse':cm,'quarter_pair_difference_mse':dm,
                                'pair_amplitude_ratio':rms(diff(pred))/max(rms(diff(target)),1e-12),
                                'common_correction_RMS':rms(common(corr)),'pair_correction_RMS':rms(diff(corr))/2,
                                'centered_history_RMS':rms(z[hist]),'observed_velocity_RMS':rms(vel[rv]) if rv.any() else 0,
                                'transport_h6_RMS':rms(transport[rv]) if rv.any() else 0,
                                'translation_correction_change_RMS':rms(changed-corr),
                                'translation_to_correction_ratio':rms(changed-corr)/max(rms(corr),1e-12)}
                            assert all(np.isfinite(v) for v in values.values());vals.append(values)
                    results.append({'task':task,'seed':seed,'selected_epoch':ck['epoch'],'split':split,'condition':condition,'pairs':len(pairs),**{k:float(np.mean([v[k] for v in vals])) for k in vals[0]}})
            assert sha(path)==expected
            del ck,model,ref
    verify_control(OLD/'control')
    result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':job,'native_allocation':allocation,'parents':states,'current_sources_verified':len(lock['files']),'checkpoint_hashes':checkpoints,'slices':results,'counterfactual_sensitivity_not_physical_performance':True,'no_optimizer_or_training':True,'GPU':0,'fresh9137_opened':False,'phase1_go':False,'script_sha256':sha(__file__)}
    with dest.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':'READONLY_DIAGNOSIS_COMPLETE','slices':len(results),'file':str(dest),'sha256':sha(dest)}),flush=True)

if __name__=='__main__':main()
