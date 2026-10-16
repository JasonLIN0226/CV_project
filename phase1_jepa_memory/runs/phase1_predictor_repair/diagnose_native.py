"""Read-only diagnosis of previously completed current models on fixed dev pairs."""
from pathlib import Path
import sys, os, json, hashlib, subprocess, datetime, copy
import numpy as np
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
OLD=Path('/scratch/xl4358/phase1_proposal_validation/fast_execution')
ROOT=Path('/scratch/xl4358/phase1_predictor_repair')
sys.path.insert(0,str(OLD/'control/src'))
from phase1.proposal_trial import verify_control, reference_model
from phase1.anchored_memory import AnchoredMemoryPredictor
from phase1.inputs import FeatureStore, collate
from phase1.model import fit_basis
from phase1.training import paired_rows
from phase1.common_anchor import recent_only_teacher_batch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rms(x):return float(x.square().mean().sqrt())

def main():
    if os.environ.get('SLURM_JOB_ACCOUNT')!='csci_ga_2271-2026fa':raise ValueError('Actual course allocation required')
    job=os.environ['SLURM_JOB_ID'];root=ROOT/'diagnostics'/('job_'+job)
    if root.exists():raise ValueError('Refuse implicit diagnostic rerun or overwrite')
    root.mkdir(parents=True)
    (root/'attempt.json').write_text(json.dumps({'job':job,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'diagnostic_protocol.json'),'no_optimizer':True})+'\n')
    native=subprocess.run(['/opt/slurm/bin/sacct','-j','7485,7486,7487','-X','--noheader','--parsable2','--format=JobID,State,ExitCode'],capture_output=True,text=True,check=True).stdout
    assert all('|COMPLETED|0:0' in line for line in native.splitlines()) and len(native.splitlines())==3
    p,lock,configs=verify_control(OLD/'control')
    assert sha(OLD/'control/source_lock.json')=='45b99b0eaa3aa4777a2e576804127ca262271b03a048146cb41b9f09d99be676'
    declared=json.loads((OLD/'training/job_7485/all_models_completion.json').read_text())
    results=[];checks={}
    for task in ['gate','motion']:
        cfg=configs[task];store=FeatureStore(cfg,max_cached=64);basis=fit_basis(cfg,store,task)
        for seed in [17,29,43]:
            reference,_=reference_model(cfg,p,task,seed,basis)
            for which in ['best','latest']:
                ckpath=OLD/'training/job_7485/predictor'/task/f'seed_{seed}'/(which+'.pt')
                expected=declared['models'][task+'/'+str(seed)][which+'_sha256'];assert sha(ckpath)==expected
                ck=torch.load(ckpath,map_location='cpu',weights_only=False)
                model=AnchoredMemoryPredictor(reference,basis);model.load_state_dict(ck['model'],strict=True);model.eval()
                checks[f'{task}/{seed}/{which}']={'SHA256':expected,'epoch':ck['epoch'],'parameter_RMS':{name:rms(value.float()) for name,value in model.side.state_dict().items() if 'weight' in name}}
                for split in ['train','validation','test_long_gap','test_distractor_shift']:
                    pairs=paired_rows([r for r in store.manifest['records'] if r['task']==task and r['split']==split])[:32]
                    rows=[r for pair in pairs for r in pair]
                    for condition in ['recent_only','privileged_small_budget','full_history']:
                        values=[]
                        for start in range(0,len(rows),32):
                            part=rows[start:start+32];samples=[store.sample(r,condition) for r in part]
                            b=collate([x[0] for x in samples],'cpu');target=torch.from_numpy(np.stack([x[1] for x in samples]))
                            with torch.inference_mode():
                                pred=model(b);anchor=model.reference(recent_only_teacher_batch(b))
                                valid=b['mask'];hist=valid&(b['type']==1)
                                projected=(b['z']-model.side.input_mean)@model.side.projection
                                visual=model.side.visual(projected)
                                time=model.side.time(torch.stack([b['age']/8.,(1-b['type']).float()],-1))
                                shifted={k:v.clone() for k,v in b.items()};shifted['age'][hist]+=16
                                changed=model(shifted)
                                assert torch.isfinite(pred).all() and torch.isfinite(changed).all()
                                values.append({'normalized_mse':float((pred-target).square().mean()),'correction_RMS':rms(pred-anchor),
                                    'predicted_pair_RMS':rms(pred[::2]-pred[1::2]),'target_pair_RMS':rms(target[::2]-target[1::2]),
                                    'visual_embedding_RMS':rms(visual[valid]),'time_embedding_RMS':rms(time[valid]),
                                    'min_history_age':float(b['age'][hist].min()) if hist.any() else 0.,
                                    'max_history_age':float(b['age'][hist].max()) if hist.any() else 0.,
                                    'age_plus16_output_change_RMS':rms(changed-pred),
                                    'age_plus16_mse_sensitivity':float((changed-target).square().mean()-(pred-target).square().mean())})
                        results.append({'task':task,'seed':seed,'checkpoint':which,'selected_epoch':ck['epoch'],'split':split,'condition':condition,'pairs':len(pairs),**{k:float(np.mean([x[k] for x in values])) for k in values[0]}})
                assert sha(ckpath)==expected
    output={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':job,'native_parents':native,'current_sources_verified':len(lock['files']),'current_checkpoint_proofs':checks,'diagnostic_slices':results,'no_model_training':True,'no_model_source_changed':True,'fresh9137_accessed':False,'scientific_acceptance_rule_changed':False,'counterfactual_age_sensitivity_only_not_performance_claim':True}
    (root/'diagnostic_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'status':'READONLY_DIAGNOSIS_COMPLETE','slices':len(results),'phase1_go':False}),flush=True)

if __name__=='__main__':main()
