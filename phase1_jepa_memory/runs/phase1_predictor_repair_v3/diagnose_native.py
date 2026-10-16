"""Read-only completed7574 diagnosis. No optimizer, fitting, GPU, or fresh data."""
import datetime, hashlib, json, os, socket, subprocess, sys
from pathlib import Path
import numpy as np
import torch

OLD = Path('/scratch/xl4358/phase1_predictor_repair_v2/execution')
ROOT = Path('/scratch/xl4358/phase1_predictor_repair_v3')
sys.path.insert(0, str(OLD/'control/src'))
from phase1.proposal_trial import verify_control, reference_model
from phase1.centered_transport import CenteredTransportPredictor
from phase1.inputs import FeatureStore, collate
from phase1.model import fit_basis
from phase1.training import paired_rows
from phase1.common_anchor import recent_only_teacher_batch
from phase1.data import load_episode
from phase1.probes import LinearReadout

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def mse(x,y): return float((x-y).square().mean())

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    job = sys.argv[1]
    native = subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',job],text=True)
    fields = dict(v.split('=',1) for v in native.split() if '=' in v)
    assert fields['JobState']=='RUNNING' and fields['Account']=='csci_ga_2271-2026fa'
    assert fields['NodeList']==socket.gethostname().split('.')[0] and 'gres/gpu' not in fields['AllocTRES']
    destination = ROOT/'diagnostic_results.json'
    assert not destination.exists() and not (ROOT/'diagnostic_attempt.json').exists()
    lock=read(ROOT/'diagnostic_source_lock.json')
    assert all(sha(ROOT/k)==v for k,v in lock['files'].items())
    (ROOT/'diagnostic_attempt.json').write_text(json.dumps({'job':job,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_lock_sha256':sha(ROOT/'diagnostic_source_lock.json')})+'\n')
    protocol=read(ROOT/'diagnostic_protocol.json')
    assert sha(OLD/'final_native_verification.json')==protocol['parent_final_verification_sha256']
    states=subprocess.check_output(['/opt/slurm/bin/sacct','-n','-X','-P','-j','7574,7575,7576','--format=JobID,State,ExitCode'],text=True)
    assert set(states.strip().splitlines())=={str(j)+'|COMPLETED|0:0' for j in (7574,7575,7576)}
    p,source,configs=verify_control(OLD/'control')
    assert sha(OLD/'control/source_lock.json')=='0118e9b61f13d6ae0b271f3faa77bf276d6e7af1f6c97992ae0974902e6ba2d0'
    completed=read(OLD/'training/job_7574/all_models_completion.json')
    results=[];checkpoints={}
    for task in ('gate','motion'):
        cfg=configs[task];store=FeatureStore(cfg,max_cached=64);basis=fit_basis(cfg,store,task)
        probe=read(Path(cfg['output'])/'probes/metrics.json')
        dp=Path(cfg['output'])/'probes'/f'{task}_state_decoder.npz'
        assert sha(dp)==probe['checkpoints'][dp.name]
        decoder=LinearReadout.load(dp)
        for seed in (17,29,43):
            reference,_=reference_model(cfg,p,task,seed,basis)
            cp=OLD/'training/job_7574/predictor'/task/f'seed_{seed}/best.pt'
            expected=completed['models'][f'{task}/{seed}']['best_sha256'];assert sha(cp)==expected
            ck=torch.load(cp,map_location='cpu',weights_only=False)
            model=CenteredTransportPredictor(reference,basis);model.load_state_dict(ck['model'],strict=True);model.eval()
            checkpoints[f'{task}/{seed}']={'sha256':expected,'selected_epoch':ck['epoch']}
            for split in protocol['splits']:
                pairs=paired_rows([r for r in store.manifest['records'] if r['task']==task and r['split']==split])[:64]
                rows=[r for pair in pairs for r in pair]
                for condition in protocol['conditions']:
                    chunks=[]
                    for start in range(0,len(rows),32):
                        rr=rows[start:start+32];samples=[store.sample(row,condition) for row in rr]
                        batch=collate([s[0] for s in samples],'cpu');target=torch.from_numpy(np.stack([s[1] for s in samples]))
                        frame_coeff=[];transport_coeff=[]
                        hf=model.frame_values.register_forward_hook(lambda m,i,o:frame_coeff.append(o.detach().clone()))
                        ht=model.transport_values.register_forward_hook(lambda m,i,o:transport_coeff.append(o.detach().clone()))
                        with torch.inference_mode():
                            pred=model(batch);anchor=model.reference(recent_only_teacher_batch(batch))
                            frame=anchor+torch.stack(frame_coeff,1)@model.output_basis.T
                            transport=anchor+torch.stack(transport_coeff,1)@model.output_basis.T
                        hf.remove();ht.remove()
                        assert torch.allclose(pred,frame+transport-anchor,atol=3e-6,rtol=3e-5)
                        common=lambda z:(z[::2]+z[1::2])/2
                        difference=lambda z:z[::2]-z[1::2]
                        states=[];secants=[];true_v=[];decoded_v=[];reflection=[];transport_norm=[]
                        for row in rr:
                            # Evaluation metadata is loaded only after all model forwards.
                            meta=load_episode(cfg,row,True);ev=meta['evidence'];a=load_episode(cfg,row,keys=('actions',))['actions']
                            z=store.features(row);q=row['query'];state=meta['states']
                            states.append(state[q+np.array([3,6])])
                            if task=='motion':
                                i,j=map(int,ev);dt=j-i;velocity=(state[j,0]-state[i,0])/dt
                                decoded=decoder.predict(z[ev])[:,1:]
                                decoded_v.append(float((decoded[1,0]-decoded[0,0])/dt));true_v.append(float(velocity))
                                ss=[];nn=[];ref=[]
                                for h in (3,6):
                                    secant=z[j]+(z[j]-z[i])/dt*(q+h-j)
                                    ss.append(secant);nn.append(float(np.sqrt(np.mean(secant**2))))
                                    actions=a[j:q+h]
                                    unreflected=state[j,0]+velocity*(q+h-j)+float((.0015*actions[:,0]+.001*actions[:,1]).sum())
                                    ref.append(abs(unreflected-float(state[q+h,0]))>1e-5)
                                secants.append(ss);transport_norm.append(nn);reflection.append(ref)
                        truth=np.asarray(states)
                        decoded_pred=decoder.predict(pred.numpy().reshape(-1,pred.shape[-1]))[:,1:].reshape(len(rr),2,2)
                        decoded_target=decoder.predict(target.numpy().reshape(-1,target.shape[-1]))[:,1:].reshape(len(rr),2,2)
                        values=[]
                        for hix,h in enumerate((3,6)):
                            y=target[:,hix];v={'horizon':h,'pred_MSE':mse(pred[:,hix],y),'reference_recent_MSE':mse(anchor[:,hix],y),'frame_only_MSE':mse(frame[:,hix],y),'transport_only_MSE':mse(transport[:,hix],y),
                                'common_MSE':mse(common(pred[:,hix]),common(y)),
                                'quarter_pair_difference_MSE':mse(difference(pred[:,hix]),difference(y))/4,
                                'reference_common_MSE':mse(common(anchor[:,hix]),common(y)),
                                'reference_quarter_pair_difference_MSE':mse(difference(anchor[:,hix]),difference(y))/4,
                                'predicted_position_MSE':float(np.mean((decoded_pred[:,hix]-truth[:,hix])**2)),
                                'true_target_feature_readout_position_MSE':float(np.mean((decoded_target[:,hix]-truth[:,hix])**2))}
                            assert np.isclose(v['pred_MSE'],v['common_MSE']+v['quarter_pair_difference_MSE'],atol=1e-6,rtol=1e-5)
                            if task=='motion':
                                v.update(feature_secant_MSE=float(np.mean((np.asarray(secants)[:,hix]-target[:,hix].numpy())**2)),feature_secant_RMS=float(np.mean(np.asarray(transport_norm)[:,hix])),
                                    visible_velocity_readout_RMSE=float(np.sqrt(np.mean((np.asarray(decoded_v)-true_v)**2))),
                                    true_visible_velocity_RMS=float(np.sqrt(np.mean(np.asarray(true_v)**2))),
                                    reflected_fraction=float(np.asarray(reflection)[:,hix].mean()))
                            assert all(np.isfinite(z) for z in v.values());values.append(v)
                        chunks.append(values)
                    for i,h in enumerate((3,6)):
                        results.append({'task':task,'seed':seed,'split':split,'condition':condition,'pairs':len(pairs),'horizon':h,
                                        **{k:float(np.mean([chunk[i][k] for chunk in chunks])) for k in chunks[0][i] if k!='horizon'}})
            assert sha(cp)==expected
            del ck,model,reference
    verify_control(OLD/'control')
    result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':job,'native_allocation':native,'parents':states,'current_scientific_files':len(source['files']),
            'checkpoint_proofs':checkpoints,'slices':results,'evaluation_only_states_never_inputs':True,'no_training_or_optimizer':True,'GPU':0,'fresh9137_opened':False,'source_lock_sha256':sha(ROOT/'diagnostic_source_lock.json')}
    with destination.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':'READONLY_DIAGNOSIS_COMPLETE','slices':len(results),'sha256':sha(destination)}),flush=True)

if __name__=='__main__':main()
