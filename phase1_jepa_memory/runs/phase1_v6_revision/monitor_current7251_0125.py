"""One compact provenance/current-boundary snapshot; no inference or fitting."""
import io,json,csv,hashlib,socket,subprocess,datetime
from pathlib import Path
import torch

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def finite(v):
 if isinstance(v,torch.Tensor):return not v.is_floating_point() or bool(torch.isfinite(v).all())
 if isinstance(v,dict):return all(finite(x) for x in v.values())
 if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
 return True

torch.set_num_threads(1)
row=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','7251','-n','-P','--format=JobIDRaw,State,Account,Partition,NodeList,AllocTRES,Start,End'],text=True).strip().split('|')
assert row[:5]==['7251','RUNNING','csci_ga_2271-2026fa','c12m85-a100-1',socket.gethostname()]
assert 'gres/gpu=1' in row[5].split(',')
control=Path('/scratch/xl4358/phase1_jepa_memory_v6_real_wd020');root=Path('/scratch/xl4358/phase1_v6_real_wd020/job_7251')
assert sha(control/'source_lock.json')=='24116c56a15ead3f47d5109be0a389eab179f01fbc63ba96fb69e9813bf3b5e5'
for name,w in json.loads((control/'source_lock.json').read_text())['files'].items():assert sha(control/name)==w
report=json.loads((root/'training_report.json').read_text());assert report['job_id']=='7251' and report['host']==socket.gethostname()
assert sha(root/'train_only_fit.npz')=='d90f99e4a91d290b7d27311de2820775981afe08dc053dc642e8a40636ed1fb3'
assert sha(root/'fit_proof.json')=='18c44903742143b482aef5a7c035aa2540fb3f60ba8963c50971be8abc5fd7aa'
seeds=[]
for seed in (17,29,43):
 p=root/f'seed_{seed}/latest.pt'
 if not p.exists():seeds.append({'seed':seed,'state':'NOT_STARTED'});continue
 payload=p.read_bytes();digest=hashlib.sha256(payload).hexdigest();state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
 assert state['seed']==seed and state['training_signature']==report['training_signature']
 assert state['global_step']==state['epoch']*323 and finite(state['model']) and finite(state['optimizer'])
 assert all(g['weight_decay']==.20 for g in state['optimizer']['param_groups'])
 import sys
 sys.path[:0]=[str(control/'src'),str(control/'scripts')]
 from phase1.v5_real_training import schedule_lr
 assert all(abs(g['lr']-schedule_lr(state['global_step']-1,9690))<1e-15 for g in state['optimizer']['param_groups'])
 assert all(int(v['step'])==state['global_step'] for v in state['optimizer']['state'].values())
 assert all(k in state for k in ('python_rng','numpy_global_rng','retention_dropout_shuffle_rng','torch_cpu_rng','torch_cuda_rng'))
 assert sha(p)==digest,'Active boundary changed; reject snapshot, not training failure'
 csvpath=root/f'seed_{seed}/training.csv';rows=list(csv.DictReader(csvpath.open()));completed=[x for x in rows if int(x['epoch'])<=state['epoch']]
 assert len(completed)==state['epoch']
 chosen=min(completed,key=lambda x:(float(x['validation_full_mse']),int(x['epoch'])))
 assert state['best_epoch']==int(chosen['epoch']) and abs(state['best_validation']-float(chosen['validation_full_mse']))<1e-12
 best_file=root/f'seed_{seed}/best.pt';best_payload=best_file.read_bytes();best_hash=hashlib.sha256(best_payload).hexdigest();best=torch.load(io.BytesIO(best_payload),map_location='cpu',weights_only=False)
 assert best['seed']==seed and best['training_signature']==report['training_signature'] and finite(best['model'])
 assert best['epoch']==state['best_epoch'] and abs(best['validation']-state['best_validation'])<1e-12
 assert sha(best_file)==best_hash
 assert sha(p)==digest
 seeds.append({'seed':seed,'epoch':state['epoch'],'steps':state['global_step'],'best_epoch':state['best_epoch'],'best_validation':state['best_validation'],'actual_weight_optimizer_finite_RNG_step_and_020_decay':True,'latest_sha256_at_stable_read_boundary':digest})
out=root/'root_partial_monitor_20261014T0125.json';assert not out.exists()
snapshot={'status':'ACTUAL_GPU_TRAINING_PARTIAL_PROVEN_NOT_FINAL_EVALUATION','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'native_sacct':row,'current_seeds':seeds,'training_report_snapshot':report,'all_current_source_hashes_match':True,'actual_fit_npz_and_proof_bytes_equal_original7209':True,'new_inference_fit_training':False,'all6_final_models_proven':False}
out.write_text(json.dumps(snapshot,indent=2)+'\n');print(json.dumps({'output':str(out),'sha256':sha(out),'seeds':seeds}))
