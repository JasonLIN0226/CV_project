"""Independent completion boundary before any downstream model inference."""
from pathlib import Path
import hashlib
import json
import subprocess
import torch
from memory_jepa.utils.hashing import canonical_hash,state_dict_hash
from memory_jepa.utils.modeling import build_predictor

COURSE='csci_ga_2271-2026fa'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def completed_proof(training_submission,training_root):
    sub=json.loads(Path(training_submission).read_text());job=int(sub['job_id'])
    state=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j',str(job),'-n','-P','--format=JobIDRaw,State,ExitCode,Account'],text=True)
    if not any(row.split('|')[:4]==[str(job),'COMPLETED','0:0',COURSE] for row in state.splitlines()):
        raise RuntimeError('actual predictor course parent not COMPLETED0:0')
    training_root=Path(training_root);lock=json.loads((training_root/'source_lock.json').read_text())
    for name,value in lock['files'].items():
        if (training_root/name).is_symlink() or sha(training_root/name)!=value:raise RuntimeError('completed training source differs')
    protocol_path=training_root/'paper_training_protocol.json';protocol=json.loads(protocol_path.read_text())
    report_path=Path(sub['output'])/'training_report.json';report=json.loads(report_path.read_text())
    if report['elapsed_seconds']>13800:
        raise RuntimeError('completed training exceeds the frozen science budget')
    if report['resources']['job_id']!=job or report['resources']['visible_gpus']!=1 or 'A100' not in report['resources']['gpu']:
        raise RuntimeError('actual training resource proof differs')
    if report['protocol_sha256']!=sha(protocol_path) or report['source_lock_sha256']!=sha(training_root/'source_lock.json'):
        raise RuntimeError('executed training binding differs')
    data_path=Path(protocol['data_report']);data=json.loads(data_path.read_text())
    if sha(data_path)!=protocol['data_report_sha256']:raise RuntimeError('data proof changed')
    if data['status']!='OFFICIAL_TSM_TRAIN_VAL_PREPARED_AND_AUDITED' or data['test_payload_deserialized']:
        raise RuntimeError('complete test-isolated input required')
    manifest_path=Path(data['manifest'])
    if sha(manifest_path)!=protocol['manifest_sha256']:raise RuntimeError('manifest changed')
    for row in data['features']:
        f=manifest_path.parent/(row['video_id']+'.pt')
        if f.is_symlink() or f.stat().st_size!=row['bytes'] or sha(f)!=row['sha256']:raise RuntimeError('official features differ')
    all_files=[];seed_proofs=[]
    for seed,executed in zip((0,1,2),report['completed_models'],strict=True):
        if executed['seed']!=seed or executed['epochs']!=30 or executed['steps']!=executed['expected_steps']:
            raise RuntimeError('incomplete seed training counts')
        folder=report_path.parent/f'seed_{seed}';history_path=folder/'predictor_history.json'
        if sha(history_path)!=executed['history_sha256']:raise RuntimeError('training history differs')
        history=json.loads(history_path.read_text())
        if [r['epoch'] for r in history]!=list(range(1,31)):raise RuntimeError('incomplete epoch history')
        earliest=min(range(30),key=lambda i:history[i]['validation']['loss'])+1
        if earliest!=executed['best_epoch']:raise RuntimeError('not the earliest minimum validation checkpoint')
        for tag in ('best','latest'):
            f=folder/f'predictor_{tag}.pt';before=sha(f)
            if before!=executed[tag+'_sha256'] or f.is_symlink():raise RuntimeError('actual saved model hash differs')
            ck=torch.load(f,map_location='cpu',weights_only=True)
            wanted_epoch=earliest if tag=='best' else 30
            if ck['seed']!=seed or ck['epoch']!=wanted_epoch or ck['global_step']!=history[wanted_epoch-1]['steps']:
                raise RuntimeError('checkpoint/history boundary differs')
            if tag=='latest' and not ck['complete']:raise RuntimeError('latest checkpoint incomplete')
            if ck['source_lock_sha256']!=sha(training_root/'source_lock.json') or ck['protocol_sha256']!=sha(protocol_path) or ck['manifest_sha256']!=sha(manifest_path):
                raise RuntimeError('checkpoint source/data binding differs')
            model=build_predictor(ck['config'],int(ck['d_in']))
            model.load_state_dict(ck['model'],strict=True)
            if ck['config_hash']!=canonical_hash(ck['config']) or ck['predictor_hash']!=state_dict_hash(model):
                raise RuntimeError('actual model/config numerical hash differs')
            tensors=list(ck['model'].values())+[v for state in ck['optimizer']['state'].values() for v in state.values() if isinstance(v,torch.Tensor)]
            if not all(torch.isfinite(t).all() for t in tensors):raise RuntimeError('nonfinite model/optimizer')
            if set(ck['rng'])!={'python','numpy','torch_cpu','torch_cuda'} or not ck['scheduler']:
                raise RuntimeError('missing RNG or scheduler boundary')
            if not ck['optimizer']['state'] or ck['scheduler']['last_epoch']!=ck['global_step']:
                raise RuntimeError('missing optimizer state or wrong schedule boundary')
            if ck['optimizer']['param_groups'][0]['weight_decay']!=.05:raise RuntimeError('paper optimizer differs')
            if sha(f)!=before:raise RuntimeError('completed checkpoint changed while auditing')
            all_files.append({'seed':seed,'tag':tag,'path':str(f),'sha256':before,'bytes':f.stat().st_size,
                              'epoch':wanted_epoch,'step':ck['global_step'],'model_hash':ck['predictor_hash'],'finite_model_optimizer':True})
            del ck,tensors,model
        seed_proofs.append({'seed':seed,'epochs':30,'steps':executed['steps'],'earliest_best_epoch':earliest})
    return {'status':'ALL_SIX_COMPLETED_PAPER_PREDICTOR_MODELS_BOUND','training_job_id':job,
            'training_report_sha256':sha(report_path),'training_source_lock_sha256':sha(training_root/'source_lock.json'),
            'training_protocol_sha256':sha(protocol_path),'data_report':str(data_path),'data_report_sha256':sha(data_path),
            'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),'seeds':seed_proofs,'models':all_files,
            'heldout_model_inference_before_proof':False,'test_confirmation_opened':False}
