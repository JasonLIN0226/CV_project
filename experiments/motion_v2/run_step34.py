"""One-shot train/validation-only Motion steps3/4. Never submits a job itself."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
REVISION = 'f157467ea509bc356ff9f61fd3c0d840eec5e04e'


def load(name):
    key = 'motion_step34_entry_' + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(name+'.py'))
        obj = importlib.util.module_from_spec(spec)
        sys.modules[key] = obj
        spec.loader.exec_module(obj)
    return sys.modules[key]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def committed(path):
    path=Path(path).resolve();relative=path.relative_to(REPO).as_posix()
    content=subprocess.run(['git','show','HEAD:'+relative],cwd=REPO,capture_output=True,check=True).stdout
    if content != path.read_bytes():raise ValueError('Uncommitted registration/source: '+relative)
    return path


def registration(path):
    protocol=read(committed(path))
    if protocol.get('status')!='FROZEN_STEP3_4_PROTOCOL' or protocol.get('registered_before_run') is not True:
        raise ValueError('Committed step3/4 preregistration required')
    for relative,checksum in protocol['source_sha256'].items():
        if sha(committed(REPO/relative))!=checksum:raise ValueError('Source hash changed: '+relative)
    for key in ('config','asset_inventory','document'):
        entry=protocol[key]
        if sha(committed(REPO/entry['path']))!=entry['sha256']:raise ValueError('Registered input changed: '+key)
    config=read(REPO/protocol['config']['path'])
    if config['budget']!={'prior_a100_seconds':3705,'total_a100_seconds':14400,
                         'remaining_a100_seconds':10695,'estimated_seconds':8500,'wall_seconds':10500}:
        raise ValueError('Budget differs from registration')
    return protocol,config


def finish(directory,status,guard=None):
    inventory={};was_complete=status['status']=='completed_step4_stop'
    for p in sorted(directory.rglob('*')):
        if p.is_file() and not p.is_relative_to(directory/'runtime'):
            inventory[p.relative_to(directory).as_posix()]=sha(p)
            if guard is not None and status['status']=='completed_step4_stop':
                try:guard()
                except Exception as exc:
                    status.update(status='stopped',error_type=type(exc).__name__,error=str(exc))
    if guard is not None:
        if status['status']=='completed_step4_stop':
            try:guard.progress(1)
            except Exception as exc:status.update(status='stopped',error_type=type(exc).__name__,error=str(exc))
        status['budget']=guard.snapshot()
    # Native check logs are mutable until the final guard call; hash them last.
    for p in sorted((directory/'runtime').rglob('*')):
        if p.is_file():inventory[p.relative_to(directory).as_posix()]=sha(p)
    status['finished_utc']=utc()
    write(directory/'status.json',status)
    inventory['status.json']=sha(directory/'status.json')
    write(directory/'artifact_sha256.json',inventory)
    with (directory/'SHA256SUMS').open('x') as f:
        for relative,checksum in inventory.items():f.write(checksum+'  '+relative+'\n')
        f.write(sha(directory/'artifact_sha256.json')+'  artifact_sha256.json\n')
    if was_complete and status['status']=='stopped':raise RuntimeError('Stopped during finalization; evidence preserved')


def environment(expected=None):
    packages=('numpy','torch','torchvision','transformers','huggingface-hub','safetensors',
              'PyYAML','Pillow','scikit-learn','scipy','pytest')
    versions={name:importlib.metadata.version(name) for name in packages}
    if expected and versions!=expected:raise ValueError('Reused venv version mismatch: '+repr(versions))
    # Import the same mandatory modules without constructing models or downloading.
    import numpy,torch,torchvision,transformers,huggingface_hub,safetensors,yaml,PIL,sklearn,scipy,pytest
    return {'python':sys.version,'platform':platform.platform(),'packages':versions,'cuda_build':torch.version.cuda}


def cpu(args,protocol,config):
    # Separate CPU registration stage: no predictor or encoder imports/weights.
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    status={'started_utc':utc(),'stage':'step34_cpu_longgap','gpu_seconds':0,'test_or_confirmation_used':False}
    try:
        data=load('step34_data')
        manifest=data.generate_longgap(out/'data',args.dataset)
        audit=data.audit_longgap(out/'data/dataset',args.dataset)
        write(out/'audit.json',audit)
        status.update(status='passed' if audit['passed'] else 'stopped_audit_failure',
                      dataset_manifest_sha256=sha(out/'data/dataset/manifest.json'),
                      pairs=96,records=len(manifest['records']),protocol_sha256=sha(args.protocol))
        if not audit['passed']:raise ValueError('Long-gap pairing/leakage audit failed; no resampling or retry')
    except BaseException as exc:
        status.update(status='stopped',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        status['finished_utc']=utc();finish(out,status)


def validate_cpu_receipt(execution,root):
    receipt=execution['cpu_receipt'];root=Path(root).resolve()
    required={'status.json','audit.json','data/dataset/manifest.json'}
    if not required <= set(receipt['artifact_sha256']):raise ValueError('Incomplete committed CPU receipt inventory')
    for relative,checksum in receipt['artifact_sha256'].items():
        p=(root/relative).resolve();p.relative_to(root)
        if not p.is_file() or sha(p)!=checksum:raise ValueError('CPU diagnostic artifact changed: '+relative)
    status=read(root/'status.json');audit=read(root/'audit.json')
    if status['status']!='passed' or audit['passed'] is not True:raise ValueError('CPU audit did not pass')
    manifest=read(root/'data/dataset/manifest.json')
    if status['dataset_manifest_sha256']!=sha(root/'data/dataset/manifest.json'):
        raise ValueError('CPU status/manifest hash disagreement')
    if len(manifest['records'])!=192 or any(r['split']!='validation' for r in manifest['records']):
        raise ValueError('Only new validation data are authorized')
    return manifest


def encode_new(assets,root,manifest,output,guard):
    import numpy as np
    import torch
    core=load('step34_core');root=Path(root);output=Path(output)
    encoder_config=dict(assets.config['encoder'],local_files_only=True)
    # Existing cache is read in HF offline mode; no alternate model/download fallback.
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    encoder=assets.reader.load_reference('encoders').FrozenIJepa(encoder_config,assets.device)
    if (encoder.metadata['resolved_revision']!=REVISION or encoder.metadata['pooling']!='grid2'
            or encoder.metadata['dtype']!='float32'):
        raise ValueError('Encoder revision/precision/pooling mismatch')
    core.write_json(output/'encoder_metadata.json',encoder.metadata)
    episodes={};frames=0;total=sum(r['frames'] for r in manifest['records'])
    for rec in manifest['records']:
        if rec['split']!='validation' or rec['id'] in assets.store.records:
            raise ValueError('Only genuinely new validation frames may be encoded')
        source=root/rec['obs']
        if sha(source)!=rec['obs_sha256']:raise ValueError('New RGB hash differs')
        with np.load(source,allow_pickle=False) as f:
            if set(f.files)!={'rgb','actions','query'}:raise ValueError('Hidden encoder input')
            rgb=f['rgb']
        if rgb.shape!=(rec['frames'],96,96,3) or rgb.dtype!=np.uint8:raise ValueError('RGB shape/dtype differs')
        chunks=[]
        for at in range(0,len(rgb),2):
            guard();z=encoder.encode(rgb[at:at+2]);torch.cuda.synchronize()
            if z.shape!=(min(2,len(rgb)-at),5120) or z.dtype!=np.float32 or not np.isfinite(z).all():
                raise ValueError('Invalid frozen encoder output')
            chunks.append(z);frames+=len(z)
            # Encoding accounts for 80% of this phase; diagnostics take the rest.
            guard.progress(int(800*frames/total))
        relative='episodes/'+rec['id']+'.npz'
        core.write_npz(output/relative,features=np.concatenate(chunks))
        episodes[rec['id']]={'path':relative,'sha256':sha(output/relative),'frames':rec['frames']}
    del encoder
    torch.cuda.empty_cache()
    index={'manifest_sha256':sha(root/'manifest.json'),'feature_dim':5120,'episodes':episodes,
           'normalization':'unchanged original train-only normalization','basis':'unchanged original train-only basis',
           'new_frames_encoded':frames,'old_frames_reencoded':0}
    core.write_json(output/'index.json',index)
    return index


def gpu(args,protocol,config):
    if args.execution is None or args.cpu_receipt is None:raise ValueError('Committed execution and CPU receipt required')
    execution=read(committed(args.execution))
    if execution['science_protocol_sha256']!=sha(args.protocol):raise ValueError('Execution binds different science')
    if execution['test_status']!='passed':raise ValueError('Registered software tests must pass before A100')
    manifest=validate_cpu_receipt(execution,args.cpu_receipt)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    status={'started_utc':utc(),'stage':'steps3_4','status':'started','test_or_confirmation_used':False,
            'predictors_retrained':False,'existing_frames_reencoded':0,'science_protocol_sha256':sha(args.protocol)}
    guard=None
    try:
        import torch
        torch.set_num_threads(4);torch.set_num_interop_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        write(out/'environment.json',environment(config['gpu_environment']))
        runtime=load('step34_runtime')
        guard=runtime.BudgetGuard(out/'runtime')
        write(out/'visible_gpu.json',runtime.verify_gpu())
        guard.gpu_probe=runtime.verify_gpu
        guard.check(force_native=True)
        def interrupted(signum,frame):raise RuntimeError('Native signal stop '+str(signum))
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,interrupted)
        core=load('step34_core')
        expected=read(REPO/protocol['asset_inventory']['path'])['artifact_sha256']
        assets=core.FrozenAssets(args.assets,args.dataset,expected,'cuda:0',guard)
        write(out/'asset_reuse.json',{'original_job':16614,'files':len(expected),
              'artifact_sha256':expected,'predictor_seeds':[17,29,43],
              'new_model_fitting':'students only; no encoder/basis/predictor/position-decoder fitting'})
        guard.set_phase('initialization',1)
        guard.progress(1)
        guard.set_phase('encoding_diagnostics',1000)
        index=encode_new(assets,args.cpu_receipt/'data/dataset',manifest,out/'longgap_features',guard)
        external=(args.cpu_receipt/'data/dataset',out/'longgap_features',index)
        ordinary=core.diagnostic(assets,assets.store.manifest['records'],('b','d','b_random'),
                                 out/'diagnostics/ordinary','ordinary')
        guard.progress(900)
        longgap=core.diagnostic(assets,manifest['records'],('a','b','d'),out/'diagnostics/long_gap','long_gap',external)
        guard.progress(1000)
        guard.set_phase('labels',25*(16384+3072))
        label_done=0
        def label_progress(units):
            nonlocal label_done
            label_done+=units;guard.progress(label_done)
        train_rows,train_inputs=core.behavior_states(assets,'train',out/'labels/train',progress=label_progress)
        val_rows,val_inputs=core.behavior_states(assets,'validation',out/'labels/validation',progress=label_progress)
        train_labels=core.deletion_labels(assets,train_rows,out/'labels/train',progress=label_progress)
        val_labels=core.deletion_labels(assets,val_rows,out/'labels/validation',progress=label_progress)
        if label_done!=25*(16384+3072):raise ValueError('Label work accounting differs')
        guard.set_phase('students',54000)
        students=core.module('step34_students').fit_students(train_inputs,val_inputs,train_labels,val_labels,
                    [r['id'] for r in train_rows],[r['id'] for r in val_rows],out/'students',assets.device,guard)
        longgap_pass=longgap['long_gap_direction_check']['passed']
        online_total=core.online_work(assets.store.manifest['records'])+(core.online_work(manifest['records']) if longgap_pass else 0)
        guard.set_phase('online',online_total)
        online_done=0
        def online_progress(units):
            nonlocal online_done
            online_done+=units;guard.progress(online_done)
        normal_result=core.evaluate_online(assets,assets.store.manifest['records'],students,out/'students',
                                          out/'online/ordinary','ordinary',progress=online_progress)
        # Scope is determined by both preregistered direction lower bounds, never by student performance.
        if longgap_pass:
            core.evaluate_online(assets,manifest['records'],students,out/'students',out/'online/long_gap','long_gap',external,
                                 progress=online_progress)
        else:
            core.write_json(out/'online/long_gap_not_evaluated.json',{
                'reason':'B-b lower bound is nonpositive at one or both horizons','claim_scope':'ordinary_only'})
        if online_done!=online_total:raise ValueError('Online work accounting differs')
        guard.set_phase('finalize',1)
        assets.verify_unchanged()
        guard.check_native(force=True)
        status.update(status='completed_step4_stop',long_gap_online_evaluated=longgap_pass,
                      gate='validation_only_no_confirmatory_claim',budget=guard.snapshot(),
                      native_final_accounting='pending terminal settlement')
    except BaseException as exc:
        status.update(status='stopped',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        if guard is not None:status['budget']=guard.snapshot()
        raise
    finally:
        status['finished_utc']=utc();finish(out,status,guard)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('cpu','gpu'),required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--assets',type=Path)
    parser.add_argument('--execution',type=Path)
    parser.add_argument('--cpu-receipt',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();protocol,config=registration(args.protocol)
    if args.output.exists() or args.output.is_symlink():raise FileExistsError('Output already exists; no resume/rerun')
    if args.mode=='cpu':cpu(args,protocol,config)
    else:gpu(args,protocol,config)


if __name__=='__main__':main()
