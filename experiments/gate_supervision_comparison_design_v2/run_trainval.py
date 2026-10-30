"""Authorized Gate matched-scorer train/validation runner; never opens test sets."""
from __future__ import annotations
import argparse, hashlib, json, os, signal, sys, time
from pathlib import Path
import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path('/scratch/xl4358/clean_future_gate_20261022/code/CV_project')
BASE = PROJECT.parents[1]
RUN = Path('/scratch/xl4358/clean_future_gate_20261022/gate_supervision_v2_20261029/run_20261029')
for p in (PACKAGE_ROOT, PROJECT, PROJECT/'experiments/predictor/control/src', PROJECT/'experiments/retention_baseline_matrix', PROJECT/'experiments/retention_online_matrix'):
    sys.path.insert(0, str(p))
from experiments.gate_supervision_comparison_design_v2.artifacts import ASSET_LOCK, digest_file, read_state_pack, write_state_pack, read_teacher_pack, write_teacher_pack
from experiments.gate_supervision_comparison_design_v2.asset_audit import audit_frozen_assets, audit_feature_files
from experiments.gate_supervision_comparison_design_v2.pack import construct_states, VerifiedManifest, train_inputs, label_one_frozen_state
from experiments.gate_supervision_comparison_design_v2.states import iter_states, load_frozen_behavior
from experiments.gate_supervision_comparison_design_v2.teacher_stage import make_teacher_arrays_checkpointed
from experiments.gate_supervision_comparison_design_v2.checkpoint_fit import fit_pair_checkpointed
from experiments.gate_supervision_comparison_design_v2.core import EvictionScorer, paired_initialization
from phase1.inputs import FeatureStore, collate, selected_evidence
from phase1.data import load_episode
from phase1.training import load_predictor
from phase1.probes import load_decoder
from phase1.evaluation import pair_diagnostic
from phase1.utils import stable_seed

PKG = Path(__file__).resolve().parent
ONLINE = PACKAGE_ROOT/'experiments/gate_retention_comparison_v1/online.py'
stop = False

def on_signal(*_):
    global stop
    stop = True
for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, on_signal)

def write_json(path, data):
    path = Path(path); temp = path.with_suffix(path.suffix + '.partial')
    with temp.open('w') as out:
        json.dump(data, out, indent=2, sort_keys=True, allow_nan=False); out.write('\n'); out.flush(); os.fsync(out.fileno())
    os.replace(temp, path)

def progress(stage, **kw):
    item = {'stage': stage, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'job_id': os.getenv('SLURM_JOB_ID'), **kw}
    write_json(RUN/'progress.json', item)
    print(json.dumps(item), flush=True)

def source_signature():
    paths = sorted(PKG.glob('*.py')) + [PKG/'PROTOCOL.md', PKG/'EXECUTION_AMENDMENT_20261029.md', ONLINE]
    return hashlib.sha256(json.dumps([(p.name, digest_file(p)) for p in paths], separators=(',', ':')).encode()).hexdigest()

def initialize():
    locked = audit_frozen_assets(BASE/'independent_test_9137_v1/model_source_lock.json', PROJECT, BASE/'evaluation_pair_repair_10011/final_audit.json')
    c = json.loads((PROJECT/'results/gate_clean_future_20261022/resolved_config.json').read_text())
    if not (Path(c['output'])/'predictor/gate/basis.npz').is_file():
        raise ValueError('frozen basis missing; forbidden to fit')
    store = FeatureStore(c, max_cached=1216)
    with np.load(Path(c['output'])/'predictor/gate/basis.npz', allow_pickle=False) as f:
        store.bases = {'gate': {k:f[k] for k in f.files}}
    return c,store,locked

def lock():
    f=RUN/'execution_lock.json'; expected={'code_sha256':source_signature(),'protocol_sha256':digest_file(PKG/'PROTOCOL.md'), 'amendment_sha256':digest_file(PKG/'EXECUTION_AMENDMENT_20261029.md'), 'asset_lock_sha256':ASSET_LOCK, 'cap_gpu_seconds':3600}
    if f.exists():
        if json.loads(f.read_text()) != expected: raise ValueError('execution lock changed')
    else:
        with f.open('x') as out: json.dump(expected,out,sort_keys=True,indent=2)
    return expected

def prepare():
    RUN.mkdir(parents=True, exist_ok=True); code=lock(); c,store,audit=initialize()
    progress('verify_cache'); audit.update(audit_feature_files(PROJECT))
    update=load_frozen_behavior(ONLINE)
    for seed in (17,29,43):
        root=RUN/f'states_{seed}'
        if root.exists():
            _,_,seal=read_state_pack(root)
            if seal['implementation_sha256']!=code['code_sha256'] or seal['seed']!=seed: raise ValueError('pack differs from execution lock')
        else:
            progress('prepare_states',seed=seed)
            pack=construct_states(store,load_episode,update,seed)
            seal=write_state_pack(root,pack,seed=seed,asset_lock_sha256=ASSET_LOCK,protocol_sha256=code['protocol_sha256'],implementation_sha256=code['code_sha256'])
            print('duplicates',seed,pack['duplicate_identity_records'],flush=True)
        if stop: raise InterruptedError('stopped during state preparation')
    write_json(RUN/'prepare_complete.json', {'passed':True,'audit':audit,'code_sha256':code['code_sha256']})
    progress('prepared',gpu_started=False)

def measured_profile(store, model, update, device):
    rows, features, seal=read_state_pack(RUN/'states_17'); verified=VerifiedManifest(rows,seal['manifest_sha256'])
    records={r['id']:r for r in store.manifest['records']}; gen=iter_states(store,load_episode,update,17)
    start=time.perf_counter(); count=0
    for expected, expected_features in zip(rows,features):
        meta,state=next(gen)
        if meta!=expected or not np.array_equal(state.features(),expected_features): raise ValueError('profile state mismatch')
        if meta['split']!='train': continue
        label_one_frozen_state(store,records[meta['episode']],state,meta,verified,load_episode=load_episode,model=model,collate=collate,device=device)
        count+=1
        if count==32: break
    if count!=32: raise ValueError('profile requires 32 train states')
    torch.cuda.synchronize(); seconds=time.perf_counter()-start
    x=torch.as_tensor(features[:64],device=device); a,d=paired_initialization(17); a=a.to(device);d=d.to(device)
    optimizers=[torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for m in (a,d)]
    y=torch.full((len(x),3),1/3,device=device); start=time.perf_counter()
    for _ in range(10):
        for m,opt in zip((a,d),optimizers):
            opt.zero_grad(set_to_none=True);loss=-(y*torch.log_softmax(m(x),dim=-1)).sum(-1).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    torch.cuda.synchronize(); fit_seconds=(time.perf_counter()-start)/10
    del a,d,optimizers,x,y
    return {'train_states_profiled':32,'teacher_seconds':seconds,'paired_step_seconds':fit_seconds,'profile_uses_train_only':True}

def train():
    code=lock(); ready=json.loads((RUN/'prepare_complete.json').read_text())
    if not ready['passed'] or ready['code_sha256']!=code['code_sha256']: raise ValueError('unverified preparation')
    c,store,_=initialize(); update=load_frozen_behavior(ONLINE); device=torch.device('cuda'); torch.set_num_threads(4)
    model,_=load_predictor(c,17,store,device,'gate'); model.eval()
    progress('profile_32_train_states')
    profile=measured_profile(store,model,update,device)
    # Includes three full label packs and paired updates, with 50% safety plus
    # 300 s fixed startup, I/O and shutdown reserve. Validation runs CPU-only.
    estimate=1.5*(profile['teacher_seconds']/32*4864*3+profile['paired_step_seconds']*3000)+300
    profile.update({'estimated_gpu_seconds':estimate,'cap_gpu_seconds':3600,'validation_device':'cpu','estimate_safety_factor':1.5})
    write_json(RUN/'profile.json',profile); progress('profile_complete',**profile)
    if estimate>3480: raise RuntimeError('full run estimate exceeds budget; no teacher pack or training started')
    for seed in (17,29,43):
        if stop: raise InterruptedError('stop before seed')
        model,_=load_predictor(c,seed,store,device,'gate'); model.eval()
        states=RUN/f'states_{seed}'; teacher=RUN/f'teacher_{seed}'
        if not teacher.exists():
            progress('teacher',seed=seed,committed=0)
            arrays,timing=make_teacher_arrays_checkpointed(states,store,load_episode,update,model,collate,device,checkpoint=RUN/f'teacher_{seed}.checkpoint.npz',stop_requested=lambda:stop,progress=lambda s,n,p:progress('teacher',seed=s,committed=n))
            write_teacher_pack(teacher,states,**arrays,timing=timing)
        rows,features,arrays,seal=read_teacher_pack(teacher,states)
        manifest=VerifiedManifest(rows,seal['manifest_sha256']); x,a,d=train_inputs(rows,features,arrays['targets_a'],arrays['targets_d'],manifest,seed)
        sig=hashlib.sha256((code['code_sha256']+digest_file(states/'seal.json')+digest_file(teacher/'seal.json')+ASSET_LOCK).encode()).hexdigest()
        progress('paired_training',seed=seed,completed_steps=0)
        fit_pair_checkpointed(x,a,d,seed=seed,signature=sig,checkpoint=RUN/f'student_{seed}.pt',device=device,stop_requested=lambda:stop,progress=lambda s,n,p:progress('paired_training',seed=s,completed_steps=n))
    audit_frozen_assets(BASE/'independent_test_9137_v1/model_source_lock.json', PROJECT, BASE/'evaluation_pair_repair_10011/final_audit.json')
    write_json(RUN/'training_complete.json', {'passed':True,'seeds':[17,29,43],'steps_per_arm':1000,'code_sha256':code['code_sha256'],'test_accessed':False})
    progress('training_complete',validation_pending=True)

def validate():
    from experiments.gate_supervision_comparison_design_v2.validation import evaluate_full_validation
    import run_matrix as base, run_online_matrix as metrics
    import importlib.util
    spec=importlib.util.spec_from_file_location('v2_old_online',ONLINE); online=importlib.util.module_from_spec(spec); spec.loader.exec_module(online)
    code=lock(); done=json.loads((RUN/'training_complete.json').read_text())
    if not done['passed'] or done['code_sha256']!=code['code_sha256']: raise ValueError('training proof mismatch')
    c,store,_=initialize(); torch.set_num_threads(4); device=torch.device('cpu'); decoder=load_decoder(c,'gate'); contexts={}
    for seed in (17,29,43):
        model,_=load_predictor(c,seed,store,device,'gate'); st,_,_,seal=read_teacher_pack(RUN/f'teacher_{seed}',RUN/f'states_{seed}')
        sig=hashlib.sha256((code['code_sha256']+digest_file(RUN/f'states_{seed}/seal.json')+digest_file(RUN/f'teacher_{seed}/seal.json')+ASSET_LOCK).encode()).hexdigest()
        ck=torch.load(RUN/f'student_{seed}.pt',map_location='cpu',weights_only=False)
        if ck['signature']!=sig or ck['seed']!=seed or ck['step']!=1000: raise ValueError('student checkpoint incomplete')
        models={}
        for name,state in zip(('attention_proxy_student','deletion_cost_student'),ck['models']):
            m=EvictionScorer();m.load_state_dict(state);m.eval();models[name]=m
        contexts[seed]={'store':store,'predictor':model,'students':models,'decoder':decoder,'device':device}
    deps={'collate':collate,'load_episode':load_episode,'selected_evidence':selected_evidence,'baseline_replay':online.replay,'sample_with_indices':base.sample_with_indices,'append_metric_rows':metrics.append_metric_rows,'pair_diagnostic':pair_diagnostic}
    progress('validation_cpu')
    result=evaluate_full_validation(contexts,dependency_factory=lambda seed:deps,seed_factory=lambda seed,rec:stable_seed(2026102317,seed,rec['id'],'reservoir'))
    write_json(RUN/'validation_results.json',result)
    audit_frozen_assets(BASE/'independent_test_9137_v1/model_source_lock.json',PROJECT,BASE/'evaluation_pair_repair_10011/final_audit.json')
    progress('validation_complete',summary=result['summary'],test_accessed=False,original_gate_passed=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','train','validate'));args=p.parse_args()
    try: globals()[args.stage]()
    except BaseException as e:
        if RUN.exists(): progress('execution_error',requested_stage=args.stage,error=repr(e))
        raise
