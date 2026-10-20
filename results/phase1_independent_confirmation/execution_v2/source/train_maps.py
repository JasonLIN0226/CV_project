"""Fresh TRAINING-only observation maps using the fixed successful recipe.

CPU preparation is cache reading and deterministic feature/geometry arithmetic.
Every scientific regression solve is CUDA-only. No test or validation target is
used to choose the bandwidth, regularizer, map, or fixed fusion coefficient.
"""
from pathlib import Path
import numpy as np
import torch

from core import maps
from guard import allow_update
from kernel import fit, predict, gram
from locked_inference import motion_context_features
from null_basis_original import null_basis
from staged_data import StageFeatureStore, verify_permit
from phase1.data import load_episode
from phase1.model import fit_basis
from phase1.probes import LinearReadout
from phase1.training import paired_rows
from phase1.utils import file_hash, read_json, write_json


def prepare(c, permit):
    verify_permit(c, permit)
    store = StageFeatureStore(c, max_cached=16)
    basis = fit_basis(c, store, 'motion')
    store.bases = {'motion': basis}
    decoder = LinearReadout.load(Path(c['output'])/'probes/motion_state_decoder.npz')
    B,A,b,right,_ = maps(basis['output_basis'], decoder.weight, decoder.std, decoder.mean, decoder.bias)
    N = null_basis(B,A)
    pairs = paired_rows([r for r in store.manifest['records'] if r['task']=='motion' and r['split']=='train'])
    if len(pairs) != 512 or N.shape != (5120,126):
        raise ValueError('Original full512 training pairs and128-output basis required')
    X,Y,ids,episodes = [],[],[],[]
    for pair in pairs:
        ids.append(pair[0]['pair'])
        for rec in pair:
            sample,target,info = store.sample(rec,'privileged_small_budget')
            times = tuple(map(int,info['indices']))
            own = load_episode(c,rec,keys=('rgb','actions'))
            if not allow_update(own['rgb'][list(times)],own['rgb'][rec['query']]):
                raise ValueError('Fixed training-map recipe requires its two visible observations')
            X.append(motion_context_features(sample,times,decoder,own['actions'],rec['query']))
            Y.append((target.astype(float)-sample['current'][None].astype(float))@N)
            episodes.append(rec['id'])
    foldorder=np.random.default_rng(0).permutation(512)
    fold=np.empty(512,int);fold[foldorder[:256]]=0;fold[foldorder[256:]]=1
    root=Path(c['output'])/'observation_maps'
    root.mkdir(parents=True,exist_ok=False)
    np.savez(root/'train_pack.npz',X=np.stack(X),Y=np.stack(Y),N=N,A=A,b=b,
        fold=np.repeat(fold,2),pair_ids=np.array(ids),episodes=np.array(episodes))
    write_json(root/'preparation.json',dict(status='TRAIN_ONLY_MAP_INPUTS_PREPARED',
        pairs=512,episodes=1024,horizons=[3,6],input_dimensions=2,target_dimensions=126,
        source_split='train',train_pack_sha256=file_hash(root/'train_pack.npz'),
        basis_sha256=file_hash(Path(c['output'])/'predictor/motion/basis.npz'),
        decoder_sha256=file_hash(Path(c['output'])/'probes/motion_state_decoder.npz'),
        no_validation_or_test_target=True,no_model_forward=True,
        input='Observed-retained-frame decoded positions plus known actions only',
        target='Training future-minus-current feature projected into frozen readout-null basis',
        fixed_fusion_weight=.5))


def fit_cuda(c, permit, sequence_guard, allocation_start):
    verify_permit(c,permit)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0):
        raise ValueError('One actual A100 required; no CPU scientific fitting')
    root=Path(c['output'])/'observation_maps'
    prep=read_json(root/'preparation.json')
    if file_hash(root/'train_pack.npz')!=prep['train_pack_sha256']:
        raise ValueError('Prepared train-only map input changed')
    with np.load(root/'train_pack.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    if d['X'].shape!=(1024,2,2) or d['Y'].shape!=(1024,2,126):
        raise ValueError('Fixed complete observation-map fitting dimensions changed')
    out=root/'fitted';out.mkdir(exist_ok=False)
    cells=[]
    for hi,h in enumerate((3,6)):
        before=np.empty(1024);after=np.empty(1024)
        for held in (0,1):
            sequence_guard.check(allocation_start)
            tr=d['fold']!=held;te=~tr
            head=fit(torch.tensor(d['X'][tr,hi],device='cuda'),torch.tensor(d['Y'][tr,hi],device='cuda'))
            head['N']=d['N'];np.savez(out/f'cv_{h}_{held}.npz',**head)
            prediction=predict(d['X'][te,hi],head)
            before[te]=np.mean((d['Y'][te,hi]-head['target_mean'])**2,1)
            after[te]=np.mean((d['Y'][te,hi]-prediction)**2,1)
        gains=(before-after).reshape(512,2).mean(1)
        ix=np.random.default_rng(0).integers(0,512,(2000,512))
        lo,high=np.quantile(gains[ix].mean(1),[.025,.975])
        cells.append(dict(horizon=h,constant_innovation_MSE=float(before.mean()),
            map_innovation_MSE=float(after.mean()),relative_reduction=float(1-after.mean()/before.mean()),
            pair_gain_lower95=float(lo),pair_gain_upper95=float(high)))
    supported=all(v['pair_gain_lower95']>0 for v in cells)
    write_json(root/'training_crossfit.json',dict(status='SUPPORTED' if supported else 'REJECTED',
        cells=cells,train_pairs=512,paired_folds=2,fold_seed=0,
        no_hyperparameter_or_fusion_weight_search=True,no_validation_or_test=True))
    if not supported:
        raise RuntimeError('Fresh training crossfit rejected fixed observation map; no retuning')
    for hi,h in enumerate((3,6)):
        sequence_guard.check(allocation_start)
        head=fit(torch.tensor(d['X'][:,hi],device='cuda'),torch.tensor(d['Y'][:,hi],device='cuda'))
        head['N']=d['N'];np.savez(out/f'head_{h}.npz',**head)
    write_json(root/'fitted_manifest.json',dict(files={p.name:file_hash(p) for p in out.glob('*.npz')},
        training_pack_sha256=prep['train_pack_sha256'],heads=2,shared_across_base_seeds=True,
        fusion_weight=.5,all_tests_closed=True))


def audit_fits(c):
    """Independent equation replay, not a CPU regression fit."""
    root=Path(c['output'])/'observation_maps'
    manifest=read_json(root/'fitted_manifest.json')
    with np.load(root/'train_pack.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    worst=0.
    for name,sha in manifest['files'].items():
        path=root/'fitted'/name
        if file_hash(path)!=sha:raise ValueError('Fresh map checkpoint changed')
        with np.load(path,allow_pickle=False) as f:head={k:f[k] for k in f.files}
        h=int(name.split('_')[1].split('.')[0]);hi=(3,6).index(h)
        if name.startswith('cv_'):
            held=int(name.split('_')[2].split('.')[0]);tr=d['fold']!=held
        else:tr=np.ones(1024,dtype=bool)
        x,y=d['X'][tr,hi],d['Y'][tr,hi]
        if not np.array_equal(head['train_X'],x) or not np.array_equal(head['N'],d['N']):
            raise ValueError('Checkpoint input provenance changed')
        k=gram(x,x,head);kc=k-k.mean(0)[None]-k.mean(1)[:,None]+k.mean()
        residual=(kc+float(head['alpha'])*np.eye(len(k)))@head['coef']-(y-y.mean(0))
        err=float(abs(residual).max());worst=max(worst,err)
        if err>1e-7 or float(head['effective_df'])>32.000001 or abs(head['N'].T@d['A']).max()>=1e-10:
            raise ValueError('Independent fresh KKT/df/position-null verification failed')
    result=dict(status='PASS_INDEPENDENT_TRAIN_ONLY_MAP_KKT',systems=6,max_error=worst,
        manifest_sha256=file_hash(root/'fitted_manifest.json'),CPU_scientific_fitting=False)
    write_json(root/'independent_fit_audit.json',result)
    return result
