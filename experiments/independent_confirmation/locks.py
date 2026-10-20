"""Current native model/transform lock required before ANY fresh test generation."""
import copy
from pathlib import Path
import numpy as np
import torch

from phase1.model import DynamicsPredictor
from phase1.proposal_trial import verify_completed_model
from phase1.utils import read_json, write_json, file_hash, digest

EXPECTED = {f'{role}/{task}/{seed}' for role in ('reference','candidate')
            for task in ('gate','motion') for seed in (17,29,43)}


def verify_all_lock(c, permit, lock):
    if (lock.get('status')!='ALL_TWELVE_MODELS_LOCKED'
            or set(lock.get('models',{}))!=EXPECTED
            or lock.get('required_epochs')!=120 or lock.get('test_opened') is not False
            or lock.get('canonical_config_digest')!=digest(c)
            or lock.get('permit_sha256')!=file_hash(permit)
            or lock.get('observation_map_count')!=2 or lock.get('fixed_fusion_weight')!=.5
            or lock.get('reference_and_candidate_roles')!=['reference','candidate']):
        raise ValueError('Complete fresh12-model/2-map lock contract absent or changed')
    for key,proof in lock['models'].items():
        role,task,seed=key.split('/')
        sig=proof['signature']
        if (not proof['complete'] or proof['completed_epochs']!=120 or proof['final_step']!=7680
                or sig['role']!=role or sig['task']!=task or sig['seed']!=int(seed)
                or sig['canonical_config_digest']!=digest(c)
                or sig['permit_sha256']!=file_hash(permit)):
            raise ValueError('A model is incomplete or not trained under this fresh protocol')
        expected=Path(c['output'])/'models'/role/task/f'seed_{seed}'
        if proof['best_sha256']!=lock['files'].get(str(expected/'best.pt')):
            raise ValueError('Wrong model role/path/current checksum')
    for name,sha in lock['files'].items():
        if file_hash(name)!=sha:raise ValueError('Frozen model/transform changed: '+name)
    return True


def create_all_lock(c,permit):
    # This stage is CPU-only and independently reloads every actual checkpoint.
    root=Path(c['output']);path=root/'all_models_lock.json'
    if path.exists():raise ValueError('Refuse implicit relocking or replacing an existing confirmation lock')
    if (root/'confirmation_test_generation_attempt.json').exists():
        raise ValueError('Tests must not have been generated/opened before all models are locked')
    budget=read_json(root/'sequence_budget.json')
    if set(budget['epochs'])!=EXPECTED or any(v!=120 for v in budget['epochs'].values()):
        raise ValueError('All12 full native training boundaries required')
    receipt=read_json(root/'training_terminal.json')
    if receipt.get('state')!='COMPLETED' or receipt.get('exitcode')!='0:0':
        raise ValueError('Actual training Slurm terminal completion required')
    from train_maps import audit_fits
    audit_fits(c)
    models,files={},{}
    def bind(p):files[str(p)]=file_hash(p)
    for task in ('gate','motion'):
        folder=root/'predictor'/task
        bind(folder/'basis.npz');bind(folder/'basis.json')
        with np.load(folder/'basis.npz',allow_pickle=False) as f:basis={k:f[k] for k in f.files}
        for seed in (17,29,43):
            reference=root/'models/reference'/task/f'seed_{seed}'
            ck=torch.load(reference/'best.pt',map_location='cpu',weights_only=False)
            frozen=ck['model']
            original=DynamicsPredictor(len(basis['input_mean']),c['model'],(3,6),basis)
            original.load_state_dict(frozen,strict=True)
            for role in ('reference','candidate'):
                r=root/'models'/role/task/f'seed_{seed}'
                latest=torch.load(r/'latest.pt',map_location='cpu',weights_only=False)
                sig=latest['signature']
                proof=verify_completed_model(r,sig,frozen if role=='candidate' else None)
                if role=='candidate' and sig['fresh_reference_best_sha256']!=file_hash(reference/'best.pt'):
                    raise ValueError('Candidate reference origin is not this fresh same-seed model')
                models[f'{role}/{task}/{seed}']=proof
                for name in ('best.pt','latest.pt','training.json','training.csv','completion_proof.json'):
                    bind(r/name)
    for p in (root/'probes').glob('*.npz'):bind(p)
    bind(root/'features/normalization.npz');bind(root/'features/normalization.json')
    bind(root/'probes/train_validation_readout_lock.json')
    for p in (root/'observation_maps/fitted').glob('*.npz'):bind(p)
    bind(root/'observation_maps/fitted_manifest.json');bind(root/'observation_maps/independent_fit_audit.json')
    lock=dict(status='ALL_TWELVE_MODELS_LOCKED',models=models,files=files,required_epochs=120,
        test_opened=False,reference_and_candidate_roles=['reference','candidate'],
        observation_map_count=2,fixed_fusion_weight=.5,
        canonical_config_digest=digest(c),permit_sha256=file_hash(permit),
        new_train_only_transforms=True,development_models_used=False)
    verify_all_lock(c,permit,lock)
    write_json(path,lock)
    return lock
