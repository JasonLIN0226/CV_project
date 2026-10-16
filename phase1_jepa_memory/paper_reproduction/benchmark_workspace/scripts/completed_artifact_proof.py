"""Read-only numeric/binding proof for actual completed train-only artifacts."""
from pathlib import Path
import torch
from completed_predictor_proof import sha


def check_payloads(basis, feature, label, *, model_hash, manifest_hash, expected_states=1856):
    if basis['fit_split']!='train' or basis['validation_features_accessed'] or basis['test_features_accessed']:
        raise RuntimeError('basis trained outside the permitted split')
    if basis['predictor_hash']!=model_hash or len(basis['participants'])!=29 or len(set(basis['participants']))!=29:
        raise RuntimeError('basis model/participant binding differs')
    b=basis['basis'];ev=basis['eigenvalues']
    if b.shape!=(512,32) or ev.shape!=(32,) or (basis['short_lag'],basis['long_lag'],basis['regularization'])!=(1,32,.001):
        raise RuntimeError('fixed slow basis dimensions/config differ')
    if not torch.isfinite(b).all() or not torch.isfinite(ev).all():raise RuntimeError('nonfinite slow basis')
    if not torch.allclose(b.T@b,torch.eye(32,dtype=b.dtype),atol=1e-5,rtol=1e-5):raise RuntimeError('slow basis is not orthonormal')
    if torch.any(ev[1:]>ev[:-1]+1e-6):raise RuntimeError('basis eigenvalue ordering differs')
    for payload in (feature,label):
        meta=payload['metadata']
        if meta['predictor_hash']!=model_hash or meta['manifest_hash']!=manifest_hash or meta['split']!='train':
            raise RuntimeError('bank model/data/split binding differs')
        if meta['validation_features_accessed'] or meta['test_features_accessed'] or meta['states']!=expected_states:
            raise RuntimeError('bank full training-only count differs')
        if meta['memory_budget']!=16 or meta['local_context_len']!=8 or meta['states_per_participant']!=64:
            raise RuntimeError('bank paper dimensions/sampling differ')
    x=feature['features'];t=label['tensors']
    if x.shape!=(expected_states,17,3073) or t['utility_scalar'].shape!=(expected_states,17):
        raise RuntimeError('bank feature/action count differs')
    if t['proposal_rollout_delta'].shape!=(expected_states,1,17) or t['reference_evict_index'].shape!=(expected_states,):
        raise RuntimeError('bank rollout/reference axes differ')
    for value in [x,feature['targets'],*t.values()]:
        if not isinstance(value,torch.Tensor) or not torch.isfinite(value).all():raise RuntimeError('nonfinite or non-tensor bank payload')
    if (t['reference_evict_index']<0).any() or (t['reference_evict_index']>16).any():raise RuntimeError('invalid reference action')
    if not torch.equal(feature['targets'],t['proposal_rollout_delta'][:,0]):raise RuntimeError('bank feature and label target mismatch')
    return {'states':expected_states,'actions':17,'feature_dimension':3073,'basis_rank':32,
            'all_finite':True,'train_only':True,'model_hash':model_hash,'manifest_hash':manifest_hash}


def completed_artifact_proof(report, proof, output):
    result=[];output=Path(output)
    for seed in (0,1,2):
        model=next(r for r in proof['models'] if r['seed']==seed and r['tag']=='best')
        folder=output/f'seed_{seed}';bp=folder/'slow_basis.pt';before=sha(bp)
        basis=torch.load(bp,map_location='cpu',weights_only=True)
        if sha(bp)!=before:raise RuntimeError('completed basis changed while reading')
        for method in ('fifo','reservoir'):
            fp=folder/'banks'/f'{method}_features.pt';lp=folder/'banks'/f'{method}_r32_delta.pt'
            before_f,before_l=sha(fp),sha(lp)
            feature=torch.load(fp,map_location='cpu',weights_only=True);label=torch.load(lp,map_location='cpu',weights_only=True)
            row=check_payloads(basis,feature,label,model_hash=model['model_hash'],manifest_hash=proof['manifest_sha256'])
            if feature['metadata']['source_labels_sha256']!=before_l or sha(fp)!=before_f or sha(lp)!=before_l:
                raise RuntimeError('completed bank feature/label boundary changed')
            row.update({'seed':seed,'method':method,'basis_sha256':before,'feature_sha256':before_f,'label_sha256':before_l})
            result.append(row);del feature,label
    if len(result)!=6:raise RuntimeError('all six bank/basis model bindings required')
    return {'status':'SIX_CURRENT_TRAIN_ONLY_BASIS_BANKS_NUMERICALLY_PROVED','actual_artifact_job_id':report['job_id'],'rows':result,
            'heldout_inference_before_proof':False,'test_data_used':False}
