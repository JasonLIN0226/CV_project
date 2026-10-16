"""Standalone v5 reader trial; old configuration is a read-only data contract."""
import copy
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .inputs import FeatureStore, collate
from .model import fit_basis
from .training import TRAIN_CONDITIONS, paired_rows, regression_loss, validation_loss
from .utils import (digest, file_hash, read_json, write_json, write_csv, save_torch,
                    seed_all, rng_state, restore_rng)
from .v5_synthetic_reader import SyntheticCausalReader


def cpu_tree(x):
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(cpu_tree(v) for v in x)
    return copy.deepcopy(x)


def finite_tree(x):
    if torch.is_tensor(x):
        return bool(torch.isfinite(x).all())
    if isinstance(x, dict):
        return all(finite_tree(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(finite_tree(v) for v in x)
    return True


def train_epoch(c, store, model, optimizer, pairs, seed, epoch, stop):
    model.train()
    rng = np.random.default_rng(seed + 100003 * epoch)
    order = rng.permutation(len(pairs))
    counts = {k: 0 for k in TRAIN_CONDITIONS}
    totals = {k: np.zeros(3, dtype=np.float64) for k in TRAIN_CONDITIONS}
    steps = 0
    for start in range(0, len(order), c['training']['batch_size'] // 2):
        selected = [pairs[i] for i in order[start:start+c['training']['batch_size']//2]]
        for condition in rng.permutation(TRAIN_CONDITIONS):
            samples = []
            for pair in selected:
                sample_seed = int(rng.integers(2**32))
                samples.extend(store.sample(r, str(condition), np.random.default_rng(sample_seed)) for r in pair)
            batch = collate([s[0] for s in samples], 'cpu')
            target = torch.from_numpy(np.stack([s[1] for s in samples]))
            optimizer.zero_grad(set_to_none=True)
            loss, mse, paired = regression_loss(model(batch), target, 2., .5)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite v5 synthetic loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c['training']['clip'], error_if_nonfinite=True)
            optimizer.step()
            n = len(samples)
            counts[str(condition)] += n
            totals[str(condition)] += np.array([float(loss.detach()), float(mse.detach()), float(paired.detach())]) * n
            steps += 1
            if stop['requested']:
                return None
    if any(n != 2*len(pairs) for n in counts.values()):
        raise RuntimeError('Incomplete balanced pair coverage')
    return {'steps': steps, 'counts': counts,
            'means': {k: (totals[k]/counts[k]).tolist() for k in TRAIN_CONDITIONS}}


def verify_contract(control, protocol):
    control = Path(control)
    p = read_json(protocol)
    if p['task'] != 'gate' or p['seeds'] != [17,29,43] or p['epochs'] != 120:
        raise ValueError('Fixed Gate development trial required')
    source = Path(p['data_source_config'])
    if file_hash(source) != p['data_source_config_sha256']:
        raise ValueError('Old data source config changed')
    c = read_json(source)
    expected = p['original_contract']
    for key in ('data','encoder','normalization','training','decision','evaluation'):
        if c[key] != expected[key]:
            raise ValueError('Original contract changed: '+key)
    if c['experiment'] != {'stage':'development','arm':'reference','tasks':['gate']}:
        raise ValueError('Only actual original Gate reference inputs allowed')
    verification = Path(p['frozen_v3_verification'])
    if file_hash(verification) != p['frozen_v3_verification_sha256']:
        raise ValueError('Actual v3 verification changed')
    verified = read_json(verification)
    old_root = Path(c['output'])
    for seed in p['seeds']:
        for name in ('best','latest'):
            path = old_root/'predictor/gate'/f'seed_{seed}'/(name+'.pt')
            if file_hash(path) != verified['verified_models']['gate/reference'][f'gate/{seed}'][name+'_sha256']:
                raise ValueError('Actual frozen reference weights changed')
    basis_path = old_root/'predictor/gate/basis.npz'
    if not basis_path.is_file() or file_hash(basis_path) != p['basis_sha256']:
        raise ValueError('Existing basis missing or changed; fitting forbidden')
    lock = read_json(control/'source_lock.json')
    for rel, sha in lock['files'].items():
        if file_hash(control/rel) != sha:
            raise ValueError('Source changed: '+rel)
    return c, p, lock


def run(control, protocol, output):
    started = time.perf_counter()
    c, p, lock = verify_contract(control, protocol)
    root = Path(output)
    if root.exists() and not (root/'run_contract.json').exists():
        raise ValueError('Output exists without a trial contract')
    root.mkdir(parents=True, exist_ok=True)
    seed_all(17, 4)
    store = FeatureStore(c)
    rows = [r for r in store.manifest['records'] if r['task']=='gate']
    train_pairs = paired_rows([r for r in rows if r['split']=='train'])
    val_pairs = paired_rows([r for r in rows if r['split']=='validation'])
    if (len(train_pairs),len(val_pairs)) != (512,96):
        raise ValueError('Original development pair counts changed')
    store.max_cached = 1216
    basis = fit_basis(c, store, 'gate')  # Existing path was required and SHA checked above.
    if set(basis['train_ids'].tolist()) != {r['id'] for pair in train_pairs for r in pair}:
        raise ValueError('Basis train IDs differ')
    store.bases = {'gate':basis}
    signature = {'protocol_sha256':file_hash(protocol), 'source_lock_sha256':file_hash(Path(control)/'source_lock.json'),
                 'data_config_sha256':p['data_source_config_sha256'], 'cache_signature':store.ix['signature'],
                 'basis_sha256':p['basis_sha256'], 'normalization_sha256':file_hash(Path(c['output'])/'features/normalization.npz'),
                 'reader':'SyntheticCausalReader', 'task':'gate', 'selection':'equal_four_condition_two_horizon_mse_earliest_minimum'}
    contract = {'signature':signature,'protocol':p,'source_files':lock['files'], 'phase1_go':False}
    if (root/'run_contract.json').exists() and read_json(root/'run_contract.json') != contract:
        raise ValueError('Resume contract differs')
    write_json(root/'run_contract.json',contract)
    stop = {'requested':False}
    def handler(signum, frame):
        stop['requested'] = True
    previous = {s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    for s in previous:
        signal.signal(s,handler)
    print('SOURCE_DATA_BASIS_REFERENCE_VERIFIED; beginning v5 Gate CPU training',flush=True)
    try:
        for seed in p['seeds']:
            seed_all(seed,4)
            model = SyntheticCausalReader(basis).to('cpu')
            opt = torch.optim.AdamW(model.parameters(),lr=c['training']['lr'],weight_decay=c['training']['weight_decay'])
            scheduler = torch.optim.lr_scheduler.StepLR(opt,c['training']['lr_step'],c['training']['lr_gamma'])
            target_root = root/'predictor/gate'/f'seed_{seed}'
            target_root.mkdir(parents=True,exist_ok=True)
            epoch,step,best,logs,seconds = 0,0,float('inf'),[],0.
            if (target_root/'latest.pt').exists():
                ck = torch.load(target_root/'latest.pt',map_location='cpu',weights_only=False)
                if ck['signature'] != signature or ck['seed'] != seed or ck['resume_granularity'] != 'epoch_boundary':
                    raise ValueError('Checkpoint provenance mismatch')
                if not finite_tree(ck['model']) or not finite_tree(ck['optimizer']):
                    raise ValueError('Non-finite checkpoint')
                model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
                restore_rng(ck['rng'])
                epoch,step,best,logs,seconds = (ck[k] for k in ('epoch','step','best','logs','seconds'))
            def snapshot():
                return {'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(opt.state_dict()),
                        'scheduler':scheduler.state_dict(),'rng':rng_state(),'epoch':epoch,'step':step,
                        'best':best,'logs':copy.deepcopy(logs),'seconds':seconds,'seed':seed,'task':'gate',
                        'signature':signature,'resume_granularity':'epoch_boundary','horizons':[3,6],
                        'feature_dim':len(store.norm['mean']),'checkpoint_selection':signature['selection']}
            if not (target_root/'latest.pt').exists():
                save_torch(target_root/'latest.pt',snapshot())
            while epoch < 120:
                boundary = snapshot()
                began = time.perf_counter()
                print(f'TRAINING seed={seed} epoch={epoch+1}/120',flush=True)
                result = train_epoch(c,store,model,opt,train_pairs,seed,epoch,stop)
                if result is None:
                    boundary['discarded_partial_epoch'] = epoch
                    save_torch(target_root/'latest.pt',boundary)
                    return {'status':'INTERRUPTED','seed':seed,'restart_epoch':epoch,'phase1_go':False}
                val = validation_loss(c,store,model,val_pairs,'cpu','gate')
                score = float(np.mean([val[k]['mse'] for k in TRAIN_CONDITIONS]))
                if not np.isfinite(score):
                    raise RuntimeError('Non-finite validation')
                elapsed = time.perf_counter()-began
                scheduler.step(); epoch += 1; step += result['steps']; seconds += elapsed
                logs.append({'epoch':epoch,'step':step,'validation_mse':score,'epoch_seconds':elapsed,'seconds':seconds,
                             'mix_counts':str(result['counts']), 'lr':opt.param_groups[0]['lr'],
                             **{f'train_{k}_{name}':v[i] for k,v in result['means'].items() for i,name in enumerate(('loss','mse','pair_mse'))},
                             **{f'val_{k}_{name}':v for k,means in val.items() for name,v in means.items()}})
                improved = score < best; best = min(best,score)
                ck = snapshot()
                if not finite_tree(ck['model']) or not finite_tree(ck['optimizer']):
                    raise RuntimeError('Non-finite actual weights/optimizer')
                save_torch(target_root/'latest.pt',ck)
                if improved: save_torch(target_root/'best.pt',ck)
                write_csv(target_root/'training.csv',logs)
                write_json(target_root/'training.json',{'signature':signature,'seed':seed,'logs':logs,'phase1_go':False})
                print(f'COMPLETE_EPOCH seed={seed} epoch={epoch} step={step} validation_mse={score:.8f} seconds={elapsed:.2f}',flush=True)
                if seed == 17 and epoch == 1:
                    projection = time.perf_counter()-started + elapsed*359
                    write_json(root/'first_epoch_budget.json',{'observed_epoch_seconds':elapsed,'projected_total_seconds':projection,
                               'soft_budget_seconds':p['soft_budget_seconds'],'passed':projection<=p['soft_budget_seconds'], 'science_gate':False})
                    if projection > p['soft_budget_seconds']:
                        return {'status':'COMPUTE_BUDGET_STOP','models_complete':False,'phase1_go':False}
                if stop['requested']:
                    return {'status':'INTERRUPTED','seed':seed,'restart_epoch':epoch,'phase1_go':False}
        proofs = {}
        for seed in p['seeds']:
            target_root = root/'predictor/gate'/f'seed_{seed}'
            latest = torch.load(target_root/'latest.pt',map_location='cpu',weights_only=False)
            best_ck = torch.load(target_root/'best.pt',map_location='cpu',weights_only=False)
            if latest['signature'] != signature or best_ck['signature'] != signature or latest['epoch'] != 120 or latest['step'] != 7680:
                raise ValueError('Incomplete current weights')
            scores = [r['validation_mse'] for r in latest['logs']]
            if best_ck['epoch'] != scores.index(min(scores))+1 or not finite_tree(latest['model']) or not finite_tree(best_ck['model']):
                raise ValueError('Best selection or finite proof failed')
            proofs[str(seed)] = {'best_sha256':file_hash(target_root/'best.pt'),'latest_sha256':file_hash(target_root/'latest.pt'),
                                'epoch':120,'steps':7680,'best_epoch':best_ck['epoch'],'finite':True,'signature':signature}
        write_json(root/'all_models_completion.json',{'complete':True,'models':proofs,'phase1_go':False,'held_out_evaluation_executed':False})
        return {'status':'TRAINING_COMPLETE_EVALUATION_PENDING','phase1_go':False}
    finally:
        for s,old in previous.items(): signal.signal(s,old)
