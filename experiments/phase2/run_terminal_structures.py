"""Frozen staged terminal R/D/S reader, baselines and student-export pipeline.

No stage silently opens confirmation. profile is generator-only and does not
encode/train. freeze records exact source/config before prepare creates data.
Reader input construction and target-free scoring have separate interfaces.
Future labels/semantic metadata are never passed to a retention selector.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments/predictor/control/src')]
H = (3, 6)
TASKS = ('R', 'D', 'S')


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def emit(stage, **kw):
    print(json.dumps({'stage': stage, **kw}), flush=True)


def source_files():
    names = ['experiments/phase2/run_terminal_structures.py',
             'experiments/phase2/terminal_structures.py',
             'experiments/phase2/prediction_retention.py',
             'experiments/phase2/memory.py',
             'experiments/phase2/encode_fresh_gate_diagnostic.py',
             'experiments/phase2/run_gate2_pilot.py',
             'experiments/phase2/run_reader_scale_probe.py',
             'experiments/phase2/fit_terminal_students.py',
             'experiments/phase2/summarize_terminal_students.py',
             'experiments/phase2/submit_terminal_structures.py',
             'docs/TERMINAL_STRUCTURES_PREREGISTRATION_20261025.md']
    names += ['experiments/predictor/control/src/phase1/'+x for x in
              ['model.py', 'inputs.py', 'utils.py', 'config.py', 'encoders.py']]
    return {n: sha(ROOT/n) for n in names}


def locked(out):
    p = read(out/'protocol.json')
    if p['source_sha256'] != source_files():
        raise ValueError('Frozen source changed')
    return p['config']


def freeze(config_path, out):
    if out.exists() and any(out.iterdir()):
        raise ValueError('freeze requires empty output')
    out.mkdir(parents=True, exist_ok=True)
    c = read(config_path)
    if c['horizons'] != list(H) or c['candidate_count'] != 3 or c['persistent_K'] != 2:
        raise ValueError('Unsupported scientific interface')
    write(out/'protocol.json', {'config': c, 'config_file_sha256': sha(config_path),
          'source_sha256': source_files(), 'frozen_unix': time.time(),
          'full_experiment_data_generated': False, 'confirmation_opened': False,
          'profile_scope': 'generator-only sizing may be run separately; this lock does not claim no prior structural preflight'})
    emit('frozen', protocol_sha256=sha(out/'protocol.json'))


def make_banks(c):
    from experiments.phase2 import terminal_structures as env
    tc = c['splits']['train']
    families = [env.family(tc['seed'], i, 'train') for i in range(tc['families'])]
    selected = np.random.default_rng(c['reference_plan_seed']).choice(len(families), c['reference_plans'], replace=False)
    refs, sources = [], []
    for i in selected:
        r = env.records_for_family(families[int(i)])[0]
        refs.append(env.reference_plan(r)); sources.append(r['family'])
    evaluation = np.random.default_rng(c['evaluation_plan_seed']).uniform(c['action_prior_uniform_low'],
        c['action_prior_uniform_high'],(c['evaluation_plans'],6,2)).astype(np.float32)
    return {'reference': np.asarray(refs, np.float32), 'evaluation': np.asarray(evaluation, np.float32)}, sources


class RGBRegistry:
    """Stream unique RGB to disk; keep only hashes/indices in RAM."""
    def __init__(self, path):
        self.path, self.lookup, self.frames = path, {}, 0
        self.file = path.open('wb')
        self.shape = None

    def add(self, frame):
        x = np.asarray(frame)
        if x.dtype != np.uint8 or x.shape != (96, 96, 3):
            raise ValueError('Expected uint8 96x96 RGB')
        key = hashlib.sha256(x.tobytes()).hexdigest()
        if key not in self.lookup:
            self.lookup[key] = self.frames
            self.file.write(x.tobytes()); self.frames += 1
        return self.lookup[key]

    def close(self):
        self.file.close()


def fifo_parent_trace(query):
    memory=[];trace=[]
    for t in range(query+1):
        incoming=t-4
        if incoming<0:continue
        before=list(memory);memory.append(incoming);dropped=None
        if t<query and len(memory)>2:dropped=memory.pop(0)
        trace.append({'time':t,'before':before,'incoming':incoming,'prefix_deleted':dropped,'after':list(memory)})
    if memory!=[query-6,query-5,query-4]:raise ValueError('Actual FIFO prefix does not form terminal parent')
    return trace


def generate(c, data, splits, profile=False, existing_banks=None):
    from experiments.phase2 import terminal_structures as env
    if data.exists() and any(data.iterdir()):
        raise ValueError('Do not overwrite generated dataset')
    data.mkdir(parents=True, exist_ok=True)
    banks, sources = make_banks(c) if existing_banks is None else (existing_banks, [])
    np.savez(data/'action_banks.npz', **banks)
    write(data/'action_bank_sources.json', {'reference_train_families': sources,
          'reference_seed': c['reference_plan_seed'], 'evaluation_seed': c['evaluation_plan_seed'],
          'same_reference_bank_all_parents': True})
    rgb = RGBRegistry(data/'unique_rgb.bin')
    records, actions, futures = [], {}, {}
    counts = {};prefix_traces={}
    try:
        for split in splits:
            n = 1 if profile else c['splits'][split]['families']
            before = rgb.frames
            for index in range(n):
                family = env.family(c['splits'][split]['seed'], index, split)
                prefix_traces[family['family_id']]=fifo_parent_trace(family['query'])
                for r in env.records_for_family(family):
                    ep = env.episode(r)
                    q = int(r['query'])
                    if ep['actions'].shape != (len(ep['rgb'])-1, 2) or np.any(ep['actions'][:q]):
                        raise ValueError('Executed actions/prefix contract')
                    if list(r['evidence']) != [q-6, q-5, q-4]:
                        raise ValueError('Terminal candidate times differ')
                    r['rgb_indices'] = [rgb.add(f) for f in ep['rgb']]
                    actions[r['id']] = ep['actions']
                    use = ('reference',) if split == 'train' else (('evaluation',) if split == 'validation' else ('reference', 'evaluation'))
                    r['plan_targets'] = {}
                    for name in use:
                        key = (r['family'], r['task_family'], r['outcome_class'], name)
                        if key not in futures:
                            futures[key] = [[rgb.add(x) for x in env.target_frames(r, plan)] for plan in banks[name]]
                        r['plan_targets'][name] = futures[key]
                    records.append(r)
                emit('generating', split=split, family=index+1, total=n, unique_rgb=rgb.frames)
            counts[split] = {'families_generated': n, 'records': sum(r['split'] == split for r in records),
                             'additional_unique_rgb': rgb.frames-before}
    finally:
        rgb.close()
    write(data/'records.json', records)
    write(data/'prefix_traces.json',prefix_traces)
    np.savez_compressed(data/'actions.npz', **actions)
    write(data/'rgb_index.json', rgb.lookup)
    info = {'unique_rgb': rgb.frames, 'rgb_shape': [rgb.frames, 96, 96, 3],
            'rgb_bytes': rgb.frames*96*96*3, 'one_feature_array_bytes': rgb.frames*5120*4,
            'estimated_raw_plus_normalized_feature_bytes': rgb.frames*5120*8,
            'profile_only': profile, 'counts': counts,
            'records_sha256': sha(data/'records.json'), 'action_banks_sha256': sha(data/'action_banks.npz')}
    if profile:
        estimate = sum(counts[s]['additional_unique_rgb']*c['splits'][s]['families'] for s in splits)
        info['linear_profile_unique_rgb_estimate'] = estimate
        info['linear_profile_disk_bytes_rgb_plus_two_features'] = estimate*(96*96*3+5120*8)
        info['estimate_limit'] = 'one family per split; different appearance/geometry may change deduplication'
    write(data/'generation.json', info)
    emit('prepared', **info)


def encode(c, data, cache, device, normalization_source=None):
    from phase1.encoders import FrozenIJepa
    from experiments.phase2.encode_fresh_gate_diagnostic import ENCODER_CONFIG, EXPECTED_ENCODER_METADATA_SHA256, digest
    import torch
    if (data/'features.npy').exists():
        raise ValueError('Do not overwrite encoded features')
    info = read(data/'generation.json')
    if info['profile_only']:
        raise ValueError('Profile data cannot be encoded as experiment')
    records = read(data/'records.json')
    bank = np.memmap(data/'unique_rgb.bin', dtype=np.uint8, mode='r', shape=tuple(info['rgb_shape']))
    raw = np.lib.format.open_memmap(data/'features_raw.npy', mode='w+', dtype=np.float32, shape=(len(bank), 5120))
    enc = FrozenIJepa(dict(ENCODER_CONFIG, cache_dir=str(cache), batch_size=c['encoder_batch_size']), torch.device(device))
    if digest(enc.metadata) != EXPECTED_ENCODER_METADATA_SHA256:
        raise ValueError('Pinned encoder metadata mismatch')
    began = time.perf_counter()
    for at in range(0, len(bank), c['encoder_batch_size']):
        z = enc.encode(np.array(bank[at:at+c['encoder_batch_size']]))
        if z.shape[1:] != (5120,) or not np.isfinite(z).all():
            raise ValueError('Invalid encoded features')
        raw[at:at+len(z)] = z
        if at % (c['encoder_batch_size']*25) == 0:
            emit('encoding', done=at+len(z), total=len(bank))
    raw.flush()
    if normalization_source is None:
        counts = np.zeros(len(bank), np.int64)
        for r in records:
            if r['split'] == 'train':
                np.add.at(counts, r['rgb_indices'][:r['query']+1], 1)
        if counts.sum() == 0:
            raise ValueError('TRAIN prefixes absent for normalization')
        mean, second = np.zeros(5120, np.float64), np.zeros(5120, np.float64)
        for at in range(0, len(raw), 1024):
            x = np.asarray(raw[at:at+1024], np.float64); w = counts[at:at+len(x), None]
            mean += (w*x).sum(0); second += (w*x*x).sum(0)
        mean /= counts.sum(); second /= counts.sum()
        std = np.maximum(np.sqrt(np.maximum(second-mean*mean, 0)), c['normalization_std_floor'])
        np.savez(data/'normalization.npz', mean=mean.astype(np.float32), std=std.astype(np.float32), train_frame_counts=counts)
    else:
        a = np.load(normalization_source, allow_pickle=False)
        mean, std = a['mean'], a['std']
        np.savez(data/'normalization.npz', mean=mean, std=std)
    a = np.load(data/'normalization.npz', allow_pickle=False); mean, std = a['mean'], a['std']
    normalized = np.lib.format.open_memmap(data/'features.npy', mode='w+', dtype=np.float32, shape=raw.shape)
    for at in range(0, len(raw), 1024):
        normalized[at:at+1024] = (raw[at:at+1024]-mean)/std
        if not np.isfinite(normalized[at:at+1024]).all():
            raise ValueError('Nonfinite normalized features')
    normalized.flush()
    write(data/'encoding.json', {'seconds': time.perf_counter()-began, 'metadata': enc.metadata,
          'metadata_sha256': digest(enc.metadata), 'features_sha256': sha(data/'features.npy'),
          'normalization_sha256': sha(data/'normalization.npz'), 'normalization_source': str(normalization_source)})


def grouped(records, split):
    groups = defaultdict(list)
    for r in records:
        if r['split'] == split:
            groups[(r['task_family'], r['counterfactual_group'])].append(r)
    return {task: [sorted(v, key=lambda r:r['id']) for (t, _), v in sorted(groups.items()) if t == task] for task in TASKS}


def basis_fit(c, records, z, out, device):
    import torch
    train = [r for r in records if r['split'] == 'train']
    count = np.zeros(len(z), np.int64)
    for r in train:
        np.add.at(count, r['rgb_indices'][:r['query']+1], 1)
    ids = np.flatnonzero(count)
    x = np.asarray(z[ids], np.float32)
    im = np.average(x, axis=0, weights=count[ids]).astype(np.float32)
    x = (x-im)*np.sqrt(count[ids]/count[ids].mean()).astype(np.float32)[:, None]
    y = np.concatenate([z[np.asarray(r['rgb_indices'])[r['query']+np.asarray(H)]]-z[r['rgb_indices'][r['query']]] for r in train])
    tm = y.mean(0)
    diffs = []
    for gg in grouped(records, 'train').values():
        for group in gg:
            targets = [z[np.asarray(r['rgb_indices'])[r['query']+np.asarray(H)]] for r in group]
            diffs.extend(2*(a-b) for a,b in itertools.combinations(targets,2))
    rows = np.concatenate([y-tm]+diffs)
    if len(rows) > c['basis_max_output_rows']:
        rows = rows[np.random.default_rng(c['reader_seed']).choice(len(rows), c['basis_max_output_rows'], replace=False)]
    def pca(a, whiten):
        torch.manual_seed(c['reader_seed'])
        v = torch.from_numpy(np.asarray(a, np.float32)).to(device)
        _, singular, vec = torch.pca_lowrank(v, q=min(c['reader']['rank'], min(v.shape)-1), center=False, niter=3)
        if whiten:
            vec = vec/(singular/np.sqrt(len(a))).clamp_min(.05)
        return vec.cpu().numpy()
    basis = {'input_mean': im, 'projection': pca(x, True), 'target_mean': tm, 'output_basis': pca(rows, False)}
    if not all(np.isfinite(v).all() for v in basis.values()):
        raise ValueError('Invalid TRAIN basis')
    np.savez(out/'basis.npz', **basis)
    write(out/'basis_provenance.json', {'train_ids': [r['id'] for r in train], 'train_prefix_weighted': True,
          'output_rows_used': len(rows), 'basis_sha256': sha(out/'basis.npz'), 'no_validation_or_dev_fit': True})
    return basis


class Data:
    def __init__(self, path, basis):
        self.path, self.basis = path, basis
        self.records = read(path/'records.json')
        self.z = np.load(path/'features.npy', mmap_mode='r', allow_pickle=False)
        self.actions = np.load(path/'actions.npz', allow_pickle=False)
        self.banks = np.load(path/'action_banks.npz', allow_pickle=False)

    def project(self, z):
        return ((z-self.basis['input_mean'])@self.basis['projection']).astype(np.float32)

    def capsule(self, r):
        q = r['query']; z = self.z[r['rgb_indices'][:q+1]]; ac = self.actions[r['id']][:q]
        return causal_capsule(z, ac, q)

    def actual_plan(self, r):
        return self.actions[r['id']][r['query']:r['query']+6].copy()

    def target(self, r):
        return self.z[np.asarray(r['rgb_indices'])[r['query']+np.asarray(H)]].copy()


def causal_capsule(observed_features, executed_actions, clock):
    """Only already observed prefix and executed controls cross this boundary."""
    if observed_features.shape[0] != clock+1 or executed_actions.shape != (clock, 2):
        raise ValueError('Expected causal prefixes only')
    times = np.r_[np.arange(clock-3, clock+1), np.arange(clock-6, clock-3)]
    prefix = np.concatenate([np.zeros((1,2), np.float32), executed_actions.cumsum(0)])
    return {'z': np.asarray(observed_features[times], np.float32), 'age': (clock-times).astype(np.float32),
            'past_actions': (prefix[clock]-prefix[times]).astype(np.float32),
            'current': np.asarray(observed_features[clock], np.float32)}


def reader_sample(capsule, keep, plan, basis, history_override=None, ages_override=None, projection_cache=None):
    """Pure tensor adapter; accepts no role, split, target, id or future observation."""
    if set(capsule) != {'z','age','past_actions','current'}:
        raise ValueError('Causal capsule field whitelist')
    ix = [0,1,2,3]+[4+i for i in keep]
    z, age, past = capsule['z'][ix].copy(), capsule['age'][ix].copy(), capsule['past_actions'][ix].copy()
    if history_override is not None:
        z[4:] = history_override
    if ages_override is not None:
        age[4:] = ages_override
    if projection_cache is not None and history_override is not None:
        raise ValueError('Changed content cannot reuse native projection')
    projected=((z-basis['input_mean'])@basis['projection']).astype(np.float32) if projection_cache is None else projection_cache[0][ix]
    current=((capsule['current']-basis['input_mean'])@basis['projection']).astype(np.float32) if projection_cache is None else projection_cache[1]
    return {'z': z, 'age': age, 'past_actions': past, 'current': capsule['current'],
            'type': np.asarray([0]*4+[1]*len(keep), np.int64), 'actions': np.asarray(plan, np.float32),
            'projected_z': projected,'projected_current': current}


def group_loss(p, y):
    """Proper conditional-mean target for every history size, including XOR.

    No pair or ranking term is used. Complete balanced factorial groups ensure
    identical insufficient inputs receive both future outcomes without a loss
    that rewards pushing them away from their conditional mean.
    """
    return (p-y).square().mean()


def train(c, out, device):
    import torch
    from phase1.model import DynamicsPredictor
    from phase1.inputs import collate
    from phase1.utils import seed_all
    if c['training_loss']!='pure_feature_mse_all_history_sizes':
        raise ValueError('Only the preregistered proper reader loss is supported')
    if (out/'reader_selected.pt').exists():
        raise ValueError('Do not overwrite trained reader')
    z = np.load(out/'data/features.npy', mmap_mode='r', allow_pickle=False)
    records = read(out/'data/records.json')
    seed_all(c['reader_seed'], 4)
    basis = basis_fit(c, records, z, out, device)
    data = Data(out/'data', basis)
    model = DynamicsPredictor(z.shape[1], c['reader'], H, basis).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [c['steps']//3, 2*c['steps']//3], gamma=.5)
    tg, vg = grouped(records, 'train'), grouped(records, 'validation')
    rng = np.random.default_rng(c['reader_seed'])
    def make(group, size, random):
        keep = sorted(random.choice(3,size,replace=False).tolist())
        samples = [reader_sample(data.capsule(r),keep,data.actual_plan(r),basis) for r in group]
        return collate(samples, device), torch.from_numpy(np.stack([data.target(r) for r in group])).to(device)
    def validation():
        model.eval(); cells = {}
        with torch.inference_mode():
            for ti, task in enumerate(TASKS):
                for size in range(4):
                    vrng = np.random.default_rng(17000+100*ti+size)
                    vals=[]
                    for group in vg[task]:
                        batch,y=make(group,size,vrng); vals.append(float((model(batch)-y).square().mean()))
                    cells[task+'/'+str(size)] = float(np.mean(vals))
        return {'equal_task_size_mse':float(np.mean(list(cells.values()))),'cells':cells}
    best, logs, started = float('inf'), [], time.perf_counter()
    for step in range(c['steps']+1):
        if step%c['validate_every']==0 or step==c['steps']:
            v=validation(); logs.append({'step':step,**v})
            if v['equal_task_size_mse']<best:
                best=v['equal_task_size_mse']
                torch.save({'model':model.state_dict(),'step':step,'validation':v,'basis_sha256':sha(out/'basis.npz'),'model_config':c['reader']},out/'reader_selected.pt')
            write(out/'learning_curve.json',logs);emit('reader_training',step=step,validation=v['equal_task_size_mse'])
        if step==c['steps']:break
        model.train(); optimizer.zero_grad(set_to_none=True)
        task_losses=[]
        for task in TASKS:
            gl=[]
            for gi in rng.choice(len(tg[task]),c['groups_per_task'],replace=False):
                batch,y=make(tg[task][int(gi)],step%4,rng); gl.append(group_loss(model(batch),y))
            task_losses.append(torch.stack(gl).mean())
        loss=torch.stack(task_losses).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite reader loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step()
    state=torch.load(out/'reader_selected.pt',map_location='cpu',weights_only=False)
    write(out/'reader_locked.json',{'selected_step':state['step'],'validation':state['validation'],
          'selected_sha256':sha(out/'reader_selected.pt'),'basis_sha256':sha(out/'basis.npz'),
          'seconds':time.perf_counter()-started,'origin':'query only','history_sizes':[0,1,2,3],
          'development_used_for_selection':False,'confirmation_opened':False})


class Predictor:
    def __init__(self, c, out, device):
        import torch
        from phase1.model import DynamicsPredictor
        self.torch,self.device,self.c = torch,device,c
        self.basis=dict(np.load(out/'basis.npz',allow_pickle=False))
        state=torch.load(out/'reader_selected.pt',map_location=device,weights_only=False)
        if state['basis_sha256']!=sha(out/'basis.npz'):raise ValueError('Reader/basis binding')
        self.model=DynamicsPredictor(len(self.basis['input_mean']),c['reader'],H,self.basis).to(device)
        self.model.load_state_dict(state['model']);self.model.eval().requires_grad_(False)
        self.calls,self.examples,self.seconds,self.reconstruction_max,self.attention_queries=0,0,0.,0.,0
    def predict(self,samples):
        from phase1.inputs import collate
        values,coefficients=[],[]
        for at in range(0,len(samples),self.c['predict_batch_size']):
            chunk=samples[at:at+self.c['predict_batch_size']];capture=[]
            hook=self.model.output.register_forward_hook(lambda mod,inputs,output:capture.append(output.detach().cpu().numpy()))
            started=time.perf_counter()
            try:
                with self.torch.inference_mode():p=self.model(collate(chunk,self.device)).cpu().numpy()
            finally:hook.remove()
            self.seconds+=time.perf_counter()-started;self.calls+=1;self.examples+=len(chunk)
            if len(capture)!=2 or not np.isfinite(p).all():raise ValueError('Prediction/capture contract')
            cf=np.stack(capture,1)
            reconstruction=np.stack([x['current'] for x in chunk])[:,None]+self.basis['target_mean']+cf@self.basis['output_basis'].T
            self.reconstruction_max=max(self.reconstruction_max,float(np.max(abs(reconstruction-p))))
            values.append(p);coefficients.append(cf)
        return np.concatenate(values),np.concatenate(coefficients)
    def attention(self,candidates,query):
        return self.attention_many(candidates,np.asarray(query)[None])[0]
    def attention_many(self,candidates,queries):
        self.attention_queries+=len(queries)
        p=((candidates-self.basis['input_mean'])@self.basis['projection']).astype(np.float32)
        q=((queries-self.basis['input_mean'])@self.basis['projection']).astype(np.float32)
        a=self.torch.from_numpy(p).to(self.device)[None].expand(len(q),-1,-1)
        b=self.torch.from_numpy(q).to(self.device)[:,None]
        with self.torch.inference_mode():
            score=self.model.score(self.torch.cat([a,a-b],-1)).squeeze(-1)+self.torch.log1p((a-b).square().mean(-1))
            return self.torch.softmax(score,-1).cpu().numpy()


def classification(p, prototypes, target, tolerance):
    distances=((np.asarray(p,np.float64)[None]-prototypes)**2).mean((1,2))
    minima=np.flatnonzero(distances<=distances.min()+tolerance).tolist()
    return {'class':int(distances.argmin()),'min_set':minima,'strict_correct':minima==[target],
            'tie_fraction_credit':1/len(minima) if target in minima else 0.,'distances':distances.tolist()}


def qualify(c,data,predictor,records,dest):
    """Fixed-age content controls, native K2/K3 and single-payload interventions."""
    by_group=defaultdict(list)
    for r in records:by_group[r['counterfactual_group']].append(r)
    rows=[];saved_coefficients=[]
    for ri,r in enumerate(records):
        cap=data.capsule(r);cand=cap['z'][4:];group=by_group[r['counterfactual_group']]
        classes={x['outcome_class']:data.target(x) for x in group}
        prototypes=np.stack([classes[i] for i in sorted(classes)])
        requests=[];meta=[]
        for drop in range(3):
            keep=[i for i in range(3) if i!=drop]
            requests.append(reader_sample(cap,keep,data.actual_plan(r),data.basis));meta.append(('native_'+str(drop),drop,r['outcome_class']))
            requests.append(reader_sample(cap,keep,data.actual_plan(r),data.basis,ages_override=[6,4]));meta.append(('fixed_'+str(drop),drop,r['outcome_class']))
            if drop in r['semantic_correct_delete_set']:
                for changed in keep:
                    candidates=[x for x in group if x['outcome_class']!=r['outcome_class'] and
                        all(x['rgb_indices'][x['evidence'][j]]==r['rgb_indices'][r['evidence'][j]] for j in keep if j!=changed)]
                    if candidates:
                        other=candidates[0];hist=cand[keep].copy()
                        hist[keep.index(changed)]=data.z[other['rgb_indices'][other['evidence'][changed]]]
                        requests.append(reader_sample(cap,keep,data.actual_plan(r),data.basis,hist,[6,4]))
                        meta.append(('flip_'+str(drop)+'_'+str(changed),drop,other['outcome_class']))
        requests.append(reader_sample(cap,[0,1,2],data.actual_plan(r),data.basis));meta.append(('full',None,r['outcome_class']))
        for i in range(3):
            requests.append(reader_sample(cap,[0,1],data.actual_plan(r),data.basis,np.stack([cand[i],cap['current']]),[6,4]))
            meta.append(('single_'+str(i),None,r['outcome_class']))
        ps,cs=predictor.predict(requests)
        for (arm,drop,target),p,coeff in zip(meta,ps,cs):
            cl=classification(p,prototypes,target,c['tolerance'])
            rows.append({'id':r['id'],'family':r['family'],'task_family':r['task_family'],'condition':r['condition'],
                'arm':arm,'drop':drop,'drop_role':r['roles'][drop] if drop is not None else None,
                'target_class':target,'original_outcome_class':r['outcome_class'],'coefficient_index':len(saved_coefficients),
                'semantic_complete':drop in r['semantic_correct_delete_set'] if drop is not None else arm=='full',
                'mse':float(((p.astype(np.float64)-prototypes[target])**2).mean()),
                'mse_to_original_target':float(((p.astype(np.float64)-data.target(r))**2).mean()),
                'midpoint_mse':float(((p.astype(np.float64)-prototypes.mean(0))**2).mean()),**cl})
            saved_coefficients.append(coeff)
        if (ri+1)%128==0:emit('qualifying',records=ri+1,total=len(records))
    gates={};summary={};cells={}
    for task in TASKS:
        rr=[r for r in rows if r['task_family']==task]
        k2=[r for r in rr if r['arm'].startswith(('native_','fixed_')) and r['semantic_complete']]
        flip=[r for r in rr if r['arm'].startswith('flip_')];full=[r for r in rr if r['arm']=='full']
        missing=[r for r in rr if r['arm'].startswith('fixed_') and not r['semantic_complete']]
        single=[r for r in rr if r['arm'].startswith('single_')]
        mean=lambda xs,key:float(np.mean([x[key] for x in xs])) if xs else None
        # Fixed-time matched complete-vs-missing MSE, not a global loss threshold.
        complete_fixed=[r for r in k2 if r['arm'].startswith('fixed_')]
        summary[task]={'K2_complete_accuracy':mean(k2,'strict_correct'),'flip_accuracy':mean(flip,'strict_correct'),
            'K3_accuracy':mean(full,'strict_correct'),'fixed_complete_mse':mean(complete_fixed,'mse'),
            'fixed_missing_mse':mean(missing,'mse'),'single_mse':mean(single,'mse'),
            'missing_tie_credit_accuracy':mean(missing,'tie_fraction_credit'),
            'missing_midpoint_mse':mean(missing,'midpoint_mse'),
            'missing_equal_distance_fraction':float(np.mean([len(x['min_set'])>1 for x in missing]))}
        s=summary[task]
        gates[task+'/K2_content']=s['K2_complete_accuracy']>=c['qualification_accuracy'] and s['flip_accuracy'] is not None and s['flip_accuracy']>=c['qualification_accuracy'] and s['fixed_complete_mse']<s['fixed_missing_mse']-c['qualification_mse_margin']
        gates[task+'/K3_content']=s['K3_accuracy']>=c['qualification_accuracy']
        for condition in sorted({r['condition'] for r in rr}):
            cell=[r for r in rr if r['condition']==condition];key=task+'/'+condition
            complete=[r for r in cell if r['arm'].startswith('fixed_') and r['semantic_complete']]
            missing=[r for r in cell if r['arm'].startswith('fixed_') and not r['semantic_complete']]
            flips=[r for r in cell if r['arm'].startswith('flip_')]
            native=[r for r in cell if r['arm'].startswith('native_') and r['semantic_complete']]
            full=[r for r in cell if r['arm']=='full']
            roles=sorted({r['drop_role'] for r in complete})
            role_means={role:mean([r for r in complete if r['drop_role']==role],'mse') for role in roles}
            role_acc={role:mean([r for r in complete if r['drop_role']==role],'strict_correct') for role in roles}
            missing_means=[mean([r for r in missing if r['drop_role']==role],'mse') for role in sorted({r['drop_role'] for r in missing})]
            cells[key]={'fixed_complete_mse_by_deleted_role':role_means,'fixed_complete_accuracy_by_deleted_role':role_acc,
                'missing_mse_by_deleted_role':missing_means,'flip_accuracy':mean(flips,'strict_correct'),
                'native_complete_accuracy':mean(native,'strict_correct'),'full_accuracy':mean(full,'strict_correct')}
            good=all(a>=c['qualification_accuracy'] for a in role_acc.values()) and mean(native,'strict_correct')>=c['qualification_accuracy']
            good=good and bool(flips) and mean(flips,'strict_correct')>=c['qualification_accuracy']
            good=good and all(x<y-c['qualification_mse_margin'] for x in role_means.values() for y in missing_means)
            gates[task+'/K2_content'] = gates[task+'/K2_content'] and good
            gates[task+'/K3_content'] = gates[task+'/K3_content'] and mean(full,'strict_correct')>=c['qualification_accuracy']
            gates[key+'/K2_content']=good;gates[key+'/K3_content']=mean(full,'strict_correct')>=c['qualification_accuracy']
    result={'summary':summary,'by_condition':cells,'gates':gates,'all_task_k2_qualified':all(gates[t+'/K2_content'] for t in TASKS),
            'all_task_k3_qualified':all(gates[t+'/K3_content'] for t in TASKS),
            'missing_information_chance_is_generator_property_not_reader_gate':True,
            'S_prototype_classes':2,'R_D_prototype_classes':4,'rows':len(rows)}
    np.save(dest/'reader_qualification_coefficients.npy',np.stack(saved_coefficients))
    write(dest/'reader_qualification_rows.json',rows);write(dest/'reader_qualification.json',result)
    emit('reader_qualification',**result)
    return result


def causal_features(cap,basis):
    p=((cap['z']-basis['input_mean'])@basis['projection']).astype(np.float32)
    current=((cap['current']-basis['input_mean'])@basis['projection']).astype(np.float32)
    items=np.concatenate([p[4:],p[4:]-current,np.log1p(cap['age'][4:,None]),cap['past_actions'][4:]],1)
    context=np.concatenate([p[:4].reshape(-1),cap['age'][:4],cap['past_actions'][:4].reshape(-1)])
    return items.astype(np.float32),context.astype(np.float32)


def appearance_features(data,r):
    info=read(data.path/'generation.json')
    rgb=np.memmap(data.path/'unique_rgb.bin',dtype=np.uint8,mode='r',shape=tuple(info['rgb_shape']))
    result=[]
    for at in r['evidence']:
        im=np.asarray(rgb[r['rgb_indices'][at]],np.float32)/255
        hist=np.concatenate([np.histogram(im[:,:,j],bins=8,range=(0,1),density=False)[0]/(96*96) for j in range(3)])
        result.append(np.r_[hist,im.mean((0,1)),im.std((0,1))])
    return np.asarray(result,np.float32)


def latency_probe(c,data,predictor,records,dest):
    from experiments.phase2.prediction_retention import medoid,full_set_sensitivity
    plans=data.banks['reference']
    def run(name,cap):
        start=time.perf_counter();pc=(data.project(cap['z']),data.project(cap['current']))
        count=3 if name=='medoid' else 4
        samples=[reader_sample(cap,[j for j in range(3) if j!=d] if d<3 else [0,1,2],plan,data.basis,projection_cache=pc)
                 for plan in plans for d in range(count)]
        p,_=predictor.predict(samples);p=p.reshape(len(plans),count,2,-1)
        if name=='medoid':medoid(p)
        else:full_set_sensitivity(p[:,:3],p[:,3])
        return time.perf_counter()-start
    before=predictor.examples
    for name in ('medoid','full_sensitivity'):run(name,data.capsule(records[0]))
    times={name:[] for name in ('medoid','full_sensitivity')}
    measured=records[1:17]
    for r in measured:
        cap=data.capsule(r)
        for name in times:times[name].append(run(name,cap))
    result={'warmup_parent':records[0]['id'],'measured_parent_ids':[r['id'] for r in measured],
        'outcome_based_selection':False,'device':predictor.device,'reference_plan_count':len(plans),
        'costs':{name:{'forecasts_per_decision':len(plans)*(3 if name=='medoid' else 4),
            'seconds':value,'median_seconds':float(np.median(value)),'p95_seconds':float(np.quantile(value,.95))} for name,value in times.items()},
        'extra_forecast_examples_including_warmup':predictor.examples-before,
        'includes':'projection, input construction, batched reader, CPU transfer, coefficient-capture/reconstruction audit and selection kernel',
        'excludes':'checkpoint loading, encoder, feature disk access, target generation and evaluation',
        'limitation':'instrumented current implementation, not optimized deployment latency; first deterministic parents, no hardware-wide claim'}
    write(dest/'method_latency.json',result)
    return result


def export_and_evaluate(c,out,data_path,split,device,do_qualification):
    from experiments.phase2.prediction_retention import medoid,full_set_sensitivity
    from experiments.phase2.run_gate2_pilot import diversity_choice,uniform_choice
    predictor=Predictor(c,out,device);data=Data(data_path,predictor.basis)
    records=[r for r in data.records if r['split']==split]
    dest=out/split;dest.mkdir(exist_ok=True)
    if (dest/'baseline_decisions.json').exists():raise ValueError('Do not overwrite scored split')
    qualification=qualify(c,data,predictor,records,dest) if do_qualification else None
    qualification_examples=predictor.examples
    latency=latency_probe(c,data,predictor,records,dest) if split in ('development','confirmation') else None
    before_main_examples=predictor.examples
    export=out/'exports'/split;export.mkdir(parents=True,exist_ok=True)
    items=[];contexts=[];appearance=[];costs=[];attention_labels=[];ids=[];decisions=[]
    bank_names=['reference'] if split=='train' else (['evaluation'] if split=='validation' else ['reference','evaluation'])
    coefficient_stores={name:np.lib.format.open_memmap(dest/(name+'_coefficients.npy'),mode='w+',dtype=np.float32,
        shape=(len(records),len(data.banks[name]),4,2,data.basis['output_basis'].shape[1])) for name in bank_names}
    started=time.perf_counter()
    for index,r in enumerate(records):
        cap=data.capsule(r);it,co=causal_features(cap,data.basis);items.append(it);contexts.append(co)
        projection_cache=(data.project(cap['z']),data.project(cap['current']))
        appearance.append(appearance_features(data,r));ids.append(r['id'])
        # Explicit boundary mutation check: only past controls enter capsule.
        ac=data.actions[r['id']].copy();ac[r['query']:]=12345
        mutated=causal_capsule(data.z[r['rgb_indices'][:r['query']+1]],ac[:r['query']],r['query'])
        assert all(np.array_equal(cap[k],mutated[k]) for k in cap)
        evaluated={};selectors={}
        for name in bank_names:
            samples=[reader_sample(cap,[j for j in range(3) if j!=d] if d<3 else [0,1,2],plan,data.basis,projection_cache=projection_cache)
                     for plan in data.banks[name] for d in range(4)]
            p,cf=predictor.predict(samples);p=p.reshape(len(data.banks[name]),4,2,-1)
            coefficient_stores[name][index]=cf.reshape(len(data.banks[name]),4,2,-1)
            if name=='reference':
                # Selectors receive predictions only, before evaluator-only targets.
                selectors={'medoid':medoid(p[:,:3]),'full_sensitivity':full_set_sensitivity(p[:,:3],p[:,3])}
            target=data.z[np.asarray(r['plan_targets'][name])]
            losses=((p.astype(np.float64)-target[:,None])**2).mean((2,3))
            teacher=predictor.attention_many(cap['z'][4:],target.reshape(-1,target.shape[-1])).mean(0)
            evaluated[name]={'losses':losses,'cost':losses[:,:3].mean(0),'attention':teacher}
        label_bank='reference' if split=='train' else 'evaluation'
        costs.append(evaluated[label_bank]['cost']);attention_labels.append(evaluated[label_bank]['attention'])
        scores=predictor.attention(cap['z'][4:],cap['current'])
        picks={'fifo':0,'terminal_uniform':uniform_choice(r['family']),
               'visual_diversity':diversity_choice(cap['z'][4:],r['evidence'],r['query']),
               'reader_attention':int(scores.argmin())}
        picks.update({name:x['drop'] for name,x in selectors.items()})
        loss=evaluated[label_bank]['cost'];sem=r['semantic_correct_delete_set']
        decisions.append({'id':r['id'],'family':r['family'],'task_family':r['task_family'],'condition':r['condition'],
            'current_time':r['query'],'candidate_times':r['evidence'],'semantic_correct_delete_set':sem,
            'teacher_query_times':[r['query']+h for h in H],
            'teacher_candidate_ages':[[r['query']+h-t for t in r['evidence']] for h in H],
            'teacher_executed_actions':'actual hypothetical-plan prefix up to q+h; score formula ignores this field and ages',
            'attention':scores.tolist(),'selection_details':selectors,'label_bank':label_bank,
            'expected_losses':loss.tolist(),'per_plan_losses':{k:v['losses'].tolist() for k,v in evaluated.items()},
            'oracle_delete_set':np.flatnonzero(loss<=loss.min()+c['tolerance']).tolist(),
            'policies':{name:{'drop':int(d),'mse':float(loss[d]),'regret':float(loss[d]-loss.min()),'semantic_correct':d in sem,
                'reader_qualified_for_claim':None if qualification is None else
                    qualification['gates'][r['task_family']+'/K2_content'] and
                    (name!='full_sensitivity' or qualification['gates'][r['task_family']+'/K3_content'])}
                for name,d in picks.items()}})
        if (index+1)%64==0:emit('export_evaluation',split=split,records=index+1,total=len(records))
    for value in coefficient_stores.values():value.flush()
    np.savez_compressed(export/'inputs.npz',items=np.stack(items),context=np.stack(contexts))
    np.savez_compressed(export/'labels.npz',cost=np.stack(costs),attention=np.stack(attention_labels))
    np.savez_compressed(export/'appearance.npz',items=np.stack(appearance),context=np.empty((len(records),0),np.float32))
    write(export/'ids.json',ids)
    task_counts={t:sum(r['task_family']==t for r in records) for t in TASKS}
    weights=np.asarray([1/task_counts[r['task_family']] for r in records],np.float64);weights/=weights.sum()
    np.save(export/'sampling_weights.npy',weights)
    write(export/'metadata.json',{'records':[{k:r[k] for k in ['id','family','task_family','condition']} for r in records],
        'input_fields':['items','context'],'metadata_is_not_model_input':True,'equal_task_family_sampling':True,
        'cost_label_bank':label_bank,'teacher_attention_formula':c['teacher_attention'],
        'teacher_distribution_limit':c['teacher_distribution_limit'],'basis_sha256':sha(out/'basis.npz'),
        'reader_sha256':sha(out/'reader_selected.pt'),'reference_action_bank_bytes':data.banks['reference'].nbytes,
        'inputs_sha256':sha(export/'inputs.npz'),'labels_sha256':sha(export/'labels.npz')})
    write(dest/'baseline_decisions.json',decisions)
    summary={}
    for task in TASKS:
        rr=[r for r in decisions if r['task_family']==task]
        summary[task]={name:{'records':len(rr),'semantic_loss_count':sum(not r['policies'][name]['semantic_correct'] for r in rr),
            'mse':float(np.mean([r['policies'][name]['mse'] for r in rr])),
            'regret':float(np.mean([r['policies'][name]['regret'] for r in rr]))} for name in rr[0]['policies']}
    result={'split':split,'summary':summary,'seconds':time.perf_counter()-started,'reader_examples':predictor.examples,
        'reader_examples_by_stage':{'qualification':qualification_examples,
            'latency_including_warmup':before_main_examples-qualification_examples,
            'shared_offline_prediction_evaluation':predictor.examples-before_main_examples},
        'reader_batches':predictor.calls,'reader_inference_seconds':predictor.seconds,
        'attention_score_query_count_including_teachers':predictor.attention_queries,'method_latency':latency,
        'coefficient_reconstruction_max_abs_difference':predictor.reconstruction_max,
        'coefficient_axes':['record','action_plan','drop0/drop1/drop2/full','horizon','basis_coefficient'],
        'reconstruction':'current + target_mean + coefficients @ output_basis.T',
        'raw_losses_preserved':True,'actual_future_actions_in_selector':False,
        'qualification':qualification,'no_training_in_evaluation':True}
    write(dest/'summary.json',result);emit('split_complete',**result)


def confirm_open(c,out,manifest):
    if (out/'confirmation_opened.json').exists():raise ValueError('Confirmation already opened')
    gate=read(out/'development/reader_qualification.json')
    if not gate['all_task_k2_qualified']:raise ValueError('Development K2 reader gate failed')
    if not manifest.is_file():raise ValueError('Frozen students manifest required')
    import torch
    students=read(manifest)
    if students['status']!='completed_fixed_student_training' or not students['arms']:
        raise ValueError('Student training incomplete')
    if sha(manifest.parent/'prefit_manifest.json')!=students['prefit_manifest_sha256']:
        raise ValueError('Student prefit binding changed')
    prefit=read(manifest.parent/'prefit_manifest.json')
    if sha(ROOT/'experiments/phase2/fit_terminal_students.py')!=prefit['source_sha256']:
        raise ValueError('Student source differs from prefit lock')
    if prefit['config']!=c['students']:
        raise ValueError('Student hyperparameters differ from scientific config')
    expected={f'{arch}_{label}_seed{seed}' for arch in ('independent','set')
              for label in ('cost','future_attention','shuffle') for seed in c['students']['seeds']}
    expected.update(f'appearance_cost_seed{seed}' for seed in c['students']['seeds'])
    if set(students['arms'])!=expected or students['appearance_control']!='completed':
        raise ValueError('Not all preregistered student/control arms completed')
    from experiments.phase2.fit_terminal_students import load_export,_array_hash
    for split in ('train','validation'):
        export=load_export(out/'exports'/split)
        if {k:_array_hash(v) for k,v in export.inputs.items()}!=prefit['inputs_hashes'][split]:
            raise ValueError('Student inputs differ from prefit lock')
        if {k:_array_hash(getattr(export,k)) for k in ('cost','attention')}!=prefit['labels_hashes'][split]:
            raise ValueError('Student supervision differs from prefit lock')
        weight_key='sampling_weights_sha256' if split=='train' else 'validation_weights_sha256'
        if _array_hash(export.sampling_weights)!=prefit[weight_key]:
            raise ValueError('Student task sampling weights changed')
        if 'appearance_inputs_hashes' in prefit and {k:_array_hash(v) for k,v in export.appearance_inputs.items()}!=prefit['appearance_inputs_hashes'][split]:
            raise ValueError('Appearance control inputs changed')
        metadata=read(out/'exports'/split/'metadata.json')
        if metadata['reader_sha256']!=sha(out/'reader_selected.pt') or metadata['basis_sha256']!=sha(out/'basis.npz'):
            raise ValueError('Student exports came from a different reader/basis')
    if sha(manifest.parent/'supervision_audit.npz')!=prefit['supervision_audit_sha256']:
        raise ValueError('Supervision audit artifact changed')
    checkpoint_hashes={}
    for name,arm in students['arms'].items():
        path=manifest.parent/arm['checkpoint']
        if sha(path)!=arm['checkpoint_sha256']:raise ValueError('Student checkpoint changed')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if sha(manifest.parent/checkpoint['normalization_file'])!=checkpoint['normalization_sha256']:
            raise ValueError('Student normalization changed')
        checkpoint_hashes[name]=arm['checkpoint_sha256']
    write(out/'confirmation_opened.json',{'opened_unix':time.time(),'students_manifest':str(manifest.resolve()),
          'students_manifest_sha256':sha(manifest),'development_report_sha256':sha(out/'development/summary.json'),
          'reader_sha256':sha(out/'reader_selected.pt'),'source_sha256':source_files(),
          'student_checkpoint_sha256':checkpoint_hashes,'one_time':True})
    banks=dict(np.load(out/'data/action_banks.npz',allow_pickle=False))
    generate(c,out/'confirmation_data',['confirmation'],existing_banks=banks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['freeze','profile','prepare','encode','train','evaluate','labels','confirm-open'])
    p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda']);p.add_argument('--cache-dir',type=Path)
    p.add_argument('--split',default='development',choices=['train','validation','development','confirmation'])
    p.add_argument('--students-manifest',type=Path)
    a=p.parse_args();out=a.output.resolve()
    if a.stage in ('freeze','profile'):
        if a.config is None:p.error('--config required')
        if a.stage=='freeze':freeze(a.config,out)
        else:generate(read(a.config),out,['train','validation','development'],profile=True)
        return
    c=locked(out)
    if a.stage=='prepare':generate(c,out/'data',['train','validation','development'])
    elif a.stage=='encode':
        if a.cache_dir is None:p.error('--cache-dir required')
        confirm=a.split=='confirmation'
        if confirm and not (out/'confirmation_opened.json').exists():raise ValueError('Confirmation not opened')
        encode(c,out/('confirmation_data' if confirm else 'data'),a.cache_dir.resolve(),a.device,
               out/'data/normalization.npz' if confirm else None)
    elif a.stage=='train':train(c,out,a.device)
    elif a.stage in ('evaluate','labels'):
        confirm=a.split=='confirmation'
        if confirm and not (out/'confirmation_opened.json').exists():raise ValueError('Confirmation not opened')
        export_and_evaluate(c,out,out/('confirmation_data' if confirm else 'data'),a.split,a.device,a.stage=='evaluate')
    else:
        if a.students_manifest is None:p.error('--students-manifest required')
        confirm_open(c,out,a.students_manifest.resolve())


if __name__=='__main__':main()
