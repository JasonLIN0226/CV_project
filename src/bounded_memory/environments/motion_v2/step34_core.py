"""Frozen-reader inference and irreversible Motion B=2 replay.

The archive belongs to the offline evaluator. Selectors receive only three
candidate projections and the four recent observations. No selector can ask
the archive for a discarded observation, hidden state, or a future frame.
Importing this module performs no experiment or filesystem writes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

SEEDS = (17, 29, 43)
BEHAVIORS = ('fifo', 'reservoir', 'temporal_buckets', 'diversity')
BASELINES = (*BEHAVIORS, 'surprise', 'attention', 'keep_first')
MODEL_KEYS = {'z', 'age', 'type', 'past_actions', 'actions', 'current',
              'projected_z', 'projected_current'}


def module(name):
    key = '_motion_step34_' + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(name+'.py'))
        obj = importlib.util.module_from_spec(spec)
        sys.modules[key] = obj
        spec.loader.exec_module(obj)
    return sys.modules[key]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)


def groups(records, split):
    if split not in ('train', 'validation'):
        raise ValueError('Only train/validation are authorized')
    pairs = {}
    for r in records:
        if r['split'] == split:
            pairs.setdefault(r['pair'], []).append(r)
    for pair in pairs.values():
        if len(pair) != 2 or pair[0]['sibling'] != pair[1]['id'] or pair[1]['sibling'] != pair[0]['id']:
            raise ValueError('Incomplete sibling pair')
    return list(pairs.values())


def seed_for(*values):
    return int(digest(values)[:16], 16) % 2**32


def sample_clocks(query):
    """Four fixed equally spaced overflow clocks, with no hidden-state filter."""
    if type(query) is not int or query < 9:
        raise ValueError('Insufficient query length')
    return tuple(6 + j*(query-6)//3 for j in range(4))


def causal_features(projected, prefix, clock, pool):
    """Same 259/524-dimensional feature construction as the 10/26 students."""
    if (not isinstance(clock,(int,np.integer)) or clock < 6 or
            any(not isinstance(t,(int,np.integer)) for t in pool)):
        raise ValueError('Integer causal clock and frame indices required')
    pool = np.asarray(pool, np.int64)
    if pool.shape != (3,) or len(set(pool)) != 3 or pool.min() < 0 or pool.max() > clock-4:
        raise ValueError('Student requires exactly three legal historical candidates')
    times = np.arange(clock-3, clock+1)
    p = projected[pool]
    items = np.concatenate([p, p-projected[clock], np.log1p(clock-pool)[:, None],
                            prefix[clock]-prefix[pool]], axis=1).astype(np.float32)
    context = np.concatenate([projected[times].reshape(-1), (clock-times),
                              (prefix[clock]-prefix[times]).reshape(-1)]).astype(np.float32)
    if items.shape != (3, 259) or context.shape != (524,):
        raise ValueError('Causal scorer input shape changed')
    return items, context


def observable_update(name, previous, arrival, clock, vectors, anchor, scores, rng):
    """Pure bounded transition. No archive, metadata, targets, or query supplied."""
    pool = list(previous) + [arrival]
    if (not isinstance(clock, (int, np.integer)) or clock < 4 or
            not isinstance(arrival, (int, np.integer)) or arrival != clock-4 or
            any(not isinstance(t, (int, np.integer)) or not 0 <= t < arrival for t in previous) or
            len(previous) > 2 or len(set(pool)) != len(pool)):
        raise ValueError('Illegal B=2 transition')
    if len(pool) <= 2:
        return sorted(pool)
    pairs = list(itertools.combinations(pool, 2))
    if name == 'fifo':
        kept = sorted(pool)[-2:]
    elif name == 'keep_first':
        kept = sorted(pool)[:2]
    elif name == 'reservoir':
        j = int(rng.integers(0, arrival+1))
        kept = list(previous)
        if j < 2:
            kept[j] = arrival
    elif name == 'temporal_buckets':
        def key(pair):
            ages = [clock-t for t in pair]
            return (len({int(np.floor(np.log2(max(1, a/4)))) for a in ages}),
                    max(ages)-min(ages), sum(pair), tuple(pair))
        kept = max(pairs, key=key)
    elif name == 'diversity':
        if set(vectors) != set(pool):
            raise ValueError('Diversity must see exactly its candidate pool')
        def key(pair):
            a, b = (vectors[t] for t in pair)
            return (float(np.linalg.norm(anchor-a)+np.linalg.norm(anchor-b)+np.linalg.norm(a-b)), tuple(pair))
        kept = max(pairs, key=key)
    elif name in ('attention', 'surprise', 'student'):
        if set(scores) != set(pool) or not np.isfinite(list(scores.values())).all():
            raise ValueError('Selector scores must be finite for exactly three candidates')
        kept = sorted(pool, key=lambda t: (scores[t], t), reverse=True)[:2]
    else:
        raise ValueError('Unknown deployable policy')
    if len(kept) != 2 or not set(kept) <= set(pool):
        raise ValueError('Resurrection or budget violation')
    return sorted(kept)


def mean_attention(raw_scores):
    """Mean of each frozen reader's candidate softmax, never softmax(mean logit)."""
    x = np.asarray(raw_scores, np.float64)
    if x.ndim != 2 or x.shape[0] != 3 or not np.isfinite(x).all():
        raise ValueError('Three finite reader salience vectors required')
    e = np.exp(x-x.max(1, keepdims=True))
    return (e/e.sum(1, keepdims=True)).mean(0)


def continue_attention(pool, drop, clock, query, score_tables):
    if (len(pool) != 3 or len(set(pool)) != 3 or
            any(not isinstance(t, (int, np.integer)) or not 0 <= t <= clock-4 for t in pool) or
            not isinstance(drop, (int, np.integer)) or drop not in range(3) or
            not isinstance(clock, (int, np.integer)) or clock < 6 or
            not isinstance(query, (int, np.integer)) or query < clock):
        raise ValueError('Illegal forced-deletion parent or clock')
    kept = [t for j, t in enumerate(pool) if j != drop]
    if len(kept) != 2 or max(kept) > clock-4:
        raise ValueError('Illegal forced deletion')
    for now in range(clock+1, query+1):
        parent = kept + [now-4]
        values = mean_attention(score_tables[:, now, parent])
        kept = observable_update('attention', kept, now-4, now, {}, None,
                                 dict(zip(parent, values)), None)
    return kept


class FrozenAssets:
    def __init__(self, output, dataset, expected, device, budget):
        self.output, self.dataset = Path(output).resolve(), Path(dataset).resolve()
        self.device, self.budget = torch.device(device), budget
        self.verified = {}
        for relative, checksum in expected.items():
            p = self.output / relative
            if not p.is_file() or sha(p) != checksum:
                raise ValueError('Frozen artifact hash differs: '+relative)
            self.verified[str(p)] = (p.stat().st_size, p.stat().st_mtime_ns, checksum)
            budget()
        self.config = json.loads((self.output/'resolved_config.json').read_text())
        self.config['output'] = str(self.output)
        self.reader = module('reader_v2')
        manifest = json.loads((self.dataset/'manifest.json').read_text())
        self.store = self.reader.V2FeatureStore(self.config, self.dataset, self.output/'features', manifest)
        with np.load(self.output/'predictor/motion/basis.npz', allow_pickle=False) as f:
            self.basis = {k: f[k] for k in f.files}
        self.store.bases['motion'] = self.basis
        self.decoder = self.reader.load_reference('probes').LinearReadout.load(self.output/'decoder/train_position_decoder.npz')
        self.models = []
        for seed in SEEDS:
            selection = json.loads((self.output/f'predictor/motion/seed_{seed}/selection.json').read_text())
            # Selection path is checked against its fixed basename, not trusted as an arbitrary path.
            ck = self.output/f'predictor/motion/seed_{seed}'/Path(selection['checkpoint']).name
            if str(ck) not in self.verified:
                raise ValueError('Selected checkpoint absent from preregistered asset inventory')
            checkpoint = torch.load(ck, map_location='cpu', weights_only=True)
            if checkpoint['seed'] != seed:
                raise ValueError('Predictor seed mismatch')
            model = self.reader.load_reference('model').DynamicsPredictor(5120, self.config['model'], [3, 6], self.basis).to(self.device)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.models.append(model)
        self.collate = self.reader.load_reference('inputs').collate
        self.cache = {}

    def verify_unchanged(self):
        self.budget()
        self.store.verify_boundaries()
        self.budget()
        for name, (size, mtime, checksum) in self.verified.items():
            p = Path(name)
            if p.stat().st_size != size or p.stat().st_mtime_ns != mtime or sha(p) != checksum:
                raise ValueError('Frozen artifact changed during stage3/4: '+name)
            self.budget()

    def episode(self, rec, external=None):
        key = rec['id']
        if key in self.cache:
            return self.cache[key]
        if external is None:
            z = self.store.features(rec)
            p = self.store.projected_features(rec)
            actions = self.store._observed_actions(rec)
            meta = self.store.read_evaluation(rec, ('states', 'visible', 't_c', 'end', 'sufficient'))
        else:
            root, feature_root, index = external
            root, feature_root = Path(root).resolve(), Path(feature_root).resolve()
            if rec['split'] != 'validation' or rec['task'] != 'motion':
                raise ValueError('External diagnostic data must be validation Motion only')
            entry = index['episodes'][key]
            fpath = feature_root/entry['path']
            if entry['path'] != f'episodes/{key}.npz' or sha(fpath) != entry['sha256']:
                raise ValueError('New validation feature index/hash differs')
            with np.load(fpath, allow_pickle=False) as f:
                if set(f.files) != {'features'}:
                    raise ValueError('Unexpected new feature fields')
                z = ((f['features']-self.store.norm['mean'])/self.store.norm['std']).astype(np.float32)
            p = self.reader.load_reference('model').project(z, self.basis)
            for kind, directory in (('obs', 'observations'), ('eval', 'evaluation_only')):
                expected_path = f'{directory}/validation/{key}.npz'
                if rec[kind] != expected_path or sha(root/expected_path) != rec[kind+'_sha256']:
                    raise ValueError('External diagnostic data path/hash differs')
            with np.load(root/rec['obs'], allow_pickle=False) as f:
                actions = f['actions']
            with np.load(root/rec['eval'], allow_pickle=False) as f:
                meta = {k: f[k] for k in ('states','visible','t_c','end','sufficient')}
        if z.shape != (rec['frames'], 5120) or not np.isfinite(z).all():
            raise ValueError('Invalid feature array')
        prefix = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(actions, axis=0)])
        obj = dict(rec=rec, z=z, p=p, actions=actions, prefix=prefix, meta=meta)
        self.cache[key] = obj
        return obj

    def sample(self, ep, clock, kept, held_action=False):
        kept = np.asarray(kept, np.int64)
        if clock < 3 or len(kept) not in (0,2) or len(set(kept)) != len(kept):
            raise ValueError('Only R=4 with zero or two history frames is supported')
        if len(kept) and (kept.min() < 0 or kept.max() > clock-4):
            raise ValueError('History crosses causal recent boundary')
        times = np.concatenate([np.arange(clock-3,clock+1),np.sort(kept)])
        a = ep['actions']
        future_actions = np.tile(a[clock-1], (6,1)) if held_action else a[clock:clock+6].copy()
        result = {'z':ep['z'][times].copy(), 'age':(clock-times).astype(np.float32),
                  'type':np.r_[np.zeros(4,np.int64),np.ones(len(kept),np.int64)],
                  'past_actions':ep['prefix'][clock]-ep['prefix'][times],
                  'actions':future_actions, 'current':ep['z'][clock].copy(),
                  'projected_z':ep['p'][times].copy(), 'projected_current':ep['p'][clock].copy()}
        if set(result) != MODEL_KEYS or future_actions.shape != (6,2) or not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError('Invalid observable predictor input')
        return result

    @torch.inference_mode()
    def losses(self, requests, reader_index=None, progress=None):
        """Requests are (episode, kept_indices), always evaluated at its true q."""
        indices = range(3) if reader_index is None else [reader_index]
        feature = np.empty((len(indices), len(requests), 2), np.float64)
        position = np.empty_like(feature)
        for out_i, mi in enumerate(indices):
            for at in range(0, len(requests), 64):
                batch = requests[at:at+64]
                samples = [self.sample(ep, ep['rec']['query'], kept) for ep, kept in batch]
                predicted = self.models[mi](self.collate(samples,self.device)).cpu().numpy()
                target = np.stack([ep['z'][ep['rec']['query']+np.array([3,6])] for ep,_ in batch])
                truth = np.stack([ep['meta']['states'][ep['rec']['query']+np.array([3,6])] for ep,_ in batch])
                decoded = self.decoder.predict(predicted.reshape(-1,5120)).reshape(-1,2,2)
                feature[out_i,at:at+len(batch)] = np.square(predicted.astype(np.float64)-target).mean(-1)
                position[out_i,at:at+len(batch)] = np.square(decoded-truth.astype(np.float64)).mean(-1)
                if progress:progress(len(batch))
                self.budget()
        if not np.isfinite(feature).all() or not np.isfinite(position).all():
            raise ValueError('Nonfinite frozen-reader loss')
        return dict(feature=feature,position=position)

    @torch.inference_mode()
    def salience(self, candidates, anchors):
        """Broadcastable projections; returns three raw score arrays."""
        p, current = np.broadcast_arrays(np.asarray(candidates,np.float32),np.asarray(anchors,np.float32))
        pt = torch.as_tensor(p.copy(),device=self.device)
        delta = pt-torch.as_tensor(current.copy(),device=self.device)
        x = torch.cat([pt,delta],-1)
        bonus = torch.log1p(delta.square().mean(-1))
        values = np.stack([(model.score(x).squeeze(-1)+bonus).cpu().numpy() for model in self.models])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite attention score')
        return values

    def attention_tables(self, ep):
        if 'scores' not in ep:
            q = ep['rec']['query']
            p = ep['p'][:q+1]
            # Offline accelerator; selectors only receive entries for their current pool.
            ep['scores'] = self.salience(p[None,:,:],p[:,None,:])
        return ep['scores']

    @torch.inference_mode()
    def surprise(self, ep, progress=None):
        if 'surprise' not in ep:
            q = ep['rec']['query']
            out = np.full(q+1,-1.,np.float64)
            clocks = list(range(3,q-6+1))
            if clocks:
                samples = [self.sample(ep,t,[],held_action=True) for t in clocks]
                batch = self.collate(samples,self.device)
                losses = []
                for model in self.models:
                    prediction = model(batch)[:,0].cpu().numpy()
                    losses.append(np.square(prediction.astype(np.float64)-ep['z'][np.array(clocks)+3]).mean(-1))
                    if progress:progress(len(clocks)*20)
                out[np.array(clocks)+3] = np.mean(losses,axis=0)
            ep['surprise'] = out
            self.budget()
        return ep['surprise']


def behavior_states(assets, split, output, progress=None):
    rows, items, contexts = [], [], []
    for pair in groups(assets.store.manifest['records'],split):
        for rec in pair:
            ep = assets.episode(rec)
            q = rec['query']
            for policy in BEHAVIORS:
                rng = np.random.default_rng(seed_for(20261103,rec['pair'],policy))
                kept = []
                for clock in range(4,q+1):
                    arrival = clock-4
                    pool = kept+[arrival]
                    if clock in sample_clocks(q):
                        x,c = causal_features(ep['p'],ep['prefix'],clock,pool)
                        parent_id = digest([rec['id'],policy,clock,pool])
                        rows.append(dict(id=parent_id,episode_id=rec['id'],pair_id=rec['pair'],
                                         behavior=policy,clock=clock,pool=pool.copy()))
                        items.append(x);contexts.append(c)
                        if progress:progress(1)
                    vectors = {t:ep['p'][t].copy() for t in pool} if policy=='diversity' else {}
                    kept = observable_update(policy,kept,arrival,clock,vectors,ep['p'][clock],{},rng)
            assets.budget()
    expected = 16384 if split=='train' else 3072
    if len(rows) != expected or len({r['id'] for r in rows}) != expected:
        raise ValueError('Behavior state count or identities differ from protocol')
    inputs = dict(items=np.stack(items),context=np.stack(contexts))
    write_npz(Path(output)/'inputs.npz',**inputs)
    write_json(Path(output)/'states.json',{'records':rows,'split':split,'multiplicity_preserved':True,
               'unique_episode_clock_pool_states':len({(r['episode_id'],r['clock'],tuple(r['pool'])) for r in rows})})
    return rows,inputs


def deletion_labels(assets, rows, output, progress=None):
    records = assets.store.records
    requests = {'local':[],'use_time':[]}
    attention, endpoints = [], []
    for at,r in enumerate(rows):
        ep = assets.episode(records[r['episode_id']]);q=ep['rec']['query'];pool=r['pool']
        tables = assets.attention_tables(ep)
        finals=[]
        for drop in range(3):
            remaining=[t for j,t in enumerate(pool) if j!=drop]
            final=continue_attention(pool,drop,r['clock'],q,tables)
            requests['local'].append((ep,remaining));requests['use_time'].append((ep,final));finals.append(final)
        values=[]
        for h in (3,6):
            values.append(mean_attention(assets.salience(ep['p'][pool],ep['p'][q+h])))
        attention.append(np.mean(values,axis=0));endpoints.append(finals)
        if progress:progress(6)
        if at%64==0: assets.budget()
    labels={'future_attention':np.asarray(attention,np.float64)}
    raw={}
    for moment, reqs in requests.items():
        values=assets.losses(reqs,progress=progress)
        for metric, loss in values.items():
            shaped=loss.reshape(3,len(rows),3,2)
            raw[moment+'_'+metric]=shaped
            labels[moment+'_'+metric]=shaped.mean((0,3))
    write_npz(Path(output)/'raw_teacher_losses.npz',**raw)
    write_npz(Path(output)/'labels.npz',**labels)
    write_npz(Path(output)/'continuation_endpoints.npz',times=np.asarray(endpoints,np.int64))
    write_json(Path(output)/'label_provenance.json',{'raw_cost':'retained-pair loss; no full-parent forward',
        'reader_seeds':list(SEEDS),'aggregation':'mean per-reader loss then mean h=3/6',
        'local':'original retained pair evaluated at actual episode query',
        'use_time':'force deletion then fixed ensemble current-attention irreversible continuation to query',
        'future_attention':'mean reader/horizon candidate softmax anchored at true q+h projection',
        'state_rows':len(rows),'no_hidden_fields_in_student_inputs':True})
    return labels


def chosen_reference(ep, condition):
    q=ep['rec']['query'];m=ep['meta'];tc=int(m['t_c']);end=int(m['end'])
    post=np.flatnonzero(m['visible'][tc:end+1])+tc
    old=np.flatnonzero(m['visible'][:tc])
    if condition=='a':return []
    if condition=='b':return [int(post[0]),end]
    if condition=='d':return [int(old[-1]),end]
    if condition=='b_random':
        return list(module('step34_data').random_post_pair(m['visible'],tc,end,ep['rec']['pair'],20261102))
    if condition=='last_visible':return list(np.flatnonzero(m['visible'][:q-3])[-2:].astype(int))
    raise ValueError(condition)


def diagnostic(assets, records, conditions, output, split, external=None):
    pairlist=groups(records,'validation')
    if len(pairlist)!=96:raise ValueError('Diagnostics require96pairs')
    episodes=[assets.episode(r,external) for pair in pairlist for r in pair]
    results={};choices={}
    for condition in conditions:
        kept=[chosen_reference(ep,condition) for ep in episodes]
        values=assets.losses(list(zip(episodes,kept)))
        results[condition]={k:v.reshape(3,96,2,2) for k,v in values.items()}
        choices[condition]=[list(map(int,x)) for x in kept]
    positions=np.stack([ep['meta']['states'][ep['rec']['query']+np.array([3,6])] for ep in episodes]).reshape(96,2,2,2)
    true_features=np.stack([ep['z'][ep['rec']['query']+np.array([3,6])] for ep in episodes])
    decoded=assets.decoder.predict(true_features.reshape(-1,5120)).reshape(96,2,2,2)
    decoder_loss=np.square(decoded-positions).mean(-1)
    write_json(Path(output)/'frame_choices.json',{'episode_ids':[ep['rec']['id'] for ep in episodes],
                                               'choices':choices,'split':split})
    write_npz(Path(output)/'raw_loss_arrays.npz',
        **{method+'_'+metric:value for method,values in results.items() for metric,value in values.items()},
        true_positions=positions,decoder_true_losses=decoder_loss,
        pair_ids=np.asarray([pair[0]['pair'] for pair in pairlist]))
    return module('step34_metrics').summarize_diagnostics(results,positions,decoder_loss,output,
        split=split,pair_ids=[pair[0]['pair'] for pair in pairlist])


def online_kept(assets,ep,policy,random_seed,student=None,progress=None):
    """Full irreversible trajectory. The only candidate set is old2 + outgoing1."""
    q=ep['rec']['query'];kept=[];trace=[]
    rng=np.random.default_rng(seed_for(20261105,ep['rec']['pair'],policy,random_seed))
    scores_table=assets.attention_tables(ep) if policy=='attention' else None
    surprise=assets.surprise(ep,progress=progress) if policy=='surprise' else None
    for clock in range(4,q+1):
        arrival=clock-4;pool=kept+[arrival];scores={};vectors={}
        if len(pool)==3:
            if policy=='attention':scores=dict(zip(pool,mean_attention(scores_table[:,clock,pool])))
            elif policy=='surprise':scores={t:float(surprise[t]) for t in pool}
            elif policy=='diversity':vectors={t:ep['p'][t].copy() for t in pool}
            elif policy=='student':
                model,norm,metadata=student;x,c=causal_features(ep['p'],ep['prefix'],clock,pool)
                value=module('step34_students').normalize_inputs({'items':x[None],'context':c[None]},norm)
                with torch.inference_mode():
                    logits=model(torch.as_tensor(value['items'],device=assets.device),
                                 torch.as_tensor(value['context'],device=assets.device)).cpu().numpy()[0]
                # Student emits deletion logits; selector keeps largest negative logits.
                scores=dict(zip(pool,-logits))
        selected=observable_update(policy,kept,arrival,clock,vectors,ep['p'][clock],scores,rng)
        trace.append((clock,tuple(kept),arrival,tuple(selected)))
        kept=selected
        if progress:progress(60 if policy=='student' and len(pool)==3 else 1)
    return kept,trace


def posthoc_best(assets,ep,progress=None):
    """Privileged offline reference: joint mean position MSE over readers/horizons."""
    candidates=list(itertools.combinations(range(ep['rec']['query']-3),2))
    losses=assets.losses([(ep,list(k)) for k in candidates],progress=progress)
    energy=losses['position'].mean((0,2))
    winner=int(np.argmin(energy))
    return list(candidates[winner]),{k:v[:,winner] for k,v in losses.items()}


def online_work(records):
    rows=[r for pair in groups(records,'validation') for r in pair]
    arrivals=sum(r['query']-3 for r in rows)
    student_units=sum(60*(r['query']-5)+2 for r in rows)
    oracle_samples=sum((r['query']-3)*(r['query']-4)//2 for r in rows)
    surprise_samples=sum(max(0,r['query']-8) for r in rows)
    # 9 baseline trajectories, 54 students, 2 cheap references, one exact oracle.
    return 9*arrivals + 54*student_units + 20*3*(65*len(rows)+oracle_samples+surprise_samples)


def evaluate_online(assets,records,student_summary,student_output,output,split,external=None,progress=None):
    pairs=groups(records,'validation');episodes=[assets.episode(r,external) for pair in pairs for r in pair]
    if len(pairs)!=96:raise ValueError('Online validation requires96pairs')
    results={};steps={};replicated=[];trace_inventory={}
    paths=module('step34_students').checkpoint_entries(student_summary,student_output)
    method_map={}
    for entry in paths:
        method='student/'+entry['architecture']+'/'+entry['label']
        method_map.setdefault(method,[]).append(entry)
    methods=['baseline/'+p for p in BASELINES]+list(method_map)+['reference/last_visible','reference/b','reference/posthoc_best']
    for method in methods:
        final_runs=[];fl=[];pl=[];success=[]
        runs=3 if method in method_map or method=='baseline/reservoir' else 1
        entries=sorted(method_map.get(method,[]),key=lambda e:SEEDS.index(e['seed']))
        if entries:steps[method]=[int(e['selected_step']) for e in entries]
        for ri in range(runs):
            student=module('step34_students').load_student(entries[ri]['path'],assets.device) if entries else None
            kept_all=[];rows=[];oracle=[]
            for ep in episodes:
                if method.startswith('reference/'):
                    condition=method.split('/')[1]
                    if condition=='posthoc_best':
                        kept,loss=posthoc_best(assets,ep,progress=(lambda n:progress(20*n)) if progress else None)
                        oracle.append(loss)
                    else:kept=chosen_reference(ep,condition)
                    trace=[]
                else:
                    policy='student' if entries else method.split('/')[1]
                    kept,trace=online_kept(assets,ep,policy,SEEDS[ri],student,progress=progress)
                kept_all.append(kept)
                rows.append({'episode_id':ep['rec']['id'],'kept':list(map(int,kept)),
                             'trace':trace})
                assets.budget()
            loss=({k:np.stack([x[k] for x in oracle],axis=1) for k in ('feature','position')}
                  if oracle else assets.losses(list(zip(episodes,kept_all)),
                       progress=(lambda n:progress(20*n)) if progress else None))
            fl.append(loss['feature'].reshape(3,96,2,2));pl.append(loss['position'].reshape(3,96,2,2))
            success.append(np.array([len(kept)==2 and bool(np.all(ep['meta']['sufficient'][kept]))
                                     for ep,kept in zip(episodes,kept_all)]).reshape(96,2))
            trace_path=Path(output)/'traces'/method.replace('/','_')/f'seed_{SEEDS[ri]}.json'
            write_json(trace_path,{'logical_B':2,'recent_R':4,'rows':rows,'offline_archive_not_physical_memory_claim':True})
            final_runs.append(sha(trace_path))
        if runs==1:
            fl=fl*3;pl=pl*3;success=success*3;replicated.append(method)
        results[method]={'feature':np.stack(fl),'position':np.stack(pl),'success':np.stack(success)}
        write_npz(Path(output)/'raw_losses'/(method.replace('/','_')+'.npz'),
                  **results[method],pair_ids=np.asarray([pair[0]['pair'] for pair in pairs]))
        trace_inventory[method]=final_runs
    write_json(Path(output)/'trace_inventory.json',trace_inventory)
    return module('step34_metrics').summarize_online(results,output,split=split,trained_steps=steps,
        replicated_methods=replicated,pair_ids=[pair[0]['pair'] for pair in pairs])
