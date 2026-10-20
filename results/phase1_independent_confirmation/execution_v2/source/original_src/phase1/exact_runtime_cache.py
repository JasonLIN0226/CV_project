"""Explicit exact-data cache adapter; original sampler and model remain intact."""
import ast
import inspect
import textwrap
from pathlib import Path
import numpy as np
import torch
from .inputs import FeatureStore, selected_evidence, memory_indices, collate
from .data import load_episode
from .utils import stable_seed, digest
from .training import TRAIN_CONDITIONS, regression_loss, rank_weight

class CachedFeatureStore(FeatureStore):
    def __init__(self, c, max_cached=1216):
        super().__init__(c, max_cached=max_cached)
        self.metadata = {}
        self.boundaries = {}
        self.validation_batches = None
        self.verify_sample_source()
        legal = [r for r in self.manifest['records'] if r['split'] in ('train','validation') and r['task'] in c['experiment']['tasks']]
        if len(legal) != 1216:
            raise ValueError('Exactly original legal1216 train/validation episodes required')
        for rec in legal:
            for metadata,key in ((True,'evidence'),(False,'actions')):
                path = Path(c['output'])/'dataset'/rec['eval' if metadata else 'obs']
                before = path.stat()
                array = load_episode(c, rec, metadata, keys=(key,))[key]
                after = path.stat()
                if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                    raise ValueError('Source changed during exact metadata cache load')
                if not np.isfinite(array).all():
                    raise ValueError('Nonfinite legal metadata')
                array.flags.writeable = False
                self.metadata[(rec['id'],metadata,key)] = array
                self.boundaries[str(path)] = (after.st_size,after.st_mtime_ns)
        self.metadata_fingerprint = digest({str(k): {'dtype':str(v.dtype),'shape':list(v.shape),'values':v.tolist()} for k,v in self.metadata.items()})

    @staticmethod
    def verify_sample_source():
        original = ast.parse(textwrap.dedent(inspect.getsource(FeatureStore.sample)))
        cached = ast.parse(textwrap.dedent(inspect.getsource(CachedFeatureStore.sample)))
        class RestoreLoader(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if isinstance(node.value,ast.Name) and node.value.id=='self' and node.attr=='read_episode':
                    return ast.copy_location(ast.Name(id='load_episode',ctx=ast.Load()),node)
                return self.generic_visit(node)
        if ast.dump(original,include_attributes=False) != ast.dump(RestoreLoader().visit(cached),include_attributes=False):
            raise ValueError('Cached sampler changes more than pure metadata reading')

    def read_episode(self, c, rec, metadata=False, keys=None):
        if c is not self.c or keys not in (('evidence',),('actions',)) or (metadata,keys[0]) not in ((True,'evidence'),(False,'actions')):
            raise ValueError('Cache only exposes original legal evidence timestamps or actions')
        key = (rec['id'],bool(metadata),keys[0])
        if key not in self.metadata:
            raise ValueError('Held-out or unknown metadata read is forbidden in training cache')
        return {keys[0]:self.metadata[key].copy()}

    def verify_boundaries(self):
        for name,expected in self.boundaries.items():
            st=Path(name).stat()
            if (st.st_size,st.st_mtime_ns) != tuple(expected):
                raise ValueError('Legal observed metadata changed after cache proof')

    def verify_all_inputs(self):
        original = FeatureStore(self.c,max_cached=self.max_cached)
        original.lru,original.projected,original.bases = self.lru,self.projected,self.bases
        checked=0
        rows=[r for r in self.manifest['records'] if (r['id'],True,'evidence') in self.metadata]
        for i,rec in enumerate(rows):
            for condition in TRAIN_CONDITIONS:
                a_rng=np.random.default_rng(90210+i);b_rng=np.random.default_rng(90210+i)
                a=original.sample(rec,condition,a_rng);b=self.sample(rec,condition,b_rng)
                if a_rng.bit_generator.state!=b_rng.bit_generator.state or set(a[0])!=set(b[0]) or set(a[2])!=set(b[2]):
                    raise ValueError('Cache changed input keys or random process')
                for left,right in ((a[0],b[0]),(a[2],b[2])):
                    for key,value in left.items():
                        if not np.array_equal(value,right[key]) or np.asarray(value).dtype!=np.asarray(right[key]).dtype:
                            raise ValueError('Original/cached sample differs: '+key)
                if not np.array_equal(a[1],b[1]) or a[1].dtype!=b[1].dtype:
                    raise ValueError('Original/cached target differs')
                checked+=1
        self.verify_boundaries()
        return {'episodes':len(rows),'conditions':4,'exact_samples_targets_info_rng':checked,'metadata_fingerprint':self.metadata_fingerprint,'no_held_out_rows':True,'sampler_AST_equal_except_pure_loader':True}

    def verify_validation_inputs(self,pairs):
        original=FeatureStore(self.c,max_cached=self.max_cached)
        original.lru,original.projected,original.bases=self.lru,self.projected,self.bases
        checked=0
        for condition in TRAIN_CONDITIONS:
            for i,(cached,target,n,versions) in enumerate(self.validation_batches[condition]):
                start=i*(self.c['training']['batch_size']//2)
                rows=[r for pair in pairs[start:start+self.c['training']['batch_size']//2] for r in pair]
                samples=[original.sample(row,condition) for row in rows]
                actual=collate([x[0] for x in samples],'cpu')
                actual_target=torch.from_numpy(np.stack([x[1] for x in samples]))
                if any(not torch.equal(actual[k],cached[k]) for k in actual) or not torch.equal(actual_target,target) or n!=len(rows):
                    raise ValueError('Original/cached validation tensor differs')
                checked+=1
        return checked

    def prepare_validation(self,pairs,device):
        batches={}
        for condition in TRAIN_CONDITIONS:
            entries=[]
            for start in range(0,len(pairs),self.c['training']['batch_size']//2):
                rows=[r for pair in pairs[start:start+self.c['training']['batch_size']//2] for r in pair]
                samples=[self.sample(r,condition) for r in rows]
                batch=collate([s[0] for s in samples],device)
                target=torch.from_numpy(np.stack([s[1] for s in samples])).to(device)
                entries.append((batch,target,len(rows),{key:value._version for key,value in batch.items()}))
            batches[condition]=entries
        self.validation_batches=batches

    def sample(self, rec, condition, rng=None, dropout=0):
        c, q = self.c, rec["query"]
        R, B = c["data"]["recent"], c["data"]["budget"]
        # Whitelist metadata: only evidence timestamps, never label/state.
        evidence = selected_evidence(c, rec, self.read_episode(c, rec, True, keys=("evidence",))["evidence"])
        if rng is None:
            # Same pair -> same random subset, so recent-only/random pair comparisons stay exact.
            rng = np.random.default_rng(stable_seed(c["evaluation"]["random_seed"], rec["pair"], condition))
        idx = memory_indices(q, R, B, evidence, condition, rng)
        if dropout and rng.random() < dropout:
            idx = idx[:0]
        z = self.features(rec)
        memory = z[idx].copy()
        if condition == "wrong_evidence":
            sibling = self.records[rec["sibling"]]
            sibling_z = self.features(sibling)
            for j, ti in enumerate(idx):
                if ti in evidence:
                    memory[j] = sibling_z[ti]
        times = np.concatenate([np.arange(q-R+1, q+1), idx])
        # Predictor training reads actions only; never decompresses RGB or accesses future images.
        o = self.read_episode(c, rec, keys=("actions",))
        # All controls executed since each observation are known at query time.
        # This is a legal summary of additive controls, not simulator state.
        prefix = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(o["actions"][:q], axis=0)])
        prev_actions = prefix[q]-prefix[times]
        sample = {"z": np.concatenate([z[q-R+1:q+1], memory]), "age": (q-times).astype(np.float32),
                  "type": np.concatenate([np.zeros(R, np.int64), np.ones(len(idx), np.int64)]),
                  "past_actions": prev_actions, "actions": o["actions"][q:q+max(c["data"]["horizons"])].copy(),
                  "current": z[q].copy()}
        if rec['task'] in self.bases:
            p = self.projected_features(rec)
            sample['projected_z'] = p[times].copy()
            if condition == 'wrong_evidence':
                sibling_p = self.projected_features(self.records[rec['sibling']])
                for j, ti in enumerate(idx):
                    if ti in evidence:
                        sample['projected_z'][R+j] = sibling_p[ti]
            sample['projected_current'] = p[q].copy()
        target = z[q+np.asarray(c["data"]["horizons"])].copy()
        info = {"indices": idx, "history_count": len(idx), "evidence_hit": bool(set(evidence.tolist()).issubset(idx.tolist()))}
        return sample, target, info

def cached_validation_loss(c,store,model,pairs,device,task):
    if store.validation_batches is None:
        raise ValueError('Exact original validation tensors must be prepared')
    model.eval()
    results={}
    ranking=rank_weight(c,task)
    with torch.inference_mode():
        for condition in TRAIN_CONDITIONS:
            losses=[]
            for batch,target,n,versions in store.validation_batches[condition]:
                loss,mse,pair_mse=regression_loss(model(batch),target,c['training']['pair_weight'],ranking)
                if any(batch[k]._version!=v for k,v in versions.items()):
                    raise ValueError('Model mutated reusable validation inputs')
                losses.append((float(loss),float(mse),float(pair_mse),n))
            results[condition]={key:sum(row[i]*row[3] for row in losses)/sum(row[3] for row in losses)
                                for i,key in enumerate(('loss','mse','pair_mse'))}
    return results
