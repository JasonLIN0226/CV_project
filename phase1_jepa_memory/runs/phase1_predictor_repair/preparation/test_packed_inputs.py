import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent/'provisional_control/src'))
from phase1.inputs import selected_evidence
from phase1.exact_runtime_cache import CachedFeatureStore
from packed_inputs import PackedTrainInputs, compare_original


class FakeStore:
    sample = CachedFeatureStore.sample

    def __init__(self):
        self.c={'data':{'recent':4,'budget':2,'horizons':[3,6]},'evaluation':{'random_seed':901}}
        self.bases={'gate':{}}
        self.arrays={}
        self.projections={}
        self.meta={}
        self.records={}
        rng=np.random.default_rng(123)
        self.pairs=[]
        for i,q in enumerate([11,8,14,7,16]):
            pair=[]
            for sibling in (0,1):
                ident=f'{i}_{sibling}'
                rec={'id':ident,'pair':str(i),'query':q,'task':'gate','split':'train',
                     'sibling':f'{i}_{1-sibling}'}
                pair.append(rec);self.records[ident]=rec
                self.arrays[ident]=rng.normal(size=(q+7,10)).astype(np.float32)
                self.projections[ident]=rng.normal(size=(q+7,3)).astype(np.float32)
                self.meta[ident]={'actions':rng.normal(size=(q+7,2)).astype(np.float32),
                                  'evidence':np.array([1,3],np.int64)}
            self.pairs.append(pair)

    def read_episode(self,c,rec,metadata=False,keys=None):
        return {k:self.meta[rec['id']][k].copy() for k in keys}

    def features(self,rec):return self.arrays[rec['id']]
    def projected_features(self,rec):return self.projections[rec['id']]


@pytest.mark.parametrize('condition',['recent_only','full_history','privileged_small_budget','random_small_budget'])
def test_variable_query_exact_tensor_target_and_indices(condition):
    store=FakeStore();packed=PackedTrainInputs(store,store.pairs,'cpu')
    for order in ([4,0,2],[1,3],[2]):
        pairs=[store.pairs[i] for i in order]
        assert compare_original(store,packed,pairs,condition,[77+i for i in order],'cpu')==2*len(order)


def test_no_test_or_unknown_row():
    store=FakeStore();store.pairs[0][0]['split']='test_id'
    with pytest.raises(ValueError): PackedTrainInputs(store,store.pairs,'cpu')


def test_cache_mutation_is_rejected():
    store=FakeStore();packed=PackedTrainInputs(store,store.pairs,'cpu')
    packed.base['z'].add_(1)
    with pytest.raises(ValueError):packed.verify_unchanged()
