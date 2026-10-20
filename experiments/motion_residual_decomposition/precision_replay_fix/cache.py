"""Only load existing validation arrays and frozen transforms. No model modules."""
import collections,csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))

class Cache:
    def __init__(self):
        self.p=read(ROOT/'protocol.json');self.parent=Path(self.p['parent']);self.guard=Path(self.p['guard']);self.data=Path(self.p['data']);self.assets={}
        for root,key in ((self.parent,'parent'),(self.guard,'guard')):
            assert sha(root/'summary.json')==self.p[key+'_summary_sha256']
        self.check(self.parent/'baseline_vectors.npz',self.p['parent_vectors_sha256'])
        self.check(self.parent/'raw_deletion_constraint.csv',self.p['parent_raw_sha256'])
        self.check(self.guard/'raw_guard.csv',self.p['guard_raw_sha256'])
        for root,key in ((self.parent,'parent'),(self.guard,'guard')):
            self.check(root/'asset_ledger.json',self.p[key+'_asset_ledger_sha256'])
        self.expected=read(self.guard/'asset_ledger.json')
        self.check(self.data/'dataset/manifest.json');self.manifest=read(self.data/'dataset/manifest.json')
        self.check(self.data/'features/index.json');self.index=read(self.data/'features/index.json')
        self.check(self.data/'features/normalization.npz')
        with np.load(self.data/'features/normalization.npz',allow_pickle=False) as f:self.norm={k:f[k] for k in f.files}
        assert self.index['dataset_signature']==self.manifest['signature'] and str(self.norm['signature'])==self.index['signature']
        assert set(self.norm['train_ids'].tolist())=={r['id'] for r in self.manifest['records'] if r['split']=='train'}
        self.records={r['id']:r for r in self.manifest['records'] if r['task']=='motion' and r['split']=='validation'}
        assert len(self.records)==192 and len({r['pair'] for r in self.records.values()})==96
        self.check(self.data/'predictor/motion/basis.npz');self.check(self.data/'probes/motion_state_decoder.npz')
        with np.load(self.data/'predictor/motion/basis.npz',allow_pickle=False) as f:self.B=f['output_basis']
        with np.load(self.data/'probes/motion_state_decoder.npz',allow_pickle=False) as f:self.decoder={k:f[k] for k in f.files}
        with np.load(self.parent/'baseline_vectors.npz',allow_pickle=False) as f:self.pred=f['predictions']
        assert self.pred.shape==(20736,5120) and self.pred.dtype==np.float32 and np.isfinite(self.pred).all()
        self.raw=sorted([r for r in rows(self.guard/'raw_guard.csv') if r['arm']=='observed_guard'],key=lambda r:int(r['witness_index']))
        assert [int(r['witness_index']) for r in self.raw]==list(range(20736))
        for r in self.raw:
            assert r['split']=='validation' and r['episode'] in self.records and int(r['seed']) in (17,29,43) and int(r['horizon']) in (3,6)
        self.lru=collections.OrderedDict()
    def check(self,path,expected=None):
        path=Path(path);value=sha(path)
        if expected:assert value==expected,str(path)
        if hasattr(self,'expected') and str(path) in self.expected:assert value==self.expected[str(path)],str(path)
        self.assets[str(path)]=value
    def episode(self,eid):
        assert eid in self.records
        if eid not in self.lru:
            r=self.records[eid];feature=self.data/'features'/self.index['episodes'][eid]['path'];truth=self.data/'dataset'/r['eval']
            self.check(feature,self.index['episodes'][eid]['sha256']);self.check(truth,r['eval_sha256'])
            with np.load(feature,allow_pickle=False) as f:
                assert str(f['signature'])==self.index['signature'];z=f['z'];assert z.dtype==np.float32
            z=((z-self.norm['mean'])/self.norm['std']).astype(np.float32)
            with np.load(truth,allow_pickle=False) as f:states=f['states']
            self.lru[eid]=(z,states)
            if len(self.lru)>2:self.lru.popitem(last=False)
        self.lru.move_to_end(eid);return self.lru[eid]
    def target(self,r):
        z,states=self.episode(r['episode']);time=int(r['query'])+int(r['horizon']);return z[time],states[time]
    def verify(self):
        assert all(sha(k)==v for k,v in self.assets.items())
