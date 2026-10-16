import hashlib
from fractions import Fraction
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from phase1.v5_encoding import encode_record, work_projection, prefetch


def fixture_frames(future_value=7, count=256):
    for i in range(count):
        yield Fraction(i,2), Fraction(i,2), np.full((4,4,3),i if i<2 else future_value,dtype=np.uint8)


def timestamp_hash(count=256):
    h=hashlib.sha256()
    for i in range(count):
        stamp=Fraction(i,2)
        h.update(f"{i//256}:{i%256}:{stamp.numerator}/{stamp.denominator}\n".encode())
    return h.hexdigest()


class DummyEncoder:
    def encode(self,batch):
        assert batch.shape==(2,4,4,3)
        return np.repeat(batch.mean((1,2,3))[:,None],5120,axis=1).astype(np.float32)


def test_published_cache_prefix_independent_of_later_pixels(tmp_path):
    counts=[]
    for value in (7,99):
        path=tmp_path/f"{value}.npy"
        meta=encode_record(DummyEncoder(),fixture_frames(value),path,256,timestamp_hash(),counts.append,lambda:None)
        assert meta['forwards']==128 and meta['shape']==[256,5120]
    a,b=np.load(tmp_path/'7.npy'),np.load(tmp_path/'99.npy')
    np.testing.assert_array_equal(a[:2],b[:2])
    assert not np.array_equal(a[2:],b[2:])
    assert sum(counts)==512


def test_encoder_failure_never_publishes_complete_cache(tmp_path):
    class Fail(DummyEncoder):
        def encode(self,batch):raise RuntimeError('real failure')
    with pytest.raises(RuntimeError):
        encode_record(Fail(),fixture_frames(),tmp_path/'final.npy',256,timestamp_hash(),lambda n:None,lambda:None)
    assert not (tmp_path/'final.npy').exists()
    assert (tmp_path/'final.partial.npy').exists()


def test_wrong_timestamp_or_nonfinite_encoder_rejected(tmp_path):
    with pytest.raises(ValueError,match='timestamps'):
        encode_record(DummyEncoder(),fixture_frames(),tmp_path/'bad_time.npy',256,'0'*64,lambda n:None,lambda:None)
    assert not (tmp_path/'bad_time.npy').exists()
    class Nan(DummyEncoder):
        def encode(self,batch):return np.full((2,5120),np.nan,dtype=np.float32)
    with pytest.raises(ValueError,match='features'):
        encode_record(Nan(),fixture_frames(),tmp_path/'bad_feat.npy',256,timestamp_hash(),lambda n:None,lambda:None)
    assert not (tmp_path/'bad_feat.npy').exists()


def test_worker_failure_is_delivered_without_hanging():
    def failed():
        yield 1
        raise ValueError('decoder failure')
    stream=prefetch(failed(),1)
    assert next(stream)==1
    with pytest.raises(ValueError,match='decoder failure'):next(stream)


def test_fixed_budget_projection_cannot_shrink_or_repeat_preflight():
    assert work_projection(20,100,50688,512)==9920
    with pytest.raises(ValueError):work_projection(20,100,500,512)
    with pytest.raises(ValueError):work_projection(20,100,50688,1024)
