import importlib.util
from pathlib import Path
import sys
import types
import unittest
import numpy as np
import torch

sys.modules.setdefault('lmdb', types.SimpleNamespace(Environment=object))
from strided_tsm_reader import read_recording

spec=importlib.util.spec_from_file_location('upstream_prepare',
    Path(__file__).parents[1]/'SPACE/scripts/prepare_assembly101_tsm.py')
upstream=importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)


class Cursor:
    def __init__(self,env):self.env=env;self.keys=sorted(env.data);self.index=0
    def set_range(self,prefix):
        self.index=next((i for i,k in enumerate(self.keys) if k>=prefix),len(self.keys))
        return self.index<len(self.keys)
    def key(self):return self.keys[self.index]
    def value(self):self.env.fetched+=1;return self.env.data[self.key()]
    def next(self):self.index+=1;return self.index<len(self.keys)


class Env:
    def __init__(self,data):self.data=data;self.fetched=0
    def begin(self,**kwargs):return self
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def cursor(self):return Cursor(self)
    def get(self,key):self.fetched+=1;return self.data.get(key)


def fixture():
    rng=np.random.default_rng(51)
    return {f'r/C10095_rgb/C10095_rgb_{i}.jpg'.encode():
            rng.normal(size=2048).astype(np.float32).tobytes() for i in range(1,78)}


class StridedReaderTests(unittest.TestCase):
    def test_numeric_sort_bitwise_matches_author(self):
        for stride in (1,15,20):
            a=Env(fixture());b=Env(fixture())
            x,f=upstream.read_recording(a,'r',stride)
            y,g=read_recording(b,'r',stride)
            self.assertTrue(torch.equal(x,y));self.assertTrue(torch.equal(f,g))
            self.assertEqual(b.fetched,len(g));self.assertEqual(a.fetched,77)
    def test_boundary_does_not_read_adjacent_recording(self):
        data=fixture();data[b's/C10095_rgb/C10095_rgb_1.jpg']=b'invalid'
        env=Env(data);x,f=read_recording(env,'r',15)
        self.assertEqual(env.fetched,6);self.assertEqual(len(f),6)
    def test_selected_invalid_shape_rejected(self):
        data=fixture();data[b'r/C10095_rgb/C10095_rgb_1.jpg']=np.zeros(12,np.float32).tobytes()
        with self.assertRaises(ValueError):read_recording(Env(data),'r',15)
    def test_missing_recording_rejected(self):
        with self.assertRaises(KeyError):read_recording(Env(fixture()),'absent',15)
    def test_invalid_stride_rejected(self):
        with self.assertRaises(ValueError):read_recording(Env(fixture()),'r',0)


if __name__=='__main__':unittest.main()
