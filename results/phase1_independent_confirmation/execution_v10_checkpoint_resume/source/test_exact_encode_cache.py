import numpy as np
import pytest
from exact_encode_cache import ExactEncodeCache
class Base:
 metadata={'frozen':'fixture'}
 def __init__(self):self.calls=0
 def encode(self,x):
  self.calls+=1
  return np.asarray(x,dtype=np.float32).reshape(len(x),-1)[:,:4].copy()
def rgb(n=2):return np.arange(n*2*2*3,dtype=np.uint8).reshape(n,2,2,3)
def test_bitwise_order_boundaries_and_output_mutation():
 b=Base();c=ExactEncodeCache(b);x=rgb();z=c.encode(x);z[:]=-9
 assert np.array_equal(c.encode(x),Base().encode(x)) and b.calls==1
 c.encode(x[::-1]);c.encode(x[:1]);assert b.calls==3
 assert c.metrics()['hit_frames']==2
 x[:]=0;c.encode(x);assert b.calls==4
 assert np.array_equal(c.encode(rgb()),Base().encode(rgb()))
def test_bounded_eviction_and_large_entry_bypass():
 x=rgb();b=Base();c=ExactEncodeCache(b,max_bytes=1)
 c.encode(x);c.encode(x);assert b.calls==2 and c.bytes==0
 c=ExactEncodeCache(b,max_bytes=70);c.encode(x);c.encode(x+1);assert c.bytes<=70
 c.encode(x);assert c.misses==3
@pytest.mark.parametrize('x',[rgb().astype(np.float32),np.zeros((3,2,2,3),np.uint8),np.zeros((2,2,2),np.uint8)])
def test_invalid_source_fails(x):
 with pytest.raises(ValueError):ExactEncodeCache(Base()).encode(x)
def test_wrong_output_fails():
 class Bad(Base):
  def encode(self,x):return np.full((len(x),4),np.nan,np.float32)
 with pytest.raises(ValueError):ExactEncodeCache(Bad()).encode(rgb())
def test_zero_bound_fails():
 with pytest.raises(ValueError):ExactEncodeCache(Base(),0)
