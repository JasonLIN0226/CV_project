"""Bounded memoization of byte-identical ordered RGB microbatches.

This is a pure frozen encoder execution cache, never a learned memory selector.
Batch boundaries, dtype, precision, frame order and numerical encoder are unchanged.
"""
from collections import OrderedDict
import hashlib,time
import numpy as np

class ExactEncodeCache:
 def __init__(self,base,max_bytes=256*1024*1024):
  self.base=base;self.metadata=base.metadata;self.max_bytes=int(max_bytes)
  if self.max_bytes<1:raise ValueError('Positive bounded cache required')
  self.entries=OrderedDict();self.bytes=0
  self.hits=self.misses=self.hit_frames=self.miss_frames=0
  self.hash_seconds=self.forward_seconds=0.
 def encode(self,rgb):
  x=np.asarray(rgb)
  if x.dtype!=np.uint8 or x.ndim!=4 or x.shape[-1]!=3 or not 1<=len(x)<=2:
   raise ValueError('Original ordered uint8 RGB batches of size1 or2 only')
  t=time.perf_counter();raw=np.ascontiguousarray(x).tobytes()
  key=(x.shape,x.dtype.str,hashlib.sha256(raw).digest());self.hash_seconds+=time.perf_counter()-t
  if key in self.entries:
   saved,output,size=self.entries.pop(key)
   if saved!=raw:raise ValueError('Hash collision or mutated encoder cache input')
   self.entries[key]=(saved,output,size);self.hits+=1;self.hit_frames+=len(x)
   return output.copy()
  t=time.perf_counter();z=self.base.encode(rgb);self.forward_seconds+=time.perf_counter()-t
  z=np.asarray(z)
  if z.dtype!=np.float32 or z.ndim!=2 or len(z)!=len(x) or not np.isfinite(z).all():
   raise ValueError('Original finite float32 feature output required')
  self.misses+=1;self.miss_frames+=len(x);size=len(raw)+z.nbytes
  if size<=self.max_bytes:
   while self.bytes+size>self.max_bytes:
    _,(_,_,old)=self.entries.popitem(last=False);self.bytes-=old
   out=z.copy();out.flags.writeable=False
   self.entries[key]=(raw,out,size);self.bytes+=size
  return z.copy()
 def metrics(self):
  return dict(hits=self.hits,misses=self.misses,hit_frames=self.hit_frames,
   forwarded_frames=self.miss_frames,cache_bytes=self.bytes,cache_max_bytes=self.max_bytes,
   hash_seconds=self.hash_seconds,forward_seconds=self.forward_seconds,
   input_identity='shape dtype SHA256 plus exact RGB bytes',batch_size_unchanged=2,
   precision_unchanged='float32',learned_policy=False)
