"""Synthetic exact-cache fixtures; no research models trained."""
import unittest,numpy as np,torch
from phase1.exact_runtime_cache import CachedFeatureStore,cached_validation_loss
from phase1.training import validation_loss
class ExactCacheTests(unittest.TestCase):
 def test_sampler_ast_identity(self): CachedFeatureStore.verify_sample_source()
 def test_whitelist_and_copy(self):
  s=CachedFeatureStore.__new__(CachedFeatureStore);s.c={};v=np.array([[1,2],[3,4]],np.float32);v.flags.writeable=False;s.metadata={('r',False,'actions'):v}
  out=s.read_episode(s.c,{'id':'r'},keys=('actions',));out['actions'][0,0]=99
  self.assertEqual(s.metadata[('r',False,'actions')][0,0],1)
  with self.assertRaises(ValueError):s.read_episode(s.c,{'id':'r'},keys=('rgb',))
  with self.assertRaises(ValueError):s.read_episode(s.c,{'id':'held_out'},keys=('actions',))
 def test_validation_arithmetic_identical(self):
  torch.set_num_threads(1)
  c={'data':{'tasks':['gate']},'experiment':{'stage':'development','arm':'reference','tasks':['gate']},'training':{'batch_size':2,'pair_weight':2.,'rank_weight':.5}}
  class Store:
   def sample(self,r,condition):
    z=np.arange(16,dtype=np.float32).reshape(4,4)+r['i']
    b={'z':z,'age':np.arange(4,dtype=np.float32),'type':np.zeros(4,np.int64),'past_actions':np.zeros((4,2),np.float32),'actions':np.zeros((6,2),np.float32),'current':z[-1]}
    return b,np.stack([z[0],z[1]]),{}
  s=Store();s.c=c;s.validation_batches=None;pairs=[[{'i':0,'task':'gate'},{'i':1,'task':'gate'}]]
  class Model(torch.nn.Module):
   def forward(self,b):return b['current'][:,None].expand(-1,2,-1)*.3
  m=Model();a=validation_loss(c,s,m,pairs,'cpu','gate');CachedFeatureStore.prepare_validation(s,pairs,'cpu');b=cached_validation_loss(c,s,m,pairs,'cpu','gate');self.assertEqual(a,b)
if __name__=='__main__':unittest.main()
