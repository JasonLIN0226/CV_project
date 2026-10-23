"""Frozen-encoder execution proof on revealed6301 training RGB only."""
from pathlib import Path
import time
import numpy as np
import torch
from phase1.utils import read_json,write_json,file_hash
from exact_encode_cache import ExactEncodeCache

def benchmark_and_wrap(base,root,training_only):
 examples=[];sources={}
 for task in ('gate','motion'):
  D=Path('/scratch/xl4358/phase1_v3_dev')/task/'reference'
  c=read_json(D/'resolved_config.json');assert c['data']['seed']==6301
  m=read_json(D/'dataset/manifest.json');sources[str(D/'dataset/manifest.json')]=file_hash(D/'dataset/manifest.json')
  recs=[r for r in m['records'] if r['task']==task and r['split']=='train'][:2]
  for r in recs:
   p=D/'dataset'/r['obs'];assert file_hash(p)==r['obs_sha256']
   with np.load(p,allow_pickle=False) as f:rgb=f['rgb']
   examples.extend([rgb[:2].copy(),rgb[-2:].copy(),rgb[:1].copy()])
   sources[str(p)]=r['obs_sha256']
 assert len(examples)==12 and not base.model.training and all(not p.requires_grad for p in base.model.parameters())
 versions=tuple(p._version for p in base.model.parameters())
 base.encode(examples[0]);base.encode(examples[-1]);torch.cuda.synchronize()
 cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
 stream=examples+examples+examples
 start=time.perf_counter();expected=[base.encode(x) for x in stream];torch.cuda.synchronize();baseline=time.perf_counter()-start
 cache=ExactEncodeCache(base);start=time.perf_counter();actual=[cache.encode(x) for x in stream];torch.cuda.synchronize();cached=time.perf_counter()-start
 assert all(np.array_equal(a,b) and a.dtype==b.dtype for a,b in zip(expected,actual)), 'Exact execution cache changed features'
 assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state()),'Encoder execution consumed RNG'
 assert tuple(p._version for p in base.model.parameters())==versions
 assert cache.hits>0 and cache.miss_frames>0
 enabled=cached<baseline*.98
 proof=dict(status='PASS_NATIVE_ENCODER_EXECUTION_EQUIVALENCE',all_outputs_bitwise_equal=True,
  stream_batches=len(stream),logical_frames=sum(len(x) for x in stream),baseline_seconds=baseline,cached_seconds=cached,
  cache_enabled=enabled,speedup_on_explicit_repeated_input_fixture=baseline/cached,
  benchmark_scope='Revealed6301 training RGB only; repetition fixture does not predict full dataset hit rate',
  unchanged_RGB_processor_model_precision_batch_order=True,encoder_batch_size=2,dtype='float32',
  CPU_CUDA_RNG_unchanged=True,parameter_versions_unchanged=True,fixture_sources=sources,cache=cache.metrics())
 write_json(Path(root)/('encoder_efficiency_train.json' if training_only else 'encoder_efficiency_test.json'),proof)
 # Fresh stage statistics start at zero; do not count development fixtures as scientific data.
 return ExactEncodeCache(base) if enabled else base
