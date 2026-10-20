from pathlib import Path
import ast,importlib.util
import numpy as np
p=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('val_boundary',p/'experiments/gate_validation_deletion/boundary.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def fixture():
 g=np.random.default_rng(0);return g.normal(size=(35,6)).astype('float32'),g.normal(size=(35,2)).astype('float32'),g.normal(size=(35,2)).astype('float32'),20
def test_all_physical_delete_paths_equal_direct_packing():
 z,pr,a,q=fixture();pool=(2,6,10)
 for j,d in enumerate(pool):
  direct=m.pack(z,pr,a,q,tuple(t for t in pool if t!=d));removed=m.remove(m.pack(z,pr,a,q,pool),j);assert m.same(direct,removed)
  assert np.array_equal(direct['current'],z[q]) and direct['actions'].shape==(6,2)
def test_canonical_old_two_frame_sample_unchanged():
 # Execute only the actual old frozen sample definition as a numerical fixture.
 text=(p.parent/'phase1_jepa_memory/runs/phase1_deletion_assay/run_assay.py').read_text();tree=ast.parse(text);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='sample');ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'old_sample_fixture','exec'),ns)
 z,pr,a,q=fixture();assert m.same(m.pack(z,pr,a,q,(2,10)),ns['sample'](z,pr,a,q,(2,10)))
def test_reject_duplicate_future_and_recent_deletions():
 import pytest
 z,pr,a,q=fixture()
 for bad in ((2,2),(10,2),(-1,2),(2,17),(2,21)):
  with pytest.raises(ValueError):m.pack(z,pr,a,q,bad)
 with pytest.raises(ValueError):m.remove(m.pack(z,pr,a,q,(2,10)),2)
def test_pair_common_contrast_identity_and_no_response():
 p=np.array([[2.,1.],[2.,1.]]);t=np.array([[1.,2.],[-1.,-2.]])
 v=m.paired_errors(p,t);assert np.isclose(v['total'],v['common']+v['contrast']) and v['prediction_sibling_maxabs']==0
def test_analysis_distinguishes_same_capacity_composition_penalty():
 sp=importlib.util.spec_from_file_location('val_analysis',p/'experiments/gate_validation_deletion/analysis.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
 rows=[]
 for seed in (17,29,43):
  for pair in range(4):
   for h in (3,6):
    for v in ('recent_only','full_history','privileged_small_budget','random_small_budget'):
     rows.append(dict(seed=seed,pair=str(pair),horizon=h,variant=v,cue_present=int(v!='recent_only'),native_privileged=int(v=='privileged_small_budget'),feature_mse=2 if v=='recent_only' else 1,task_error=.5 if v=='recent_only' else 0,common=.2,contrast=1.8 if v=='recent_only' else .8))
    for gi,g in enumerate(a.GROUPS):
     for j in range(3):
      cue=j!=gi;native=g=='cue_oldest' and j==1;f=2 if not cue else (1 if native else 3)
      rows.append(dict(seed=seed,pair=str(pair),horizon=h,variant=f'{g}/delete_{j}',cue_present=int(cue),native_privileged=int(native),feature_mse=f,task_error=.5 if not cue else (0 if native else .25),common=.2 if not cue or native else 1.5,contrast=f-(.2 if not cue or native else 1.5)))
 s=a.summarize(rows,pairs=4,repeats=100)
 assert all(c['nonnative_minus_native_penalty']['feature_mse']['mean']==2 for c in s['comparisons'])
 import pytest
 with pytest.raises(AssertionError):a.summarize(rows[:-1],pairs=4,repeats=100)
