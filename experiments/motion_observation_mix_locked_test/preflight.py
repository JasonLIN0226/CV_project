"""Synthetic software axes/independent-statistics check, not scientific evidence."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import deletion_analysis as primary
from deletion_independent import audit_rows
primary.GROUPS={'motion':('clock_all',)}
def main():
 rows=[]
 for seed in (17,29,43):
  for split in ('test_id','test_long_gap','test_distractor_shift'):
   for i in range(4):
    pair=f'{split}_{i}';q=30;key=json.dumps((20261017,'clock_only',pair),sort_keys=True,separators=(',',':')).encode();rng=np.random.default_rng(int(hashlib.sha256(key).hexdigest()[:16],16)%(2**32));triples=set()
    while len(triples)<6:triples.add(tuple(sorted(map(int,rng.choice(q-3,3,replace=False)))))
    for sibling in range(2):
     for pi,pool in enumerate(sorted(triples)):
      for deleted in pool:
       for h in (3,6):rows.append(dict(task='motion',seed=seed,split=split,pair=pair,episode=f'{pair}_{sibling}',query=q,evidence_end=14,evidence='2;5',pool=f'clock_{pi}',pool_group='clock_all',parent=';'.join(map(str,pool)),kept=';'.join(str(t) for t in pool if t!=deleted),deleted=deleted,deleted_role='start' if deleted==2 else 'end' if deleted==5 else 'other',history_count=2,feature_dim=5120,horizon=h,feature_mse=.3+.001*deleted+.0001*seed+.0002*h,task_error=.01+.0001*deleted+.00001*h))
 compact,_=primary.aggregate(rows,pairs=4);summary=primary.summarize(compact,repeats=32,pairs=4);a=audit_rows(rows,compact,summary,pairs=4,repeats=32)
 assert a['status']=='INDEPENDENT_RAW_SCALAR_AUDIT_PASS' and a['independent_scalar_statistics']==288
 result=dict(status='SOFTWARE_FIXTURE_PASS_NOT_SCIENCE',raw_rows=len(rows),paired_rows=len(compact),independent_statistics=288,fixture_pairs=4,fixture_bootstrap=32,scientific_bootstrap_unchanged=10000,GPU=0)
 Path(__file__).with_name('software_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
