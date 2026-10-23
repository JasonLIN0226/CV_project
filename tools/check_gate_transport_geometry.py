"""Posthoc legal-time geometry only; no predictor calls or fitted thresholds."""
from pathlib import Path
import collections,csv,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/gate_validation_deletion';P=R/'native_completed/output/job_8321/paired_validation.csv'
def main():
 s=json.loads((R/'native_completed/output/job_8321/summary.json').read_text());assert hashlib.sha256(P.read_bytes()).hexdigest()==s['paired_sha256']
 with P.open() as f:rows=list(csv.DictReader(f))
 by=collections.defaultdict(list);seen=set()
 for r in rows:
  if r['variant']!='privileged_small_budget' and not (r['family']=='deletion' and int(r['cue_present']) and not int(r['native_privileged'])):continue
  # Geometry repeats across model seeds. Count it once, not three times.
  key=(r['pair'],r['variant'],r['horizon'])
  if key in seen:continue
  seen.add(key);t=tuple(map(int,r['kept'].split(';')));assert len(t)==2 and t[1]>t[0]
  gain=(int(r['query'])-t[1]+int(r['horizon']))/(t[1]-t[0]);bucket='native_privileged' if r['variant']=='privileged_small_budget' else r['pool_group']
  by[bucket].append(gain)
 stats={k:dict(unique_pair_variant_horizon_records=len(v),minimum=float(np.min(v)),median=float(np.median(v)),p95=float(np.quantile(v,.95)),maximum=float(np.max(v))) for k,v in by.items()}
 result=dict(status='POSTHOC_TIMESTAMP_GEOMETRY_COMPLETE',declaration_sha256=hashlib.sha256((R/'transport_geometry_declaration.json').read_bytes()).hexdigest(),paired_source_sha256=s['paired_sha256'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),transport_gain=stats,new_model_calls=0,GPU=0,interpretation='Larger factor permits stronger amplification of observed feature differences; not a measured learned branch norm and not causal proof',repair_candidate_not_executed='One Gate-only frozen forward ablation of transport contribution on original validation; preserve all weights and inputs, evaluate content branch and frozen recent reference without transport; do not adopt absent validation benefit and original gates')
 (R/'transport_geometry.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
