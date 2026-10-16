"""Data-only label phase sensitivity. Does not choose a join or certify origin."""
from pathlib import Path
from fractions import Fraction
import json,hashlib,socket,subprocess
from collections import Counter

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
row=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','7243','-n','-P','--format=JobIDRaw,State,Account,Partition,NodeList'],text=True).strip().split('|')
assert row[:5]==['7243','RUNNING','csci_ga_2271-2026fa','interactive',socket.gethostname()]
base=Path('/scratch/xl4358/phase1_v5_data/annotation_runs/job_6945')
report=base/'annotation_availability_report.json'
assert sha(report)=='06cbeda4e0b1f2a9c4225553ea292fb458ba4dd520c8ce97ccccb00e9f7977b4'
audit=json.loads(report.read_text())
file=base/'development_intervals_unaligned.jsonl'
assert not file.is_symlink() and file.is_file()
assert sha(file)=='564e0b6471058b0168e05d945ba22f0c29221cf5503ce0fd4e5ed266bb0efde4'
for item in audit['source_files']:
 assert sha(base/'quarantine'/item['path'])==item['sha256']
encoding=Path('/scratch/xl4358/phase1_v5_real_encoding_frame/job_7187/encoding_report.json')
assert sha(encoding)=='8c13d5a6a9f417b23bcea184ccebc401315a7f0da47cdfa86991889731bbb50e'
cache=json.loads(encoding.read_text());rows=[json.loads(s) for s in file.read_text().splitlines()]
by={}
for r in rows:by.setdefault(r['video'],[]).append(r)
counts=Counter();changes=[]
for e in cache['completed_recordings']:
 video=e['rgb_path'].removeprefix('recordings/')
 assert video in by
 for episode in e['episodes']:
  # Hypotheses only: full-video uniform30fps lattice first sample at0 or1/60s.
  # This audit does not establish that the released extraction used either.
  start=Fraction(*episode['start_seconds'])
  for t in range(256):
   clock=start+Fraction(t,2)
   ids=[int((clock-phase)*30//1) for phase in (Fraction(0),Fraction(1,60))]
   labels=[]
   for index in ids:
    labels.append(sorted({r['labels'][1] for r in by[video] if r['labels'] is not None and r['start_frame']<=index<=r['end_frame']}))
   key=e['split'];counts[key+'_times']+=1
   if labels[0]!=labels[1]:
    counts[key+'_verb_union_disagreements']+=1
    if len(changes)<20:changes.append({'video':video,'clock':str(clock),'indices':ids,'verb_union_differs':True})
output=Path('/scratch/xl4358/phase1_v6_revision_metadata_7243/phase_ambiguity_audit.json')
assert not output.exists()
result={'status':'ACTUAL_DEVELOPMENT_PHASE_SENSITIVITY_NO_EXACT_ALIGNMENT_PROOF','clock_scope':'All50688fixeddevelopmenttimes','counts':dict(counts),'first20_disagreement_identifiers_no_label_payload':changes,'annotations_sha256':sha(file),'official6945_report_sha256':sha(report),'encoding_report_sha256':sha(encoding),'exact_raw_rgb_to_label_phase_proved':False,'label_to_feature_join_chosen':False,'conditional_hypotheses_only':['phase0','phase1/60s'],'unproven_assumption':'No crop/time origin shift and uniform30fps resampling; retained as hypotheses, never factual alignment','sealed_numeric_rows_read':False,'model_inference_fit_training':False}
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'output':str(output),'sha256':sha(output),'counts':dict(counts)}))
