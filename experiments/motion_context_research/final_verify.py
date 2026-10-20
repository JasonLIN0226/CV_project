"""One bounded 0GPU terminal-evidence receiver, after the scientific CPU job."""
from pathlib import Path
import datetime,hashlib,json,os,socket,subprocess,sys,time
ROOT=Path('/scratch/xl4358/phase1_motion_context_locked_test')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 job=int(os.environ['SLURM_JOB_ID']);print('JOB',job,'HOST',socket.gethostname(),'IP',socket.gethostbyname(socket.gethostname()),flush=True);print(Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text(),flush=True)
 native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);n=dict(x.split('=',1) for x in native.split() if '=' in x);assert n['JobState']=='RUNNING' and n['Account']=='csci_ga_2271-2026fa' and n['Partition']=='n2c48m24' and 'gres/gpu' not in n['AllocTRES']
 ledger=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','8505,8511,8512,8513','-n','-P','--format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,AllocTRES'],text=True);rows={int(r.split('|')[0]):r.split('|') for r in ledger.splitlines() if r.split('|')[0].isdigit()};assert rows[8511][1].startswith('CANCELLED') and int(rows[8511][3])==312 and rows[8512][1:4]==['COMPLETED','0:0','27'];total=12765+int(rows[8511][3])+int(rows[8512][3]);assert total==13104 and total<=28800
 target=ROOT/'output/job_8513';success=rows[8513][1:3]==['COMPLETED','0:0'];sources=read(ROOT/'source_lock.json')['files'];assert all(sha(ROOT/k)==v for k,v in sources.items());lock=read(ROOT/'candidate_lock.json');assert all(sha(k)==v for k,v in lock['weights_and_transforms'].items())
 names=['source_lock.json','protocol.json','candidate_lock.json','execution_attempt.json','execution_receipt.json','execution.out','execution.err','review-8513.out','review-8513.err','submission.json','output/job_8513/summary.json','output/job_8513/independent_audit.json','output/job_8513/paired_deletions.csv','output/job_8513/fixed_reference_pairs.csv','output/job_8513/witness_metadata.csv','output/job_8513/asset_ledger.json','output/job_8513/original/development_comparison.json','output/job_8513/original/independent_scalar_audit.json']
 files={name:{'sha256':sha(ROOT/name),'bytes':(ROOT/name).stat().st_size} for name in names if (ROOT/name).exists()}
 if success:
  s=read(target/'summary.json');a=read(target/'independent_audit.json');receipt=read(ROOT/'execution_receipt.json');assert receipt['status']=='COMPLETE_WITH_INDEPENDENT_AUDIT' and a['status']=='PASS_INDEPENDENT_LOCKED_MOTION_AUDIT';assert s['raw_prediction_rows']==34560 and s['raw_deletion_rows']==82944 and s['paired_deletion_rows']==2304
 else:s={'status':'EXECUTION_INCOMPLETE_OR_FAILURE'};a={}
 out=dict(status='NATIVE_TERMINAL_SOURCE_MODELS_AND_RECEIPTS_VERIFIED' if success else 'NATIVE_TERMINAL_PARTIAL_REVIEW',parent_job=8513,evidence_job=job,native_evidence=native,terminal_sacct=ledger,source_lock_sha256=sha(ROOT/'source_lock.json'),scientific_status=s['status'],prediction=s.get('prediction',{}),deletion_positive=s.get('deletion_positive'),deletion_total=s.get('deletion_total'),independent_audit=a.get('status'),GPU_settled_seconds=total,GPU_remaining_seconds=28800-total,GPU_total_cap_seconds=28800,compact_files=files,original_gates_changed=False,fresh9137=False,phase1_go=False,source_sha256=sha(__file__))
 (ROOT/'final_native.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('compact_files','prediction','native_evidence','terminal_sacct')}),flush=True)
 # Transfer receipt is a release signal only, never scientific input.
 for _ in range(120):
  if (ROOT/'final_transfer_done.json').exists():return
  time.sleep(5)
if __name__=='__main__':main()
