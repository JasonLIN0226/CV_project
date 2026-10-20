"""One bounded 0GPU terminal-evidence receiver, after the scientific CPU job."""
from pathlib import Path
import datetime,hashlib,json,os,socket,subprocess,sys,time
ROOT=Path('/scratch/xl4358/phase1_motion_observation_mix_locked_test')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 job=int(os.environ['SLURM_JOB_ID']);parent=int(read(ROOT/'submission.json')['job']);print('JOB',job,'HOST',socket.gethostname(),'IP',socket.gethostbyname(socket.gethostname()),flush=True);print(Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text(),flush=True)
 native=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);n=dict(x.split('=',1) for x in native.split() if '=' in x);assert n['JobState']=='RUNNING' and n['Account']=='csci_ga_2271-2026fa' and n['Partition']=='n2c48m24' and 'gres/gpu' not in n['AllocTRES']
 ledger=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','8518,'+str(parent),'-n','-P','--format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,AllocTRES'],text=True);rows={int(r.split('|')[0]):r.split('|') for r in ledger.splitlines() if r.split('|')[0].isdigit()};assert rows[8518][1:4]==['COMPLETED','0:0','27'];total=13104+int(rows[8518][3]);assert total==13131 and total<=28800
 target=ROOT/f'output/job_{parent}';success=rows[parent][1:3]==['COMPLETED','0:0'];sources=read(ROOT/'source_lock.json')['files'];assert all(sha(ROOT/k)==v for k,v in sources.items());lock=read(ROOT/'candidate_lock.json');assert all(sha(k)==v for k,v in lock['weights_and_transforms'].items())
 names=['source_lock.json','protocol.json','candidate_lock.json','execution_attempt.json','execution_receipt.json','execution.out','execution.err',f'review-{parent}.out',f'review-{parent}.err','submission.json',f'output/job_{parent}/summary.json',f'output/job_{parent}/independent_audit.json',f'output/job_{parent}/paired_deletions.csv',f'output/job_{parent}/fixed_reference_pairs.csv',f'output/job_{parent}/witness_metadata.csv',f'output/job_{parent}/asset_ledger.json',f'output/job_{parent}/original/development_comparison.json',f'output/job_{parent}/original/independent_scalar_audit.json']
 files={name:{'sha256':sha(ROOT/name),'bytes':(ROOT/name).stat().st_size} for name in names if (ROOT/name).exists()}
 if success:
  s=read(target/'summary.json');a=read(target/'independent_audit.json');receipt=read(ROOT/'execution_receipt.json');assert receipt['status']=='COMPLETE_WITH_INDEPENDENT_AUDIT' and a['status']=='PASS_INDEPENDENT_LOCKED_MOTION_AUDIT';assert s['raw_prediction_rows']==34560 and s['raw_deletion_rows']==82944 and s['paired_deletion_rows']==2304
 else:s={'status':'EXECUTION_INCOMPLETE_OR_FAILURE'};a={}
 out=dict(status='NATIVE_TERMINAL_SOURCE_MODELS_AND_RECEIPTS_VERIFIED' if success else 'NATIVE_TERMINAL_PARTIAL_REVIEW',parent_job=parent,evidence_job=job,native_evidence=native,terminal_sacct=ledger,source_lock_sha256=sha(ROOT/'source_lock.json'),scientific_status=s['status'],prediction=s.get('prediction',{}),deletion_positive=s.get('deletion_positive'),deletion_total=s.get('deletion_total'),independent_audit=a.get('status'),GPU_settled_seconds=total,GPU_remaining_seconds=28800-total,GPU_total_cap_seconds=28800,compact_files=files,original_gates_changed=False,fresh9137=False,phase1_go=False,source_sha256=sha(__file__))
 (ROOT/'final_native.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('compact_files','prediction','native_evidence','terminal_sacct')}),flush=True)
 # Transfer receipt is a release signal only, never scientific input.
 for _ in range(120):
  if (ROOT/'final_transfer_done.json').exists():return
  time.sleep(5)
if __name__=='__main__':main()
