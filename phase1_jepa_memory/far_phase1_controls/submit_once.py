"""One-shot submission; any unknown outcome blocks replay. No science retry."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess
D=Path(__file__).resolve().parent
S='/opt/slurm/bin/'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(D/n).write_text(json.dumps(v,indent=2)+'\n')
def main():
 for n in ['submission_attempt.json','evaluation_submission.json','analysis_attempt.json','analysis_submission.json']:
  assert not (D/n).exists(),'Existing submission/outcome marker: '+n
 entries=json.loads((D/'source_lock.json').read_text())['files']
 assert all(sha(D/n)==h for n,h in entries.items())
 assert json.loads((D/'protocol.json').read_text())['budget']['new_wall_seconds']==7200
 active=subprocess.check_output([S+'squeue','-u','xl4358','-h','-o','%i|%a|%P|%T|%b|%j'],text=True)
 assert not any('gpu' in x.lower() or 'a100' in x.lower() for x in active.splitlines()),active
 for job in [7426,7435]:
  state=subprocess.check_output([S+'sacct','-j',str(job),'-n','-P','-o','JobID,State,ExitCode'],text=True)
  assert any(x.split('|')[:3]==[str(job),'COMPLETED','0:0'] for x in state.splitlines())
 with (D/'submission_attempt.json').open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unknown_outcome_refuse_replay':True,'source_lock_sha256':sha(D/'source_lock.json'),'squeue_before':active},f)
 r=subprocess.run([S+'sbatch','--parsable',str(D/'controls.sbatch')],capture_output=True,text=True)
 save('submission_outcome.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 assert r.returncode==0 and r.stdout.strip().split(';')[0].isdigit()
 job=int(r.stdout.strip().split(';')[0]);receipt={'job_id':job,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_lock_sha256':sha(D/'source_lock.json'),'protocol_sha256':sha(D/'protocol.json'),'stdout':r.stdout,'actual_scontrol':subprocess.check_output([S+'scontrol','show','job',str(job)],text=True)}
 save('evaluation_submission.json',receipt)
 with (D/'analysis_attempt.json').open('x') as f:json.dump({'parent':job,'dependency':'afterany','unknown_outcome_refuse_replay':True},f)
 r=subprocess.run([S+'sbatch','--parsable','--dependency=afterany:'+str(job),str(D/'analysis.sbatch')],capture_output=True,text=True)
 save('analysis_outcome.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 assert r.returncode==0 and r.stdout.strip().split(';')[0].isdigit()
 audit=int(r.stdout.strip().split(';')[0]);save('analysis_submission.json',{'job_id':audit,'parent':job,'dependency':'afterany:'+str(job),'source_lock_sha256':sha(D/'source_lock.json'),'stdout':r.stdout,'actual_scontrol':subprocess.check_output([S+'scontrol','show','job',str(audit)],text=True)})
 print(json.dumps({'science_job':job,'analysis_job':audit,'source_lock_sha256':receipt['source_lock_sha256']}))
if __name__=='__main__':main()
