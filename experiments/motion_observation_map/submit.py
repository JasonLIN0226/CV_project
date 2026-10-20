"""One GPU->read-only CPU chain, with explicit unknown-outcome boundaries."""
from pathlib import Path
import hashlib,json,os,socket,subprocess,sys
ROOT=Path('/scratch/xl4358/phase1_motion_observation_map')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main(job):
 s=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',str(job)],text=True);n=dict(v.split('=',1) for v in s.split() if '=' in v);assert n['JobState']=='RUNNING' and n['Account']=='csci_ga_2271-2026fa' and 'gres/gpu' not in n['AllocTRES'] and n['NodeList']==socket.gethostname().split('.')[0]
 assert all(sha(ROOT/k)==v for k,v in read(ROOT/'source_lock.json')['files'].items());prep=read(ROOT/'preparation.json');assert prep['status']=='READ_ONLY_OBSERVATION_INNOVATION_CACHE_PASS' and sha(ROOT/'train.npz')==prep['train_sha256'] and sha(ROOT/'validation.npz')==prep['validation_sha256'];assert 13104+600+14490<=28800
 with (ROOT/'submission_attempt.json').open('x') as f:json.dump({'job':job,'unknown_outcome_no_retry':True,'source_lock_sha256':sha(ROOT/'source_lock.json')},f)
 response=subprocess.check_output(['/opt/slurm/bin/sbatch','--parsable',str(ROOT/'fit.sbatch')],text=True).strip();gid=int(response.split(';')[0]);receipt={'GPU_job':gid,'GPU_response':response,'source_lock_sha256':sha(ROOT/'source_lock.json')};(ROOT/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
 response=subprocess.check_output(['/opt/slurm/bin/sbatch','--parsable','--dependency=afterany:'+str(gid),str(ROOT/'review.sbatch')],text=True).strip();receipt.update(CPU_job=int(response.split(';')[0]),CPU_response=response);(ROOT/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main(int(sys.argv[1]))
