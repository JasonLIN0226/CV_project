"""One recorded submission only, after successful read-only CPU packing."""
import sys
from common import *
def main(job):
 native(job,False);source_check();assert read(ROOT/'preparation.json')['status']=='READ_ONLY_PREFLIGHT_PASS';assert read(ROOT/'authorization.json')['new_head_fitting_approved'];p=read(ROOT/'protocol.json');assert p['prior_GPU_seconds']+600+p['provisional_confirmation_reserve_seconds']<=28800
 with (ROOT/'submission_attempt.json').open('x') as f:json.dump(dict(status='UNKNOWN_DO_NOT_RETRY',CPU_job=job,source_lock_sha256=sha(ROOT/'source_lock.json')),f)
 (ROOT/'logs').mkdir(exist_ok=True)
 r=subprocess.run(['/opt/slurm/bin/sbatch','--parsable',f'--output={ROOT}/logs/fit-%j.out',f'--error={ROOT}/logs/fit-%j.err',str(ROOT/'fit.sbatch')],capture_output=True,text=True)
 assert r.returncode==0,r.stderr;gid=int(r.stdout.strip().split(';')[0]);write(ROOT/'gpu_submission.json',dict(job=gid,CPU_job=job,actual_stdout=r.stdout,source_lock_sha256=sha(ROOT/'source_lock.json'),GPU_hard_cap_seconds=600,prior_settled_seconds=12723));print(json.dumps(dict(job=gid,submitted_once=True)),flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
