import os,subprocess,sys,traceback
from common import *
def main(job):
 native(job,False);source_check()
 with (ROOT/'review_attempt.json').open('x') as f:json.dump(dict(job=job,no_retry=True),f)
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
 for script in ('audit_fit.py','validate.py','audit_validation.py'):
  if script=='validate.py' and read(ROOT/'fit_independent_audit.json')['scientific_status']=='TRAIN_PREDICTABILITY_REJECTED':
   write(ROOT/'outcome.json',dict(status='TRAIN_PREDICTABILITY_REJECTED',validation_evaluated=False,test_opened=False,phase1_go=False));return
  subprocess.run(['/scratch/xl4358/phase1_env/bin/python',str(ROOT/script),str(job)],check=True,env=env)
 s=read(ROOT/'validation/summary.json');write(ROOT/'outcome.json',dict(status='VALIDATION_SUPPORTED_PENDING_LOCKED_TEST' if s['validation_eligible'] else 'VALIDATION_REJECTED_STOP',test_opened=False,phase1_go=False))
if __name__=='__main__':
 try:main(int(sys.argv[1]))
 except Exception:
  write(ROOT/'review_failure.json',dict(error=traceback.format_exc(),no_automatic_retry=True));raise
