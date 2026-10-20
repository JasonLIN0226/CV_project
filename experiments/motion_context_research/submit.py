"""Single declared submission; unknown outcome is never retried."""
import sys
from common import *
def main(job):
 native(job,False);source_check();assert read(ROOT/'authorization.json')['new_small_calibration_fitting_authorized']
 assert read(ROOT/'preparation.json')['status']=='CACHED_CAUSAL_INPUT_PREFLIGHT_PASS'
 a=read(ROOT.parent/'association.json');assert all(c['contexts']['causal_scene_centre']['permutation_p']<.05 for c in a['cells'])
 with (ROOT/'submission_attempt.json').open('x') as f:json.dump({'CPU_job':job,'source_lock_sha256':sha(ROOT/'source_lock.json'),'no_retry_on_unknown_outcome':True},f)
 response=subprocess.check_output(['/opt/slurm/bin/sbatch','--parsable',str(ROOT/'fit.sbatch')],text=True).strip();gid=int(response.split(';')[0]);write(ROOT/'submission.json',dict(job=gid,response=response,source_lock_sha256=sha(ROOT/'source_lock.json'),protocol_sha256=sha(ROOT/'protocol.json')));print(response)
if __name__=='__main__':main(int(sys.argv[1]))
