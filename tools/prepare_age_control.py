#!/usr/bin/env python3
"""Freeze locally tested new execution sources. No connection, model or submission."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/age_control'
FILES=('PROTOCOL.md','protocol.json','counterbalanced_pools.py','analysis.py','run.py','independent_audit.py','run.sbatch','deploy.py','local_software_preflight.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    status=ROOT/'results/age_control/status.json'
    if status.exists() and json.loads(status.read_text()).get('science_submitted'):
        raise SystemExit('Science already submitted; never alter its frozen source')
    fixture=json.loads((BASE/'local_software_preflight.json').read_text())
    assert fixture['status']=='FULL_EXPECTED_AXES_SOFTWARE_FIXTURE_PASS' and fixture['synthetic_raw_rows']==124416
    assert fixture['test_source_sha256']==sha(ROOT/'tests/test_age_control.py')
    lock={'files':{name:sha(BASE/name) for name in FILES}}
    (BASE/'source_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    folder=ROOT/'build/age_control';folder.mkdir(parents=True,exist_ok=True)
    archive=folder/'phase1_age_control.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in (*FILES,'source_lock.json'):z.write(BASE/name,'phase1_age_control/'+name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    receipt={'stage':'READY_TO_DEPLOY_AWAITING_HUMAN_LOGIN','science_submitted':False,'job_id':None,
        'source_files':len(FILES),'source_lock_sha256':sha(BASE/'source_lock.json'),'protocol_sha256':sha(BASE/'protocol.json'),
        'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
        'software_tests_passed':31,'expected_raw_rows':124416,'expected_pair_rows':9216,'expected_statistics':1296,
        'execution':'One course n2c48m24 CPU batch;4 science threads16G30min; no GPU or training',
        'authentication':'SSH_KEY_NOT_LOADED_MASTER_ABSENT','human_login_requested_once':True,
        'new_GPU_seconds':0,'github_updated':False,'do_not_resubmit_unknown_outcome':True}
    status.parent.mkdir(parents=True,exist_ok=True);status.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
