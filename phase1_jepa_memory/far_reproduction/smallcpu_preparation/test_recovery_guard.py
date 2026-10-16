"""Fake scheduler and receipts; no allocation or real data/model execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import register_recovered_evaluation as ctl
from recover_dependencies import require_failure_boundary


class Guard(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.root=self.base/'reference';self.own=self.base/'recovery';self.root.mkdir();self.own.mkdir()
        self.old=(ctl.ROOT,ctl.OWN);ctl.ROOT,ctl.OWN=self.root,self.own
        self.env=patch.dict(os.environ,{'SLURM_JOB_ACCOUNT':'csci_ga_2271-2026fa','SLURM_JOB_ID':'9001'});self.env.start()
        (self.root/'reference_protocol.json').write_text('{}')
        (self.root/'evaluation_source_lock.json').write_text(json.dumps({'files':{'reference_protocol.json':hashlib.sha256(b'{}').hexdigest()}}))
        (self.own/'recovery_source_lock.json').write_text(json.dumps({'files':{}}))
        (self.own/'preparation_submission.json').write_text(json.dumps({'job_id':9000}))
        self.report=self.own/'output/job_9000/preparation_report.json';self.report.parent.mkdir(parents=True)
        self.d={'status':'PUBLIC_RELEASE_FILES_AND_DEPENDENCIES_PREPARED','test_clips_declared':3840,'no_redownload_or_reextract':True,'model_inference_or_training':False,'recovery_source_sha256':hashlib.sha256((self.own/'recovery_source_lock.json').read_bytes()).hexdigest()}
        self.report.write_text(json.dumps(self.d))
        self.state='9000|COMPLETED|0:0|csci_ga_2271-2026fa\n';self.queue='9001 binding N/A\n'
        realsha=ctl.sha
        self.sha=patch.object(ctl,'sha',side_effect=lambda p:'a941ab106917c3efdb7516806b1f1a21faa870dc47d5e229e2fec10d512df0f9' if Path(p)==self.root/'evaluation_source_lock.json' else realsha(p));self.sha.start()

    def tearDown(self):
        self.sha.stop();self.env.stop();ctl.ROOT,ctl.OWN=self.old;self.tmp.cleanup()

    def execute(self,value='9002\n'):
        with patch.object(subprocess,'check_output',side_effect=lambda a,**kw:self.state if a[0].endswith('sacct') else self.queue),patch.object(subprocess,'run',return_value=subprocess.CompletedProcess([],0,value,'')) as submit:
            try:ctl.main()
            finally:self.calls=submit.call_count

    def test_native_failed_parent_is_rejected(self):
        self.state=self.state.replace('COMPLETED','FAILED')
        with self.assertRaises(RuntimeError):self.execute()
        self.assertEqual(self.calls,0)

    def test_partial_report_is_rejected(self):
        self.d['test_clips_declared']=100;self.report.write_text(json.dumps(self.d))
        with self.assertRaises(RuntimeError):self.execute()
        self.assertEqual(self.calls,0)

    def test_source_changed_is_rejected(self):
        (self.root/'reference_protocol.json').write_text('{"changed":true}')
        with self.assertRaises(RuntimeError):self.execute()
        self.assertEqual(self.calls,0)

    def test_parallel_allocation_is_rejected(self):
        self.queue+='8999 training gpu:1\n'
        with self.assertRaises(RuntimeError):self.execute()
        self.assertEqual(self.calls,0)

    def test_unknown_marker_is_never_retried(self):
        (self.root/'evaluation_submission_attempt.json').write_text('UNKNOWN')
        with self.assertRaises(RuntimeError):self.execute()
        self.assertEqual(self.calls,0)

    def test_exact_once_and_bound_recovery_id(self):
        self.execute();self.assertEqual(self.calls,1)
        d=json.loads((self.root/'evaluation_submission.json').read_text());self.assertEqual((d['job_id'],d['preparation_job'],d['original_failed_preparation_job']),(9002,9000,7385))
        with self.assertRaises(RuntimeError):self.execute()
        self.assertEqual(self.calls,0)

    def test_unknown_submission_outcome_is_never_retried(self):
        with self.assertRaises(RuntimeError):self.execute('unknown receipt')
        self.assertEqual(self.calls,1)
        with self.assertRaises(RuntimeError):self.execute()
        self.assertEqual(self.calls,0)

    def test_original_failure_and_cancel_boundary(self):
        require_failure_boundary('7385|FAILED|1:0|csci_ga_2271-2026fa\n7386|CANCELLED by 3850228|0:0|csci_ga_2271-2026fa\n')
        with self.assertRaises(RuntimeError):require_failure_boundary('7385|COMPLETED|0:0|csci_ga_2271-2026fa\n7386|CANCELLED|0:0|csci_ga_2271-2026fa\n')


if __name__=='__main__':unittest.main()
