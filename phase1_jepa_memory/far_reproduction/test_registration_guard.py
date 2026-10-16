"""Failure-boundary tests: fake Slurm replies, never submit a real job."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import register_reference_evaluation as controller


class Guard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.previous = controller.ROOT
        controller.ROOT = self.root
        self.env = patch.dict(os.environ, {"SLURM_JOB_ACCOUNT": "csci_ga_2271-2026fa",
                                          "SLURM_JOB_ID": "9001"})
        self.env.start()
        (self.root / "reference_protocol.json").write_text("{}")
        (self.root / "evaluation_source_lock.json").write_text(json.dumps({"files": {
            "reference_protocol.json": hashlib.sha256(b"{}").hexdigest()}}))
        prep = self.root / "preparation/job_7385"
        prep.mkdir(parents=True)
        self.report = prep / "preparation_report.json"
        self.report.write_text(json.dumps({"status": "PUBLIC_RELEASE_FILES_AND_DEPENDENCIES_PREPARED",
                                          "test_clips_declared": 3840}))
        self.sacct = "7385|COMPLETED|0:0|csci_ga_2271-2026fa|cpu=24\n"
        self.queue = "9001 cv-far-reference-bind N/A\n"

    def tearDown(self):
        controller.ROOT = self.previous
        self.env.stop()
        self.tmp.cleanup()

    def output(self, args, **kw):
        return self.sacct if args[0].endswith("sacct") else self.queue

    def execute(self, stdout="9002\n"):
        with patch.object(subprocess, "check_output", side_effect=self.output), \
             patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout, "")) as submit:
            try:
                controller.main()
            finally:
                self.submission_calls = submit.call_count

    def test_actual_failure_never_reaches_submission(self):
        self.sacct = self.sacct.replace("COMPLETED", "FAILED")
        with self.assertRaises(RuntimeError): self.execute()
        self.assertEqual(self.submission_calls, 0)

    def test_wrong_course_never_reaches_submission(self):
        self.sacct = self.sacct.replace("csci_ga_2271-2026fa", "other_course")
        with self.assertRaises(RuntimeError): self.execute()
        self.assertEqual(self.submission_calls, 0)

    def test_partial_test_split_never_reaches_submission(self):
        self.report.write_text(json.dumps({"status": "PUBLIC_RELEASE_FILES_AND_DEPENDENCIES_PREPARED",
                                          "test_clips_declared": 100}))
        with self.assertRaises(RuntimeError): self.execute()
        self.assertEqual(self.submission_calls, 0)

    def test_unknown_attempt_is_not_retried(self):
        (self.root / "evaluation_submission_attempt.json").write_text('{"status":"UNKNOWN"}')
        with self.assertRaises(FileExistsError): self.execute()
        self.assertEqual(self.submission_calls, 0)

    def test_parallel_allocation_is_rejected(self):
        self.queue += "8999 other-science gpu:1\n"
        with self.assertRaises(RuntimeError): self.execute()
        self.assertEqual(self.submission_calls, 0)

    def test_changed_protocol_is_rejected(self):
        (self.root / "reference_protocol.json").write_text('{"changed":true}')
        with self.assertRaises(RuntimeError): self.execute()
        self.assertEqual(self.submission_calls, 0)

    def test_exact_receipt_is_recorded_once(self):
        self.execute()
        self.assertEqual(self.submission_calls, 1)
        self.assertEqual(json.loads((self.root / "evaluation_submission.json").read_text())["job_id"], 9002)
        with self.assertRaises(FileExistsError): self.execute()
        self.assertEqual(self.submission_calls, 0)

    def test_malformed_receipt_remains_unknown(self):
        with self.assertRaises(RuntimeError): self.execute("unexpected result")
        self.assertEqual(self.submission_calls, 1)
        self.assertFalse((self.root / "evaluation_submission.json").exists())
        with self.assertRaises(FileExistsError): self.execute()
        self.assertEqual(self.submission_calls, 0)


if __name__ == "__main__":
    unittest.main()
