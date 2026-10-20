# Phase 1 independent confirmation: current handoff

Updated 2026-10-20 03:xx UTC. This file summarizes execution state; the scientific
contract remains in `execution_v10_checkpoint_resume/source/protocol.json`.

## Scientific objective

Test whether retaining more history improves future prediction beyond a
recent-frame-only predictor on a held-out seed. Gate tests a binary pass
property. Motion tests long occlusion, including the continued trajectory of
an object hidden behind a box. In the Motion figure, the diagonal arrow marks
the observed pre-occlusion trajectory; the hidden interval contains no visual
evidence. The paired endpoint is generated from the known action sequence, so
one endpoint is consistent with the arrow and the other is reversed. A model
that understands the trajectory should recover the hidden future endpoint;
a model using only the recent visible frame remains at chance.

The independent confirmation uses seed 9137 with 512 train, 96 validation, and
three 256-pair test splits per task. Reference and candidate are evaluated
across seeds 17, 29, and 43 with 120 epochs and 7680 steps per model. All
original thresholds remain fixed: Gate 51/51 prediction checks, Motion 51/51
prediction checks, 32/32 fixed-reference checks per task, and all 72 deletion
criteria. No threshold, sample count, seed, epoch count, or test split may be
changed after seeing the result.

## Execution state

- Training job 9242 completed successfully in 1:00:10 with exit code 0:0.
- All 12 independent-confirmation models completed 120 epochs and 7680 steps.
- All 62,418 required train/validation frames were encoded once. The cache is
  reused; do not encode the training data again.
- The CPU lock stage 9254 completed successfully and wrote
  `fresh/all_models_lock.json`.
- The lock contains 12 model proofs, 83 frozen files, `test_opened=false`, and
  passed the full model/config/permit/file checksum verifier.
- The fresh test manifest was generated only after the lock. Its generation
  receipt is `fresh/confirmation_test_generation_complete.json`.
- The remaining automatic chain is:
  9255 `encode_test` (A100) -> 9256 `evaluate_cpu` ->
  9257 `audit_cpu` -> 9258 `evidence_cpu`.
- Jobs 9255-9258 were queued at the last check. Job 9259 is a temporary
  zero-GPU Jupyter check session and is not part of the scientific chain.
- A stale 9249 lock attempt failed because a valid lock already existed. The
  failure is preserved; it did not alter the lock or open the test set.

## Recovery logic

The frozen scientific source is not modified. Execution-only wrappers under
`execution_v10_checkpoint_resume/execution_repair/final_continuation/` reuse
the completed training, validate the existing lock, and prequeue the remaining
stages. `submit_train_resume.py` and earlier failed attempts must not be rerun.

## Immediate next action

Wait for 9255 to run. Do not resubmit it. When it completes, inspect:

1. `stage_completed/lock_cpu.json`
2. `stage_completed/encode_test.json`
3. `stage_completed/evaluate_cpu.json`
4. `stage_completed/audit_cpu.json`
5. `stage_completed/evidence_cpu.json`
6. `fresh/primary_outcome.json` and the final audit/evidence manifests

Fetch only compact JSON, scalar CSV, small logs, and report assets. Keep
weights, feature arrays, raw NPZ files, and full prediction matrices on HPC.
Do not treat a bitwise-equivalence or execution-completion receipt as a
scientific pass.

## Budget

The user approved a cumulative 28,800 GPU-second cap, then added four hours.
The current scientific guard reports 69,299 seconds cap and 47,925 seconds
settled before the current chain, leaving 21,374 seconds. The queued GPU test
encoding stage has a 7,200-second hard limit. All startup, failed attempts,
encoding, and later GPU work count against the remaining authorization.

## Final interpretation

Only a fully completed independent confirmation with every original criterion
met supports `Phase1_GO`. A failed scientific criterion must be reported as a
standard failure. Missing execution evidence or an exhausted budget must be
reported as evidence insufficiency, never as a pass.

## Latest execution blocker

Authenticated OOD job details show both 9255 and 9259 PENDING with reason `user env retrieval failed requeued held`. This supersedes the ordinary queue interpretation above. The SSH master is absent and one normal BatchMode connection was denied; a single login-refresh request is pending. Preserve the existing jobs and all scientific locks. Diagnose native submission environment and allocation attempts before repairing the hold. No fresh test result is available yet. Current GPU accounting requires final duplicate-aware native settlement.

## Native repair completed at 2026 October 20 0406 UTC

SSH restored and CPU9266 actualRUNNING b-10-146 verified via authenticated Notebook public ED25519 plus exact Torch keyscan. Original9255 nativeCANCELLED by0 after env retrieval failure. Cloud Slurm rejected dependency update and release. The unallocated held9268 and old pending9256–9258 were explicitly cancelled and verified; they did no model work. One corrected chain was submitted with explicit ALL environment and no dependency on scheduler-purged completed9254, whose completion and model-lock hash were independently verified: 9270 encode_test → 9271 evaluate_cpu → 9272 audit_cpu → 9273 evidence_cpu. At0406UTC9270 is PENDING Priority, not held; provisional scheduler start0654UTC is not guaranteed. Seven compact files match native-before/local/native-after SHA. Science115 source and all12 locks unchanged. Accounting overlay preserves requeued allocation seconds, with conservative9255 charge floor3853s covering its full submission-to-cancellation interval; remaining cap lower bound17521s. This floor is not an actual GPU consumption claim. No scientific result yet. CPU9266 deadline041410UTC, use only while actualRUNNING; cancel after evidence and do not reuse onceEnded. Future9270 route must be pinned from normal actual stdout.

## Native encoding active at 2026 October 20 0418 UTC

Authenticated OOD shows 9270 `p1-env-encode_test` RUNNING on `b-19-30`
(`10.144.0.137`). Its stdout public ED25519 matched a fresh Torch
`ssh-keyscan` exactly and is pinned as `phase1-confirm10-encode9270`. Native
output shows test encoding progressing at 4,288 / 5,504 episodes at 04:25:00
UTC. Jobs 9271–9273 remain queued behind it. Training, the 62,418-frame
train/validation cache, all 12 model locks, and the frozen 9137 test manifest
remain unchanged. No evaluation result or scientific conclusion exists yet.

At 04:46:29 UTC, test encoding had completed and the automatic chain moved to
9271 `evaluate_cpu`, running on `b-10-102` (`10.144.0.120`). A fresh
`ssh-keyscan` matched the public output key exactly and is pinned as
`phase1-confirm10-eval9271`. The first progress record is reference/gate
seed29 on `test_id`. Jobs 9272 and 9273 remain queued behind it.

By 04:55 UTC both the reference and candidate Gate evaluations had completed
62,208 scalar rows each, covering all seeds, horizons, conditions, and test
splits. The same 9271 job is continuing into Motion evaluation; no failure
receipt exists and 9272–9273 remain queued.

## Final independent result at 2026 October 20 0556 UTC

Jobs 9270–9273 completed. The authenticated final artifact reports
`PHASE1_ORIGINAL_STANDARD_NOT_MET` with `phase1_go=false` and independent audit
status `PASS_INDEPENDENT_FRESH_PROTOCOL_AUDIT`. Gate passed the complete
155/155 registered check set. Motion passed 51/51 original prediction and
32/32 fixed-reference checks, but 64/72 deletion checks, for 147/155 total.
The eight failures are all Motion long-gap cross-horizon choices against the
reject-newest baseline: feature error improves 9.4–11.1%, while decoded
position-error improvement is near zero and every confidence interval crosses
zero. These are scientific failures, not execution failures. Phase 1 is not
passed and retention-policy training must not start.

The verified report is
`reports/phase1_independent_confirmation_20261020/Phase1_Independent_Confirmation_Report.docx`.
It was rendered to 10 pages and visually inspected page by page. Settlement
reports 53,544 of 69,299 approved GPU seconds, leaving 15,755 seconds unused.
