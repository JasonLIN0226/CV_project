# V9 execution interruption: evidence and recovery boundary

Authenticated normal portal Views show both CPU guardians8949/8950 received termination at20:21:02 UTC. GPU9030 subsequently raised the frozen heartbeat guard at20:21:30. Sequence accounting records reference/Gate/seed17 at80/120 epochs; no full model and no scientific acceptance result exist. These are portal observations, not native terminal settlement or checkpoint verification. Fresh tests remain closed.

The initial cancellation cause is unknown. Both jobs declared12-hour walls; do not label this their wall timeout, an authentication failure, heartbeat latency, or a user action without native evidence. Local SSH availability does not control Slurm allocation lifetime.

A separate, directly observed cleanup defect exists: both guardians aborted their cancellation loops when scontrol could no longer resolve completed GPU8951. The recovery implementation must independently reconcile terminal receipts through native accounting, isolate cancellation errors per job, attempt every still-owned active GPU, persist all errors, and verify terminal states. Missing or ambiguous accounting must remain unresolved, never treated as zero charge or success. This cleanup defect does not establish the cause of the original simultaneous CPU cancellation.

Before any GPU recovery:

1. Restore the existing normal NYU authentication; do not reuse Ended routes. Use one necessary short CPU receiver and its normally exposed public identity with an exact strict host-key pin.
2. Recover native duplicate/attempt accounting for8951/9030 and both guardians, including allocation starts, ends, states, exit codes, cancellation attribution/reasons where available, restart counts and configured walls. Reconcile observed startup wall with native accounting; do not charge only batch runtime.
3. Verify frozen109source lock, original train-only transforms/cache signatures, actual latest/best checkpoint hashes, epoch80/step5120 if actually present, all finite state, AdamW moments, scheduler, all RNG states,80 log rows and earliest validation minimum. The epoch/step expectation is an audit assertion, not a current proof.
4. Preserve old sources, receipts, failure logs and partials. A separate declared recovery may resume only an audited exact epoch boundary with the same optimizer/schedule/RNG, retaining original permit/source provenance. Do not regenerate or re-encode accepted inputs or restart completed stages unnecessarily.
5. Freeze and verify the targeted execution changes, test failure cleanup and serialization/resume equivalence, settle remaining budget under69299 seconds, and prove the complete remaining12-model/map/test sequence fits before unique submission. No threshold/sample/seed/epoch or scientific model change; no fresh-test access before all locks.
