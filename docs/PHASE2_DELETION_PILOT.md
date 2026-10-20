# Frozen equal-capacity deletion pilot

`generate_deletion_pilot.py` uses the newly trained Gate/Motion predictor and real official I-JEPA cache to produce a bounded development deletion dataset. It does not resume unavailable repaired Phase 1 weights, train a retention scorer, run all six baseline comparisons, or validate PushT planning.

After both tasks in `train_predictor_pilot.py` have completed, run on CPU in an existing allocation:

```bash
"$TASK_ROOT/venv/bin/python" "$TASK_ROOT/repo/experiments/phase2/generate_deletion_pilot.py" \
  --data-root "$TASK_ROOT/data" \
  --encoded-report "$TASK_ROOT/runs/ACTUAL_COMPLETED_ENCODER_JOB/encoded" \
  --predictor-root "$TASK_ROOT/runs/cpu-predictor-pilot" \
  --output "$TASK_ROOT/runs/NEW_DELETION_OUTPUT" \
  --seed 17 --threads 4 --timeout-seconds 600
```

Use an actual completed encoded-report path and a new output directory. This command submits no job. Its worker hides CUDA, bounds wall time, and preserves partial files on failure; only a completed manifest establishes dataset completion. Enclosing Slurm allocation use must still be taken from scheduler accounting.

The model loader verifies completed training/encoding provenance, frozen checkpoint, basis and normalization identities. It reconstructs the unchanged supplied predictor, explicitly switches to evaluation mode, freezes parameters, and checks live parameters/buffers before and after label construction. The checkpoint's `frozen_verified` flag alone is insufficient.

The pilot uses the existing new development dataset: 64 training and 16 validation episodes across both tasks. It never loads `test_*` feature arrays. Each episode is replayed continuously from frame zero to its declared query under FIFO, reservoir, temporal buckets and visual diversity. The four policies generate parents; they are not the proposal's six model-bound baseline results. There is no fabricated surprise or attention score. The policy RNG is stable per task/group/policy and shared across paired siblings.

Immediately before the query's eviction, the parent consists of the two previously retained entries and the oldest frame leaving the recent buffer. The new recent window contains four distinct later entries. All three deletions retain exactly two real entries, with identical recent context, checkpoint and offline action continuation. Runtime action metadata retains only each resident entry's cumulative-prefix value plus the current prefix; no action or evicted-observation archive is given to the memory policy or predictor callback. Offline feature storage and output shards are not inference memory sources.

There are at most 320 policy-parent observations under these defaults (256 train and 64 validation), before deduplicating identical parents. The fixed rule selects only each episode's existing query and does not inspect a loss or future target to choose it. Duplicate parents reached by multiple policies are saved once with all origins. Missing horizons are masked; all-missing parents are excluded with a reason. This pilot rule and dataset size are engineering choices, not proposal requirements or performance gates.

For every parent, labels retain the full three-deletion by two-horizon matrix of raw MSE in the normalized full feature space. Predictions and targets are float32; residual/reduction computation is float64. No horizon aggregation, preference threshold or learned score is substituted for raw values. Deterministically selected parents are recomputed exactly, and each shard records checksums. Near ties remain raw numerical values, not assumed semantic preferences.

Outputs are physically separated:

- `scorer_inputs/<task>_<split>.npz`: candidate/recent features, timestamps and legal executed-action summaries, with parent IDs.
- `labels/<task>_<split>.npz`: raw deletion losses, horizon masks, deleted/retained timestamps and matching parent IDs.
- `metadata/<task>_<split>.json`: episode/group/split identity, parent-policy origins/RNG, resource counters and checkpoint hashes.
- `summary.json` and `run_manifest.json`: completed/partial status, provenance, shard hashes, exclusions, counts and explicit remaining scientific deliverables.

The scorer-input files contain neither future features nor future action sequences nor simulator state. Future actions are supplied only to the offline predictor call, limited to the requested horizon; future target features remain in the outer loss computation. Validation labels are separate from scorer-training labels, and sibling forks keep their original group and split.

Local tests use synthetic vectors and explicitly labeled fixtures. Passing them establishes software contracts only. Actual dataset claims require the real frozen-checkpoint run and its generated artifacts. A successful bounded pilot does not demonstrate predictor quality, a learned-retention improvement, complete Phase 2 baseline results or Phase 1 planning completion.

## Recorded 20 October run

The actual CPU pilot completed in **10.21 seconds** at `4ff9529d892f6ae6a255647493704d5d8e23e55c`, using the newly trained Gate/Motion models after A100 job 9285. Four continuous parent policies produced 320 policy-origin records, deduplicated into **238 parents**: **190 train / 48 validation**. Each parent has three equal-K=2 deletions and horizons 3/6: **714 deletion forks / 1,428 finite loss values**. Thirty preselected parents were recomputed exactly. No test feature arrays were loaded and there were no missing horizons or excluded episodes.

The downloaded new-development backup includes all raw feature arrays, bases, checkpoints and supervision shards. All **472 file hashes** and the archive SHA-256 were checked locally. Independent shard checks confirm matching parent IDs, input/label separation, causal timestamp bounds, exactly two retained entries per fork and finite loss values. Source hashes match the executed commit. See [portable proof](../results/phase2_hpc/20261020/deletion_pilot.json).

These labels come from the small new predictor pilot whose trained validation error was slightly worse than its initialization. Their availability establishes the supervision pipeline; it does not establish useful ranking signal, a trained scorer, or superior retention. The scorer remains a later proposal phase.
