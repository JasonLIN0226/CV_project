# Six retention policies: frozen-predictor validation pilot

This bounded CPU pilot runs FIFO, reservoir, temporal buckets, visual diversity, prediction surprise and attention with the same newly trained frozen checkpoint per task. It uses the 16 existing validation episodes (eight independent sibling pairs across Gate/Motion), yielding 96 policy/episode rows with horizons 3 and 6. Every policy replays each complete episode; the scored query is the dataset's fixed declared query. Checkpoint selection already used this validation split, so results are development observations, not untouched-test confirmation.

```bash
"$TASK_ROOT/venv/bin/python" "$TASK_ROOT/repo/experiments/phase2/run_retention_baseline_pilot.py" \
  --data-root "$TASK_ROOT/data" \
  --encoded-report "$TASK_ROOT/runs/9285/encoded" \
  --predictor-root "$TASK_ROOT/runs/cpu-predictor-pilot" \
  --output "$TASK_ROOT/runs/NEW_SIX_POLICY_OUTPUT" \
  --seed 17 --threads 4 --timeout-seconds 600
```

Use the actual completed report path and a new output directory. The worker requests no GPU, uses at most four CPU threads, and stops at its time limit. Partial files do not establish completion. Scheduler accounting remains necessary for any enclosing allocation.

Attention is the actual predictor's learned content-salience score plus its existing novelty term and softmax normalization over the K+1 current candidates. It is conditioned on the current feature and candidate contents only. This predictor has no Transformer heads; the report does not describe this weight as multihead attention.

Prediction surprise at time `t` is the h=3 normalized-feature prediction error from the context at `q=t-3`, using only controls actually executed during `[q,t)`. At `q`, the original model's action-independent context is compressed into `5*d+3` float32 sufficient statistics. The three observed controls are accumulated as they occur; once `z_t` arrives, the same frozen action/output layers produce the prediction and the outer wrapper computes its error. The `q` raw current feature remains in the R=4 recent buffer until this calculation; it is never recovered from an evicted-feature archive. The scalar error stays attached to the observation when that observation later becomes a memory candidate.

The first complete recent context occurs at frame 3, so surprise becomes available at frame 6. Earlier observations have no score and receive the existing policy's lowest priority, with deterministic oldest-first tie handling. Pending contexts are limited to three and discarded at episode reset. Tests compare reconstructed h3 outputs with the original predictor forward computation; failure is a software blocker rather than a reason to substitute a synthetic score.

The shared real-observation capacity is K=2 with R=4, at most one additional transient observation. **Total memory bytes are unequal across policies.** Surprise additionally stores three compressed contexts and short realized-action prefixes. The report separately counts statistic floats, origin times, pending controls, resident action-prefix metadata, persistent logical payload and transient prefix storage. Each prepared context is `(5*d+3)*4+8` bytes including its timestamp (2,580 bytes at d=128), before its partial action prefix. Shared model parameters/buffers are reported separately. Python allocator overhead and transient tensor workspaces are excluded; these logical counters are not process RSS or a fully byte-matched comparison.

Retentions receive observations one at a time and at most the immediately executed action. Full future action prefixes enter only a separate offline query-prediction call; targets enter only the outer metric computation. No simulator state, cue labels, future feature vectors, output trace or feature-store reference enters a retention callback. Only validation feature arrays are loaded by this runner.

Outputs include the complete online retention trace, per-episode query predictions, raw per-horizon feature MSE, retained timestamps, grouped-pair means and sample standard deviations, latency, memory counters and frozen checkpoint/cache/source hashes. Siblings and horizons do not count as independent samples. The training record's initial versus selected validation MSE is preserved: successful training execution is not evidence of predictive improvement, and a positive reported percentage means worse MSE.

This run trains neither a retention scorer nor another predictor and does not run PushT planning. It supplies limited reproducible six-policy prediction results if execution completes. It is not a claim of learned-policy benefit, generalization, equal auxiliary memory budgets or complete scientific Phase 2.

## Recorded execution

The actual Linux CPU pilot completed in **7.51 seconds** at `6ac378b3d24a39611cb55c862e934e06381cbae0`, after **18 targeted Linux tests passed**. It produced 96 episode/policy rows, 192 per-horizon losses, and 12 task/policy aggregate cells. No test feature arrays were loaded; both checkpoint/state identities remained unchanged. Full raw predictions and online traces were downloaded and all nine exported file hashes verified.

The following values are mean paired validation MSE (lower is better), with **four independent scene pairs per task**. FIFO and temporal buckets tied for the lowest means in this small run; the result does not establish generalization or a winner on fresh test data.

| Policy | Gate h3 | Gate h6 | Motion h3 | Motion h6 |
|---|---:|---:|---:|---:|
| fifo | 0.105594 | 0.106590 | 0.155914 | 0.151745 |
| reservoir | 0.116191 | 0.124455 | 0.158084 | 0.153838 |
| temporal_buckets | 0.105594 | 0.106590 | 0.155914 | 0.151745 |
| visual_diversity | 0.107369 | 0.107626 | 0.156560 | 0.152019 |
| prediction_surprise | 0.106218 | 0.106605 | 0.156195 | 0.152155 |
| attention | 0.106049 | 0.107500 | 0.158172 | 0.155372 |

Maximum reported persistent logical payload was 123,615–123,629 bytes for the other policies and 131,466 bytes for surprise, plus shared predictor parameters/buffers (5,452,160 bytes). Pending surprise contexts never exceeded three. This is an equal observation-capacity comparison, **not an equal total-byte comparison**.

An independent post-download audit recomputed all 192 losses exactly from saved predictions and raw normalized targets, checked all 12 pair aggregates, 2,316 online events and 42 artifact hashes, and matched checkpoint state hashes. That audit did not rerun model inference. See [portable run evidence](../results/phase2_hpc/20261020/six_retention_baselines.json).
