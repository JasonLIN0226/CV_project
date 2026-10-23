# Literature baseline audit — 2026-10-22

This independent review covers the WR-inspired and PaFu-inspired mechanism
adaptations, their online interfaces, and the CV `dev-v3` artifacts. It does not
certify paper reproduction, independent generalization, or A100 execution.

## Scientific contract

- WR preserves standardized Gaussian L1 surprise, past-only MAD, delayed peak
  confirmation and causal suppression. The four-frame resident window and
  event-first bounded replacement rule are explicit adaptations; generic JEPA
  prediction residual is a separate control. Its scalar history is bounded and
  counted; it cannot reconstruct evicted feature vectors.
- CV PaFu uses the frozen predictor's actual single-head additive pooling
  attention, not invented cosine/QKV attention. Offline probes extend through
  decision time +6. Earlier-than-key probes are outside the causal reader's
  training domain. Frame-level balancing, probability normalization and
  probability SmoothL1 are adaptations of the original method.
- Online students receive only resident candidate/recent features and ages.
  Future queries, target residuals and deletion losses remain in the offline
  harness. All retained K items enter the same frozen predictor.

Primary sources: [Worth Remembering, section 3](https://arxiv.org/html/2606.03787)
and [PaFu-KV, section 3](https://arxiv.org/html/2601.21896v3).

## CV validation

Reviewed `literature_baselines.py`, `run_literature_baselines.py`, their existing
causal/oracle tests, and the inherited online retention code. Independent
artifact recomputation of `dev-v3` found the following. Its portable
[archive](../results/literature_baselines/dev-20261022/README.md) retains
[source/artifact hashes](../results/literature_baselines/dev-20261022/manifest.json),
[protocol](../results/literature_baselines/dev-20261022/protocol.json), and
[per-case results](../results/literature_baselines/dev-20261022/episode_results.json).
Prediction vectors and checkpoints stay outside this small archive; their
original hashes are recorded.

- All 72 prediction-vector rows reproduce reported feature MSE exactly.
- Every online query retains exactly K=2. The hindsight evaluator enumerates
  `C(query-3, 2)` subsets; its reported minimum equals the recorded subset
  minimum. The local oracle compares the same K+1 parent and shares one
  deletion across h3/h6. Neither is an online policy.
- Training metadata matches the 16 declared training pairs, excludes the four
  validation pairs, and records six genuinely future query steps. Checkpoint
  selection is fixed to the final epoch, with explicit training-group checks.
- All five recorded source hashes match the reviewed source files.
- The evidence metric now distinguishes the annotated start timestamp from
  exact copies of the same observed feature. This reporting correction leaves
  all prediction vectors unchanged from dev-v2.
- WR timing now includes Gaussian/MAD processing; PaFu's 82,308 parameter
  bytes are separate from dynamic state. Equal K is **not** equal total memory
  or compute. CPU timings are descriptive, not a controlled speed benchmark.

The [PaFu diagnostic](../results/literature_baselines/dev-20261022/pafu_teacher_diagnostic.json)
records that student and offline attention teacher choose the same deletion in 18/24
development rows. Their mean local regrets are 0.005365 and 0.006248,
respectively. This small reused-development result does not support assuming
that better attention-target imitation necessarily improves task prediction.

The [WR scalar traces](../results/literature_baselines/dev-20261022/wr_warmup_trace.json)
explain all six cue-loss episodes: the cue starts at t=2
or t=3, with surprise about 6,028–6,187 but no initialized threshold. The two
t=4 cues are confirmed and retained. Thus the demonstrated WR failure is
threshold warm-up, not low-surprise evidence displaced by dynamic distractors.
After correcting the evidence metric, attention, visual diversity and PaFu all
retain an exact feature copy of the cue in every evaluated episode/seed row.
Their remaining prediction regret cannot be attributed simply to losing it.

## Predictive-memory bank boundary

The separate `predictive-memory-v6-h800/experiments/bank_reference` code was
reviewed for read-before-write behavior, source-disjoint fit, true future-query
teacher inputs, and same-parent deletion enumeration. Its K+1 oracle is local;
it does not enumerate all historical K-subsets. Its future-only conditional
masked-mean attention target is a further, explicitly weaker correspondence to
PaFu's original balanced teacher.

The bank data contains frozen residual/mask/key streams, not complete L0 or RGB.
Replayed A2 gates were produced from a richer original input; archived A1/A2/A6
are contextual references, not input-matched bank competitors. A common reserved
controller envelope is a cap, not equal actual allocation; weights and actual
state must remain itemized. The all-observations mean is a larger fixed-size
sufficient-statistic reference, not growing unbounded storage. Adjacent CHANGE
streams still share sources within development, limiting independence of CIs.

Independent artifact checks of `br-0.1-dev-seed17` verified all recorded source
hashes, the reported 1,152 archived-reference checks, all 864 same-parent oracle
records (K+1 alternatives, selected/minimum loss and regret), genuinely future
teacher timestamps, train-only teacher sources, development-only result rows,
and actual dynamic usage within each declared reserved cap. This was artifact
recomputation, not a second model training run.

Only 3, 4 and 39 of 276 candidate training parents per K=4/16/64 yield nonempty
attention targets. All bank policies have worse POST MPE than zero prediction
in all three conditions. These results expose a weak residual-bank/teacher
interface; they do not establish that the original papers fail or that a new
retention policy is warranted. Actual bytes differ despite a common cap, and
each PaFu student additionally has 44,548 parameter bytes.

## Interpretation limits

The CV result uses four previously reused validation pairs. Surviving exact
task-defined evidence timestamps is a trace diagnostic, not a task-success
criterion. Positive oracle regret is not automatically a learnable opportunity:
unobserved future actions or queries can create an irreducible gap. Compare
failure rates and paired regret, preserve positive controls, and separate
eviction failures from teacher-target mismatch and predictor/reader limits.
