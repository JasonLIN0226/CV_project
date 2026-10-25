# Gate reader sample-size pilot: protocol and prior-result audit

The 24 October single-seed pilot **completed encoding, CUDA reader training and
evaluation** in A100 job 11026. Increasing training scenes substantially reduced
the retained-cue prediction failure on 48 new scene pairs. This is a reader
diagnostic, not a demonstrated retention-policy improvement or a formal Phase1
confirmation. The execution source is
`experiments/phase2/run_reader_scale_probe.py`.

Local run evidence is under `build/reader-scale-a100-20261024/run/`; portable
paired summaries and figures are under
[`results/literature_diagnosis/reader-scale-pilot-20261024/analysis/`](../results/literature_diagnosis/reader-scale-pilot-20261024/analysis/paired_summary.json).
The saved rows, aggregate summary, checkpoint hashes and validation selections
were independently reconciled. A subsequent hash-bound target export enabled
[independent NumPy arithmetic verification](../results/literature_diagnosis/reader-scale-pilot-20261024/analysis/arithmetic_verification.json)
of all 1,920 saved predictions against 96 episode targets: maximum MSE error
was 1.11e-16, and the 80 aggregate values agreed within 5.55e-17. Raw-feature
MSE, alpha and its direction fraction also matched. This verifies saved-result
arithmetic and artifact binding; it does not independently rerun the encoder or
model, or turn the feature-direction proxy into a physical-lane metric.

## Why this pilot

The reader used in the WR/PaFu and cue diagnosis is the independently trained
Gate **development** reader: 16 training pairs, or 32 episodes, selected at epoch
35 on four reused validation pairs. It is not either official Phase1 model
family. The old normalization uses 64 training episodes across Gate and Motion;
the Gate basis uses 32 Gate training episodes. The fixed unit-column input
projection has shape 5120 by 128, and the output basis has shape 5120 by 95.

The official handoff has 512 training pairs per task, or 1,024 episodes, and is
available separately in the project's verified Release. Its final Gate model
is a `CenteredTransportPredictor` with a matching frozen `DynamicsPredictor`
reference, and the final inference repair disables its transport contribution.
Comparing it directly with the small development reader changes architecture,
training, fitted transforms, and data volume together. It cannot isolate sample
size. See `handoff/README.md`, `handoff/inference.py`, and
`handoff/training_identity.json`.

Both families specify `facebook/ijepa_vith14_1k`, revision
`f157467ea509bc356ff9f61fd3c0d840eec5e04e`, float32 and `grid2` pooling, with
5120-dimensional raw encoder output. A future official-model reference run can
reuse the existing `unique_features.npz::raw` bank and apply that family's own
normalization. It must not pass the old normalized feature arrays directly into
the official model or mix its basis with the development checkpoint. The
official Release excludes its large RGB/feature caches and encoder weights.
No official checkpoint is used by this sample-size pilot.

## Completed A100 results

All metrics below use the same 96 episodes / 48 scene pairs and the same
cue-plus-filler K=2 control. Direction fraction is the mean of `alpha_h > 0`
over 192 episode-horizon outcomes; it is not decoded physical lane accuracy.
The episode counts are paired observations, not independent trials.

| Reader | Cue MSE | Correct cue worse than matched wrong cue | At least one negative alpha | Both alphas negative | Direction fraction |
|---|---:|---:|---:|---:|---:|
| Old epoch-35, 16-pair reference | 0.108929 | 24/96 | 34/96 | 16/96 | 73.96% |
| Retrained 16 pairs | 0.104284 | 19/96 | 33/96 | 12/96 | 76.56% |
| Retrained 64 pairs | 0.073858 | 2/96 | 10/96 | 5/96 | 92.19% |
| Retrained 512 pairs | 0.057564 | 2/96 | 4/96 | 2/96 | 96.88% |

The 512-pair reader reduces cue MSE by about **44.8%** relative to the retrained
16-pair reader. This is not simply uniform improvement in every input condition:
no-cue K=2 MSE changes from 0.123436 to 0.112789, while cue K=2 changes from
0.104284 to 0.057564. The correct-cue advantage over no-cue therefore grows from
0.019153 to 0.055226. Of the 0.046720 cue-MSE decrease, 0.040712 (about 87.1%)
is in the paired-future contrast component; 0.006007 is orthogonal to that axis.

The trend holds in the aggregate of each of the three conditions, each with
only 16 independent scene pairs:

| Condition | Cue MSE, 16 pairs | Cue MSE, 64 pairs | Cue MSE, 512 pairs | 512-pair episodes with at least one negative alpha |
|---|---:|---:|---:|---:|
| ID | 0.102725 | 0.070206 | 0.054868 | 3/32 |
| Long gap | 0.098199 | 0.062073 | 0.052016 | 1/32 |
| Distractor shift | 0.111927 | 0.089296 | 0.065806 | 0/32 |

The remaining two 512-pair episodes where correct cue is worse than wrong cue
are **one sibling scene pair**, `dfada2d1a6aa90938540b005`, with first query action
`+0.0664757714`. Both siblings have negative alpha at both horizons. The other
two episodes with any negative alpha are much closer to the contrast boundary:
one ID episode has alpha at h=3 of -0.006685, and one long-gap episode has alpha
at h=6 of -0.011323. These remaining cases are legitimate diagnosis targets, but
neither their frequency nor their proximity to an action boundary establishes
the causal mechanism.

All three initial state hashes match. All arms completed 512 updates and saw
all their available training pairs. Validation selected steps **128, 448, 448**
for the 16/64/512 arms respectively, with equal-mode validation MSE
0.110589, 0.101926, 0.094816. Thus maximum optimization budget was controlled;
the selected models do not have equal optimizer ages. The interpretation is a
fixed training-and-selection protocol with different available scene sets,
not a pure causal estimate that holds every selected checkpoint's update count
fixed. The saved paired bootstrap describes scenario uncertainty conditional
on this one seed; it does not measure variation across training seeds.

The hardware receipt identifies `NVIDIA A100-SXM4-40GB`, job 11026, host
`b-19-53`, UUID `GPU-32696669-0cff-0b88-1315-809240c3f81b`. The run records
273.111 seconds of encoding and approximately 47.38 / 43.41 / 42.41 seconds of
training for the three arms. Slurm `sacct` records `COMPLETED`, exit `0:0`,
539 elapsed seconds and one allocated GPU (approximately 0.150 GPU hours).
The component timings exclude some preparation and evaluation overhead and
must not be substituted for that total allocation duration. The accounting
receipt is retained under the run's `execution-evidence/sacct-11026.txt`.

**What this changes:** the retained-cue failure pattern exhibited by the old
reader on these new 48 pairs is substantially less frequent with expanded
training under the same architecture and fixed feature transforms. The original
casebook episodes were not replayed by this pilot, so this does not establish
that each previously displayed case has been repaired. These failures should
not be promoted directly into a new retention-method claim. The next retention comparison must first freeze a
reader that demonstrably benefits from the evidence, then compare all policies
with that same reader. This pilot did not rerun WR, PaFu, attention or a learned
retention scorer, did not establish usable oracle regret, and did not fix the
early-time local-cost labels or their horizon mismatch. It also did not identify
whether action-sign composition, irrelevant action summaries, or another aspect
of coverage explains the remaining failures.

A subsequent scaled-student feasibility test is **pending**: fix the 512-pair
reader and compare local-cost, use-time, future-attention and shuffled-label
students trained on 512 pairs. The reader outcomes of the current 48 evaluation
pairs have already been inspected; any subsequent student comparison on them is
development analysis, not a new independent confirmation. No completed result
from that test is claimed here. The earlier small-data CPU use-time student's
negative development result remains unchanged.

## Corrections to the existing confirmation analysis

The following counts were recomputed from saved rows in
`results/literature_confirmation/gate-seed20261023/evaluation-v1/online_results.json`
and the existing confirmation action arrays. No new model calls were needed.

The earlier bin counts refer to the sign of **mean alpha over h=3 and h=6**.
They do not count episodes with at least one horizon on the wrong side.

| Absolute first query action | Episodes | Mean alpha below 0 | At least one alpha below 0 |
|---|---:|---:|---:|
| below 0.1 | 4 | 2 | 2 |
| 0.1 to below 0.2 | 18 | 3 | 7 |
| 0.2 to below 0.35 | 28 | 13 | 13 |
| 0.35 to below 0.6 | 52 | 5 | 14 |
| at least 0.6 | 90 | 9 | 14 |

Alpha is a projection onto the evaluator's paired-future feature contrast:
positive points toward this episode's true feature target, negative toward its
sibling's target. It is neither decoded physical lane accuracy nor a probability.
The saved `feature_branch_correct` averages the two indicators `alpha_h > 0`.
The evaluator definition is `diagnose_gate_oracle_metric.py::paired_components`.

The old 16 independent training scenes have no first query action between
**-0.3812747598 and -0.1353571564**. In confirmation:

- The rounded interval `(-0.38, -0.14)` contains **30 episodes**, with **11/30**
  mean-alpha-negative and **12/30** having at least one negative alpha.
- The narrower interval `(-0.35, -0.14)` contains **20 episodes**, with **10/20**
  under both definitions. The quoted 10/20 result therefore must not be described
  as covering the whole training gap.
- Of the masked-cost arm's 53 cue-loss episodes occurring at least six steps
  before query, 11 have lower MSE than reader-attention. All 11 attention results
  have at least one negative alpha: four have one negative horizon and seven
  have two. This is consistent with reader mistakes allowing cue deletion to
  appear helpful; it does not establish the cause of those mistakes.

These bins and intervals are posthoc descriptions of an already inspected set,
not independent confirmation of an action-boundary or coverage hypothesis.

## Fixed pilot comparison

All three readers retain the exact `DynamicsPredictor` implementation, old
normalization, old unit-column input projection, old output basis, loss weights,
optimizer, initialization seed 17, R=4, K=2, and prediction horizons 3 and 6.
No encoder or fitted transform is retrained. No deletion-utility scorer is trained.

| Arm | Training pairs | Composition |
|---|---:|---|
| scale 16 | 16 | Exact existing Gate training pairs |
| scale 64 | 64 | Old 16 plus first 48 new pairs |
| scale 512 | 512 | Old 16 plus all 496 new pairs |

New training scenes use seed 20261025. Common validation uses 32 new pairs with
seed 20261026. Evaluation uses 48 new pairs with seed 20261027: 16 each of ID,
long gap, and distractor shift. Generator dynamics are unchanged. Scene seeds
and episode IDs are checked against the old dataset and both previous fresh
datasets. The actual portable input export contains 512 excluded metadata rows
(128 old, 192 fresh development, 192 previous confirmation).

The completed invocation used **512 optimizer updates per arm**, 16 pairs per
update, and validates every 64 updates, including initialization and the terminal
step. It selects the earliest minimum of equal-weight validation MSE over the
same six history sampling modes. Thus 16/64/512 arms have respectively
512/128/16 equivalent dataset passes. Exact unique pair coverage and selected
steps are recorded. This equal-update pilot is not an equal-epoch comparison.

Each update uses the existing causal sampler with 0.15 memory dropout and cycles
through random fixed-size, random subset, FIFO, reservoir, temporal buckets and
visual diversity. Training samples remain at the declared query. This pilot
**does not repair or test early-time local-cost reader distribution shift**, and
does not test delayed-value retention supervision.

All three selected readers are saved and recorded in `readers_locked.json`
before evaluation. The previously selected 16-pair epoch-35 reader is a separate
fourth reference, not the scale-16 retraining result. Every reader receives the
same five evaluation controls: recent-only; two recent historical non-cue frames;
cue plus a common filler; matched wrong cue plus that filler; and full history.
Wrong-cue and full-history controls are evaluator-only; full history exceeds K.

The resulting comparisons test whether useful cue reading improves with more
training scenes under a fixed representational bottleneck. They cannot establish
a new retention method, general performance across model seeds, or that sample
coverage is the only cause of the original bad case. If the larger reader fails,
the fixed old basis remains a possible limitation. Any subsequent baseline
comparison must run all policies against the same chosen new reader and use a
separately fixed evaluation protocol.

## Portable execution

The local export was created and verified at
`build/reader-scale-probe-20261024-inputs-v2/`. It contains 38 hashed files: 32
old normalized training-episode archives with their actions, records, complete
basis, normalization, old-reader checkpoint, model config, excluded identities,
and manifest. The obsolete first export lacks the complete output basis and must
not be used. The v2 package passed SHA checks, strict old-reader loading, and an
actual finite CPU forward with shape `[1, 2, 5120]`; this is an interface check,
not by itself an A100 experiment. The completed A100 evidence is reported above.

Rebuild the inputs from the verified local assets if needed:

```bash
python experiments/phase2/run_reader_scale_probe.py export \
  --snapshot build/hpc-20261020/development-assets \
  --followup build/history-diagnosis-20261020/followup-blas1 \
  --exclude-records build/literature-baselines-20261022/fresh-data/records.json \
    build/literature-confirmation-20261023/data/records.json \
  --output NEW_PORTABLE_INPUT_DIRECTORY
```

The remote bundle needs the portable input directory, all
`experiments/phase2/*.py`, and all
`experiments/predictor/control/src/phase1/*.py`, preserving this directory tree.
Runtime dependencies are NumPy, CUDA PyTorch, Pillow, Transformers and PyYAML,
plus the already cached pinned I-JEPA model and processor. The remote run does
not depend on the historical PortableSnapshot filesystem layout or the
teammate's `/scratch/xl4358` paths. Preserve an immutable source hash inventory
with the submitted bundle.

Inside a Slurm job with exactly one visible A100:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python experiments/phase2/run_reader_scale_probe.py run \
  --inputs /path/to/portable-inputs-v2 \
  --cache-dir /scratch/xy3257/cv-phase2-20261020/cache/ijepa \
  --output /path/to/new-reader-scale-run \
  --steps 512 --validate-every 64 --model-seed 17 --encoder-batch-size 2
```

The script verifies the physical CUDA GPU UUID against `nvidia-smi`, generates
new RGB, encodes each distinct frame once, trains all three readers on CUDA, and
evaluates them. It writes model locks, curves, individual metric rows, saved
prediction vectors, hardware identity and file hashes. A scheduler allocation or
encoding completion alone is not evidence that reader training or evaluation
finished. The final `completion.json` is written only after all stages succeed.
