# Gate reader sample-size pilot: protocol and prior-result audit

This document records the prepared 24 October pilot, not a completed GPU result.
Check the run's `completion.json`, hardware receipt, `readers_locked.json`, and
`evaluation_summary.json` before reporting execution or a scientific outcome.
The execution source is `experiments/phase2/run_reader_scale_probe.py`.

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

The intended invocation fixes **512 optimizer updates per arm**, 16 pairs per
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
not an A100 experiment.

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
