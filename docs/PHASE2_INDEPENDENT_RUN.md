# Phase 2 independent work — 20 October 2026

Subsequent real-encoder result: the official frozen I-JEPA processed the paired PushT trace successfully on **local Apple MPS**, with finite `(25,5120)` features per variant. See [the separate encoder evidence](PHASE2_LOCAL_MPS.md). The CPU runs below retain their original scope and hashes. Baseline-model planning and scientific Phase 2 remain incomplete.

The user asked to proceed with work that does not require the teammate's unavailable artifacts. This report records local CPU software checks, the real PushT observation interface, and newly generated controlled RGB data. It does not report real JEPA encoding, predictor training, trained retention baselines, a scientific scorer dataset, or baseline-model planning. HPC and MPS execution status is tracked separately.

## Executed result

The final unified runner completed on local macOS CPU with Python 3.12.14: **138 tests passed, 0 failures, 0 errors, 0 skipped**. Counts below were read from each run's `tests.xml`; all three runs are preserved.

| Run on 20 October 2026 (UTC start) | Passed tests | Phase 2 / inherited | Added coverage | Portable record |
|---|---:|---:|---|---|
| Initial, 04:28:07 | 80 | 54 / 26 | Memory, deletion contracts and PushT interface | [`initial_checks.json`](../results/phase2_independent/20261020/initial_checks.json) |
| Followup, 04:39:11 | 133 | 107 / 26 | Asset validation, controlled data and training sample adapter | [`followup_checks.json`](../results/phase2_independent/20261020/followup_checks.json) |
| Final, 04:50:57 | 138 | 112 / 26 | Five encoded-sample pipeline fixture checks | [`final_checks.json`](../results/phase2_independent/20261020/final_checks.json) |

Both supplied Phase 1 evidence verification scripts and `pip check` passed in each run. Each real `gym_pusht/PushT-v0` paired trace completed 24 actions / 25 RGB observations. The final run took 19.31 seconds overall; its PushT manifest has the same hash as the initial deterministic trace. These local runners submitted no GPU allocations and used zero GPU allocation seconds. Those fields describe these executions only, not separate HPC or MPS work.

Portable evidence is saved in [`results/phase2_independent/20261020/`](../results/phase2_independent/20261020/), including the three check records, `initial_pusht.json`, the contact sheet and controlled-data audits. Portable command paths replace the local absolute checkout path with `<REPO>`; source and original log hashes remain unchanged. `final_checks.json` also records the original manifest/XML hashes, XML-derived counts and a provenance note. Full local manifests, XML, logs, environment listings and PushT arrays remain in `build/phase2-independent-20261020/{verified,followup,final}/`. The earlier pre-pin smoke remains in `pre-pin-pusht/`. `build/` is excluded from Git.

The initial upstream source was `JasonLIN0226/CV_project` commit `2c73c6bca11a9396cdb0546b8bf78b52445e7b22`. The followup and final manifests record Git bases `5de49da085de8d3f97ea575789901be8d67dce50` and `514012d2325b29b78882a0850b72186e0774a3a4`. Because new files may be uncommitted at execution, the per-file hashes identify the exact checked source. Inherited scientific source/protocol files were not altered.

After the final run, `prepare_controlled_data.py` received a resource-accounting correction: it records `SLURM_JOB_ID` and reports GPU allocation seconds as unknown under Slurm, to be resolved through scheduler accounting. Local CPU execution still reports zero. This changes provenance fields, not data-generation semantics. The final 138-test run and the existing local dataset used the earlier source hash; their archived hashes were deliberately preserved. The revised runner separately passed its five controlled-data tests; that focused check is not an additional full-suite run.

## Requirement status

| Proposal requirement | Implemented and checked here | Remaining |
|---|---|---|
| Common bounded-memory interface | `StreamingMemory` automatically moves the oldest frame leaving the recent buffer into the historical candidate set; exactly one K+1→K deletion, no archive, causal timestamps, reset, immutable snapshots and logical memory accounting | Execute on real JEPA features with the shared predictor; measure real runtime overhead |
| Six retention baselines | FIFO, standard reservoir, candidate-only temporal buckets and visual diversity; surprise selection from an already causally computed scalar; attention selection via an explicitly required model callback | Bind real predictor attention and causal surprise generation; run all six under identical predictor weights and data. Synthetic callback tests are not trained-model baseline results |
| Varied historical subsets and memory dropout | New adapter implements varied historical subsets, explicit stochastic whole-memory dropout, group/episode split safeguards, causal input versus future-target separation and deterministic validation; 29 adapter tests pass | Run actual shared-predictor training with this mixture; save logs and checkpoint. The published job7574 training did not enable explicit stochastic dropout |
| Equal-capacity deletion supervision | Every K+1 deletion is enumerated at exactly K; immutable past-only scorer input is separate from offline future actions/targets; raw per-horizon feature MSE, missing-horizon mask and checkpoint identity contract | Load and independently verify a frozen predictor; generate a split/group-aware scientific dataset from actual bounded streaming states |
| PushT interface | Actual pixel observations and absolute actions, RGB masking before the encoder boundary, matched unoccluded control, later-visible targets, reset/termination checks, diagnostic state separation | Execute the pinned real encoder; validate that history is genuinely necessary |
| Baseline-model planning | Interface/action incompatibilities identified and documented | Train or obtain a compatible visual PushT baseline and execute a model-driven closed loop. Random actions in the trace are data collection, not planning |

No extra T8, all-checks, oracle-headroom, or mandatory dataset-size gate was added. These CPU checks establish implementation behavior; they do not establish research improvement or Phase 2 completion.

## New controlled data and sample preparation

A new Gate/Motion development dataset was generated locally with seed **20261020**: **128 episodes, 64 pairs and 3,616 RGB frames**. Each task has 16 training pairs and four pairs in each other split. The retained generator's dynamics remain unchanged, including 96×96 RGB, recent length 4, memory capacity 2, horizons 3/6 and Motion time steps 3/6. Native data validation and the observation-boundary audit passed. The observation index exposes only identity/split metadata and the RGB/actions/query files; simulator states and evidence labels remain in separate evaluation-only files. All `test_*` splits are explicitly development smoke splits, not untouched confirmation data. See [`controlled_data.json`](../results/phase2_independent/20261020/controlled_data.json) and its adjacent audit records; full data are in `build/phase2-data-20261020/`.

[`training_samples.py`](../experiments/phase2/training_samples.py) accepts encoded episode arrays and separates recent/history/executed actions from offline future targets/action continuations. It offers fixed-size or variable-size random historical subsets plus four asset-independent policy choices, with configurable whole-memory dropout on training samples. Validation disables dropout and fixes sample selection. These six sample modes are not the proposal's six model-bound retention baseline results. Surprise/attention integration still needs genuine causal model outputs. Per-query prefix replays provide offline augmentation; they must not be used as one persistent online retention rollout. The historical predictor bridge has been checked with a small random model fixture only. See [the adapter contract](PHASE2_TRAINING_SAMPLES.md).

[`prepare_encoded_samples.py`](../experiments/phase2/prepare_encoded_samples.py) now connects fresh prepared data to the unchanged official pinned I-JEPA cache implementation, exact-signature cache resume, train-only normalization, and the new sampler. It saves a separate encoder configuration, checks source/data/feature provenance, loads all encoded episodes, and generates sample summaries only for train/validation. It does not fit a predictor or scorer. Its five tests use an explicitly named `smoke_pixels_NOT_JEPA` fixture, including partial-cache resume and normalization/split checks; they do not download or validate real JEPA weights. No completed real-encoder cache or real-feature sample output is claimed in this report. The executable command and remaining bindings are documented in [sample preparation](PHASE2_TRAINING_SAMPLES.md#prepare-samples-from-real-frozen-encoder-features).

## PushT observation trace

The chosen initial interface variant blacks out timestamps 8–12 inclusive. Control and masked runs use seed 17 and identical absolute XY target actions. Their observed transitions/rewards/termination agree except for the prescribed pixel mask. Queries 11 and 12 have a fully masked recent buffer of four frames, with visible targets at horizons 3 and 6. State information is saved only in the separate diagnostic file.

The contact sheet shows unoccluded control above masked input. Columns are timestamps 0, 7, 8, 11, 12, 13, 17. Real JEPA encoding, memory necessity and planning flags remain false.

![Paired PushT RGB trace](../results/phase2_independent/20261020/paired_trace.png)

## Re-run the independent work

Use Python 3.10–3.12. The tested local runtime is Python 3.12 on macOS arm64; installed package versions are captured in the evidence. This requirements file pins packages for these checks and is not a validated A100 deployment environment.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-phase2-cpu.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python experiments/phase2/run_independent.py --output build/phase2-independent-rerun
```

Use a new empty output path. The runner does not submit a job, download encoder weights, call the old deployment scripts, or infer completion from a successful unit test. Review individual logs when it exits nonzero.

## Asset and course-connection boundary

The GitHub checkout contains no model/cache binaries. The 19 October repaired-result summaries are newer than the supplied runnable predictor; the corresponding latest repair source/maps are absent. See [the asset audit](PHASE2_ASSET_AUDIT.md) and [the exact known asset manifest](../experiments/phase2/required_phase1_assets.json).

During the initial asset search, the course portal was reached in Chrome incognito. A direct read of the documented teammate reference path returned “does not exist” through the portal; this does not establish deletion or permission status on every storage system. The user confirmed no additional shared paths were available and directed the independent work above. This local report does not describe subsequent live HPC sessions or job status.

When artifacts arrive, verify their hashes and compatibility before binding the callbacks and trainer. Alternatively, rebuilding data/encoding/predictors from the supplied source is a separately identified training run; it must not be reported as continuing the unavailable latest repaired weights.
