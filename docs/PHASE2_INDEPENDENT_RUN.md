# Phase 2 independent work — 20 October 2026

The user asked to proceed with work that does not require the teammate's unavailable artifacts. This run implements and verifies software contracts and the real PushT observation interface. It does not train a predictor, evaluate trained retention baselines, generate a scientific scorer dataset, or verify baseline-model planning.

## Executed result

The unified runner completed on local macOS CPU with Python 3.12.14. **80 tests passed, 0 failed, 0 skipped**: 54 new Phase 2 tests and 26 inherited tests. Both supplied Phase 1 evidence verification scripts and `pip check` also passed. A real `gym_pusht/PushT-v0` paired trace completed 24 actions / 25 RGB observations. No GPU allocation was submitted; GPU allocation use was zero.

Evidence is in [`build/phase2-independent-20261020/verified/`](../build/phase2-independent-20261020/verified/): `manifest.json`, `tests.xml`, test/verifier logs, the installed environment listing, and `pusht/manifest.json`, `paired_trace.npz`, `paired_trace.png`, `diagnostics_only.json`. The manifest records exact hashes of the new source and tests, commands, exit codes, elapsed times and explicit incomplete scientific stages. `build/` is deliberately excluded from Git. The earlier pre-pin smoke is preserved in a sibling `pre-pin-pusht` folder; the `verified` run is the reported result.

Base source: `JasonLIN0226/CV_project` commit `2c73c6bca11a9396cdb0546b8bf78b52445e7b22`. New source is on local branch `phase2/independent-interfaces-20261020`. Inherited scientific source/protocol files were not altered.

## Requirement status

| Proposal requirement | Implemented and checked here | Remaining |
|---|---|---|
| Common bounded-memory interface | `StreamingMemory` automatically moves the oldest frame leaving the recent buffer into the historical candidate set; exactly one K+1→K deletion, no archive, causal timestamps, reset, immutable snapshots and logical memory accounting | Bind to actual encoded episodes and the shared predictor; measure real runtime overhead |
| Six retention baselines | FIFO, standard reservoir, candidate-only temporal buckets and visual diversity; surprise selection from an already causally computed scalar; attention selection via an explicitly required model callback | Bind real predictor attention and causal surprise generation; run all six under identical predictor weights and data. Synthetic callback tests are not trained-model baseline results |
| Varied historical subsets and memory dropout | Audit found inherited random K-subset exposure; dropout/random-subset functions exist but the published job7574 training did not enable explicit stochastic dropout | Enable and run the Phase 2 training mixture; save logs and checkpoint |
| Equal-capacity deletion supervision | Every K+1 deletion is enumerated at exactly K; immutable past-only scorer input is separate from offline future actions/targets; raw per-horizon feature MSE, missing-horizon mask and checkpoint identity contract | Load and independently verify a frozen predictor; generate a split/group-aware scientific dataset from actual bounded streaming states |
| PushT interface | Actual pixel observations and absolute actions, RGB masking before the encoder boundary, matched unoccluded control, later-visible targets, reset/termination checks, diagnostic state separation | Execute the pinned real encoder; validate that history is genuinely necessary |
| Baseline-model planning | Interface/action incompatibilities identified and documented | Train or obtain a compatible visual PushT baseline and execute a model-driven closed loop. Random actions in the trace are data collection, not planning |

No extra T8, all-checks, oracle-headroom, or mandatory dataset-size gate was added. These CPU checks establish implementation behavior; they do not establish research improvement or Phase 2 completion.

## PushT observation trace

The chosen initial interface variant blacks out timestamps 8–12 inclusive. Control and masked runs use seed 17 and identical absolute XY target actions. Their observed transitions/rewards/termination agree except for the prescribed pixel mask. Queries 11 and 12 have a fully masked recent buffer of four frames, with visible targets at horizons 3 and 6. State information is saved only in the separate diagnostic file.

The contact sheet shows unoccluded control above masked input. Columns are timestamps 0, 7, 8, 11, 12, 13, 17. Real JEPA encoding, memory necessity and planning flags remain false.

![Paired PushT RGB trace](../build/phase2-independent-20261020/verified/pusht/paired_trace.png)

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

The course portal was reached in Chrome incognito as `xy3257`. A direct read of the documented `/scratch/xl4358/phase1_v3_dev/gate/reference` path returned “does not exist” through the portal; this does not establish deletion or permission status on every storage system. The portal's default shell terminated. No new A100 job was submitted. The user confirmed no additional shared paths were available and directed the asset-independent work above.

When artifacts arrive, verify their hashes and compatibility before binding the callbacks and trainer. Alternatively, rebuilding data/encoding/predictors from the supplied source is a separately identified training run; it must not be reported as continuing the unavailable latest repaired weights.
