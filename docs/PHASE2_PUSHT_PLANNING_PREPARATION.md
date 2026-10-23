# PushT data preparation for a learned planning baseline — 20 October 2026

**Completed: a new CPU-only RGB/action corpus with a fixed collection protocol.**
This removes a data-preparation dependency for the proposal's PushT baseline
planning work. It does not establish a trained PushT predictor, history necessity,
a goal image, or model-driven planning. The new corpus has not yet been encoded
by I-JEPA. Earlier encoding of the separate 25-frame interface trace is not
encoding of these 48 episodes.

The proposal asks for pre-encoding visual occlusion, later visible prediction
targets, an unoccluded control, and verification that history helps. Timeline v2
puts an end-to-end baseline pipeline in Phase 1, fuller baseline-retention/MPC
integration in Phase 3, and formal PushT comparisons in Phase 4. The dataset size
below is a bounded implementation choice, not an added scientific gate.

## Actual preparation result

| Split | Independent environment seeds | Frames per variant | Executed actions per variant |
|---|---:|---:|---:|
| Train | 32 | 3,104 | 3,072 |
| Validation | 8 | 776 | 768 |
| Reserved evaluation | 8 | 776 | 768 |
| Total | 48 | 4,656 | 4,608 |

Each episode has 96 actions and 97 observations, including reset. Both masked
and unoccluded variants were collected, yielding **9,312 RGB frames** total.
The successful attempt used **26.390 seconds** for collection and **36.685
seconds** including structural audit and hashing, on local CPU. It submitted no
Slurm job and used no GPU. Saved artifacts occupy **9,773,064 bytes**, excluding
the preparation manifest; each variant's arrays occupy 128,840,064 bytes before
compression. Timing does not include the preserved failed attempt or diagnostic
replays, whose total duration was not recorded.

All **148 saved artifact checksums and four collector/environment source
checksums** were verified after execution. The eight collection-contract tests
passed in 1.27 seconds. Two additional tests for the permanent TRAIN-only audit
passed in 0.07 seconds, including rejection of any validation/evaluation diagnostic
read. These tests establish software boundaries, not predictor quality or planning
performance.

## Fixed collection and observation contracts

The protocol and source receipt were written before the first environment was
created. The complete episode list and planned-action hashes are preserved in
[protocol.json](../results/phase2_history/20261020/pusht_preparation/protocol.json).

- Environment: installed `gym-pusht==0.1.8`, `gym_pusht/PushT-v0`, `obs_type="pixels"`,
  96×96 RGB uint8 observations from `reset`/`step`. `render()` is not used because
  it can add the last-action overlay and use another resolution.
- Split seeds: train `2026102000..2026102031`, validation `2026103000..2026103007`,
  evaluation `2026104000..2026104007`. Entire episodes and seeds are disjoint.
- The action sequence is fully computed before reset. The first 48 controls are
  seeded horizontal or vertical sweeps, followed by 12 random-walk targets each
  held for four controls. Actions depend only on the predeclared policy seed.
  Pixels, reward, block position, contact information and goal coverage never
  choose an action. No collection policy was retuned after outcome inspection.
- Native actions are **absolute target XY positions in [0,512]**, not
  displacements or velocities. This collection keeps targets within [40,472].
  Normalization is `absolute / 256 - 1`; the ordered sequence is preserved.
  The environment runs at 10 control steps per second with 0.01-second physics
  steps. Gate/Motion's additive-control summary is not compatible with this API.
- Each episode has a predeclared full-image blackout lasting five observations,
  starting at 8, 16, 24 or 32. Masking happens on RGB before any future encoder.
  A separate same-seed simulator executes the exact same action sequence for
  the unoccluded control. Masked pixels equal the prescribed transformation of
  control pixels exactly; physical diagnostic values agree as described below.
- With recent length four, offline target metadata identifies horizons 3 and 6
  that land on a later visible observation. This metadata is kept with diagnostics.
  It is not a future-visibility input to a retention policy or planner.

`observations/*.npz` contains only `rgb`, `timestamps`, `actions_absolute` and
`actions_normalized`. Unoccluded RGB lives in separate `unoccluded_control/*.npz`
files. The observation index explicitly identifies the separate control arm;
future masked-model training must not substitute control pixels for masked ones.

Privileged simulator state, reward, coverage, contacts and target audits live
under `evaluation_only/`, including diagnostics from training episodes. They are
not model inputs. The later TRAIN-only audit reads only training diagnostic files;
it does not inspect validation or evaluation outcome quality. Recorded future
actions can support offline prediction training, but a live planner must generate
its own candidate actions and a retention scorer must not read continuations.

## Numerical repair and preserved first attempt

The first attempt stopped after six training episodes because an exact floating
comparison detected coverage `0.12306193831688134` versus
`0.12306193831688140` at training episode 006, timestamp 10. Pixels matched.
An identical-policy diagnostic replay localized the difference to approximately
`6e-17` in coverage/reward from separate simulator geometry calculations.

The repaired validation permits **absolute error at most 1e-12 for floating
physical diagnostics only**. Pixels, actions, flags and integer counts still
require exact equality. Both diagnostic streams are saved, along with the maximum
difference. The observed maximum on the 32 training episodes was
`2.220446049250313e-16`.

Policy actions, masks, episode seeds and split membership were unchanged. Only
the diagnostic-comparison contract was added to the frozen protocol. The original
partial output and original collector source remain in
`build/pusht-planning-development-20261020-attempt1-exact-diagnostics/`; the compact
[failure record](../results/phase2_history/20261020/pusht_preparation/initial_failed_attempt.json)
is also preserved. The successful attempt did not overwrite it.

## TRAIN-only interaction audit

A separate reproducible audit checked that this fixed policy actually encountered
objects. It used simulator diagnostics only for this postcollection description,
not for collection, fitting or selection.

| Measure, across 32 training episodes | Observed result |
|---|---:|
| Episodes with at least one contact | 32 |
| Episodes with translation greater than 1e-6 native units | 32 |
| Episodes with wrapped rotation greater than 1e-6 radians | 32 |
| Episodes with nonzero goal coverage | 23 |
| Contact timesteps | 1,065 |
| Median maximum translation from initial block position | 188.53 native units |
| Median maximum wrapped rotation from initial orientation | 0.9808 radians |
| Episodes with later-visible targets | 32 |
| Successful collection episodes | 0 |

These observations establish that the training corpus contains physical object
interactions. They do not establish sufficient training diversity, identifiable
action effects, a need for history, or learned-model accuracy. **Zero collection
successes is not a planner evaluation:** no planner ran. Validation and reserved
evaluation diagnostic outcomes were not summarized or used to alter the protocol.

## Reproduction and provenance

From the repository root, run the existing tests and use new output paths:

```bash
.venv/bin/python -m pytest -q tests/test_phase2_prepare_pusht_planning_data.py
.venv/bin/python -m pytest -q tests/test_phase2_audit_pusht_training_data.py

.venv/bin/python experiments/phase2/prepare_pusht_planning_data.py \
  --output build/pusht-planning-development-NEW

.venv/bin/python experiments/phase2/audit_pusht_training_data.py \
  --input build/pusht-planning-development-NEW \
  --output build/pusht-train-coverage-NEW.json
```

The collector refuses a nonempty output directory, and the audit refuses to
replace an existing report. The audit CLI verifies the frozen protocol and each
training diagnostic hash against the collection manifest before reading it.
For the existing saved run, the exact audit invocation was:

```bash
.venv/bin/python experiments/phase2/audit_pusht_training_data.py \
  --input build/pusht-planning-development-20261020 \
  --output build/history-diagnosis-20261020/planning/pusht_train_coverage_reproducible.json
```

[Portable evidence](../results/phase2_history/20261020/pusht_preparation/portable_evidence.json)
records every copied file's hash, source hashes, commands, the original-attempt
source/protocol hashes and the comparison showing unchanged actions/masks/splits.
The [raw preparation manifest copy](../results/phase2_history/20261020/pusht_preparation/preparation_manifest.json)
contains all 148 artifact hashes, package versions and the exact installed
environment source hashes. The recorded Git base predates these new source files;
content hashes identify the code actually executed.

The collector source hash is
`0176218bd693868a9bcee5b7fad56bfb056b7c0fe767a903279bdf5b69c80466`.
The training audit source hash is
`17755a22af87ca1fd34c3f372c6565ffc0415db1822de5ee24fb71a9c910c1d0`.
The successful protocol hash is
`96e5ac3df252f376d230092aa59699270170ccf3b4c5b2be012d790528b207fa`.

Portable JSON files are byte-identical copies of the retained raw evidence.
**No path redaction was necessary:** these files contain relative paths only.
Raw RGB/action binaries remain under ignored
`build/pusht-planning-development-20261020/`; they are not committed to Git.
Floating simulator geometry may differ at machine precision across reruns, so
fresh diagnostic/NPZ hashes need not equal this run's hashes. Preserve each run's
manifest rather than treating bitwise rerun identity as a scientific result.

## Remaining work under the proposal

The independently available next stages are: encode these new frames using the
pinned frozen I-JEPA; fit any normalization/basis on training episodes only;
train a PushT-specific ordered absolute-action-conditioned predictor; compare
history conditions; and connect model predictions to a fixed-budget MPC planner.
These stages have not run here.

A separate, fixed exogenous goal RGB also remains pending. The installed
`reset_to_state` uses a legacy position/rotation assignment order, so blindly
resetting the T block to the numerical goal pose would not verify a correct goal
image. A separately verified goal observation must be saved before evaluation;
the live planner may receive its RGB/features, never the simulator's block or goal
pose. An image-goal cost can then be checked for agent/background confounding.

Any later planning smoke must log its trained checkpoint, visual goal, generated
candidate action sequences, prediction-based costs, selected action and real
transition. The same planner, objective and search budget must be held fixed for
retention comparisons. Simulator rollout scores, random collection actions or a
Gate/Motion checkpoint cannot substitute for this learned PushT planning path.
