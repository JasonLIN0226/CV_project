# New controlled data prepared without external Phase 1 assets

`experiments/phase2/prepare_controlled_data.py` uses the retained, unchanged
`phase1.envs` and `phase1.data` generators to produce a modest **new CPU development
dataset**. It does not reproduce the teammate's original results, recover their
checkpoints/caches, run an encoder, or execute the reserved independent confirmation.

## Run and scope

```bash
.venv/bin/python -m pytest -q tests/test_phase2_controlled_data.py
.venv/bin/python experiments/phase2/prepare_controlled_data.py \
  --output build/phase2-data-20261020 --seed 20261020 --train-pairs 16 --eval-pairs 4
```

The output must be empty. To regenerate, use a new output directory with the same
arguments; the native audit also regenerates both siblings of every scene and
checks RGB/actions for exact agreement. The run records source-file hashes,
Git commit, dependency versions, resolved configuration, original configuration
identity, data/manifest checksums, and CPU elapsed time. No GPU is allocated.

Defaults produce **128 episodes in 64 sibling pairs**: for each of Gate and Motion,
16 training pairs and 4 pairs in each of validation, ordinary development, long-gap
development, and distractor-shift development. Seed `20261020` is new; the old
`6301` seed and reserved `9137` confirmation seed are rejected by this runner.
This size is an explicit engineering smoke choice, not a proposal requirement or
a statistical sufficiency claim.

Dynamics come from `results/reference/gate/resolved_config.json`: 96×96 RGB,
recent length 4, K=2, horizons 3/6, ordinary gap 5–10, long gap 20–32,
cue start 2–7, cue duration 1–3, and Motion interval 3–6. Only the base data seed
and pair counts change. The generator produces entire trajectories, preserves
both siblings within one split, and derives distinct environment seeds by
task/split/scene index. Long-gap and distractor settings remain separate development
conditions. Fields copied for encoder/training/decision settings are **inactive**;
they do not authorize new training or impose new Phase 2 acceptance gates.

## Audits and model boundary

The inherited data audit verifies checksums, matched recent RGB and recorded actions
within sibling pairs, distinct historical visual evidence, later visible/distinct
targets, legal evidence/target timestamps, label balance, disjoint groups/seeds,
and exact regeneration. These are structural data checks, not proof that a JEPA
encoder or predictor has learned to use the evidence.

The additional observation-boundary audit requires each model-readable NPZ to have
exactly `rgb`, `actions`, and scalar `query`, with verified shape/dtype/checksum.
Privileged states, labels, visibility and evidence timestamps remain in the
separate `dataset/evaluation_only` files. They cannot be smuggled into an observation
archive or the model-facing index. Pair/episode/task/split/seed/query identifiers
in `observation_index.json` are dataset bookkeeping, not predictor features.

The full observed-action trajectory is stored offline. A later sample adapter must
separate past context from the future action/target block; recorded future controls
are legal for offline prediction supervision, never for deployed retention decisions.
Controlled-task actions are additive exogenous controls in [-1,1], unlike PushT's
absolute target coordinates. Do not interchange their action preprocessing.

`split_roles.json` explicitly maps the inherited filenames `test_id`, `test_long_gap`,
and `test_distractor_shift` to **development_smoke**. The structural audit inspects
all these splits; this output makes no untouched-test or fresh-confirmation claim.
Any future normalization, feature basis, predictor fitting or scorer-label training
must use only its declared training membership. Validation is reserved for selection.

## Outputs and next connection

- `resolved_config.json`: exact generator configuration, with inactive downstream sections identified in provenance.
- `dataset/manifest.json` and `dataset/split_audit.json`: native records, checksums and structural audit.
- `dataset/observations/`: RGB/actions/query archives.
- `dataset/evaluation_only/`: privileged diagnostics, kept out of the model boundary.
- `observation_index.json`: observation-only paths and group identity for the Phase 2 sample adapter.
- `observation_boundary_audit.json`, `split_roles.json`, `preparation_manifest.json`: extra checks, declared roles and source/run provenance.
- `figures/pair_*.png`: paired diagnostic visualizations from the native audit.

The next independent connection is to encode these RGB observations with the actual
frozen, pinned I-JEPA checkpoint, fit preprocessing only on training membership,
then build varied-history/dropout predictor samples through the Phase 2 adapter.
Those stages remain separate executions; data generation alone is not completed
Phase 2 baseline evaluation or deletion-supervision generation.

## Executed local preparation, 2026-10-20

The command above completed: 128 episodes, 64 sibling pairs, 64 distinct scene
seeds, and 3,616 RGB frames. Native structural checks and the observation-boundary
audit passed. Generation plus audits took 4.12 seconds of local CPU time measured
as wall time; compressed dataset files total 964,910 bytes. Five focused boundary
tests passed, and the Gate/Motion training-pair contact sheets were inspected.
No encoder, feature normalization/basis, predictor training, or GPU allocation ran.
The local `preparation_manifest.json` is the machine-readable evidence for these
counts and exact source/configuration identities.
