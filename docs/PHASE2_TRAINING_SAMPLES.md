# Predictor training sample adapter

`experiments/phase2/training_samples.py` implements the proposal's varied historical
subsets and explicit memory dropout at the **sample preparation** level. It can
consume real encoded trajectories when available. The current tests use small
software fixtures; they do not establish predictor training, baseline performance,
or a scientific deletion-supervision dataset.

The historical Phase 1 trainer used recent-only, full-history, privileged K=2, and
random K=2 conditions. Its sampler had a dropout argument, but the published
`repair_trial.py` calls did not enable it. This addition keeps historical source
locks untouched and records actual dropout decisions in new sample metadata.

## Inputs and split identity

`EpisodeIdentity.from_record(record)` accepts the observation-index schema:
`id`, `pair`, `split`, `task`, and `seed`. It ignores simulator state, evidence
timestamps, evaluation-file paths, and other metadata. Register every episode
before sampling:

```python
from experiments.phase2.training_samples import (
    EpisodeIdentity, SplitRegistry, FeatureEpisode, SamplingConfig,
    sample_predictor, phase1_arrays,
)

identities = [EpisodeIdentity.from_record(record) for record in index["records"]]
registry = SplitRegistry(identities)
episode = FeatureEpisode(
    identity=identities[0],
    features=encoded_features,       # float32 [T,D], already frozen/normalized
    actions=observed_actions,        # float32 [T-1,A], a_t takes frame t to t+1
    action_semantics="additive_controls",
    feature_space_id="actual encoder + preprocessing + normalization identity",
)
sample = sample_predictor(
    episode, registry, SamplingConfig(memory_dropout=0.15),
    query=recorded_query, base_seed=17, epoch=0, draw=0,
)
```

`SplitRegistry` rejects duplicate episode IDs, a sibling/group spanning splits,
and a task/environment seed spanning splits even under different group IDs.
Optional `dynamics_split={"long_gap_setting": "test_long_gap"}` enforces declared
held-out dynamics settings when identities supply those IDs. Arrays and declared
IDs still require external source/checksum verification; this registry cannot
prove their generation history.

By default, `train` means training, `validation` means validation, and other splits
mean `development_evaluation`. A caller can propagate the fresh controlled RGB
runner's `development_smoke` role through `split_roles`. Historical `test_*`
names must not be presented as untouched confirmation. This adapter neither
creates new splits nor fits normalization or a basis.

## History selection and dropout

The default mixture is uniform over six available sample modes:

| Mode | Construction |
| --- | --- |
| `random_fixed` | Sample min(K, available history) timestamps without replacement. |
| `random_subset` | Draw a count uniformly from 0 through min(K, available history), then sample that many timestamps. |
| `fifo` | Replay the observed prefix through the shared irreversible memory. |
| `reservoir` | The same replay, using standard reservoir sampling. |
| `temporal_buckets` | The same replay, using documented geometric age buckets. |
| `visual_diversity` | The same replay, using candidate-only visual diversity. |

These are **training sample modes**, not six completed model-bound baseline
experiments. `recent_only` is also available explicitly. Random subsets are an
offline training augmentation; their access to the historical cache must not
be copied into an online policy.

No model-dependent policy is required to create the other samples. To sample a
real `attention` or `prediction_surprise` history, pass an externally generated
`SuppliedHistory(policy, snapshot)` together with the matching `history_mode`.
The adapter checks timestamps, episode/features, recent window, and capacity. It
does not certify how the supplied policy generated that snapshot. Missing policy
weights or scoring callbacks never silently fall back to FIFO or random scores.

Whole-memory dropout independently removes all selected history with probability
`memory_dropout`. It leaves recent observations, actions and prediction targets
unchanged. Metadata records the probability, event, number removed, and history
before/after dropout. An empty sampled history is distinguished from an actual
dropout removal. The 0.15 default is an explicit engineering starting choice,
not a proposal-mandated hyperparameter or an empirically selected value.

Random streams are derived from base seed, task, sibling-group ID, query, epoch,
draw and purpose. Siblings share random subset/dropout choices; content-dependent
policies can legitimately choose different histories. Epoch/draw changes produce
varied training samples without depending on future targets. Non-training splits
disable dropout and require `epoch=draw=0` for stable validation sampling.

The seed includes the query time, so each sample's baseline mode independently
replays a complete causal prefix. Adjacent samples do **not** continue the same
reservoir state and may contain an observation absent from an earlier sample.
This is permitted for independent offline training/validation examples; it is
not a streaming baseline evaluation. Formal online evaluation must advance one
persistent `StreamingMemory` per episode and never refill it from this sampler.

## Causal and future boundaries

`sample.causal` contains current time, recent entries, selected historical entries,
already executed actions `a_0 ... a_(t-1)`, and declared action semantics. All
arrays are copied into immutable storage; a view cannot expose a later action
suffix through its backing array. This complete action prefix is an **offline
training input**, not a claim of a bounded-memory deployment implementation.

`sample.future` separately contains horizons, feature targets, availability masks,
and the recorded action continuation. Missing near-end horizons receive NaN
targets with a false mask; a query with no valid target is rejected. Never feed
this future object or the merged world-model dictionary to a retention scorer.
Simulator state, cue labels, evidence roles and future target values are not
used to select history or decide dropout.

Metadata holds IDs, split role, feature-space identity and sampling decisions.
It is for grouping/audit, not model features. Check preprocessing was fitted on
the training split, and confirm feature dimensions/dtypes and encoder/basis hashes
before combining episodes or using a checkpoint. Merely providing a
`feature_space_id` is not a provenance verification.

## Historical predictor bridge and remaining work

`phase1_arrays(sample)` returns `(model_input_dictionary, target_array)` compatible
with `phase1.inputs.collate`. It requires recent length 4, horizons (3,6), both
targets available, float32 features/actions, and two-dimensional additive controls.
It preserves the original recent-then-history token order and computes executed
additive-control sums from each retained timestamp to the query. The predictor
receives the recorded future action continuation as supervised world-model
conditioning; the target features remain a separate array.

Absolute-coordinate PushT actions are explicitly rejected by this bridge. They
can be stored/sampled with `action_semantics="absolute_targets"`, but need a
compatible sequence-aware PushT predictor adapter. Summing absolute XY targets
would not reproduce the environment's control semantics.

The tiny CPU integration test checks shape/mask compatibility with the existing
`DynamicsPredictor` and `CenteredTransportPredictor` using random toy parameters.
That test is **not** a trained baseline, a checkpoint replay, or a performance
result. Actual next steps still require verified frozen JEPA features, training
normalization/basis, model weights or an explicitly new training initialization,
an optimizer/training loop, validation checkpoint selection, and logging.

Run only the independent adapter checks with:

```bash
.venv/bin/python -m pytest -q tests/test_phase2_training_samples.py
```
