# Gate memory-retention comparison — latest validation

This Gate-only comparison asks how different online rules select two history frames for the same frozen future predictor. It reports performance on the existing train/validation dataset, not a fresh independent test. The original Gate assessment is historical context, not a prerequisite for this comparison; these results do not retroactively change it.

## Latest results

All methods see the same 192 validation episodes (96 paired scenes), frozen predictor, four recent frames, two history slots, and horizons 3 and 6. Values below pool three predictor seeds and both horizons. The 1,152 rows per method are repeated measurements, **not independent examples**. An event hit means at least one retained history frame falls in the cue interval; it is not the same measure as route accuracy.

| Method | Route/state accuracy | Cue event hit | Normalized latent MSE ↓ |
|---|---:|---:|---:|
| Manual cue-aware reference (privileged) | 95.57% | 100.00% | 0.09783 |
| Deletion-cost student | 94.53% | 100.00% | 0.09934 |
| Visual diversity | 71.79% | 47.40% | 0.11385 |
| Prediction surprise | 66.15% | 34.55% | 0.11765 |
| Reservoir | 57.38% | 16.84% | 0.12362 |
| Temporal buckets | 56.60% | 15.62% | 0.12256 |
| FIFO | 56.51% | 13.54% | 0.12423 |
| Feature-attention-proxy student | 56.08% | 13.72% | 0.12332 |
| Recent frames only (negative control) | 50.00% | 0.00% | 0.12497 |

The retired task-trained-score-plus-novelty method is omitted from the current result table. The original run artifacts and source are unchanged for provenance: the frozen historical validation runner computed it at the time, and removing it from this table is a presentation change rather than a new run.

## Data design

![Example clean-future Gate episode](../results/gate_retention_comparison_20261029/data_preview.png)

The structured Gate scenes are 96 × 96 RGB, with two parallel pale-gray arrows and cue position sampled over the legal history. The cue rotates both arrows; paired sibling episodes share actions and nuisance masks but have different cue/target outcomes. Before and including the query frame, 7–14 pale-gray vertical lines are distributed over the whole image, preserving the cue and scene objects. Frames after the query use the corresponding clean rendering without the random lines. There are 512 training pairs (1,024 episodes) and 96 validation pairs (192 episodes), disjoint by scene/pair. The source dataset seed is 6301. No new data were generated for this comparison: it reused the already encoded clean-future train/validation cache.

The exact generation entry point is [`materialize.py`](../experiments/gate_clean_future/materialize.py), using [`structured_design.py`](../experiments/retention_structured_noise_audit/structured_design.py) and [`design.py`](../experiments/retention_position_noise_audit/design.py). A readable configuration with its output path replaced by a placeholder is [here](../results/gate_retention_comparison_20261029/dataset_config_template.json); it is a portable template, not the byte-identical HPC resolved configuration. [Dataset preflight](../results/gate_retention_comparison_20261029/data_preflight.json) records source-data checks.

## Models and fairness

Both new students are freshly initialized, identical 45,633-parameter eviction scorers. Each update chooses one item to remove from exactly two retained history entries plus one arriving entry. The student sees past-only projected features, relative position/age, recent observations and observed actions; it does not receive cue labels or future frames/actions at selection. A deterministic tie discards the oldest timestamp. Shared behavior rollouts and minibatch order, AdamW settings, 1,000 updates per arm, and seeds 17/29/43 are fixed. The frozen downstream predictor and its encoder are not retrained.

The **feature-attention-proxy** target uses similarity to the future feature offline. The **deletion-cost** target uses the same frozen predictor and future training targets to score the loss of deleting each candidate. These teacher signals and their preparation costs differ: matching student architecture and update count does not make the full training costs or supervision equal. The predictor-specific deletion target can favour the predictor used for evaluation. Manual selection is a privileged reference because it knows cue positions. The attention proxy is a project-defined baseline, not a complete reproduction of a particular Transformer attention method.

Independent replay checked 4,608 online policy/episode traces and 115,200 state transitions without resurrection. Offline replay still stores historical prefixes, so this establishes the logical B=2 selection constraint, not a measured physical memory bound. Both students exceed the recent-only control on pooled validation, but this is not evidence of independent generalization or statistically confirmed superiority.

## Code and artifacts

- [Frozen experiment protocol](../experiments/gate_supervision_comparison_design_v2/PROTOCOL.md), [authorized execution amendment](../experiments/gate_supervision_comparison_design_v2/EXECUTION_AMENDMENT_20261029.md), and [scorer implementation](../experiments/gate_supervision_comparison_design_v2/core.py).
- [Teacher construction](../experiments/gate_supervision_comparison_design_v2/teachers.py) and [matched checkpointed training](../experiments/gate_supervision_comparison_design_v2/checkpoint_fit.py).
- [Machine-readable latest summary](../results/gate_retention_comparison_20261029/summary.json) and [Chinese report](../results/gate_retention_comparison_20261029/RESULTS_ZH.md).
- The 39 MB raw validation output, trained student checkpoints, predictor weights and cached feature arrays are **not included in this commit**. The experiment was run on NYU HPC, and recorded `/scratch/xl4358/...` paths identify provenance only; they are not a teammate-accessible download. SHA-256 of the raw output is `a2a6c3c2ebca33c5540e60de2e50ab474415b91ad978c850f8c864073dc40b04`, and its independent audit summary is `b4981009bff1373c75c6ea2b4c7f67455deb214bfdb57ea066c822ea62581a7a`. The v2 runner is an exact course-HPC snapshot with absolute original paths and is not claimed to be portable without configuration.
