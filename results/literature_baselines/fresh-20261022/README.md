# Fresh Gate literature-baseline development evidence

192 episodes / 96 new scene pairs; three conditions, 32 pairs each. Fixed old
predictor and frozen dev-v3 PaFu students; no fresh fitting. These are explicit
WR/PaFu mechanism adaptations, not full paper reproductions or a held-out benchmark claim.

- [Summary](summary.json), [protocol](protocol.json), [per-case rows](episode_results.json).
- [Independent arithmetic audit](audit.json), [oracle enumeration](oracle_enumeration_audit.json).
- [Statistics and pair CIs](statistics.json), [history controls](history_controls.json).
- [WR strata and traces](wr_classification.json), [PaFu teacher diagnostics](pafu_teacher_diagnostic.json).
- [Actual RGB bad cases](bad_cases.json), [data provenance](data_manifest.json), [export manifest](manifest.json).
- [Metrics PNG](figures/fresh_metrics.png) / [PDF](figures/fresh_metrics.pdf).
- [RGB cases PNG](figures/fresh_rgb_badcases.png) / [PDF](figures/fresh_rgb_badcases.pdf).
- [A100 receipt](hpc-execution-receipt.json), [scheduler screenshot](a100-completed.png).

A100 encoding job10097 completed with 143 allocated GPU-seconds; model prediction
and offline oracle enumeration ran on CPU. Encoding time alone is not allocated
GPU wall time. Equal K is not equal total bytes. CIs resample independent scene
pairs after averaging siblings and fixed student seeds, not individual rows.
Selected RGB examples are explicitly chosen diagnostic cases, not prevalence estimates.

Weights and large arrays are excluded; their raw hashes remain recorded. Replot
from these portable JSON/PNG files, without a model or the original dataset:

```sh
python experiments/phase2/export_fresh_literature_report.py --render-only
```
