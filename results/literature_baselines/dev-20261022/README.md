# Literature baseline development evidence

Frozen Gate predictor, R4/K2; four reused validation pairs, not fresh confirmation.
PaFu rows include three student seeds. These are mechanism adaptations, not full
paper reproductions. CPU development only; equal K does not mean equal total cost.

- [Summary](summary.json), [protocol](protocol.json), [per-case results](episode_results.json).
- [Recorded oracle subsets](oracle_subsets.json): evaluator-only K2 hindsight/local references.
- [WR causal warm-up trace](wr_warmup_trace.json), [PaFu teacher diagnostics](pafu_teacher_diagnostic.json).
- [Training receipt](training_receipt.json), [provenance manifest](manifest.json).
- [Metrics figure](figures/metrics.png) ([PDF](figures/metrics.pdf)).
- [WR timeline](figures/wr_badcase_timeline.png) ([PDF](figures/wr_badcase_timeline.pdf)).

Model weights, training arrays and prediction vectors are excluded; source
hashes are retained. Nothing in this export is a new training/evaluation result.
WR examples are selected by earliest cue time, then episode ID, within failure
and success groups. Figure dots are source-pair means, not independent seed rows.

From the repository root, redraw both figures without raw data or model files:

```sh
python experiments/phase2/export_literature_report.py --render-only
```
