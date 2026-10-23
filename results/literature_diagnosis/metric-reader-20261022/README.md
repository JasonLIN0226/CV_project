# Gate predictor / metric diagnosis

This is exploratory diagnosis of the existing 192 development episodes, with
the predictor, normalization and all online retention policies frozen.

- [Complete global comparisons](global/episode_rows.json) and [actual-parent comparisons](local/episode_rows.json).
- [Diagnosis counts](diagnosis.json), [arithmetic audit](audit.json), [independent frozen-model replay](global/independent_review.json).
- [Local candidate sets](local/parents.json), [identical noncue-feature timing audit](local/noncue_timing_audit.json).
- [Position-readout validation](global/readout_validation.json), [original unstable cue probes](local/cue_probe.json).
- [Feature-error decomposition](figures/metric_decomposition.png), [reader interventions](figures/reader_interventions.png).

All candidate-set choices are tied to the original loss. Reader interventions
do not reoptimize sets and are not a repaired method. The original cue probe has
severe train-support scaling instability; consult the input-support diagnosis
before interpreting it as evidence about representation information.

Prediction arrays, fitted evaluation readout and original inputs remain under
the hash-recorded build paths. Replot portable figures without them:

```sh
python experiments/phase2/export_gate_reader_diagnosis.py --render-only
```
