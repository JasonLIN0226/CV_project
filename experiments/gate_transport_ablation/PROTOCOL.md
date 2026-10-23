# Gate frozen transport ablation

One predeclared intervention: zero the complete transport projection output, retaining the content projection, frozen recent predictor, all weights and all inputs. No fitting, training, checkpoint choice, threshold change, test inference or policy optimization.

Replay all original 96 validation pairs, seeds 17/29/43, horizons 3/6, four original contexts and nine equal-capacity 3-to-2 deletion forks. Every intervention forward is paired with the unchanged forward on exactly the same batch. Verify identical content/reference outputs and unchanged complete state digests, explicit transport subtraction identity, empty-history equality and all 14976 baseline scores against the audited parent. Caches/checkpoints/normalization/basis/readout are reused and hashed before/after.

29952 episode scores;14976 paired scores. Bootstrap base pairs10000 seed0. Reuse the parent's 10 percent relative effect and paired95percent lower>0 diagnostic criteria; show x/18 individual-seed and x/6 seed-mean cells. These are validation diagnostics, not original test acceptance. Also show native cue benefit and original context errors; a model that suppresses useful evidence does not solve deletion utility merely by reducing feature magnitude.

This removes the whole learned transport contribution including its last-observation component, not only the slope times elapsed-time component. Co-adaptation is possible. Failure of this ablation does not prove time extrapolation is harmless. One CPU-only execution, no automatic repeat or additional variants after results.
