# Use-time credit and reader reliability: diagnostic pilot

Written 2026-10-24 before the new pilot results. This is a bounded diagnostic continuation of the proposal, not a claim of a new successful method.

## Questions and order

1. Audit all 1,282 saved OLD TRAIN deletion parents without refitting. Separate both targets before the result first appears, straddling that boundary, and both after it. Count actual cue-destroying deletion preferences, ties, and loss magnitudes, distinguishing one cue copy from redundant copies. Compare cost targets with actual future-attention targets.
2. On the same parent and frozen reader, compare local h3/h6 loss with loss at the episode query after each deletion and a fixed attention continuation. All continuation steps use only currently resident memories and newly observed frames. Discarded evidence cannot be restored from the offline archive. The teacher may use the recorded future; the student may not receive future actions, query time, labels, scene state, or oracle outputs at inference.
3. Fit matched local-cost, use-time-cost, attention-label, and shuffled-label scorers using only OLD TRAIN. Use identical causal features, architecture, regularization, and train-support handling. Evaluate complete streaming episodes, cue survival, pair-level uncertainty, and overhead. An oracle ranking change alone is not a learned benefit.
4. Check reader training scale with one shared architecture and matched input interface. A formal Phase 1 checkpoint that changes architecture, normalization, projection, or transport is a separate reference, not a sample-count-only intervention. Verify an actual A100 device/UUID inside the scheduled encoding/training job and retain exit status and hardware evidence.

## Split and claim boundaries

- Keep the archived seed-20261023 confirmation artifacts and historical conclusion immutable. This seed has now informed hypotheses, so further analyses on it are explicitly exploratory; it cannot confirm a newly selected method. Never rewrite the earlier locked protocol to disguise this reuse.
- Reader/scorer tuning uses training and designated development data only. A later confirmation requires a separately frozen configuration and untouched seed. A fresh seed used for learning curves or selection is development, not confirmation.
- Report the full sample, including cases where keeping cue raises feature MSE. Feature-space paired-future direction is a diagnostic proxy, not simulator-state accuracy or a calibrated confidence measure.
- Future-use value is conditional on the frozen reader and continuation policy. A fixed attention continuation can produce zero differences when a later redundant cue repairs a deletion; preserve these ties.
- Do not assume a visually salient distractor defeats attention, diversity, or WR. A distractor experiment is conditional on a reliable reader and informative label pilot. Include matched nuisance-free and genuinely useful salient-event controls, fixed K and byte accounting, and multiple timings/intensities. Actual baseline outcomes must be measured.
- The identity of useful evidence must be inferable from past/current observations before the irreversible decision, or the experiment must explicitly quantify that ambiguity. Do not assign hidden random cue-versus-decoy roles and interpret a hindsight oracle advantage as learnable causal headroom.
- TD is deferred: a per-item survival formula is not yet a well-defined Bellman target for a set-dependent irreversible deletion policy. Define state, action, reward/cost, terminal loss, and counterfactual continuation before implementing it; do not double-count prediction losses.

## Interpretation

If local labels already favor cue retention but the learned scorer deletes it, diagnose fitting/support/generalization. If use-time labels reverse cue-destroying local preferences and a causal matched student improves on held-out development, proceed to reader-matched stress tests. If attention/diversity remain as good or better, report the negative comparison. Do not extrapolate a Gate repair to PushT planning or to the PM matrix reader.

## Related-work boundary

- [LEMN](https://arxiv.org/html/1812.04227) learns bounded episodic replacement using task reward arriving at a later query.
- [KVP](https://arxiv.org/html/2602.10238) trains rankings with a reward derived from future-attention importance; this is not counterfactual prediction-loss supervision.
- [TVT](https://arxiv.org/html/1810.06721) transports later value to earlier memory-associated actions for long-range credit assignment.

Delayed credit is established prior work. Any contribution here must come from controlled evidence about set-conditioned deletion loss, use-time extension, causal learnability, strong matched baselines, and costs. Lower variance or novelty is not assumed.
