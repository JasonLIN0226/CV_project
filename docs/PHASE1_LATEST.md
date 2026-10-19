# Phase 1: latest verified status (19 October 2026)

## Objective

The research asks whether the prediction error caused by deleting an observation can help decide what a bounded visual memory should retain. Phase 1 checks the prerequisites for studying this question: historical evidence is usable, the predictor responds to it, and controlled deletion comparisons produce meaningful signals.

## Development results

| Registered checks satisfied | Gate | Motion |
|---|---:|---:|
| Original prediction requirements | 51/51 | 51/51 |
| Additional fixed-reference requirements | 32/32 | 32/32 |
| Task-specific deletion diagnostics | 72/72 | 72/72 |

These are correlated registered comparisons, **not independent datasets or prediction accuracy percentages**. The 51 original checks include evidence decoding, prediction benefit and content interventions. The 32 fixed-reference checks compare against the frozen recent-only reference and include difficult-case requirements.

The two deletion counts have different meanings. Gate's 72 combine 54 individual-seed and 18 seed-mean critical-versus-other deletion contrasts across age-controlled pools. Motion's 72 are cross-horizon future-informed oracle comparisons against uniform, FIFO and reject-newest baselines, requiring positive paired confidence bounds for both feature and position error. They demonstrate diagnostic headroom; they do not prove that a learned past-only policy can achieve the same choices. No retention policy was trained.

The evaluated synthetic development data use seed 6301, three model seeds (17, 29, 43), horizons 3 and 6, and three test conditions: ordinary, longer gaps and distractor shifts. Each development test condition has 128 paired cases per task. These data have been repeatedly inspected during repairs, so success requires fresh confirmation before claiming Phase1_GO.

## Fixed repairs and their scope

- **Gate:** disable the complete temporal-extrapolation contribution, as selected on validation and subsequently locked for the development recheck. This addresses deletion sensitivity while retaining every original acceptance requirement.
- **Motion:** combine same-history frozen-reference visual prediction with guarded candidate position and a fixed equal-weight training-only observation-innovation map. Neural predictors stayed frozen during this repair; two small train-only maps were fitted. The complete development recheck meets the unchanged requirements above.

Native independent audits and additional local scalar replays support these results. This is a controlled synthetic **I-JEPA** prototype with a supervised readout and renderer-specific observed-RGB guard. It is not a V-JEPA or real-video confirmation result. Existing raw arrays and model weights remain external HPC assets; the published summaries contain per-comparison estimates and confidence intervals.

## Independent confirmation and execution failure

CPU equivalence passed 9,728 sample comparisons, 24 validation batches, 240 Gate prediction vectors and 468 Motion cached vectors. This verifies execution equivalence, not a new scientific result.

Fresh seed 9137 was never generated or accessed: **0/12 new neural models and 0/2 new maps** were completed. The required fresh train-only transforms, new models and locked-before-test evaluation therefore remain outstanding.

| Allocated GPU time | Seconds | Duration |
|---|---:|---:|
| Earlier settled allocations | 13,131 | 3 h 38 m 51 s |
| Job 8536 failed startup allocations | 26,520 | 7 h 22 m |
| Total | 39,651 | 11 h 00 m 51 s |
| Approved cumulative cap | 28,800 | 8 h |
| Excess | 10,851 | 3 h 00 m 51 s |

Job 8536 accumulated 57 separate allocation records (56 automatic requeues followed by cancellation) without entering the GPU experiment script. The agent's in-process guard did not cover startup/configuration failures or scheduler requeues. This is an execution-protection failure and budget overrun, not a fresh scientific rejection. Allocated time includes failed startup and is not productive training time.

All compute tasks are ended, the temporary zero-GPU evidence node was released, and the monitoring automation was deleted. No new GPU submission is authorized under the exhausted cap. Any future confirmation needs explicit additional resource authorization and separately validated startup/requeue protection; neither the acceptance criteria nor the fresh-data boundary should change.

## Conclusion

**Both development tasks meet their registered checks. Full Phase1_GO and entry into the retention-policy stage remain unestablished because independent confirmation did not run.**

## Published evidence and verification

- [Illustrated English Word report](../reports/independent_confirmation_boundary/Phase1_Confirmation_Review.docx)
- [Gate detailed estimates and confidence intervals](../results/phase1_latest/gate_summary.json)
- [Motion detailed estimates and confidence intervals](../results/phase1_latest/motion_summary.json)
- [Native budget accounting](../results/phase1_latest/budget_boundary.json) and [57 allocation records](../results/phase1_latest/startup_allocations.csv)
- [Evidence manifest and original artifact provenance](../results/phase1_latest/manifest.json)

Run `python tools/verify_phase1_latest.py` for a standard-library-only check of the published hashes, comparison counts, allocation durations and final status. This snapshot check does not rerun model inference or replace the original raw-data audits.
