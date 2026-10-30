# Reader aggregation diagnostic — source freeze, 2026-10-25

**Status: ready for source freeze before any new scientific fit or intervention.** This protocol follows [the completed terminal R/D/S study](TERMINAL_THREE_STRUCTURES_REPORT_ZH.md). The executable architecture, probe controls, cache bindings and end-to-end synthetic test have been reviewed. The associated configuration is `experiments/phase2/configs/reader_aggregation_20261025.json`. The source commit and archive hashes are recorded in the release/execution receipts; this status is not a claim that training has completed and does not authorize opening the original confirmation set.

## Question and evidence boundary

The completed study found failures in reading salient useful R/D evidence and in the S full-three-item reader. S K2 qualification passed, whereas K3 failed. These observations do **not** establish that softmax necessarily loses information, that attention is intrinsically incapable of XOR, or that the frozen encoder discarded the cue. They motivate two separate checks:

1. Are visible cue payloads and S tags linearly decodable from the frozen visual features, and does the existing TRAIN-fitted rank128 projection preserve that decodability?
2. With data, normalization, projection, prediction targets, training loss and training schedule held fixed, does changing the history aggregation improve qualified K2/K3 reading?

A successful linear probe is evidence of accessible information, not a guarantee that the dynamics reader learns to use it. A failed linear probe is not proof that all nonlinear information was destroyed. A different aggregation succeeding would implicate this implementation/optimization choice; extra parameters and inductive biases prevent a universal causal claim about softmax.

The pilot uses one paired initialization seed17. It is an exploratory architecture diagnosis, not a multi-seed method result. No student training, original CONF opening or PM experiment belongs to this run.

## Frozen inputs and split use

Reuse the completed terminal study's **exact** TRAIN48/VAL24 families and the read-only encoded arrays, TRAIN normalization, input/output basis and action banks. Bind all files by SHA256. Do not refit preprocessing per arm, change image rendering, enlarge the dataset, add semantic labels to the reader, or alter future-control visibility. The source artifact must be verified against the `terminal-structures-20261025` release; absent feature arrays must be recovered from the recorded HPC run, not replaced by silently re-encoded data.

| Dataset | Seed | Families | Use |
|---|---:|---:|---|
| Existing TRAIN |20261101|48|Reader fit; probe normalization and fit|
| Existing VAL |20261102|24|Reader checkpoint and fixed probe regularization selection|
| Existing DEV |20261103|24|Already observed diagnostic comparisons only|
| Fresh diagnostic |20261120|24|Once-only held-out diagnostic after all five checkpoints and probes are locked|
| Original CONF |20261104|48|Remains unopened; excluded from this protocol|

Seed20261120 was selected before outcomes and checked against existing generator/encoding receipts before source freeze; executable guards additionally reject every parent split seed. Its 24 families contain1728 R/D/S records. Use the unchanged generator and original action priors, with generator split identity `development` in a separate directory. Mark the external analysis role `fresh_diagnostic` so it cannot be mistaken for old DEV or original CONF. No pixels, labels, features or model results from this seed are generated before the checkpoint manifest is frozen.

Once any held-out diagnostic outcome is viewed, this seed is consumed. A later modification needs a new protocol and another unseen seed; repeated evaluations are permitted only for deterministic artifact verification without fitting or choice changes. Even a fully successful five-arm comparison remains a single-seed pilot, not the original proposal's final confirmation.

Family is the grouping and resampling unit. Report R/D/S separately, with fixed pre-existing appearance strata. Do not pool the four-times-larger S record set into an unweighted overall headline.

## CPU linear feature probes

Use two primary probe representations and audit the intermediate normalization:

- **Raw frozen I-JEPA output:** the stored5120D observation feature, before the terminal-study normalization.
- **TRAIN-normalized5120D output:** audit the exact invertible affine map to the existing preprocessing; do not count another standardized ridge fit of the same information as a third independent probe.
- **Reader input projection:** the exact existing TRAIN-fitted128D coordinates.

The raw representation name means the encoder output, not RGB. Fit any probe centering/scaling only on TRAIN and save it. Normalizing the raw vector for numerical conditioning does not create an independent representation ablation; the raw-versus-normalized comparison mainly checks preprocessing, while5120D-versus128D tests the bottleneck. Do not give the probe simulator coordinates, timestamps, future images, roles, IDs or other candidates' features.

Fit six binary probe heads: `R_a`, `R_b`, `D_a`, `D_b`, `S_payload`, and `S_tag`. The first four decode the visible payload on the indicated task/axis cue; D decoys are excluded because they are outside the task-relevant payload probe scope; the renderer does draw an arrow beneath the decoy marker. S heads decode only the visible local payload or local tag. Evaluate by cue salience, slot and label value. Roles/axes are evaluator metadata for selecting the task, never prediction inputs. No global XOR/outcome label is used to train a single-frame payload/tag probe.

Use unique evidence observations within each split rather than counting factorial copies of the same frame as extra examples. Deduplicate by `(family, RGB hash, probe head)` and require a unique target for each key, verifying that repeated cue observations have consistent visible labels. Head-specific target construction must not silently keep a contradictory label for the same cue image. Record original versus unique counts and sample weights. All observations from one base family remain in the same split. Verify that family hashes do not cross splits; identical local cue types across splits are legitimate task vocabulary, not independent identities to memorize.

Use linear ridge classification with binary targets and regularization λ∈{0.001,0.01,0.1,1,10,100}; the intercept is not penalized. The objective is mean squared classification loss plus λ times the weight norm, with the sample-weight normalization recorded so λ does not scale implicitly with sample count. Choose λ independently for each head/representation using only VAL balanced accuracy, breaking ties in favor of the largest λ. Standardization is fitted on unique TRAIN observations. Weight families equally within each head rather than counting factorial copies as independent evidence. Save normal-equation residuals/solver diagnostics, balanced accuracy, both class recalls and family bootstrap intervals; never pick the best of separate DEV regularization settings. The weighted primal/dual spectral solver and its selected-model normal-equation residual checks are implemented and tested; nonfinite results and residual violations stop execution.

Required leakage controls:

- Check that task/payload/tag factors are fully crossed as intended in S and that class balance is recorded for every stratum. Do not claim exact chance for an unbalanced R/D decoy stratum.
- Fit the same probe on the current blank/recent observation, matched to the cue target, as a negative control. The current observation must not reveal which S tag or payload belongs to a historical slot. Record that D decoy payload is outside the probe target definition, not a failed encoding result.
- Fit a fixed TRAIN label-permutation control: permute labels across deduplicated observation rows within each probe-head×salience stratum, preserving its label counts and leaving VAL/diagnostic labels genuine. Use the locked permutation seed and save the permutation indices. Do not permute labels identically across splits or construct an invertible global0/1 relabeling. Labels, roles and hashes are never prediction inputs.
- Verify that a probe API accepts only feature arrays at prediction time. Metadata is allowed solely for targets, grouping, weighting and audit.

High raw/low projected decodability motivates a future projection study, not a claim about aggregation. Low raw linear decodability leaves nonlinear representational explanations open. High projected decodability with poor reader performance motivates the architecture comparison below. These interpretations are descriptive; no posthoc probe cutoff controls which arm is reported.

## Five reader arms: agreed architecture scope

All arms retain the same current feature, age and executed-action input definitions, future action-conditioned prediction interface, rank128 basis, coefficient output and h3/h6 feature targets. The current configuration's `heads` and `layers` fields do not themselves create multi-head attention in the old `DynamicsPredictor`; a multi-head arm must implement and expose real head-wise weights.

| Arm | History aggregation | Capacity/control |
|---|---|---|
| `original` | Exact original learned single softmax, temporal summaries and action interaction | Reference implementation and forward-equivalence test |
| `attention_mean` | Original learned-softmax plus uniform weights, each producing complete temporal summaries | Two summary channels; enlarged output input |
| `multihead` | Shared first scoring layer with four output logits and four independently normalized history weights | Four summary channels; enlarged output input |
| `item_attention` | History token=p+past+learned(age/8), then one pre-LN self-attention residual block (four heads) and a2d feed-forward residual block, followed by original single-head pooling | Explicit relational preprocessing; report all added parameters |
| `wide_control` | Exactly original score, single softmax, action and temporal summaries; increase only the two output-MLP hidden widths | Choose the integer width whose total active parameters are closest to the largest new aggregation arm, before training |

For each pooling channel, retain the six d-dimensional temporal/action vectors (memory, velocity, transported memory, memory×action, past-action mean, velocity×action) and the mean-age/existence statistics. Current feature, action feature and horizon are shared; concatenate channel-specific summaries into the unchanged output-basis prediction interface. The new reader module records exact tensor shapes and the wide-control parameter-count search receipt before fitting. The original aggregation already includes velocity and action interactions, not only a weighted mean; attention weights alone cannot prove inability to represent XOR.

Copy all identically shaped shared parameters from one reference model initialized with torch seed17 **after loading the frozen basis**. Seed newly introduced parameters deterministically and archive the initial state hashes. Original training consumed torch RNG while fitting PCA before constructing its model; the paired retrained `original` therefore need not reproduce the old trained checkpoint bit for bit. Re-evaluating that archived checkpoint is a reproduction check, not the new paired baseline. No refitting PCA solely to recreate its RNG history.

A source-level count with the archived basis gives active trainable parameters: original215,297; attention_mean313,857; multihead511,364; item_attention347,905; wide_control510,927 (output hidden313,0.08546% below multihead). These are architecture counts, not trained results. Verify them in the run receipt. Report optimizer state bytes, forward examples, measured training duration, and K2/K3 latency for each arm. Same examples and updates do not imply equal parameter capacity, FLOPs, wall time or optimization difficulty. `wide_control` addresses the simplest extra-capacity explanation but does not equalize architecture, FLOPs or optimization. Its width is determined from parameter counts, never from accuracy. No claim of universal softmax failure follows even if an alternative wins.

For absent history, all arms must return finite, deterministic zero-history aggregates and must not softmax an entirely invalid sequence into a nonzero memory. Verify padding invariance and permutation behavior under coordinated content/age/action permutations.

## No-fit pooling intervention on the selected original reader

Besides the five trained arms, evaluate a **no-fit uniform intervention** on the selected retrained `original` checkpoint. Keep every weight and input fixed; replace valid-history softmax weights with1/n and recompute all derived temporal statistics, including memory, mean age, velocity and transported/action-interaction terms. Empty-history weights remain zero. Do not retrain, tune a temperature, choose an intervention strength or select a different checkpoint. Check K0/K1 equality to the ordinary forward and retain exact input/output hashes.

Compare original versus this intervention on old DEV and once on the fresh diagnostic. It directly tests the computational effect of the learned weighting at fixed downstream parameters, whereas the trained `attention_mean` arm also changes capacity and learning. Improvement supports a role for weighting in this particular reader. Failure or deterioration can reflect distribution shift in the fixed readout and does not prove the original weights optimal. Neither result establishes that one softmax or one mean cannot represent the relation; for example, a mean of suitable one-hot features can preserve tag/payload counts.

An additional archived-checkpoint intervention uses the released original `reader_selected.pt`, with its exact basis and the existing DEV only. Evaluate its native and forced-uniform forwards without fitting, data generation or fresh-diagnostic access. This answers the direct question about the already observed reader; keep it separate from the paired retrained original, whose initialization differs. Archive the same qualification rows and coefficients and bind all parent cache/source bytes.

Timing uses fixed first16 records, K2 slots[0,1] and all K3 slots, one warmup batch and five timed batches with CUDA synchronization. Per-parent seconds are batch16 throughput-normalized timing, not single-request latency; encoder time is excluded.

## Shared reader training

- Initialization seed17,1024 updates; hidden128, dropout0 and input/output rank128.
- Pure feature MSE only. Do not reintroduce pair/rank losses that can shift the missing-information conditional mean.
- Terminal query origins only; history sizes0/1/2/3 equally represented by the exact old schedule.
- Equal task weight; two complete factorial groups per task per step. Keep all payload siblings, including S outcomes that share XOR, under the same aggregation input choice.
- AdamW lr0.001, weight decay0.0001, gradient clip1; learning-rate halves at updates341 and682.
- Identical saved group/subset schedule across arms. VAL at step0 and every128 updates through1024. Select each arm's strict earliest minimum equal-task/equal-history-size VAL MSE.
- No per-arm early stopping, extra training, hidden-size sweep, output-basis refit or model selection from old DEV/fresh diagnostic.
- Archive all learning curves, selected steps and optimizer/example counts. Retrained original is the paired baseline; the old archived original checkpoint is a separate reproduction reference.

A single shared reader per arm covers all three tasks. Do not report a synthetic winner assembled from task-specific checkpoints or use accuracy on the fresh diagnostic to select an arm for the same diagnostic's headline.

## Endpoints, gates and readout diagnosis

Reapply the completed study's **unchanged executable qualification rules** to every arm on old DEV and the fresh diagnostic, with raw per-record outputs retained. Aggregate K2 accuracy includes fixed and native ages; fixed/native arms must also be shown separately. Each task and each appearance condition must pass the original>=80% complete-set and content-flip accuracy requirements and the matched complete-versus-missing MSE margin>1e−6. K3 full-set accuracy must meet>=80% separately. S has two future classes; R/D have four.

Report the same single-cue/blank conditions, complete and missing-set MSE, tie-credit versus strict classification, wrong-content flips, and distance to the conditional mean. No new hedging tolerance or gate threshold is chosen from observed results. Native-age and content-only fixed-age interventions stay distinct.

Primary diagnostic comparisons are paired per-family changes versus the retrained original in:

1. Complete K2 and K3 accuracy/MSE, split by task and appearance.
2. Native versus fixed-age content performance.
3. On S, K2-correct/K3-wrong count and incomplete-versus-complete prediction geometry.
4. Qualification flags per task and appearance, including failures.

Family bootstrap uses2000 paired resamples and a newly locked analysis seed. Intervals describe this paired-seed pilot; do not treat many within-family variants or five architectures as independent replications. If inferential significance is claimed over multiple arm/task endpoints, declare multiplicity treatment rather than selecting a favorable p-value.

This pilot ends at probe and reader qualification. Evaluate all five frozen trained arms and the fixed uniform intervention once on the fresh diagnostic regardless of which old-DEV gates passed; there is no old-DEV-based variant filtering. Baseline/medoid/full-reference comparisons, student fits and original CONF opening do not run automatically. A follow-up retaining those selectors must preserve the original16 TRAIN reference plans,32 evaluation plans and prediction-distance kernel; full-reference sensitivity requires both K2 and K3 qualification for its task.

## Decisions after this pilot

- **Features decodable, multiple aggregators fail:** the encoder-information explanation becomes less plausible, but optimization, sample size, output basis and task-rule learning remain open.
- **One aggregator improves but does not pass all relevant conditions:** retain the negative qualification; do not route around it using an overall average.
- **K3 improves and full-reference sensitivity succeeds:** evidence favors reference-reader adequacy as a practical prerequisite. It does not establish a novel memory method or universal superiority of full-reference sensitivity.
- **All qualifications pass:** stop at the completed reader diagnostic. Write a separate frozen protocol to choose/fix a reader and compare baseline/student strategies with at least three planned training seeds and an unseen confirmation set. The old CONF remains unopened unless that later protocol explicitly binds its allowed use.
- **Changing aggregation does not help:** publish the null result. Do not silently add data, losses or extra updates inside this run.

PM remains unchanged. The study can inform its requirement that world-model outputs be valid before using them for memory decisions; it does not run or validate PM S0–S2, CHANGE or BF-Delta.

## Immutable execution and publication

Use a new empty output directory. Before fitting, bind source/config hashes and the read-only parent release manifest. Do not overwrite old results or an existing partial run. An interrupted resume may continue only from the exact saved state; it is not a chance to alter hyperparameters.

The fresh diagnostic unlock must validate all five selected checkpoint hashes, the no-fit intervention definition, source/config hashes, original data/basis hashes and finalized probe checkpoints/hyperparameters. It must never invoke the old `confirm-open` stage. Save the once-only diagnostic receipt, encoder metadata, hardware/job identities, predictions and coefficient reconstruction checks. Missing arrays, failed equivalence tests or source/hash drift stop execution with explicit status.

Publish the exact negative and positive outcomes together in CV PR#1, with a bounded PM connection update in PR#6. The provisional allocation estimate is0.6–1.2GPUh for five training arms, encoding fresh diagnostic observations and qualification checks. This is an estimate, not a user-imposed cap or measured cost. Report actual scheduler allocation and stage durations; do not extrapolate the old single-reader timing as a completed result.

## Source-freeze preflight completed

1. Aggregation formulas, original-forward equivalence, shared-parameter mapping and capacity count verified in executable tests.
2. Probe ridge grid, permutation unit, deduplication, weighting, class recalls, family bootstrap and numerical residual checks implemented and tested.
3. Parent source and ten required cache files verified against the released manifest. Recovered raw features SHA256 `f1123a87c41f9bb2c8d461ca3d74a6f7d626cf39bdad7ba51573b6bd2279ac9f` and normalized features `9090806a3e03cedeecc8046a675ba8d35523380f545d021f7fb0a2410234e6b1` match the original HPC files.
4. Fresh seed uniqueness and original-CONF closure checked. CLI/manifest, input-isolation, mutation rejection and full synthetic pipeline tests pass. Synthetic tests are not scientific outcomes.
5. All selected models, shared initialization, protocol/config bytes and parent bindings are checked before the fresh diagnostic unlock; parent cache is rechecked in full at that boundary.
