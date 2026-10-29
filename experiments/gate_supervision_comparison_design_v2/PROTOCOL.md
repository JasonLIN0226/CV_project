# Gate matched retention supervision — design v2

Date: 2026-10-29, America/New_York.
Status: DESIGN_ONLY / NOT_IMPLEMENTED / NOT_TRAINED. No label generation, dataset generation, GPU submission or execution is authorized by this document.
Supersedes v1 for future implementation, while preserving v1 and every historical result. Gate remains failed at 42/51. Motion excluded.

## Purpose and names

Compare two newly initialized, identical eviction scorers differing only in their training target:
A: **Feature-attention-proxy distillation (from scratch)**.
D: **Deletion-cost distillation (from scratch)**.
A is a project-defined content-similarity attention proxy, not Transformer attention, not a named paper reproduction, and not a proven best attention baseline. Do not shorten its name to generic Attention in result tables.
Legacy Attention remains a supplementary historical comparator named **Frozen task-trained scorer + novelty**. Neither new arm calls or initializes from its score branch, and neither adds a novelty bonus.

## Shared frozen assets and restrictions

Reuse the three clean-future predictors (17/29/43), input normalization/PCA, output basis, probes and existing caches pinned in v1 protocol.json. No predictor or encoder training, re-encoding, or threshold changes. Original train has 512 pairs/1024 episodes; validation has 96 pairs/192 episodes. Split by whole scene/pair.
The frozen downstream predictors themselves inherited privileged-memory training. This shared historical condition is disclosed, not claimed removed. D uses their deletion loss and may benefit from alignment with that evaluator; positive results establish predictor-specific utility, not universal superiority.

## Deployable student: identical in both arms

B=2 retained history plus R=4 recent observations. On arrival, choose one deletion from exactly the two retained entries plus the arriving entry. Never retrieve discarded observations.
Per-candidate input: p_i (128), p_i-p_current (128), log(1+age)/log(65) without clipping (1), cumulative observed action from candidate to clock (2), p_current (128), mean recent PCA vector (128), last observed action (2): total 517.
Shared phi: Linear(517,64), GELU, Linear(64,64), GELU. Mean pool phi across the three candidates. Head: concatenate own phi and pooled phi, Linear(128,64), GELU, Linear(64,1). Total 45,633 trainable parameters.
Output is a deletion logit; discard its argmax. Exact ties discard the oldest timestamp, identically for both arms. Retain all before the third arrival. No slot embeddings, cue flags/timestamps, target features, future actions, episode IDs or query-end index in student inputs. Same past-action/current/recent access in both arms. No hidden archive or scorer state.

## Shared state manifest without legacy learned selection

Use four deterministic behavior rollouts per episode: FIFO, reservoir, temporal buckets and visual diversity. These select without task-trained scoring or cue metadata. Replace v1's legacy-scorer behavior with temporal buckets. Freeze RNG and state manifest before labels.
Capture final query q's three-entry pre-deletion pool, producing 4096 training records and 768 validation records per seed. Preserve duplicate records with identical multiplicity in both arms and report unique-state count. No cue-based filtering, balancing or manual-keyframe states.
Existing clean-future targets cover q+3 and q+6 only. Earlier cached observations must not be silently treated as clean targets. Query-only supervision is a limitation: deploy at every eviction and measure complete validation rollouts before claiming applicability. Failure does not authorize more data or tuning.

## Training-only supervision

Let p be the existing frozen input PCA representation. Future features/actions below are allowed ONLY in offline teacher-label generation; the student never receives them. Validation labels are diagnostics only.

A: for each h in {3,6}, define cosine similarity
s_ih = dot(p_i,p_(q+h)) / (norm(p_i)*norm(p_(q+h)))
when both norms exceed 1e-8; otherwise define s_ih=0.
a_ih = softmax_i(s_ih / 1.0); a_i = (a_i3+a_i6)/2.
Deletion energy E_A(i)=a_i, so low-attention entries are preferred for eviction. This teacher has zero fitted parameters, no inherited model.score, no novelty term and no cue labels. Similarity may emphasize background rather than causally useful information; this is a falsifiable baseline limitation, not justification for retuning or declaring all attention methods inferior.

D: for each candidate i, evaluate the same frozen predictor with recent R and C minus i, using the recorded future action continuation. E_D(i) is mean normalized latent MSE against the clean targets at h=3,6. Every option retains exactly two entries; three forward examples/state. No cue labels or sibling-ranking terms in the teacher. No unnecessary fourth full-pool forward.

Both teachers: deletion target t_i = softmax_i(-E_i / max(population_std(E),1e-6)). Exact equal energies give uniform targets. Both students minimize cross entropy from their deletion logits to t. Fixed temperatures, no selected target scaling or teacher variant search. High attention means keep, high student logit means delete: explicitly test this sign convention.
Both teacher generators have access to the same training futures; they use them differently by design. There is no claim of identical teacher algorithms or intrinsic preparation cost.

## Matched student training, separately disclosed teacher cost

For each seed17/29/43: paired identical random initialization and optimizer state; identical states/minibatch indices; AdamW 1000 updates, batch64 with replacement, lr0.001, wd0.0001, gradient clip1, FP32, dropout0, constant LR. Final step1000 only, no early selection or hyperparameter search. Only optimizer targets differ. 64000 state exposures per arm/seed, 6000 total updates across both arms and three seeds.
A label work: six 128D cosine comparisons/state; 24576 comparisons/seed. D: three full predictor forward examples/state; 12288/seed, each returning both horizons. Across seeds: 73728 cosine comparisons and 36864 deletion forward examples. Report actual per-teacher wall time, calls, memory, shared I/O and total physical cost; do not manufacture budget equality by dividing costs or running dummy work. Deployment architecture/parameters/calls and student training work are matched; teacher preparation cost is NOT matched.
Proposed future cap1800 GPU seconds INCLUDING labels, training, validation, startup and failures; no budget consumed or approved now. Before later authorized execution profile a fixed32-training-state batch. Stop if the complete paired run will not fit; do not shorten only one arm or automatically expand/retry.

## Validation and independent evaluation

Primary metrics: route accuracy separately on ordinary and long-gap scenes. Also report normalized latent MSE, state position, pair accuracy, event/canonical cue hit, per-seed/per-horizon outcomes. Separate cue retention from conditional prediction correctness. Manual cue-aware reference is privileged; all historical baseline costs and training signals remain disclosed.
Run full irreversible validation trajectories. Before later independent evaluation require source integrity, finite outputs, exact resource matching and causality tests. If either arm cannot outperform the recent-only negative control in validation, stop and report rather than choose alternative settings.
9137 and 2026102317 are exposed and regression-only. v1 reserved seed2026102701 is not evidence of a fresh dataset: verify it has never been generated/viewed before using it. Generate new independent64-ID-pair +64-long-gap-pair data only after student/config/source freeze and separate execution authorization; verify disjoint identities across all historical datasets. Background stress is secondary. Do not tune or select based on old stress failures.
D-A route contrasts use scene-cluster bootstrap, averaging siblings and seeds within scene, 10000 resamples seed0, 97.5% two-sided interval per condition (Bonferroni two contrasts). Both lower bounds >0 required for improvement on both conditions. Mixed/negative findings preserved. Original Gate criteria and failed conclusion unchanged.

## Dataset-shortcut controls: separately preregister before generation

Existing data suffice for a pilot, not proof that visual shortcuts are removed. A later matched dataset extension should independently randomize cue timing/duration and nuisance appearance, include salient non-predictive distractors, and use matched scene pairs separating background from task information. Cue must remain legible and nuisance changes must not alter intended dynamics. All methods share identical scenes and targets; no method-specific failing-case selection. No new renderer/distribution parameters are frozen here: those require a concrete separate protocol, not implicit dataset changes during this pilot.

## Required checks before any run (not yet performed)

- Exact parameter count, initialization, optimizer and batch equality.
- Candidate permutation equivariance and fixed deterministic tie handling.
- Student timestamp firewall; future/label metadata inaccessible, including in cached batches.
- Shared state IDs and multiplicity; no cue-based state selection or old scorer calls in A/state generation.
- Teacher sign/uniform-energy/zero-norm fixtures and D equal-B check.
- Frozen model/transform/cache hashes before/after; no test access during fit/selection.
- Independent no-resurrection replay and physical memory accounting, not just logical B.
- End-to-end timing including all selector preparation; standalone teacher costs disclosed.

This is a design, not a validated implementation or a claim that A will be weaker or stronger. Preserve v1 and all historical data. No training has been launched.
