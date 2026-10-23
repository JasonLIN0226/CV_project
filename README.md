# Bounded Visual Memory

Can the prediction error caused by deleting an observation identify what limited visual memory should retain?

## Final independent confirmation — 20 October 2026

**Phase1 did not pass the registered rule.** Gate passes 155/155 checks; Motion passes 147/155. The independent raw-data audit passed. All 12 fresh models completed 120 epochs and 7,680 steps before the fresh seed 9137 test was opened. No retention policy was trained as part of this formal Phase1 study.

| Registered checks | Gate | Motion |
|---|---:|---:|
| Original prediction (includes evidence probes) | 51/51 | 51/51 |
| Fixed-reference prediction | 32/32 | 32/32 |
| Deletion reliability | 72/72 | 64/72 |
| Total | 155/155 | 147/155 |

The 302/310 total counts correlated comparisons, not independent trials or accuracy. All eight failures concern long-gap Motion against reject-newest: feature error improves by 9.4–11.1%, but decoded-position gain has a 95% interval containing zero. This supports a task-alignment limitation; it does not identify a unique causal mechanism. Under the unchanged all-checks rule, retention-policy training cannot begin within that registered Phase1 protocol. The separate development diagnostics below do not change this conclusion or claim that its gate has passed.

## Verified model handoff

[Download the development and independent-confirmation model assets](https://github.com/JasonLIN0226/CV_project/releases/tag/phase1-asset-handoff-20261020-v2) · [Asset mapping, exact versions and verification instructions](handoff/README.md) · [Publication and loading audit](handoff/publication_audit.json).

The two official model families are separate: the original development family uses data seed 6301; the fresh independent-confirmation family uses data seed 9137. The Release provides model checkpoints, transforms, maps, metadata and frozen source. Large weights live in Release assets, not Git history. Official I-JEPA encoder weights and the large RGB/feature caches are excluded from the handoff; see its asset inventory and loading instructions for the exact scope. Historical scratch paths identify provenance and are not required local paths.

## Literature-baseline diagnostics — 22 October 2026

The current development track compares **Worth Remembering-inspired (WR)** and **PaFu-inspired** retention mechanisms with existing policies and offline oracle controls. These are explicitly documented mechanism adaptations, not reproductions of the original papers. They use the **previously frozen, independently trained Gate development predictor** from the 20 October follow-up (no whitening, unit-column projection, epoch 35). They do not load or continue the formal 12-model independent-confirmation family, and their results cannot be attributed to that family.

The CV comparison fixes the predictor, recent buffer R=4 and retained capacity K=2; all retained items enter the same reader. WR uses a causal standardized-L1/Gaussian/MAD event mechanism adapted to the recent buffer. PaFu trains a causal retention student from offline temporal reader-attention labels, including future queries; future observations are unavailable to the deployed student. This trains a literature-baseline student, **not the proposal's own deletion-utility scorer**. Equal K does not imply equal auxiliary bytes or compute. Privileged-evidence, local replacement and all-history hindsight controls have different information access and are reported separately.

- [Development results, oracle comparisons and bad-case evidence](docs/LITERATURE_BASELINE_DEVELOPMENT_REPORT.md)
- [Independent code and artifact audit](docs/LITERATURE_BASELINE_AUDIT.md)
- [Development protocol, per-episode results and provenance](results/literature_baselines/dev-20261022/README.md)
- [Fresh development evaluation and execution evidence](docs/LITERATURE_BASELINE_FRESH_REPORT.md)

The first development report reuses four previously inspected validation pairs; it is not independent generalization evidence. The fresh diagnostic protocol uses separate seed 20261022 scenes, the old normalization and already frozen predictor/student checkpoints. Its ID, longer-gap and distractor-shift conditions are new **development** data, not the official seed 9137 test and not three established failure mechanisms. The fresh report is the authority for its execution status and results.

## Historical Phase 2 development — 20 October 2026

This separate implementation track established the proposal's bounded-memory interface, generated independent controlled data, diagnosed validation degradation, and ran equal-capacity deletion forks and retention controls. The follow-up recovered useful history dependence for the independent Gate development predictor; Motion did not show a net history benefit. These experiments predate the formal model handoff and do not reproduce either official model family.

- [Predictor diagnosis, history controls and reproduction](docs/PHASE2_HISTORY_DIAGNOSIS.md)
- [Gate baseline and deletion follow-up](docs/PHASE2_GATE_FOLLOWUP_CONTINUATION.md)
- [A100 encoding and interface execution evidence](docs/PHASE2_HPC_RUN.md)
- [Predictor training contract and initial pilot](docs/PHASE2_PREDICTOR_PILOT.md)
- [Equal-capacity deletion pilot](docs/PHASE2_DELETION_PILOT.md)
- [Six-policy retention pilot](docs/PHASE2_RETENTION_BASELINE_PILOT.md)
- [PushT data preparation and remaining planning work](docs/PHASE2_PUSHT_PLANNING_PREPARATION.md)
- [Portable diagnosis evidence](results/phase2_history/20261020)

PushT preparation is not completed model-driven baseline planning or a demonstration that planning requires history. Historical reports describe the assets and checks available when they were written; use the current [model handoff](handoff/README.md) for asset availability. The proposal's deletion-utility scorer and downstream planning results must be evaluated separately from these interface and diagnostic results.

## Formal Phase1 reports and data

- [Current detailed experiment walkthrough — Word](reports/phase1_experimental_walkthrough_20261020/Phase1_Experimental_Walkthrough_and_Evaluation_Report.docx)
- [Current detailed experiment walkthrough — PDF](reports/phase1_experimental_walkthrough_20261020/Phase1_Experimental_Walkthrough_and_Evaluation_Report.pdf)
- [Independent-confirmation technical report — Word](reports/phase1_independent_confirmation_20261020/Phase1_Independent_Confirmation_Report.docx)
- [Verified numerical results](results/phase1_independent_confirmation/local_verified_results_20261020.json)
- [Current result and limitations](docs/PHASE1_LATEST.md)
- [Reproduction and external assets](docs/REPRODUCIBILITY.md)
- [Cleanup record](docs/CLEANUP_MANIFEST.json)

The formal Phase1 experiment is a controlled synthetic **I-JEPA** study, with a supervised training-only position readout and observed-RGB guard. It is not V-JEPA or real-video validation. Both tasks use 512 training pairs, 96 validation pairs, and three test conditions with 256 pairs each. Reference and candidate models use seeds 17, 29 and 43. All train-only transforms, two maps and models were locked before test generation. The separate development track above has its own data and model provenance.

## Verify and reproduce

```bash
python -m pip install -r requirements.txt
python tools/verify_independent_confirmation.py
python tools/verify_results.py
python tools/verify_phase1_latest.py
python tools/verify_phase2_history.py
python -m pytest -q
```

The first verifier checks the published Phase1 snapshot hashes and final counts without HPC access. The other verifiers check retained historical evidence. Local verification is not a repeat of HPC model inference or its raw-data audit. Literature-baseline protocols and execution receipts are linked in their reports above.

```bash
python -m pip install -r requirements-report.txt
python tools/build_phase1_experimental_walkthrough.py
```

## Repository layout

- `experiments/independent_confirmation/`: fresh Phase1 confirmation methods and statistics.
- `results/phase1_independent_confirmation/execution_v10_checkpoint_resume/source/`: final frozen Phase1 execution source; its dependencies and earlier snapshots are retained for provenance.
- `handoff/`: official model-family mapping, portable loading interface and asset verification.
- `experiments/phase2/`: independent development interfaces, predictor diagnostics and literature-baseline implementations.
- `results/literature_baselines/`: baseline protocols, compact results, figures and provenance.
- Other `experiments/` and `results/` folders: development methods, controls, negative results and audit evidence; read each dated report for its scope.
- `tests/`: local software checks; `tools/`: analysis, verification and report generation.
- `reports/`: the formal Phase1 detailed walkthrough, independent-confirmation technical report, source figures and machine-readable manifests.

Official model assets are downloadable from the handoff Release; encoded arrays and large raw HPC outputs remain external under recorded asset paths. The formal Phase1 closeout recorded 53,544 of the approved 69,299 GPU seconds and ended its HPC jobs. That ledger does not describe the later literature-baseline diagnostics; their execution evidence is recorded separately. Historical connection aliases are not live endpoints.

The formal Phase1 publication removed superseded reports, outdated prose, duplicate renders, deployment ZIPs and caches while preserving failed scientific outcomes and immutable locks. It did not change thresholds or launch retention experiments. The separately documented development work above does not revise that publication's registered result.
