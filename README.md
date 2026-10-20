# Bounded Visual Memory

Can the prediction error caused by deleting an observation identify what limited visual memory should retain?

## Final independent confirmation — 20 October 2026

**Phase1 did not pass the registered rule.** Gate passes 155/155 checks; Motion passes 147/155. The independent raw-data audit passed. All 12 fresh models completed 120 epochs and 7,680 steps before the fresh seed 9137 test was opened. No retention policy was trained.

| Registered checks | Gate | Motion |
|---|---:|---:|
| Original prediction (includes evidence probes) | 51/51 | 51/51 |
| Fixed-reference prediction | 32/32 | 32/32 |
| Deletion reliability | 72/72 | 64/72 |
| Total | 155/155 | 147/155 |

The 302/310 total counts correlated comparisons, not independent trials or accuracy. All eight failures concern long-gap Motion against reject-newest: feature error improves by 9.4–11.1%, but decoded-position gain has a 95% interval containing zero. This supports a task-alignment limitation; it does not identify a unique causal mechanism. Under the unchanged all-checks rule, retention-policy training cannot begin.

## Verified model handoff

[Download the development and independent-confirmation model assets](https://github.com/JasonLIN0226/CV_project/releases/tag/phase1-asset-handoff-20261020-v2) · [Asset mapping, exact versions and verification instructions](handoff/README.md). The two model families are separate; large weights live in Release assets, not Git history.

## Report and data

- [Complete illustrated English report — Word](reports/phase1_independent_confirmation_20261020/Phase1_Independent_Confirmation_Report.docx)
- [Complete report — PDF](reports/phase1_independent_confirmation_20261020/Phase1_Independent_Confirmation_Report.pdf)
- [Verified numerical results](results/phase1_independent_confirmation/local_verified_results_20261020.json)
- [Current result and limitations](docs/PHASE1_LATEST.md)
- [Reproduction and external assets](docs/REPRODUCIBILITY.md)
- [Cleanup record](docs/CLEANUP_MANIFEST.json)

This is a controlled synthetic **I-JEPA** study, with a supervised training-only position readout and observed-RGB guard. It is not V-JEPA or real-video validation. Both tasks use 512 training pairs, 96 validation pairs, and three test conditions with 256 pairs each. Reference and candidate models use seeds 17, 29 and 43. All train-only transforms, two maps and models were locked before test generation.

## Verify and reproduce

```bash
python -m pip install -r requirements.txt
python tools/verify_independent_confirmation.py
python tools/verify_results.py
python tools/verify_phase1_latest.py
python -m pytest -q
```

The first verifier checks the published snapshot hashes and final counts without HPC access. The other verifiers check earlier retained evidence. Local verification is not a repeat of HPC model inference or its raw-data audit.

```bash
python -m pip install -r requirements-report.txt
python tools/build_independent_confirmation_report.py
```

## Repository layout

- `experiments/independent_confirmation/`: fresh confirmation methods and statistics.
- `results/phase1_independent_confirmation/execution_v10_checkpoint_resume/source/`: final frozen execution source; its dependencies and earlier snapshots are retained for provenance.
- Other `experiments/` and `results/` folders: required development methods, controls, negative results and audit evidence. These are historical evidence, not additional current conclusions.
- `tests/`: local software checks; `tools/`: analysis, verification and report generation.
- `reports/`: one current report, its source figures and machine-readable manifest.

Weights, encoded arrays and large raw HPC outputs remain external under the recorded asset paths. They are downloadable from the model handoff Release; they are not stored in Git history. GPU settlement was 53,544 of the approved 69,299 seconds; all HPC jobs have ended. Historical connection aliases are not live endpoints.

Superseded reports, outdated prose, duplicate renders, deployment ZIPs and generated caches were removed. Failed scientific evidence and immutable source locks were preserved. No new training, threshold changes or retention-policy experiment is part of this publication.
