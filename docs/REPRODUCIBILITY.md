# Reproduce and inspect

## 1  Verify the archived numerical evidence

Use Python 3.10–3.12. From the repository root:

```bash
python -m pip install -r requirements-results.txt
python tools/verify_results.py
```

Expected: `PUBLISHED_PAIR_LEVEL_VERIFICATION_PASS`, 11,520 pair-level rows and 180 independently recomputed scalar statistics. The script verifies the copied-file SHA-256 manifest and original development coverage. It must keep `original_phase1_go` and `joint_policy_stage_ready` false.

This is verification of already-published evidence. It is not a new experiment and does not make a failed scientific gate pass. The full 55,296 fork/horizon rows remain HPC-only; their identity and original raw-level independent audit are retained. The local verifier starts from the transferred pair aggregates, so it does not re-prove the underlying model forward or raw-fork construction.

The original pure-software boundary checks can also be run without a model:

```bash
python -m pytest -q phase1_jepa_memory/runs/phase1_deletion_assay/test_assay.py
```

## 2  Rebuild the report

```bash
python -m pip install -r requirements-report.txt
# Install Poppler using the package manager for your operating system.
python tools/build_report.py
```

The wrapper preserves the original report builder and replaces only its workstation-specific `pdftoppm` executable path at execution time. It reads the published JSON/CSV evidence, regenerates figures and creates an editable Word report. It does not encode frames or train any predictor. Rebuilding may update generated artifact hashes; restore the archived files before running the byte-identity verifier if you want to verify the original publication snapshot.

The ready-to-read PDF is the checked Google Docs export; PDF page layout is not re-created merely by running the Word builder. To make a new PDF, export the rebuilt Word document with Word, LibreOffice, or Google Docs and inspect all pages.

## 3  Reproduce actual model computation

Install the experimental package and consult the frozen per-run protocol before any scientific execution:

```bash
python -m pip install -e ./phase1_jepa_memory
phase1 --help
```

The shared package is under `phase1_jepa_memory/src/phase1`. The actual complete second-repair source snapshot, its protocol and source locks are under `runs/phase1_predictor_repair_v2/execution/`. The equal-capacity assay source is under `runs/phase1_deletion_assay/`.

The published source retains original absolute workstation and `/scratch/...` paths as provenance. Deployment and Slurm scripts are historical records; they are not portable one-command launchers. Do not run old submit/deploy/login scripts expecting existing hosts or allocations. Adapt a new execution plan and verify its asset paths, data split, sources and resource budget first.

### External assets

- Official I-JEPA checkpoint and processor, pinned in the per-run protocol.
- Encoded 6301 features, train-only normalization and input basis.
- Six complete best predictor checkpoints and frozen state readouts.
- Native raw forward metrics, large traces and original scalar forks where not locally transferred.

The original `asset_ledger.json`, source locks, run contracts and checkpoint proofs record hashes and HPC origins. Availability of an identifier is not availability of the binary file. The repository does not bundle these assets or claim a clean-clone full forward replay without them.

Fresh 9137 data and the registered 12-model independent confirmation were not executed. Do not generate or open a confirmation set merely as part of checking this repository.

## Source preservation

Copied source/result bytes are unchanged, including failed outcomes and budget stops. The publication's portable helper tools and navigation documents are separate additions. A package-level ignore file is adjusted so Git retains the experiment records; that ignore file is not a frozen scientific source.
