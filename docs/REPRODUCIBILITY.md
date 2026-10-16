# Reproduction

Use Python 3.10–3.12 from the repository root.

## Verify existing results

```bash
python -m pip install -r requirements.txt
python tools/verify_results.py
python -m pytest -q experiments/deletion/test_assay.py
```

Expected: 11,520 unique finite pair-level rows, 180 recomputed statistics and 9 deletion-assay software tests. The verifier checks `results/manifest.json` and all three retained scientific source locks. Original outcomes remain unchanged.

The native raw audit covered 55,296 deletion rows. Those raw rows remain HPC-only; local recomputation starts from transferred pair aggregates and does not repeat the underlying forward pass.

## Check the predictor

```bash
python -m pip install -e '.[model]'
python -m pytest -q
```

These tests use small synthetic tensors on CPU and do not train a scientific model or access research data. The installed `phase1` package comes only from `experiments/predictor/control/src`, the retained complete predictor version. The other reference source files are evidence, not another installed package.

## Rebuild the report

```bash
python -m pip install -r requirements-report.txt
# Install Poppler so pdftoppm is available on PATH.
python tools/build_report.py --output build/report
```

The builder reads only the retained scalar evidence. It recreates the report's eleven figures and editable Word document without overwriting the delivered report. Export Word to PDF using Word, LibreOffice or Google Docs and inspect the pages if making a new PDF. The checked PDF in `reports/` remains the original Google Docs export.

## Full scientific replay

`experiments/predictor/control/run_repair.py` provides the original train/evaluate/verify stages. `experiments/deletion/run_assay.py` is the original inference-only assay. Their protocols and source hashes are unchanged. They require the original course allocation and asset layout; they are not portable one-command launchers.

External assets required:

- Official I-JEPA weights and processor at the protocol's pinned revision.
- Encoded 6301 features, train-only normalization and basis.
- Frozen recent-reference and six selected predictor checkpoints, plus state readouts.
- Native raw metrics/arrays for original raw-level auditing.

Hashes and original paths remain in the protocols, checkpoint proofs and `results/deletion/asset_ledger.json`. An asset identifier does not mean the binary is included. Required older reference probe evidence and original source files are retained because the latest results depend on them. Unrelated older experiments are available only in Git history.

Fresh 9137 data and the 12-model independent confirmation were not run. Verification does not generate them, request HPC resources, or start training.

## Attribution

I-JEPA uses `facebook/ijepa_vith14_1k`, revision `f157467ea509bc356ff9f61fd3c0d840eec5e04e`. The predictor adapts last-observation centering from NLinear and learned query pooling from Set Transformer; the original Set Transformer license is retained at `experiments/predictor/control/SET_TRANSFORMER_LICENSE.txt`. External ownership and checkpoint licenses still apply.
