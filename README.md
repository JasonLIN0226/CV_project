# Bounded Visual Memory

Can the prediction error caused by deleting an observation help decide what a limited visual memory should retain?

This repository contains the **latest complete I-JEPA predictor and equal-capacity deletion assay**. It is an I-JEPA prototype, not a V-JEPA experiment. No learned retention policy or fresh independent confirmation has been run.

## Results

| Check | Gate | Motion |
|---|---:|---:|
| Evidence readable from features | 3/3 | 3/3 |
| History improves prediction | 23/24 | 8/24 |
| Prediction responds to historical content | 24/24 | 24/24 |
| Fixed-reference checks | 32/32 | 8/32 |
| Critical deletion contrasts | 6/18 | 36/36 |
| Cross-horizon deletion checks | 18/18 | 18/18 |

Counts describe registered comparisons, not independent datasets. The original strict Phase One criteria and joint policy-stage readiness rule remain **unmet**. Motion has a consistent relative deletion signal in the tested pools; Gate deletion stability and Motion long-occlusion prediction remain limitations.

[English report](reports/Phase1_Report.pdf) · [Editable Word report](reports/Phase1_Report.docx) · [Design and test definitions](docs/EXPERIMENTS.md) · [Results explained](docs/RESULTS.md)

![Paired Gate and Motion tasks](reports/figures/01_tasks.png)

This is an explanatory drawing, not a model prediction.

## Next step

[Executed signal diagnosis and next experiment](docs/NEXT_STEP.md): the existing pools always place nuisance content newest. The added zero-GPU analysis measures pair-level stability and remaining headroom; counterbalanced pool code is prepared and tested, with no policy training or new predictor inference.

## Run the checks

```bash
python -m pip install -r requirements.txt
python tools/verify_results.py
python -m pytest -q experiments/deletion/test_assay.py
```

The verifier checks retained file hashes and independently recomputes 180 statistics from 11,520 pair-level rows. These commands do not download models, train, or use a GPU.

Predictor software checks additionally require PyTorch, Pillow and PyYAML:

```bash
python -m pip install -e '.[model]'
python -m pytest -q
```

[Reproduction and external assets](docs/REPRODUCIBILITY.md) explains report rebuilding and the limits of full model replay.

## Layout

- `experiments/predictor/`: one frozen predictor version, training/evaluation code and independent audit.
- `experiments/deletion/`: one inference-only deletion assay and its tests.
- `experiments/reference/`: three original source files required by the frozen-reference verification.
- `results/`: latest numerical evidence, six-model training records and required reference probes/configurations.
- `reports/`: one report in Word/PDF, its figures and plotted values.
- `tools/`: result verification and report rebuilding.

Unused historical experiments, duplicate implementations, old reports and monitoring/deployment records have been removed from the current tree. Earlier commits retain the archive. The frozen predictor's complete source-lock set is kept so its provenance and imports remain verifiable; legacy helpers inside that set are not additional active experiments.

Model weights, encoded arrays and HPC-only raw rows are external. No old Slurm allocation or SSH endpoint is a live connection. Proposal GPU use remains 12,723 seconds; cleanup and scalar checks add no GPU usage.
