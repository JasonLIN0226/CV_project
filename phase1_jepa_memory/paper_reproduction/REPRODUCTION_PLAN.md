# SPACE reproduction and controlled transfer

## Objective

Reproduce the closest published experiment before attributing weak effects to
our memory idea. The target is SPACE, arXiv:2609.32592, author repository
`GilliaN-HJ/SPACE`, commit `5a073bcc303f006834bfb6ae0010c11747ded615`.
This is an attempt to reproduce its Assembly101 experiment; it is not yet a
successful reproduction or evidence that our original acceptance checks pass.

## Execution order

1. Acquire the paper's official C10095 TSM release on course Cloud storage.
   Check the fixed upstream SHA, container paths, and train/validation
   participant split. Do not deserialize the nine test participants.
2. Restore the missing dataset integration with explicit assumptions, retaining
   the released predictor mathematics and paper settings: 512 hidden units,
   four layers/eight heads, 30 epochs, cosine prediction loss, BF16, recent eight
   observations, memory sixteen, and horizons 1/4/16/64. Train seeds 0/1/2.
3. Fit the released slow basis and counterfactual utility bank on training
   streams only. Run the original SPACE controller and matched Reservoir,
   FIFO, utility-only and state-only baselines.
4. Reconstruct the missing benchmark evaluation transparently: future-action
   NLL, mAP, LRAP, Recall@5, within-regime drift, boundary response and paired
   corruption recovery. Official TSM filename indices must match the official
   annotation coordinate system before semantic scores are computed.
5. Compare achieved numbers with the paper, report deviations, and then freeze
   a separate transfer experiment connecting the verified components to our
   I-JEPA task. Preserve old results and original acceptance rules.

## Known reproducibility gaps

The released tree still imports a missing `memory_jepa.data.latent_dataset`.
It provides artifact training and a controller but no complete benchmark
evaluation/decoder training entry point. A reconstructed integration is not
identical author code. Sampling, decoder fitting, and any unspecified details
must be documented; no undocumented defaults or claims of exact reproduction.

The paper's reported mean Action NLL is 3.6315 for Reservoir and 3.6291 for
SPACE. These published effects are also small; reproduction cannot guarantee
a large gain or passing our earlier checks. Scores from different feature
representations or objectives must not be compared as if on the same scale.

## Resource and evidence boundaries

Prepare and check software before GPU allocation. Data preparation uses course
Cloud CPU only. Model training uses one A100, four science threads, 60 GB, and
a four-hour allocation; project all 90 epochs from the first complete epoch
and stop if setup plus projection exceeds 13,800 seconds. No shortened seeds,
epochs, dataset, precision, or silent budget expansion. Separate downstream
stages require their own fixed protocol and evidence before execution.

All model/data payloads remain on HPC. Only named compact metadata, scalar
summaries, curves, and logs return with before/local/after hashes. No external
messages, new credentials, sealed confirmation access, or CPU model fallback.

## Primary sources

- https://arxiv.org/html/2609.32592v1
- https://github.com/GilliaN-HJ/SPACE
- https://huggingface.co/datasets/cvml-nus/assembly101
