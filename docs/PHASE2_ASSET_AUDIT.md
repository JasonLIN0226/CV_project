# Phase 2 asset handoff audit

Audit date: 20 October 2026 (America/New_York). Repository: `https://github.com/JasonLIN0226/CV_project`, inspected commit `2c73c6bca11a9396cdb0546b8bf78b52445e7b22`. This was a local, read-only scientific-source audit; no checkpoint was loaded, no remote asset was accessed, and no GPU work was performed. This document and its accompanying asset manifest are new operational documentation.

## Decision

The supplied source is enough to begin Phase 2 implementation and to construct a new, explicitly identified experiment from its generator, encoder and predictor components. It is **not enough to resume the exact latest repaired Phase 1 predictor**. The checkout contains no `.pt` or `.npz` binaries, and the latest repair implementations are absent as well.

The user's proposal and Timeline Version 2 govern the new Phase 2 scope. The teammate's historical T8/fresh-confirmation requirement is not introduced as a new prerequisite for implementing the memory interface, six retention baselines, varied-subset/memory-dropout predictor training, deletion supervision or the missing PushT/planning interface checks. A newly rebuilt experiment must not be described as inheriting the exact old checkpoints, reproducing the latest repaired result, or completing independent confirmation.

The teammate's exhausted eight-hour authorization and 39,651 allocated GPU seconds describe their earlier execution. They are neither authorization to reuse that task nor the budget for this user's separately authorized Phase 2 run. New work needs its own recorded allocation limit and actual accounting. The recorded 56 automatic requeues are a concrete reason to inspect startup and requeue behavior before launching; do not blindly execute the historical deployment scripts.

## What is present

- `experiments/predictor/control/src/phase1`: deterministic Gate/Motion scene generation, recorded actions and RGB observations, split/data audits, frozen I-JEPA encoding, train-only normalization and basis fitting, reference dynamics predictor, and the 16 October `CenteredTransportPredictor`.
- `experiments/deletion`: the original equal-capacity deletion assay and its software tests.
- Three retained source locks and historical protocols, scalar results, per-pair aggregate deletion rows, checkpoint hashes, and the newer 19 October result summaries.
- The two original reference configurations and the original reference verification record, with exact file hashes matching the protocol.

No PushT, MPC or planning implementation was found in the inspected source. The newer summaries report Gate and Motion each satisfying 51 original checks, 32 fixed-reference checks and 72 task-specific deletion comparisons on repeatedly inspected development data. These are correlated comparison counts, not accuracy percentages or independent confirmations. No retention scorer was trained.

## Latest repair source is missing

`docs/PHASE1_LATEST.md` describes a Gate repair that disables the complete temporal extrapolation contribution, and a Motion repair combining same-history frozen-reference visual prediction, guarded candidate position, and a fixed equal-weight train-only observation-innovation map. Two small train-only maps were fitted. These implementations are not present in the retained `experiments` source.

An exhaustive file-hash lookup in the inspected checkout found none of the following files referenced by the latest summaries:

| Latest artifact | Expected SHA-256 | Checkout |
|---|---|---|
| Gate source lock | `ed742dd96059c12fa4d5f0d74a4f76d086c4432f3dc6c295ce1b18637cc201d1` | Absent |
| Gate protocol | `82f4cd7e32bd5abc099d25bd4af40dcd6350495f21e3161bbe36e83634681673` | Absent |
| Motion source lock | `1df92952ea33474859a57b9cbc165ffa873be47ee16fb9bbf1e8d10a862f4326` | Absent |
| Motion protocol | `ef2419c2834ec394bbd13df179b3c24049c6ff480f6599bcef20bd9f84984b3a` | Absent |

The latest Motion execution record explicitly names `/scratch/xl4358/phase1_motion_observation_mix_locked_test/review.sbatch`. Its directory is a verified place to request the missing Motion source/protocol/asset ledger. The Gate result is job 8332; the manifest supplies the original local artifact path `results/gate_locked_test/native_completed/output/job_8332/summary.json`, but does not establish a remote Gate source directory. The two map filenames and locations are not published in the retained evidence and must be supplied by the teammate; no path is guessed here.

## Minimal handoff for direct continuation

The teammate's netid in all recorded asset paths is `xl4358`. Existing filesystem permissions must permit access or the teammate must transfer/share the files; a path in a manifest is not proof of current accessibility.

1. **Reference stores and transforms:** obtain both `/scratch/xl4358/phase1_v3_dev/gate/reference/` and `/scratch/xl4358/phase1_v3_dev/motion/reference/`. For a compatible store, preserve the `resolved_config.json`, dataset manifest and observation/evaluation files, feature index and feature arrays, feature audit, normalization NPZ/JSON, and task basis NPZ/JSON. The executable store verifies these linked provenance records; copying only a basis and a few feature arrays is insufficient. Train and validation caches are needed for Phase 2 training; held-out caches are only needed when the corresponding evaluation is actually run.
2. **Six candidate checkpoints:** `/scratch/xl4358/phase1_predictor_repair_v2/execution/training/job_7574/predictor/{gate,motion}/seed_{17,29,43}/best.pt`. This brace notation is a compact description of the six separately verified paths in the JSON manifest. Preserve training completion/provenance records when attempting the original loader.
3. **Six reference checkpoints:** `predictor/<task>/seed_<seed>/best.pt` inside each task's reference directory. The old loader separately constructs, loads and verifies its frozen reference; do not assume a candidate checkpoint alone is sufficient. Exact reference `best.pt` hashes are published in `results/reference/verification.json` and copied into the JSON manifest.
4. **Latest repair source and maps:** request the two missing source trees/protocols/locks above, two fitted maps and their provenance, and their latest asset ledgers. The 16 October source and checkpoints alone cannot instantiate the 19 October repaired model.
5. **Optional diagnostic replay:** `probes/gate_state_decoder.npz` and `probes/motion_state_decoder.npz` inside their reference directories are required for the original task-state diagnostics. Native raw metric/vector arrays are needed to re-audit the original forward-pass results, but are not required to begin new Phase 2 implementation or train a separately identified predictor.

The machine-readable [required asset manifest](../experiments/phase2/required_phase1_assets.json) lists only paths supported by the retained ledger, protocol, configurations or executable loader and includes exact published hashes where available. It is an inventory, not a downloader or a claim that the files are present. It deliberately does not invent paths for the latest maps or Gate source.

## What can be regenerated

- **Official encoder:** download `facebook/ijepa_vith14_1k` at revision `f157467ea509bc356ff9f61fd3c0d840eec5e04e` with its pinned processor. The old cache is `/scratch/xl4358/phase1_hf_cache/`, but transfer of that cache is optional.
- **Synthetic observations:** the retained `data.generate` and `envs` source can regenerate configured Gate/Motion data and establish a new manifest. The source contains split/seed/pair checks. Record the configuration, source revision and generated hashes; do not claim byte-identical reproduction without comparing the actual artifacts.
- **Feature cache, normalization and basis:** `cache.encode`, `fit_normalization`, and `model.fit_basis` can recreate these under a new output directory. Normalization/basis must use the training split only. Changing implementation, environment or transforms creates new provenance and must not be silently inserted beneath an old checkpoint's signatures.
- **Predictor:** train a separately identified Phase 2 predictor using the supplied architecture or a documented proposal-compatible implementation. This costs encoding/training compute and does not reproduce unavailable exact weights. The original `run_repair.py` is not a portable launcher: it checks hard-coded `/scratch/xl4358` paths, original sources, proof hashes and old resource records.
- **Latest learned maps and exact latest repaired outputs:** cannot be reconstructed faithfully from scalar summaries alone; their source and fitting specification are required. These missing assets do not prevent independent proposal-aligned Phase 2 engineering.

## Verification performed

`python3 tools/verify_phase1_latest.py` passed using the standard library. It verified the published latest snapshot hashes, comparison counts, 57 allocation records and the 39,651-second accounting total; it preserved `Phase1_GO: false`. This validates snapshot consistency, not model inference or a new scientific result.

A separate standard-library hash check validated all retained lock entries: predictor control **57/57**, deletion assay **8/8**, independent audit **3/3**. The four newer repair lock/protocol hashes above were absent from the checkout. No model was instantiated.

Three included provenance files match the original protocol byte-for-byte:

| Included file | SHA-256 |
|---|---|
| `results/reference/gate/resolved_config.json` | `e2b737cb814f5e09be6c0ceea6e668e0e07c9f43496e951be1d899d0269ef392` |
| `results/reference/motion/resolved_config.json` | `3ce7db93aa98f4fab7586d1a85bbb456bef5a49c97c2e80b47525cbcfb154b90` |
| `results/reference/verification.json` | `a36baa1372b736e0cddebde99cbc6d49202a10b4bd9e35b86c20728c6bedaedf` |

Sources: `docs/PHASE1_LATEST.md`, `docs/REPRODUCIBILITY.md`, `results/phase1_latest/{manifest,gate_summary,motion_summary}.json`, `results/deletion/asset_ledger.json`, `results/reference/verification.json`, `experiments/predictor/control/protocol.json`, and the retained `phase1` loader/cache/model source.
