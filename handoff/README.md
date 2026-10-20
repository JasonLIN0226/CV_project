# Phase 1 verified model handoff

Download: https://github.com/JasonLIN0226/CV_project/releases/tag/phase1-asset-handoff-20261020

Verified on the original HPC asset store on 20 October 2026. All 313 exported files match their applicable frozen source/model locks; all six archives and the original manifest passed native-before/local/native-after SHA256 checks. Every archive member was also checked against its file-level SHA256 and size. No training, fitting, model forward or encoding was performed for this export. Checkpoints were verified as bytes against their locks, not deserialized during export.

## Development and confirmation are separate

| Family | Data seed | Best checkpoints | Fitted Motion maps | Scientific outcome |
|---|---:|---:|---:|---|
| development | 6301 | 6 candidate + 6 matching frozen reference | 2 | Final development checks supported |
| independent_confirmation | 9137 | 6 new candidate + 6 new reference | 2 newly fitted | Gate 155/155; Motion 147/155; strict Phase1 not met |

Each family is divided into models_gate, models_motion, and metadata_maps_source archives. Download all three for the chosen family. The 571,583,377 compressed bytes across six model/metadata archives include best and available latest checkpoints, completion/training records, normalization, basis, readouts, maps, configs, dataset/feature identity manifests, and exact frozen source. Official I-JEPA weights and large RGB/feature caches are excluded.

Development uses the original job7574 neural candidate weights. The final Gate repair disables the whole temporal-extrapolation contribution at inference; it does not replace those weights. The final Motion repair uses the matching frozen reference visual prediction, guarded position and fixed 0.5 observation-innovation maps. Gate has no fitted observation map.

The fresh confirmation models/maps/transforms are newly trained/fitted on seed9137 and must not be interchanged with development assets. All twelve fresh models completed120epochs/7680steps and were locked before test generation. The earlier seed9137 test has now been used and is not an untouched future test set.

## Exact original roots and latest locks

- Development Gate: `/scratch/xl4358/phase1_gate_locked_test/`; source lock `ed742dd96059c12fa4d5f0d74a4f76d086c4432f3dc6c295ce1b18637cc201d1`.
- Development Motion final composite: `/scratch/xl4358/phase1_motion_observation_mix_locked_test/`; source lock `1df92952ea33474859a57b9cbc165ffa873be47ee16fb9bbf1e8d10a862f4326`.
- Development maps: `/scratch/xl4358/phase1_motion_observation_map/gpu_output/head_3.npz` and `head_6.npz`; source lock `b342ee65859436c5941121efb4500b199fba1822390f9a242b21e5720bd24e7b`.
- Development candidate checkpoints: `/scratch/xl4358/phase1_predictor_repair_v2/execution/training/job_7574/predictor/{gate,motion}/seed_{17,29,43}/best.pt`.
- Development frozen references and transforms: `/scratch/xl4358/phase1_v3_dev/{gate,motion}/reference/`.
- Fresh final source: `/scratch/xl4358/phase1_independent_confirmation_execution_v10_checkpoint_resume/`; source lock `4133e4335f87f0b719376f45c71ac9adb52e05b3e2c45550bcb22863472727d9`.
- Fresh assets: `/scratch/xl4358/phase1_independent_confirmation_execution_v9_cpu_incarnation/fresh/`; models are `models/{reference,candidate}/{gate,motion}/seed_{17,29,43}/best.pt`; production maps are `observation_maps/fitted/head_{3,6}.npz`. The v9 output and v10 source paths reflect checkpoint recovery, not mixed model families.

The full manifest records original path, resolved original path, package-relative path, bytes, SHA256, applicable pre-existing locked SHA, variant, purpose and model role/task/seed where relevant. “Latest” means the final adopted development locks and the completed fresh model lock, not whichever diagnostic file has the newest modification date.

## Download and verify

Release assets contain `manifest.json` (identical to `handoff/asset_manifest.json` in this repo), `SHA256SUMS`, six model/metadata archives, `phase1-source-988fe9e.tar.gz`, `source_archive_manifest.json`, `verify_handoff.py`, `transfer_integrity.json`, `training_identity.json`, and this README. The source archive is the exact published scientific snapshot at commit `988fe9e784225bb7b584901c82357905f854f5c0`; subsequent handoff metadata does not change that scientific source.

```bash
# Download release assets into a new directory.
gh release download phase1-asset-handoff-20261020 --repo JasonLIN0226/CV_project --dir assets
cd assets
shasum -a 256 -c SHA256SUMS
python3 verify_handoff.py --manifest manifest.json --root .
mkdir intake
for archive in development-*.tar.gz independent_confirmation-*.tar.gz; do
  tar -xzf "$archive" -C intake
done
python3 verify_handoff.py --manifest manifest.json --root intake --extracted
```

To use the teammate's older Phase2 required-assets checker, point its `--root` to `assets/intake/development`. Files preserve the tree beneath `/scratch/xl4358/` within the relevant variant. That checker covers the old development assets only; the supplied verifier covers both twelve-model families. Checks are SHA/size integrity checks, not a repeat of scientific inference.

The additional training_identity.json reconciles each family and task's 1,024 locked basis training IDs with its actual dataset manifest (512 pairs), without loading a predictor or refitting a map. Map preparation/crossfit provenance is also retained in the source archive under results/motion_observation_map and results/motion_scene_calibration; fresh map preparation is under results/phase1_independent_confirmation/execution_v9_cpu_incarnation/native_first_training/fresh/observation_maps.

## Source and loading entry points

The source archive contains the predictor architecture and its dependencies under `experiments/predictor/control/src/phase1/`, final Gate inference under `experiments/gate_locked_test/intervention.py`, final Motion composition in `experiments/motion_observation_mix_locked_test/evaluate.py`, and development map fit/load functions in `experiments/motion_observation_map/{gpu_fit,kernel}.py`. For fresh assets use the frozen v10 `locked_inference.py`, `train_maps.py`, `core.py`, `geometry.py`, `guard.py`, and `kernel.py`. `MotionInference` and `gate_transport_off` are the fresh inference entry points. Existing kernels load NPZ heads with `allow_pickle=False`; exported heads are already fitted and require no refit.

Checkpoint loading must use the matching family, task, seed, reference, basis, normalization and readout. Historical Slurm/resource guards expect the original HPC layout; this package preserves them as provenance and does not claim a new portable planning implementation. Large caches can be reconstructed from the supplied source and identity manifests, but rebuilt files need their own provenance and must not be assumed bitwise identical.

No PushT/planning experiment was implemented by this Phase1 project. The teammate's Phase2 code is independent. No retention policy was trained here. This controlled synthetic I-JEPA prototype includes a training-supervised position readout and observed-RGB guard; it is not V-JEPA or real-video validation. Independent confirmation's eight Motion long-gap deletion failures remain unchanged.
