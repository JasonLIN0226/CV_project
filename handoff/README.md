# Phase 1 verified model handoff

Download: https://github.com/JasonLIN0226/CV_project/releases/tag/phase1-asset-handoff-20261020-v2

Verified on the original HPC asset store on 20 October 2026. All 313 exported files match their applicable frozen source/model locks; all six archives and the original manifest passed native-before/local/native-after SHA256 checks. Every archive member was also checked against its file-level SHA256 and size. No training, fitting, model forward or encoding was performed for this export. After export, an independent fresh download audit strictly loaded all 24 best checkpoints and four production maps, ran finite synthetic-input forward checks, and verified exact empty-history reference identity. See `inference_smoke.json`. This is a software compatibility check, not a new scientific experiment.

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

Release assets contain `manifest.json` (identical to `handoff/asset_manifest.json` in this repo), `SHA256SUMS`, six model/metadata archives, `phase1-source.tar.gz`, `source_archive_manifest.json`, `verify_handoff.py`, `transfer_integrity.json`, `training_identity.json`, and this README. The source archive includes the corrected portable tests and inference adapter. Its exact code commit, size and SHA are in the release source_archive_manifest.json. Scientific source locks and native model archives are unchanged. The v2 release also includes development-supplemental_provenance.tar.gz and supplemental_manifest.json: the exact retained verification.json requested by the teammate, matching her pre-existing SHA. This supplemental copy is transparently marked as retained published provenance, not a new HPC read.

```bash
# Download release assets into a new directory.
gh release download phase1-asset-handoff-20261020-v2 --repo JasonLIN0226/CV_project --dir assets
cd assets
shasum -a 256 -c SHA256SUMS
python3 verify_handoff.py --manifest manifest.json --root . --supplement supplemental_manifest.json
mkdir intake
for archive in development-*.tar.gz independent_confirmation-*.tar.gz; do
  tar -xzf "$archive" -C intake
done
python3 verify_handoff.py --manifest manifest.json --root intake --extracted --supplement supplemental_manifest.json
python3 teammate_verify_assets.py --manifest teammate_required_assets.json --root intake/development --include-optional
```

To use the teammate's older Phase2 required-assets checker, point its `--root` to `assets/intake/development`. Files preserve the tree beneath `/scratch/xl4358/` within the relevant variant. That checker covers the old development assets only; the supplied verifier covers both twelve-model families. The teammate checker now matches all 21/21 required assets, including reference verification. SHA/size integrity checks are separate from the included inference smoke check.

The additional training_identity.json reconciles each family and task's 1,024 locked basis training IDs with its actual dataset manifest (512 pairs), without loading a predictor or refitting a map. Map preparation/crossfit provenance is also retained in the source archive under results/motion_observation_map and results/motion_scene_calibration; fresh map preparation is under results/phase1_independent_confirmation/execution_v9_cpu_incarnation/native_first_training/fresh/observation_maps.

## Source and loading entry points

The source archive contains the predictor architecture and its dependencies under `experiments/predictor/control/src/phase1/`, final Gate inference under `experiments/gate_locked_test/intervention.py`, final Motion composition in `experiments/motion_observation_mix_locked_test/evaluate.py`, and development map fit/load functions in `experiments/motion_observation_map/{gpu_fit,kernel}.py`. For fresh assets use the frozen v10 `locked_inference.py`, `train_maps.py`, `core.py`, `geometry.py`, `guard.py`, and `kernel.py`. `MotionInference` and `gate_transport_off` are the fresh inference entry points. Existing kernels load NPZ heads with `allow_pickle=False`; exported heads are already fitted and require no refit.

Checkpoint loading must use the matching family, task, seed, reference, basis, normalization and readout. Historical Slurm/resource guards expect the original HPC layout; this package preserves them as provenance and does not claim a new portable planning implementation. Large caches can be reconstructed from the supplied source and identity manifests, but rebuilt files need their own provenance and must not be assumed bitwise identical.

No PushT/planning experiment was implemented by this Phase1 project. The teammate's Phase2 code is independent. No retention policy was trained here. This controlled synthetic I-JEPA prototype includes a training-supervised position readout and observed-RGB guard; it is not V-JEPA or real-video validation. Independent confirmation's eight Motion long-gap deletion failures remain unchanged.

## Portable inference: no HPC account required

Use a clean checkout of the v2 source (or extract `phase1-source.tar.gz`), install the
package and CPU model dependencies, then load the extracted archives. No encoder,
GPU, refit, training, or old absolute HPC directory is required for this check.
Keep the source checkout after the editable installation. The package reads its
frozen inference code beside that checkout; it does not need the checkout to be
your working directory. This is an editable-source installation, not a standalone
wheel distribution.

```bash
python -m pip install -r requirements.txt Pillow PyYAML
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps
python handoff/inference.py --intake /absolute/path/to/assets/intake
```

```python
from handoff.inference import Predictor, pack
model = Predictor('/absolute/path/to/assets/intake', 'development', 'gate', 17)
# encoded_features: original 5120-dimensional I-JEPA output for observed frames.
z = model.normalize(encoded_features)  # skip if already normalized with this family
sample = pack(z, known_actions, query=24, retained=(2, 10))
prediction = model.predict(sample)  # normalized future features, h=3,6
```

For Motion K=2, also pass `query`, `retained_rgb` (two actual retained RGB frames),
`current_rgb`, and `own_actions` to `predict`. Guard fallback and the fixed 0.5 map
mix follow the locked inference exactly. Other history counts retain the original
candidate output. Gate disables the entire transport output for every history
count. The API consumes no hidden states, targets or future images. Input features
must come from the original I-JEPA feature representation; this is not a PushT
encoder adapter. Retention policies and planning remain the teammate's Phase2 work.

Both the source checkout and extracted `intake` can live in your own directories,
including paths with spaces. After the editable install, `from handoff.inference
import Predictor` also works in a notebook outside the checkout. Pass your own
intake path (or `~/...`) to `Predictor`; no NYU/HPC account, SSH session, VPN or
credentials are read during loading. `original_path`, `resolved_original_path`,
historical config output paths and old Slurm scripts are provenance only. Do not
run those historical HPC scripts as portable entry points, and do not create
`/scratch/xl4358` symlinks. SHA/size checks use only `relative_path` beneath intake.

## Publication audit and limitations

The first release exposed two packaging issues: one test relied on a neighboring
local project directory, and the old Phase2 inventory lacked one verification
JSON in its expected extracted path. Both are corrected in v2. Original six native
archives keep their exact SHA256 values. No model, map, sample, metric or threshold
was changed. `inference_smoke.json` records real checkpoint loading, not interface
mocks. `teammate_intake_audit.json` records the teammate's independent inventory
checker; its older unverified-scope wording is retained rather than rewritten.
The raw HPC scientific audit and current report remain the authority on results:
strict Phase1 is NOT met because eight fresh long-gap Motion checks fail.

Offline relocation evidence is in `portability_audit.json`: both model families were loaded from a new path with spaces, with an empty HOME and no HPC credentials; original paths, network connections and subprocess launches were blocked. All 24 best checkpoints and four production maps passed. Replacing original-path provenance fields with unusable paths left predictions identical. The editable install now includes the `handoff` module, and CI imports it from outside the checkout.
