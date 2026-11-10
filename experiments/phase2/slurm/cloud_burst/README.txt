Hidden-CoG PushT v4 -- I-JEPA encoding + readability probes (CV_project, Nov 7 2026)

Run (from the directory that holds the .tar.gz):
  tar xzf pusht_cog_v4_encode.tar.gz && cd pusht_cog_v4_encode && sbatch run_encode_probe.sbatch

Contents
  code/      probe_pusht_cog_features.py + the repo's phase1 encoder (FrozenIJepa), unchanged
  data/      pusht-cog-v4-20261107: train (128 families) + validation (32) observations, index,
             protocol, manifest, class map. The evaluation split is NOT included (reserved).
  hf_cache/  facebook/ijepa_vith14_1k @ f157467ea509bc356ff9f61fd3c0d840eec5e04e (loaded offline)
  requirements.txt, run_encode_probe.sbatch

The first run builds .venv (pip, needs internet) with torch 2.5.1+cu121 and the pins above.
Output: output/pusht-cog-v4-20261107-ijepa/
  encoding_manifest.json, probe/probe_results.json  (small; enough to read the result)
  features/*.npy  (~0.9 GB; needed later for predictor training)
Log: pusht_cog_v4_ijepa_<jobid>.out in this directory. Resubmitting skips a finished encoding.
