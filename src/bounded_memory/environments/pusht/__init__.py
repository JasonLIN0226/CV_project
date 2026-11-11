"""Hidden-centre-of-gravity PushT: environment variants, corpus collection and diagnostics.

No predictor and no retention result exist for PushT yet. The modules keep their original file
names because they import each other by name and record their own SHA256 in corpus provenance;
run them through scripts/prepare_data/pusht.py, which puts this folder on the import path. All
are byte-identical to experiments/phase2/ at commit fe69074 except probe_pusht_cog_features.py, where one constant
now points at the predictor source vendored in this package. They need the PushT simulator
packages (gym-pusht, pymunk, pygame).

    pusht.py                              visual adapter, action normalisation, occlusion
    pusht_cog_cue.py (v1), pusht_cog_coherent.py (v2), pusht_cog_matched.py (v3)   earlier variants, reused by v4 and v5
    pusht_cog_natural.py (v4)             natural: the cue is the block's response to a shared probe push
    pusht_cog_marker.py (v5)              marker: the centre of gravity is drawn as a dot for 3 early frames
    prepare_pusht_cog_{marker,v4,v2,v1}_data.py, prepare_pusht_planning_data.py     corpus collection
    probe_pusht_cog_features.py, diagnose_pusht_cog_pose_decoding.py, dev_pusht_cog_*.py   I-JEPA readability probes and diagnostics
    audit_pusht_training_data.py          corpus audit
"""
