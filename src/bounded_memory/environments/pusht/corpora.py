"""The hidden-CoG PushT corpora: generator, folder name and generator arguments per version.

Shared by the PushT entry scripts (scripts/prepare_data, encode, train_predictor, train_scorer).
Corpus folders live under one root (the scripts' --data-root / --output-root); I-JEPA features of a
corpus are in "<corpus>-ijepa", its predictor checkpoints in "<corpus>-ijepa/predictors".
"""
VERSIONS = {
    # the Gate-like testbeds: the centre of gravity drawn as a dot for 3 early frames
    "v5": {"generator": "prepare_pusht_cog_marker_data.py", "corpus": "pusht-cog-v5-marker-20261109", "args": []},
    "v5b": {"generator": "prepare_pusht_cog_marker_v5c_data.py", "corpus": "pusht-cog-v5b-marker-r4-t960-20261120",
            "args": ["--marker-radius", "4", "--train-families", "960"]},
    "v5c": {"generator": "prepare_pusht_cog_marker_v5c_data.py", "corpus": "pusht-cog-v5c-marker-r4-d2-t960-20261123",
            "args": ["--marker-radius", "4", "--train-families", "960", "--distractor-dots", "2"]},
    # the harder follow-up: the block's response to a shared probe push (needs the probe library: pass --library)
    "v4": {"generator": "prepare_pusht_cog_v4_data.py", "corpus": "pusht-cog-v4-20261107", "args": []},
}
