"""The pinned I-JEPA encoder: repository, revision, pooling, and the checksum of its metadata.

Extracted unchanged from experiments/phase2/encode_fresh_gate_diagnostic.py. The encoder class itself is
predictors/phase1/encoders.py::FrozenIJepa.
"""
import hashlib
import json

ENCODER_CONFIG = {"backend": "ijepa", "repository": "facebook/ijepa_vith14_1k",
                  "revision": "f157467ea509bc356ff9f61fd3c0d840eec5e04e",
                  "pooling": "grid2", "dtype": "float32", "batch_size": 2,
                  "smoke_size": 12, "local_files_only": True, "cache_dir": None}
EXPECTED_ENCODER_METADATA_SHA256 = "caaf35f0bbb97992cc49450dafdc6cfbe0e84395650bd61a48f3718f175e2fa4"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
