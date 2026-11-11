"""PushT predictor training: not available. Exits non-zero.

No predictor has been trained on PushT, and no configuration for one exists in the project. The
v5 (marker) corpus still has to be encoded first; see docs/PUSHT_COG_CUE.md, section 9.
"""
import sys

if __name__ == "__main__":
    sys.exit("PushT has no predictor: none was ever trained and no training configuration exists "
             "(configs/reproduction.yaml, results.pusht). Encode the v5 corpus and define the predictor first.")
