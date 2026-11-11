"""The project's own non-learned deletion rules on the terminal (Synergy) tasks.

Each rule reads one stage-A array (see methods/deletion_cost/terminal.py) and deletes one of the
three cards. The definitions follow results/.../qualified-reader-retention-20261025/code/evaluate_dev.py.
"""
import numpy as np


def rules(arrays):
    """name -> index of the deleted card for every record."""
    n = len(arrays["ids"])
    return {
        "fifo": np.zeros(n, int),                                    # delete the oldest card
        "attention": arrays["attention"].argmin(1),                  # delete the card the reader attends to least
        "full_set_sensitivity": arrays["sensitivity"].argmin(1),     # delete the card whose removal moves the output least
        "medoid": arrays["medoid_sum"].argmin(1),                    # keep the pair whose prediction is the medoid
    }
