from __future__ import annotations

import numpy as np


def evaluate_policy(uplift: np.ndarray, true_effect: np.ndarray) -> float:
    # correlation sanity check
    return float(np.corrcoef(uplift, true_effect)[0, 1])
