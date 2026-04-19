import numpy as np

from config import MEAN_UPLIFT_VALUE, RANDOM_SEED


class RandomPolicy:
    """Random ranking baseline: scores are a random permutation (no ties)."""

    def __init__(self, seed=RANDOM_SEED):
        self._rng = np.random.default_rng(seed)

    def fit(self, X, t, y):
        pass

    def predict_uplift(self, X):
        n = len(X)
        perm = self._rng.permutation(n).astype(np.float64)
        # Strictly monotone transform: same ranking as perm, mean scale ~O(1)
        return (perm + 0.5) / n

class MeanUplift:
    def predict(self, X):
        return np.ones(len(X)) * MEAN_UPLIFT_VALUE