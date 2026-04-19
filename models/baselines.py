import numpy as np

from config import MEAN_UPLIFT_VALUE, RANDOM_POLICY_SCORE_STD, RANDOM_SEED


class RandomPolicy:
    """
    Random ranking baseline: i.i.d. Gaussian scores (no ties almost surely).
    Default σ is RANDOM_POLICY_SCORE_STD = SD(τ) under the Beta-intent DGP in config.
    """

    def __init__(self, seed=RANDOM_SEED, score_std=RANDOM_POLICY_SCORE_STD):
        self._rng = np.random.default_rng(seed)
        self._score_std = float(score_std)

    def fit(self, X, t, y):
        pass

    def predict_uplift(self, X):
        return self._rng.normal(0.0, self._score_std, size=len(X))

class MeanUplift:
    def predict(self, X):
        return np.ones(len(X)) * MEAN_UPLIFT_VALUE