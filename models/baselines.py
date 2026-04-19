import numpy as np

from config import MEAN_UPLIFT_VALUE


class RandomPolicy:
    def fit(self, X, t, y):
        pass

    def predict_uplift(self, X):
        return np.zeros(len(X))

class MeanUplift:
    def predict(self, X):
        return np.ones(len(X)) * MEAN_UPLIFT_VALUE