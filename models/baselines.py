import numpy as np

class RandomPolicy:
    def fit(self, X, t, y):
        pass

    def predict_uplift(self, X):
        return np.zeros(len(X))

class MeanUplift:
    def predict(self, X):
        return np.ones(len(X)) * 0.03