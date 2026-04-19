from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import numpy as np

from config import GRADIENT_BOOSTING_PARAMS, PROPENSITY_CLIP_HIGH, PROPENSITY_CLIP_LOW

gb_params = GRADIENT_BOOSTING_PARAMS


class TLearner:
    def __init__(self):
        self.treated = GradientBoostingClassifier(**gb_params)
        self.control = GradientBoostingClassifier(**gb_params)

    def fit(self, X, t, y):
        self.treated.fit(X[t == 1], y[t == 1])
        self.control.fit(X[t == 0], y[t == 0])

    def predict_uplift(self, X):
        return (
            self.treated.predict_proba(X)[:, 1]
            - self.control.predict_proba(X)[:, 1]
        )


class XLearner:
    def __init__(self):
        self.m0 = GradientBoostingRegressor(**gb_params)
        self.m1 = GradientBoostingRegressor(**gb_params)

    def fit(self, X, t, y):
        X0, y0 = X[t == 0], y[t == 0]
        X1, y1 = X[t == 1], y[t == 1]

        self.m0.fit(X0, y0)
        self.m1.fit(X1, y1)

        d1 = y1 - self.m0.predict(X1)
        d0 = self.m1.predict(X0) - y0

        self.m2 = GradientBoostingRegressor(**gb_params)
        self.m3 = GradientBoostingRegressor(**gb_params)

        self.m2.fit(X1, d1)
        self.m3.fit(X0, d0)

    def predict_uplift(self, X):
        return 0.5 * (self.m2.predict(X) + self.m3.predict(X))


class DRLearner:
    """
    DR-learner style CATE: pseudo-outcome
      Γ = μ₁(X) − μ₀(X) + T(Y − μ₁)/e(X) − (1−T)(Y − μ₀)/(1−e(X))
    with separate μ₁, μ₀ (binary Y → classifiers) and final regression on Γ.
    """

    def __init__(self):
        self.propensity = GradientBoostingClassifier(**gb_params)
        self.mu1 = GradientBoostingClassifier(**gb_params)
        self.mu0 = GradientBoostingClassifier(**gb_params)
        self.final_model = GradientBoostingRegressor(**gb_params)

    def fit(self, X, t, y):
        t = np.asarray(t)
        y = np.asarray(y, dtype=float)

        self.propensity.fit(X, t)
        e = np.clip(
            self.propensity.predict_proba(X)[:, 1],
            PROPENSITY_CLIP_LOW,
            PROPENSITY_CLIP_HIGH,
        )

        self.mu1.fit(X[t == 1], y[t == 1])
        self.mu0.fit(X[t == 0], y[t == 0])

        m1 = self.mu1.predict_proba(X)[:, 1]
        m0 = self.mu0.predict_proba(X)[:, 1]

        dr = m1 - m0 + t * (y - m1) / e - (1.0 - t) * (y - m0) / (1.0 - e)
        self.final_model.fit(X, dr)

    def predict_uplift(self, X):
        return self.final_model.predict(X)
