from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from config import GRADIENT_BOOSTING_PARAMS, PROPENSITY_CLIP_HIGH, PROPENSITY_CLIP_LOW

gb_params = GRADIENT_BOOSTING_PARAMS


class TLearner:
    def __init__(self) -> None:
        self.treated = GradientBoostingClassifier(**gb_params)
        self.control = GradientBoostingClassifier(**gb_params)

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> None:
        self.treated.fit(X[t == 1], y[t == 1])
        self.control.fit(X[t == 0], y[t == 0])

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return (
            self.treated.predict_proba(X)[:, 1]
            - self.control.predict_proba(X)[:, 1]
        )


class XLearner:
    """
    Kunzel et al. X-learner: stage-two CATE surfaces are blended with propensity weights
      τ̂(x) = ê(x) τ̂₀(x) + (1 − ê(x)) τ̂₁(x)
    so the model trained on the larger / more-supported arm gets more mass when ê is skewed.
    """

    def __init__(self) -> None:
        self.m0 = GradientBoostingRegressor(**gb_params)
        self.m1 = GradientBoostingRegressor(**gb_params)
        self.propensity = GradientBoostingClassifier(**gb_params)

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> None:
        t = np.asarray(t)
        self.propensity.fit(X, t.astype(int))

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

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        e = np.clip(
            self.propensity.predict_proba(X)[:, 1],
            PROPENSITY_CLIP_LOW,
            PROPENSITY_CLIP_HIGH,
        )
        # m3 = τ̂ learned from control pseudo-effects; m2 = τ̂ from treated pseudo-effects.
        return e * self.m3.predict(X) + (1.0 - e) * self.m2.predict(X)


class DRLearner:
    """
    DR-learner style CATE: pseudo-outcome
      Γ = μ₁(X) − μ₀(X) + T(Y − μ₁)/e(X) − (1−T)(Y − μ₀)/(1−e(X))
    with separate μ₁, μ₀ (binary Y → classifiers) and final regression on Γ.
    """

    def __init__(self) -> None:
        self.propensity = GradientBoostingClassifier(**gb_params)
        self.mu1 = GradientBoostingClassifier(**gb_params)
        self.mu0 = GradientBoostingClassifier(**gb_params)
        self.final_model = GradientBoostingRegressor(**gb_params)

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> None:
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

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return self.final_model.predict(X)


class RLearner:
    """
    R-learner CATE via residual-on-residual regression:
      (Y - m(X)) ≈ (T - e(X)) * τ(X)
    We fit a final model on pseudo-target (Y - m)/(T - e) with
    sample weights (T - e)^2.
    """

    def __init__(self) -> None:
        self.propensity = GradientBoostingClassifier(**gb_params)
        self.outcome = GradientBoostingRegressor(**gb_params)
        self.final_model = GradientBoostingRegressor(**gb_params)

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> None:
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        self.propensity.fit(X, t.astype(int))
        e = np.clip(
            self.propensity.predict_proba(X)[:, 1],
            PROPENSITY_CLIP_LOW,
            PROPENSITY_CLIP_HIGH,
        )

        self.outcome.fit(X, y)
        m = self.outcome.predict(X)

        w = t - e
        # Clip denominator away from 0 to avoid exploding pseudo-targets.
        w_safe = np.where(np.abs(w) < 1e-3, np.sign(w) * 1e-3 + (w == 0) * 1e-3, w)
        pseudo = (y - m) / w_safe
        sample_weight = np.square(w)

        self.final_model.fit(X, pseudo, sample_weight=sample_weight)

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return self.final_model.predict(X)
