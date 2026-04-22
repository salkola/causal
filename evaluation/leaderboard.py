from __future__ import annotations

from typing import Any

import numpy as np

from config import SAFE_CORR_STD_EPS


# ============================================================
# SAFE CORRELATION
# ============================================================

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Stable Pearson correlation.
    Returns 0.0 when x has no variance (e.g., Random policy).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if np.std(x) < SAFE_CORR_STD_EPS or np.std(y) < SAFE_CORR_STD_EPS:
        return 0.0

    return float(np.corrcoef(x, y)[0, 1])


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    name: str,
    model: Any,
    X_train: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray,
    y_test: np.ndarray,
    true_effect_test: np.ndarray,
    propensity_test: np.ndarray,
    qini_null_median: float,
) -> dict[str, Any]:
    """
    Fit on train; metrics and predictions on holdout.
    Qini and policy (obs) use Hajek IPW with propensity_test (e.g. ê from train).
    qini_null_median: median random-ranking Qini AUC on the same test fold (see metrics).
    """
    model.fit(X_train, t_train, y_train)
    uplift = model.predict_uplift(X_test)

    from evaluation.metrics import policy_value_ipw, qini_auc, qini_curve_ipw

    xs, ys = qini_curve_ipw(y_test, uplift, t_test, propensity_test)
    qini_raw = qini_auc(xs, ys)
    qini_excess = float(qini_raw - qini_null_median)

    # Random ranking: one draw's Qini AUC is high-variance vs the null median.
    # Pin Qini to the null reference so the baseline does not "win" by Monte Carlo noise.
    if name.strip().lower() == "random":
        qini_raw = float(qini_null_median)
        qini_excess = 0.0
        # xs, ys stay from `uplift` (RandomPolicy scores) so plots match policy/corr.

    policy_score = policy_value_ipw(y_test, t_test, uplift, propensity_test)
    corr = safe_corr(uplift, true_effect_test)

    return {
        "name": name,
        "qini_auc_raw": float(qini_raw),
        "qini_auc_excess": qini_excess,
        "qini_null_median": float(qini_null_median),
        "policy_value": policy_score,
        "avg_uplift": float(np.mean(uplift)),
        "uplift": uplift,
        "xs": xs,
        "ys": ys,
        "corr": corr,
    }
