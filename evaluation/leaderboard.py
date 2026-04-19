import numpy as np

from config import SAFE_CORR_STD_EPS


# ============================================================
# SAFE CORRELATION
# ============================================================

def safe_corr(x, y):
    """
    Stable Pearson correlation.
    Returns 0.0 when x has no variance (e.g., Random policy).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if np.std(x) < SAFE_CORR_STD_EPS or np.std(y) < SAFE_CORR_STD_EPS:
        return 0.0

    return np.corrcoef(x, y)[0, 1]


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(name, model, X, t, y, true_effect):

    model.fit(X, t, y)
    uplift = model.predict_uplift(X)

    # --- Qini ---
    from evaluation.metrics import qini_curve, qini_auc, policy_value

    xs, ys = qini_curve(y, uplift, t)
    qini_score = qini_auc(xs, ys)

    # --- Policy value (observed space metric) ---
    policy_score = policy_value(y, t, uplift)

    # --- CORRELATION (FIXED) ---
    corr = safe_corr(uplift, true_effect)

    result = {
        "name": name,
        "qini_auc": qini_score,
        "policy_value": policy_score,
        "avg_uplift": np.mean(uplift),
        "uplift": uplift,
        "xs": xs,
        "ys": ys,
        "corr": corr
    }

    return result