import numpy as np


# ============================================================
# SAFE CORRELATION (fixes Random → NaN issue)
# ============================================================

def safe_corr(x, y):
    """
    Stable Pearson correlation.
    Returns 0.0 when x has no variance (e.g., Random policy).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
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


# ============================================================
# LEADERBOARD
# ============================================================

def build_leaderboard(results):

    ranked = sorted(results, key=lambda x: x["qini_auc"], reverse=True)

    print("\n================ CAUSAL ML LEADERBOARD ================\n")

    for i, r in enumerate(ranked):
        print(f"{i+1}. {r['name']}")
        print(f"   Qini AUC       : {r['qini_auc']:.4f}")
        print(f"   Policy value   : {r['policy_value']:.4f}")
        print(f"   Avg uplift     : {r['avg_uplift']:.4f}")
        print(f"   Corr (true)    : {r['corr']:.4f}")
        print("")