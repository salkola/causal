import numpy as np
import matplotlib.pyplot as plt

from config import DEFAULT_POLICY_TOP_K, METRIC_DECIMALS, REPORT_TITLE


# ============================================================
# ORACLE (TRUE EFFECT SPACE)
# ============================================================

def oracle_policy_value(true_effect, top_k=DEFAULT_POLICY_TOP_K):
    idx = np.argsort(true_effect)[::-1]
    k = max(1, int(top_k * len(true_effect)))
    return np.mean(true_effect[idx[:k]])


# ============================================================
# MODEL VALUE (TRUE EFFECT EVALUATION)
# ============================================================

def model_policy_value(true_effect, model_score, top_k=DEFAULT_POLICY_TOP_K):
    idx = np.argsort(model_score)[::-1]
    k = max(1, int(top_k * len(true_effect)))
    return np.mean(true_effect[idx[:k]])


# ============================================================
# REGRET
# ============================================================

def true_regret(model_score, true_effect, top_k=DEFAULT_POLICY_TOP_K):
    oracle_idx = np.argsort(true_effect)[::-1]
    model_idx = np.argsort(model_score)[::-1]

    k = max(1, int(top_k * len(true_effect)))

    oracle_val = np.mean(true_effect[oracle_idx[:k]])
    model_val = np.mean(true_effect[model_idx[:k]])

    return oracle_val - model_val


# ============================================================
# REPORT
# ============================================================

def generate_report(models_results, y, t, true_effect):

    oracle_val = oracle_policy_value(true_effect)

    d = METRIC_DECIMALS
    print(f"\n{REPORT_TITLE}\n")
    print(f"Oracle policy value (true): {oracle_val:.{d}f}\n")

    ranked = sorted(models_results, key=lambda x: x["qini_auc"], reverse=True)

    for r in ranked:

        regret = true_regret(r["uplift"], true_effect)

        model_val = model_policy_value(true_effect, r["uplift"])

        print(f"{r['name']}")
        print(f"  Qini AUC     : {r['qini_auc']:.{d}f}")
        print(f"  Policy value : {model_val:.{d}f}")
        print(f"  Regret       : {regret:.{d}f}")
        print(f"  Avg uplift   : {r['avg_uplift']:.{d}f}")

        if "corr" in r:
            print(f"  Corr (true)  : {r['corr']:.{d}f}")

        print("")

    # ========================================================
    # QINI PLOT
    # ========================================================
    plt.figure()

    for r in models_results:
        plt.plot(r["xs"], r["ys"], label=r["name"])

    plt.legend()
    plt.title("Qini Curves (Ranking Quality)")
    plt.xlabel("Fraction targeted")
    plt.ylabel("Incremental gain")
    plt.show()