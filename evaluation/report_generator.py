import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import EVALUATION_REPORT_TITLE, METRIC_DECIMALS, SAFE_CORR_STD_EPS
from evaluation.metrics import (
    model_policy_value,
    oracle_policy_value,
    true_regret,
)


# ============================================================
# UPLIFT DISTRIBUTION PLOT
# ============================================================

def plot_uplift_distribution(models_results):

    plt.figure()

    for r in models_results:
        plt.hist(r["uplift"], bins=50, alpha=0.4, label=r["name"], density=True)

    plt.title("Uplift Distribution by Model")
    plt.xlabel("Predicted uplift")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


# ============================================================
# UPLIFT CALIBRATION PLOT (FIXED)
# ============================================================

def plot_uplift_calibration(models_results, y, t, bins=10):

    plt.figure()

    for r in models_results:

        uplift = r["uplift"]

        # 🚨 HANDLE CONSTANT UPLIFT (Random policy)
        if np.std(uplift) < SAFE_CORR_STD_EPS:
            plt.plot(
                [0, bins - 1],
                [0, 0],
                linestyle="--",
                label=f"{r['name']} (flat)"
            )
            continue

        df = pd.DataFrame({
            "y": y,
            "t": t,
            "uplift": uplift
        })

        df["bin"] = pd.qcut(df["uplift"], bins, duplicates="drop")

        lift = df.groupby("bin").apply(
            lambda d: d[d.t == 1]["y"].mean() - d[d.t == 0]["y"].mean()
        )

        plt.plot(range(len(lift)), lift.values, marker="o", label=r["name"])

    # Optional baseline for clarity
    plt.axhline(0, linestyle="--", color="black", alpha=0.5, label="Zero uplift")

    plt.title("Uplift Calibration (by Score Decile)")
    plt.xlabel("Uplift decile (low → high)")
    plt.ylabel("Observed treatment effect proxy")
    plt.legend()
    plt.show()


# ============================================================
# REPORT (single text summary — no duplicate leaderboard)
# ============================================================

def print_evaluation_summary(models_results, true_effect):
    """
    One ranked table: Qini, observed vs true-τ policy value, regret, uplift, corr.
    """
    oracle_val = oracle_policy_value(true_effect)
    ranked = sorted(models_results, key=lambda x: x["qini_auc"], reverse=True)
    d = METRIC_DECIMALS

    print(f"\n{EVALUATION_REPORT_TITLE}\n")
    print(f"Oracle policy value (true τ, top fraction): {oracle_val:.{d}f}\n")
    print("(Ranked by Qini AUC. Policy (obs) = slice in assignment/outcome space;")
    print(" Policy (true τ) = mean true effect in top slice by model score.)\n")

    for i, r in enumerate(ranked):
        pol_true = model_policy_value(true_effect, r["uplift"])
        regret = true_regret(r["uplift"], true_effect)
        print(f"{i + 1}. {r['name']}")
        print(f"   Qini AUC          : {r['qini_auc']:.{d}f}")
        print(f"   Policy (observed) : {r['policy_value']:.{d}f}")
        print(f"   Policy (true τ)   : {pol_true:.{d}f}")
        print(f"   Regret (true τ)   : {regret:.{d}f}")
        print(f"   Avg uplift        : {r['avg_uplift']:.{d}f}")
        print(f"   Corr (true)       : {r['corr']:.{d}f}")
        print("")


def generate_report(models_results, y, t, true_effect):

    print_evaluation_summary(models_results, true_effect)

    # ========================================================
    # QINI CURVES
    # ========================================================

    plt.figure()
    for r in models_results:
        plt.plot(r["xs"], r["ys"], label=r["name"])

    plt.legend()
    plt.title("Qini Curves")
    plt.xlabel("Fraction targeted")
    plt.ylabel("Incremental gain")
    plt.show()

    # ========================================================
    # ADDITIONAL PLOTS
    # ========================================================

    plot_uplift_distribution(models_results)
    plot_uplift_calibration(models_results, y, t)