import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CORE METRICS
# ============================================================

def oracle_policy_value(true_effect, top_k=0.2):
    idx = np.argsort(true_effect)[::-1]
    k = max(1, int(top_k * len(true_effect)))
    return np.mean(true_effect[idx[:k]])


def model_policy_value(true_effect, model_score, top_k=0.2):
    idx = np.argsort(model_score)[::-1]
    k = max(1, int(top_k * len(true_effect)))
    return np.mean(true_effect[idx[:k]])


def true_regret(model_score, true_effect, top_k=0.2):
    oracle_idx = np.argsort(true_effect)[::-1]
    model_idx = np.argsort(model_score)[::-1]

    k = max(1, int(top_k * len(true_effect)))

    oracle_val = np.mean(true_effect[oracle_idx[:k]])
    model_val = np.mean(true_effect[model_idx[:k]])

    return oracle_val - model_val


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
        if np.std(uplift) < 1e-8:
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
# REPORT
# ============================================================

def generate_report(models_results, y, t, true_effect):

    oracle_val = oracle_policy_value(true_effect)

    print("\n================ CAUSAL ML REPORT (TRUE EFFECT SPACE) ================\n")
    print(f"Oracle policy value (true): {oracle_val:.4f}\n")

    for r in models_results:

        model_val = model_policy_value(true_effect, r["uplift"])
        regret = true_regret(r["uplift"], true_effect)

        print(f"{r['name']}")
        print(f"  Qini AUC     : {r['qini_auc']:.4f}")
        print(f"  Policy value : {model_val:.4f}")
        print(f"  Regret       : {regret:.4f}")
        print(f"  Avg uplift   : {r['avg_uplift']:.4f}")
        print(f"  Corr (true)  : {r['corr']:.4f}")
        print("")

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