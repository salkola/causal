from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    EVALUATION_REPORT_TITLE,
    METRIC_DECIMALS,
    QINI_NULL_DRAWS,
    QINI_PLOT_NULL_BAND_DRAWS,
    RANDOM_POLICY_SCORE_STD,
    RANDOM_SEED,
    SAFE_CORR_STD_EPS,
)
from evaluation.metrics import (
    hajek_ate,
    model_policy_value,
    oracle_policy_value,
    qini_null_band_curves,
    true_regret,
)


# ============================================================
# UPLIFT DISTRIBUTION PLOT
# ============================================================

def plot_uplift_distribution(models_results: list[dict[str, Any]]) -> None:

    plt.figure()
    bin_edges = np.arange(-0.1, 0.15 + 0.004, 0.004)

    for r in models_results:
        plt.hist(r["uplift"], bins=bin_edges, alpha=0.4, label=r["name"], density=True)

    plt.title("Uplift Distribution by Model")
    plt.xlim(-0.1, 0.15)
    plt.xlabel("Predicted uplift")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


# ============================================================
# UPLIFT CALIBRATION (Hajek IPW within score bins)
# ============================================================

def plot_uplift_calibration(
    models_results: list[dict[str, Any]],
    y: np.ndarray,
    t: np.ndarray,
    propensity: np.ndarray,
    bins: int = 10,
) -> None:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t)
    propensity = np.asarray(propensity, dtype=float)

    plt.figure()

    for r in models_results:

        uplift = r["uplift"]

        if np.std(uplift) < SAFE_CORR_STD_EPS:
            plt.plot(
                [0, bins - 1],
                [0, 0],
                linestyle="--",
                label=f"{r['name']} (flat)",
            )
            continue

        df = pd.DataFrame({"uplift": uplift})
        df["bin"] = pd.qcut(df["uplift"], bins, duplicates="drop")

        lifts = []
        for cat in df["bin"].cat.categories:
            m = (df["bin"] == cat).to_numpy()
            lifts.append(hajek_ate(y, t, propensity, m))

        plt.plot(range(len(lifts)), lifts, marker="o", label=r["name"])

    plt.axhline(0, linestyle="--", color="black", alpha=0.5, label="Zero")

    plt.title("Uplift calibration (Hajek IPW by score decile)")
    plt.xlabel("Uplift decile (low → high)")
    plt.ylabel("Hajek IPW effect in bin")
    plt.legend()
    plt.show()


# ============================================================
# REPORT
# ============================================================

def print_evaluation_summary(
    models_results: list[dict[str, Any]],
    true_effect: np.ndarray,
) -> None:
    """
    One ranked table: Qini excess vs random null, raw Qini, policies, regret, corr.
    """
    oracle_val = oracle_policy_value(true_effect)
    ranked = sorted(
        models_results,
        key=lambda x: (round(x["qini_auc_excess"], METRIC_DECIMALS), x["corr"]),
        reverse=True,
    )
    n_splits = int(models_results[0].get("n_splits", 1))
    d = METRIC_DECIMALS
    # Fixed-width numeric formatting: show '-' when negative, no '+' when positive.
    width = d + 3
    fmt = lambda v: f"{v:>{width}.{d}f}"
    null_ref = models_results[0]["qini_null_median"]

    print(f"\n{EVALUATION_REPORT_TITLE}\n")
    print(f"Oracle policy value (true τ, top fraction): {fmt(oracle_val)}\n")
    split_label = "Holdout" if n_splits == 1 else f"Holdout Monte Carlo: {n_splits} splits"
    print(f"{split_label}\n")
    print("Qini raw: Hajek-IPW Qini AUC on test.")
    print(
        f"Qini Δ: Qini raw minus the median of {QINI_NULL_DRAWS} random-ranking AUCs "
        f"({fmt(null_ref)}, averaged across splits)."
    )
    print("Models are ranked by Qini Δ (ties broken by Corr (true)).\n")
    print("Random baseline:")
    print("- Qini raw is set to that null median.")
    print("- Qini Δ is fixed at 0.")
    print(
        f"- Policy/Corr use random Gaussian scores (σ = SD(τ) under Beta DGP = "
        f"{RANDOM_POLICY_SCORE_STD:.{METRIC_DECIMALS}f})."
    )
    print("- Qini curve uses the same random scores.\n")
    print("IPW details:")
    print("- ê(X) for IPW is fit on train only.")
    print("- Policy (IPW obs): Hajek effect in the top-scored slice.")
    print("- Policy (true τ): mean simulator τ in that slice (not IPW-adjusted).\n")

    for i, r in enumerate(ranked):
        pol_true = r.get("policy_true", model_policy_value(true_effect, r["uplift"]))
        regret = r.get("regret_true", true_regret(r["uplift"], true_effect))
        print(f"{i + 1}. {r['name']}")
        print(f"   Qini Δ (vs null): {fmt(r['qini_auc_excess'])}")
        print(f"   Qini raw        : {fmt(r['qini_auc_raw'])}")
        print(f"   Policy (IPW obs): {fmt(r['policy_value'])}")
        print(f"   Policy (true τ) : {fmt(pol_true)}")
        print(f"   Regret (true τ) : {fmt(regret)}")
        print(f"   Avg uplift      : {fmt(r['avg_uplift'])}")
        print(f"   Corr (true)     : {fmt(r['corr'])}")
        print("")


def generate_report(
    models_results: list[dict[str, Any]],
    y: np.ndarray,
    t: np.ndarray,
    true_effect: np.ndarray,
    propensity: np.ndarray,
) -> None:

    print_evaluation_summary(models_results, true_effect)

    xs0, y_med, y_p5, y_p95 = qini_null_band_curves(
        y,
        t,
        propensity,
        n_draws=QINI_PLOT_NULL_BAND_DRAWS,
        seed=RANDOM_SEED + 20_021,
    )

    plt.figure()
    for r in models_results:
        plt.plot(r["xs"], r["ys"], label=r["name"])

    plt.fill_between(
        xs0,
        y_p5,
        y_p95,
        color="0.5",
        alpha=0.22,
        label=f"Null random ({QINI_PLOT_NULL_BAND_DRAWS} draws, 5–95%)",
    )
    plt.plot(
        xs0,
        y_med,
        color="0.2",
        linestyle="--",
        linewidth=1.5,
        label="Null random (median y)",
    )

    plt.legend(loc="best")
    plt.title("Qini curves (Hajek IPW, holdout)")
    plt.xlabel("Fraction targeted")
    plt.ylabel("IPW incremental effect")
    plt.show()

    plot_uplift_distribution(models_results)
    plot_uplift_calibration(models_results, y, t, propensity)
