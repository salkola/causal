from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from config import (
    CATE_INTERCEPT,
    CATE_INTENT_SLOPE,
    GRADIENT_BOOSTING_PARAMS,
    HOLDOUT_TEST_SIZE,
    MONTE_CARLO_SPLITS,
    PROPENSITY_CLIP_HIGH,
    PROPENSITY_CLIP_LOW,
    RANDOM_SEED,
)
from data.simulate_ads import generate_ads_data
from models.uplift import TLearner, XLearner, DRLearner, RLearner
from models.baselines import RandomPolicy

from evaluation.leaderboard import evaluate_model
from evaluation.metrics import model_policy_value, qini_null_median_auc, true_regret
from evaluation.report_generator import generate_report


def fit_eval_propensity(
    X_train: np.ndarray,
    t_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Propensity for IPW metrics: fit on train only, clip on test."""
    m = GradientBoostingClassifier(**GRADIENT_BOOSTING_PARAMS)
    m.fit(X_train, t_train)
    e = m.predict_proba(X_test)[:, 1]
    return np.clip(e, PROPENSITY_CLIP_LOW, PROPENSITY_CLIP_HIGH)


def main() -> None:

    df = generate_ads_data()

    X = df[["intent", "context"]].values
    t = df["treatment"].values
    y = df["conversion"].values
    true_effect = CATE_INTERCEPT + CATE_INTENT_SLOPE * df["intent"].values

    model_specs: list[tuple[str, Callable[[], Any]]] = [
        ("T-Learner", TLearner),
        ("X-Learner", XLearner),
        ("DR-Learner", DRLearner),
        ("R-Learner", RLearner),
        ("Random", RandomPolicy),
    ]
    by_model: dict[str, dict[str, list[Any]]] = {
        name: {
            "qini_auc_raw": [],
            "qini_auc_excess": [],
            "qini_null_median": [],
            "policy_value": [],
            "policy_true": [],
            "regret_true": [],
            "avg_uplift": [],
            "corr": [],
            "uplift": [],
            "xs": [],
            "ys": [],
        }
        for name, _ in model_specs
    }

    y_eval_all: list[np.ndarray] = []
    t_eval_all: list[np.ndarray] = []
    true_eval_all: list[np.ndarray] = []
    e_eval_all: list[np.ndarray] = []
    idx = np.arange(len(y))

    for split_i in range(MONTE_CARLO_SPLITS):
        split_seed = RANDOM_SEED + split_i
        idx_train, idx_test = train_test_split(
            idx,
            test_size=HOLDOUT_TEST_SIZE,
            random_state=split_seed,
            stratify=t,
        )

        X_train, X_test = X[idx_train], X[idx_test]
        t_train, t_test = t[idx_train], t[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        true_effect_test = true_effect[idx_test]
        e_test = fit_eval_propensity(X_train, t_train, X_test)
        qini_null_med = qini_null_median_auc(
            y_test,
            t_test,
            e_test,
            seed=split_seed + 10_007,
        )

        y_eval_all.append(y_test)
        t_eval_all.append(t_test)
        true_eval_all.append(true_effect_test)
        e_eval_all.append(e_test)

        for name, model_factory in model_specs:
            r = evaluate_model(
                name,
                model_factory(),
                X_train,
                t_train,
                y_train,
                X_test,
                t_test,
                y_test,
                true_effect_test,
                e_test,
                qini_null_med,
            )
            pol_true = model_policy_value(true_effect_test, r["uplift"])
            regret = true_regret(r["uplift"], true_effect_test)

            b = by_model[name]
            b["qini_auc_raw"].append(r["qini_auc_raw"])
            b["qini_auc_excess"].append(r["qini_auc_excess"])
            b["qini_null_median"].append(r["qini_null_median"])
            b["policy_value"].append(r["policy_value"])
            b["policy_true"].append(pol_true)
            b["regret_true"].append(regret)
            b["avg_uplift"].append(r["avg_uplift"])
            b["corr"].append(r["corr"])
            b["uplift"].append(r["uplift"])
            b["xs"].append(r["xs"])
            b["ys"].append(r["ys"])

    y_eval = np.concatenate(y_eval_all)
    t_eval = np.concatenate(t_eval_all)
    true_eval = np.concatenate(true_eval_all)
    e_eval = np.concatenate(e_eval_all)

    models_results: list[dict[str, Any]] = []
    for name, _ in model_specs:
        b = by_model[name]
        xs_ref = b["xs"][0]
        ys_interp = [
            ys
            if (len(xs) == len(xs_ref) and np.allclose(xs, xs_ref))
            else np.interp(xs_ref, xs, ys)
            for xs, ys in zip(b["xs"], b["ys"])
        ]
        ys_mean = np.mean(np.vstack(ys_interp), axis=0)

        models_results.append(
            {
                "name": name,
                "qini_auc_raw": float(np.mean(b["qini_auc_raw"])),
                "qini_auc_excess": float(np.mean(b["qini_auc_excess"])),
                "qini_null_median": float(np.mean(b["qini_null_median"])),
                "policy_value": float(np.mean(b["policy_value"])),
                "policy_true": float(np.mean(b["policy_true"])),
                "regret_true": float(np.mean(b["regret_true"])),
                "avg_uplift": float(np.mean(b["avg_uplift"])),
                "corr": float(np.mean(b["corr"])),
                "uplift": np.concatenate(b["uplift"]),
                "xs": xs_ref,
                "ys": ys_mean,
                "n_splits": MONTE_CARLO_SPLITS,
            }
        )

    generate_report(models_results, y_eval, t_eval, true_eval, e_eval)


if __name__ == "__main__":
    main()
