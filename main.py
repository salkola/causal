import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from config import (
    CATE_INTERCEPT,
    CATE_INTENT_SLOPE,
    GRADIENT_BOOSTING_PARAMS,
    HOLDOUT_TEST_SIZE,
    PROPENSITY_CLIP_HIGH,
    PROPENSITY_CLIP_LOW,
    RANDOM_SEED,
)
from data.simulate_ads import generate_ads_data
from models.uplift import TLearner, XLearner, DRLearner
from models.baselines import RandomPolicy

from evaluation.leaderboard import evaluate_model
from evaluation.metrics import qini_null_median_auc
from evaluation.report_generator import generate_report


def fit_eval_propensity(X_train, t_train, X_test):
    """Propensity for IPW metrics: fit on train only, clip on test."""
    m = GradientBoostingClassifier(**GRADIENT_BOOSTING_PARAMS)
    m.fit(X_train, t_train)
    e = m.predict_proba(X_test)[:, 1]
    return np.clip(e, PROPENSITY_CLIP_LOW, PROPENSITY_CLIP_HIGH)


def main():

    df = generate_ads_data()

    X = df[["intent", "context"]].values
    t = df["treatment"].values
    y = df["conversion"].values
    true_effect = CATE_INTERCEPT + CATE_INTENT_SLOPE * df["intent"].values

    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=t,
    )

    X_train, X_test = X[idx_train], X[idx_test]
    t_train, t_test = t[idx_train], t[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    true_effect_test = true_effect[idx_test]

    e_test = fit_eval_propensity(X_train, t_train, X_test)

    qini_null_med = qini_null_median_auc(
        y_test, t_test, e_test, seed=RANDOM_SEED + 10_007
    )

    models_results = []

    models_results.append(
        evaluate_model(
            "T-Learner",
            TLearner(),
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
    )

    models_results.append(
        evaluate_model(
            "X-Learner",
            XLearner(),
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
    )

    models_results.append(
        evaluate_model(
            "DR-Learner",
            DRLearner(),
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
    )

    random_model = RandomPolicy()
    models_results.append(
        evaluate_model(
            "Random",
            random_model,
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
    )

    generate_report(models_results, y_test, t_test, true_effect_test, e_test)


if __name__ == "__main__":
    main()
