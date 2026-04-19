import numpy as np

from data.simulate_ads import generate_ads_data, true_ate
from models.uplift import TLearner, XLearner, DRLearner
from models.baselines import RandomPolicy

from evaluation.leaderboard import evaluate_model, build_leaderboard


def main():

    df = generate_ads_data()

    X = df[["intent", "context"]].values
    t = df["treatment"].values
    y = df["conversion"].values

    true_effect = 0.01 + 0.10 * df["intent"].values

    models_results = []

    # ============================================================
    # MODELS
    # ============================================================

    models_results.append(
        evaluate_model("T-Learner", TLearner(), X, t, y, true_effect)
    )

    models_results.append(
        evaluate_model("X-Learner", XLearner(), X, t, y, true_effect)
    )

    models_results.append(
        evaluate_model("DR-Learner", DRLearner(), X, t, y, true_effect)
    )

    # ============================================================
    # RANDOM POLICY (NOW ALSO HAS CORR)
    # ============================================================

    random_model = RandomPolicy()

    models_results.append(
        evaluate_model("Random", random_model, X, t, y, true_effect)
    )

    # ============================================================
    # LEADERBOARD
    # ============================================================

    build_leaderboard(models_results)


if __name__ == "__main__":
    main()