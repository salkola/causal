from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    BETA_INTENT_A,
    BETA_INTENT_B,
    CATE_INTERCEPT,
    CATE_INTENT_SLOPE,
    CONTEXT_MEAN,
    CONTEXT_STD,
    N_SAMPLES_DEFAULT,
    OUTCOME_BASE,
    OUTCOME_CONTEXT_COEF,
    OUTCOME_INTENT_COEF,
    PROB_CLIP_MAX,
    PROB_CLIP_MIN,
    RANDOM_SEED,
    TREATMENT_PROB_INTERCEPT,
    TREATMENT_PROB_SLOPE,
)


# -----------------------------
# Structural data generating process
# -----------------------------
def outcome(
    intent: np.ndarray,
    context: np.ndarray,
    treatment: np.ndarray,
) -> np.ndarray:
    base = OUTCOME_BASE + OUTCOME_INTENT_COEF * intent + OUTCOME_CONTEXT_COEF * context
    treatment_effect = CATE_INTERCEPT + CATE_INTENT_SLOPE * intent
    p = base + treatment * treatment_effect
    return p


def generate_ads_data(
    n: int = N_SAMPLES_DEFAULT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    np.random.seed(seed)

    intent = np.random.beta(BETA_INTENT_A, BETA_INTENT_B, n)
    context = np.random.normal(CONTEXT_MEAN, CONTEXT_STD, n)

    # treatment assignment bias (selection bias)
    treatment_prob = TREATMENT_PROB_INTERCEPT + TREATMENT_PROB_SLOPE * intent
    treatment = np.random.binomial(1, treatment_prob)

    # observed outcome
    p = outcome(intent, context, treatment)
    conversion = np.random.binomial(1, np.clip(p, PROB_CLIP_MIN, PROB_CLIP_MAX))

    return pd.DataFrame({
        "intent": intent,
        "context": context,
        "treatment": treatment,
        "conversion": conversion
    })


# -----------------------------
# TRUE causal quantity (ATE)
# -----------------------------
def true_ate() -> float:
    """
    Analytical ATE:
    E[ (0.01 + 0.10 * intent) ]
    intent ~ Beta(2, 5)
    """
    return CATE_INTERCEPT + CATE_INTENT_SLOPE * (
        BETA_INTENT_A / (BETA_INTENT_A + BETA_INTENT_B)
    )
