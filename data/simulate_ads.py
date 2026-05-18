from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    BETA_INTENT_A,
    BETA_INTENT_B,
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
    cate,
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
    p = base + treatment * cate(intent, context)
    return p


def generate_ads_data(
    n: int = N_SAMPLES_DEFAULT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    np.random.seed(seed)

    intent = np.random.beta(BETA_INTENT_A, BETA_INTENT_B, n)
    context = np.random.normal(CONTEXT_MEAN, CONTEXT_STD, n)

    # Assignment depends on intent only (not on context-driven τ).
    treatment_prob = TREATMENT_PROB_INTERCEPT + TREATMENT_PROB_SLOPE * intent
    treatment = np.random.binomial(1, treatment_prob)

    p = outcome(intent, context, treatment)
    conversion = np.random.binomial(1, np.clip(p, PROB_CLIP_MIN, PROB_CLIP_MAX))

    return pd.DataFrame({
        "intent": intent,
        "context": context,
        "treatment": treatment,
        "conversion": conversion,
    })


def true_ate(n_mc: int = 200_000, seed: int = RANDOM_SEED) -> float:
    """Monte Carlo E[τ(X)] under the simulator."""
    rng = np.random.default_rng(seed)
    intent = rng.beta(BETA_INTENT_A, BETA_INTENT_B, n_mc)
    context = rng.normal(CONTEXT_MEAN, CONTEXT_STD, n_mc)
    return float(np.mean(cate(intent, context)))
