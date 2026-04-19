import numpy as np
import pandas as pd


# -----------------------------
# Structural data generating process
# -----------------------------
def outcome(intent, context, treatment):
    base = 0.02 + 0.08 * intent + 0.01 * context
    treatment_effect = 0.01 + 0.10 * intent
    p = base + treatment * treatment_effect
    return p


def generate_ads_data(n=50000, seed=42):
    np.random.seed(seed)

    intent = np.random.beta(2, 5, n)
    context = np.random.normal(0, 1, n)

    # treatment assignment bias (selection bias)
    treatment_prob = 0.1 + 0.7 * intent
    treatment = np.random.binomial(1, treatment_prob)

    # observed outcome
    p = outcome(intent, context, treatment)
    conversion = np.random.binomial(1, np.clip(p, 0, 1))

    return pd.DataFrame({
        "intent": intent,
        "context": context,
        "treatment": treatment,
        "conversion": conversion
    })


# -----------------------------
# TRUE causal quantity (ATE)
# -----------------------------
def true_ate():
    """
    Analytical ATE:
    E[ (0.01 + 0.10 * intent) ]
    intent ~ Beta(2, 5)
    """
    return 0.01 + 0.10 * (2 / (2 + 5))