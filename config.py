"""Project-wide constants for simulation, models, and evaluation."""

from __future__ import annotations

import math

# Randomness
RANDOM_SEED = 0

# Data generation (ads DGP)
N_SAMPLES_DEFAULT = 20_000

BETA_INTENT_A = 2
BETA_INTENT_B = 5

CONTEXT_MEAN = 0.0
CONTEXT_STD = 1.0

TREATMENT_PROB_INTERCEPT = 0.1
TREATMENT_PROB_SLOPE = 0.3

OUTCOME_BASE = 0.02
OUTCOME_INTENT_COEF = 0.08
OUTCOME_CONTEXT_COEF = 0.01

CATE_INTERCEPT = 0.01
CATE_INTENT_SLOPE = 0.10

PROB_CLIP_MIN = 0.0
PROB_CLIP_MAX = 1.0

# Gradient boosting (uplift learners)
GB_N_ESTIMATORS = 100
GB_LEARNING_RATE = 0.05
GB_MAX_DEPTH = 3
GB_SUBSAMPLE = 0.8

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": GB_N_ESTIMATORS,
    "learning_rate": GB_LEARNING_RATE,
    "max_depth": GB_MAX_DEPTH,
    "subsample": GB_SUBSAMPLE,
    "random_state": RANDOM_SEED,
}

PROPENSITY_CLIP_LOW = 0.01
PROPENSITY_CLIP_HIGH = 0.99

# Evaluation / metrics
HOLDOUT_TEST_SIZE = 0.4
MONTE_CARLO_SPLITS = 50
DEFAULT_POLICY_TOP_K = 0.2
QINI_N_BINS = 20
# Smallest *nominal* fraction on the grid (actual first point may be larger if min prefix applies).
QINI_FRAC_MIN = 0.05
QINI_FRAC_MAX = 1.0
# Hajek IPW is volatile in tiny slices; require at least this many rows in the prefix for the grid.
QINI_MIN_PREFIX_SAMPLES = 500
# Monte Carlo median AUC of random rankings for Qini excess (variance control)
QINI_NULL_DRAWS = 100
# Draws for Qini plot: median + 5–95% band of random-ranking curves
QINI_PLOT_NULL_BAND_DRAWS = 100

SAFE_CORR_STD_EPS = 1e-8

METRIC_DECIMALS = 3
EVALUATION_REPORT_TITLE = "================ CAUSAL ML EVALUATION REPORT ================"

# Figures written by `evaluation/report_generator.py` (also referenced in README)
OUTPUT_DIR = "output"
OUTPUT_QINI_CURVES = f"{OUTPUT_DIR}/qini-curves.png"
OUTPUT_UPLIFT_DISTRIBUTION = f"{OUTPUT_DIR}/uplift-distribution.png"
OUTPUT_UPLIFT_CALIBRATION = f"{OUTPUT_DIR}/uplift-calibration.png"

# Decision table
UPLIFT_BUCKET_COUNT = 5

# Baselines
MEAN_UPLIFT_VALUE = 0.03


def _beta_variance(a: float, b: float) -> float:
    return (a * b) / ((a + b) ** 2 * (a + b + 1))


# Simulator CATE: τ = CATE_INTERCEPT + CATE_INTENT_SLOPE * intent, intent ~ Beta(BETA_INTENT_A, BETA_INTENT_B).
# SD(τ) = |CATE_INTENT_SLOPE| * SD(intent). RandomPolicy uses N(0, SD(τ)) so score spread matches heterogeneity scale.
RANDOM_POLICY_SCORE_STD = abs(CATE_INTENT_SLOPE) * math.sqrt(
    _beta_variance(BETA_INTENT_A, BETA_INTENT_B)
)
