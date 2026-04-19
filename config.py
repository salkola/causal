"""Project-wide constants for simulation, models, and evaluation."""

# --- Randomness ---
RANDOM_SEED = 42

# --- Data generation (ads DGP) ---
N_SAMPLES_DEFAULT = 50_000

BETA_INTENT_A = 2
BETA_INTENT_B = 5

CONTEXT_MEAN = 0.0
CONTEXT_STD = 1.0

TREATMENT_PROB_INTERCEPT = 0.1
TREATMENT_PROB_SLOPE = 0.7

OUTCOME_BASE = 0.02
OUTCOME_INTENT_COEF = 0.08
OUTCOME_CONTEXT_COEF = 0.01

CATE_INTERCEPT = 0.01
CATE_INTENT_SLOPE = 0.10

PROB_CLIP_MIN = 0.0
PROB_CLIP_MAX = 1.0

# --- Gradient boosting (uplift learners) ---
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

# --- Evaluation / metrics ---
DEFAULT_POLICY_TOP_K = 0.2
QINI_N_BINS = 20
QINI_FRAC_MIN = 0.01
QINI_FRAC_MAX = 1.0

SAFE_CORR_STD_EPS = 1e-8

METRIC_DECIMALS = 4
EVALUATION_REPORT_TITLE = "================ CAUSAL ML EVALUATION REPORT ================"

# --- Decision table ---
UPLIFT_BUCKET_COUNT = 5

# --- Baselines ---
MEAN_UPLIFT_VALUE = 0.03
