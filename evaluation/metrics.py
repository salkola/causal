from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc

from config import (
    DEFAULT_POLICY_TOP_K,
    QINI_FRAC_MAX,
    QINI_FRAC_MIN,
    QINI_MIN_PREFIX_SAMPLES,
    QINI_N_BINS,
    QINI_NULL_DRAWS,
    RANDOM_SEED,
    SAFE_CORR_STD_EPS,
)


def hajek_ate(
    y: np.ndarray,
    t: np.ndarray,
    propensity: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Hajek (stabilized IPW) difference in means on the subset where mask is True.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t)
    e = np.asarray(propensity, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    y_s, t_s, e_s = y[mask], t[mask], e[mask]
    if mask.sum() == 0:
        return 0.0

    w1 = t_s / e_s
    w0 = (1.0 - t_s) / (1.0 - e_s)
    den1 = np.sum(w1)
    den0 = np.sum(w0)
    if den1 < SAFE_CORR_STD_EPS or den0 < SAFE_CORR_STD_EPS:
        return 0.0

    return float(np.sum(w1 * y_s) / den1 - np.sum(w0 * y_s) / den0)


def policy_value_ipw(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    propensity: np.ndarray,
    top_k: float = DEFAULT_POLICY_TOP_K,
) -> float:
    """Top-k by score; Hajek IPW effect in that slice (needs e(X) per row)."""
    y = np.asarray(y)
    t = np.asarray(t)
    score = np.asarray(score)
    propensity = np.asarray(propensity)

    idx = np.argsort(score)[::-1]
    k = max(1, int(top_k * len(score)))
    mask = np.zeros(len(y), dtype=bool)
    mask[idx[:k]] = True
    return hajek_ate(y, t, propensity, mask)


def policy_value(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    top_k: float = DEFAULT_POLICY_TOP_K,
) -> float:
    idx = np.argsort(score)[::-1]
    k = max(1, int(top_k * len(score)))

    selected = idx[:k]

    treated = (t[selected] == 1)
    control = (t[selected] == 0)

    if treated.sum() == 0 or control.sum() == 0:
        return 0.0

    treated_rate = y[selected][treated].mean()
    control_rate = y[selected][control].mean()

    return float(treated_rate - control_rate)


def qini_curve(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    n_bins: int = QINI_N_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({
        "y": y_true,
        "uplift": uplift,
        "t": treatment
    })

    df = df.sort_values("uplift", ascending=False).reset_index(drop=True)

    xs, ys = [], []
    global_control_rate = df[df.t == 0]["y"].mean()

    for k in np.linspace(QINI_FRAC_MIN, QINI_FRAC_MAX, n_bins):
        top = df.iloc[:int(k * len(df))]

        treated = top[top.t == 1]["y"]
        control = top[top.t == 0]["y"]

        treated_rate = treated.mean() if len(treated) > 0 else 0
        control_rate = control.mean() if len(control) > 0 else global_control_rate

        xs.append(k)
        ys.append(float(treated_rate - control_rate))

    return np.array(xs), np.array(ys)


def qini_curve_ipw(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    n_bins: int = QINI_N_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Qini-style curve with Hajek IPW effect at each targeted fraction.

    Early fractions are noisy when the targeted prefix is tiny; the grid starts at
    max(QINI_FRAC_MIN, QINI_MIN_PREFIX_SAMPLES / n) so the first point uses enough rows.
    """
    y_true = np.asarray(y_true, dtype=float)
    uplift = np.asarray(uplift)
    treatment = np.asarray(treatment)
    propensity = np.asarray(propensity, dtype=float)

    order = np.argsort(-uplift)
    y_s = y_true[order]
    t_s = treatment[order]
    e_s = propensity[order]
    n = len(y_s)

    min_frac = max(QINI_FRAC_MIN, min(QINI_FRAC_MAX, QINI_MIN_PREFIX_SAMPLES / max(n, 1)))
    fracs = np.linspace(min_frac, QINI_FRAC_MAX, n_bins)
    min_m = min(int(QINI_MIN_PREFIX_SAMPLES), n)

    xs, ys = [], []
    prev_m = 0
    for frac in fracs:
        m = max(1, int(frac * n), min_m)
        m = min(max(m, prev_m + 1), n)
        if m <= prev_m:
            continue
        prev_m = m
        mask = np.zeros(n, dtype=bool)
        mask[:m] = True
        xs.append(m / n)
        ys.append(hajek_ate(y_s, t_s, e_s, mask))

    return np.array(xs), np.array(ys)


def qini_auc(xs: np.ndarray, ys: np.ndarray) -> float:
    return float(auc(xs, ys))


def qini_null_band_curves(
    y: np.ndarray,
    t: np.ndarray,
    propensity: np.ndarray,
    n_draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack Hajek-IPW Qini curves from random rankings; return xs and median / p5 / p95 y.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t)
    propensity = np.asarray(propensity, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    xs_ref = None
    ys_list = []
    for _ in range(n_draws):
        u = (rng.permutation(n) + 0.5) / n
        xs, ys_c = qini_curve_ipw(y, u, t, propensity)
        if xs_ref is None:
            xs_ref = xs
        elif not np.allclose(xs, xs_ref):
            raise ValueError("qini_curve_ipw x grid must match across draws")
        ys_list.append(ys_c)
    mat = np.vstack(ys_list)
    assert xs_ref is not None
    return (
        xs_ref,
        np.median(mat, axis=0),
        np.percentile(mat, 5, axis=0),
        np.percentile(mat, 95, axis=0),
    )


def qini_null_median_auc(
    y: np.ndarray,
    t: np.ndarray,
    propensity: np.ndarray,
    n_draws: int = QINI_NULL_DRAWS,
    seed: int = RANDOM_SEED,
) -> float:
    """
    Median Hajek-IPW Qini AUC over random rankings (same y, t, e as evaluation).
    Used as a variance anchor: single random permutations can beat models by chance.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t)
    propensity = np.asarray(propensity, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_draws):
        u = (rng.permutation(n) + 0.5) / n
        xs, ys_c = qini_curve_ipw(y, u, t, propensity)
        aucs.append(qini_auc(xs, ys_c))
    return float(np.median(aucs))


def oracle_policy_value(
    true_effect: np.ndarray,
    top_k: float = DEFAULT_POLICY_TOP_K,
) -> float:
    idx = np.argsort(true_effect)[::-1]
    k = max(1, int(top_k * len(true_effect)))
    return float(np.mean(true_effect[idx[:k]]))


def model_policy_value(
    true_effect: np.ndarray,
    model_score: np.ndarray,
    top_k: float = DEFAULT_POLICY_TOP_K,
) -> float:
    idx = np.argsort(model_score)[::-1]
    k = max(1, int(top_k * len(true_effect)))
    return float(np.mean(true_effect[idx[:k]]))


def true_regret(
    model_score: np.ndarray,
    true_effect: np.ndarray,
    top_k: float = DEFAULT_POLICY_TOP_K,
) -> float:
    oracle_idx = np.argsort(true_effect)[::-1]
    model_idx = np.argsort(model_score)[::-1]
    k = max(1, int(top_k * len(true_effect)))
    oracle_val = float(np.mean(true_effect[oracle_idx[:k]]))
    model_val = float(np.mean(true_effect[model_idx[:k]]))
    return oracle_val - model_val
