import numpy as np
import pandas as pd
from sklearn.metrics import auc

from config import (
    DEFAULT_POLICY_TOP_K,
    QINI_FRAC_MAX,
    QINI_FRAC_MIN,
    QINI_N_BINS,
)


def policy_value(y, t, score, top_k=DEFAULT_POLICY_TOP_K):
    idx = np.argsort(score)[::-1]
    k = max(1, int(top_k * len(score)))

    selected = idx[:k]

    treated = (t[selected] == 1)
    control = (t[selected] == 0)

    if treated.sum() == 0 or control.sum() == 0:
        return 0.0

    treated_rate = y[selected][treated].mean()
    control_rate = y[selected][control].mean()

    return treated_rate - control_rate


def qini_curve(y_true, uplift, treatment, n_bins=QINI_N_BINS):
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
        ys.append(treated_rate - control_rate)

    return np.array(xs), np.array(ys)


def qini_auc(xs, ys):
    return auc(xs, ys)


def oracle_policy_value(true_effect, top_k=DEFAULT_POLICY_TOP_K):
    idx = np.argsort(true_effect)[::-1]
    k = max(1, int(top_k * len(true_effect)))
    return float(np.mean(true_effect[idx[:k]]))


def model_policy_value(true_effect, model_score, top_k=DEFAULT_POLICY_TOP_K):
    idx = np.argsort(model_score)[::-1]
    k = max(1, int(top_k * len(true_effect)))
    return float(np.mean(true_effect[idx[:k]]))


def true_regret(model_score, true_effect, top_k=DEFAULT_POLICY_TOP_K):
    oracle_idx = np.argsort(true_effect)[::-1]
    model_idx = np.argsort(model_score)[::-1]
    k = max(1, int(top_k * len(true_effect)))
    oracle_val = float(np.mean(true_effect[oracle_idx[:k]]))
    model_val = float(np.mean(true_effect[model_idx[:k]]))
    return oracle_val - model_val