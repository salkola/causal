import numpy as np
import pandas as pd
from sklearn.metrics import auc


def policy_value(y, t, score, top_k=0.2):
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


def qini_curve(y_true, uplift, treatment, n_bins=20):
    df = pd.DataFrame({
        "y": y_true,
        "uplift": uplift,
        "t": treatment
    })

    df = df.sort_values("uplift", ascending=False).reset_index(drop=True)

    xs, ys = [], []
    global_control_rate = df[df.t == 0]["y"].mean()

    for k in np.linspace(0.01, 1.0, n_bins):
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