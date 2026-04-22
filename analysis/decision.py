from __future__ import annotations

import numpy as np
import pandas as pd

from config import UPLIFT_BUCKET_COUNT


def build_decision_table(df: pd.DataFrame, uplift: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["uplift"] = uplift

    summary = df.groupby(pd.cut(df["uplift"], UPLIFT_BUCKET_COUNT)).agg({
        "conversion": "mean",
        "treatment": "mean"
    })

    return summary
