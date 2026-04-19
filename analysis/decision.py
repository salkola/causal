import pandas as pd
import numpy as np

from config import UPLIFT_BUCKET_COUNT


def build_decision_table(df, uplift):
    df = df.copy()
    df["uplift"] = uplift

    summary = df.groupby(pd.cut(df["uplift"], UPLIFT_BUCKET_COUNT)).agg({
        "conversion": "mean",
        "treatment": "mean"
    })

    return summary