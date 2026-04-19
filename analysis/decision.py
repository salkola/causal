import pandas as pd
import numpy as np

def build_decision_table(df, uplift):
    df = df.copy()
    df["uplift"] = uplift

    summary = df.groupby(pd.cut(df["uplift"], 5)).agg({
        "conversion": "mean",
        "treatment": "mean"
    })

    return summary