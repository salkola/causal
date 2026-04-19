import numpy as np

def evaluate_policy(uplift, true_effect):
    # correlation sanity check
    return np.corrcoef(uplift, true_effect)[0, 1]