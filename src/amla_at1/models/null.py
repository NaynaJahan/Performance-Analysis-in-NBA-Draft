import numpy as np
import pandas as pd

class NullModel:
    """
    Constant-probability baseline equals y.mean() from training.
    """
    def __init__(self):
        self.p_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame):
        if self.p_ is None:
            raise RuntimeError("Call fit() first")
        n = len(X)
        p1 = np.full((n, 1), self.p_, dtype=float)
        p0 = 1.0 - p1
        return np.hstack([p0, p1])
