from typing import Dict, List
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

def metrics_from_proba(oof: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    o = np.asarray(oof, dtype=float).ravel()
    y = np.asarray(y_true).ravel()
    return {"auroc": float(roc_auc_score(y, o)), "brier": float(brier_score_loss(y, o))}

def weighted_blend(preds: List[np.ndarray], weights: List[float]) -> np.ndarray:
    assert len(preds) == len(weights) and len(preds) > 0
    wsum = float(sum(weights))
    out = np.zeros_like(preds[0], dtype=float)
    for p, w in zip(preds, weights):
        out += np.asarray(p, dtype=float).ravel() * (w / wsum)
    return out
