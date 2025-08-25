from __future__ import annotations
import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def pop_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract target variable from dataframe; return (X, y).
    """
    if df is None:
        raise AttributeError("df is None")
    if not target_col:
        raise KeyError("target_col is None or empty")
    df_copy = df.copy()
    y = df_copy.pop(target_col)   # raises KeyError if column missing
    return df_copy, y

def save_sets(
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    X_val:   Optional[np.ndarray] = None,
    y_val:   Optional[np.ndarray] = None,
    X_test:  Optional[np.ndarray] = None,
    y_test:  Optional[np.ndarray] = None,
    path: str = "data/processed/",
) -> None:
    """
    Save any provided sets as .npy in 'path'.
    """
    os.makedirs(path, exist_ok=True)
    if X_train is not None: np.save(os.path.join(path, "X_train"), X_train)
    if y_train is not None: np.save(os.path.join(path, "y_train"), y_train)
    if X_val   is not None: np.save(os.path.join(path, "X_val"),   X_val)
    if y_val   is not None: np.save(os.path.join(path, "y_val"),   y_val)
    if X_test  is not None: np.save(os.path.join(path, "X_test"),  X_test)
    if y_test  is not None: np.save(os.path.join(path, "y_test"),  y_test)

def load_sets(path: str = "data/processed/"):
    """
    Load sets saved via save_sets(); missing files return None.
    """
    def _maybe(name):
        p = os.path.join(path, f"{name}.npy")
        return np.load(p, allow_pickle=True) if os.path.isfile(p) else None

    X_train = _maybe("X_train")
    y_train = _maybe("y_train")
    X_val   = _maybe("X_val")
    y_val   = _maybe("y_val")
    X_test  = _maybe("X_test")
    y_test  = _maybe("y_test")
    return X_train, y_train, X_val, y_val, X_test, y_test

def subset_x_y(target: pd.Series, features: pd.DataFrame, start_index: int, end_index: int):
    """
    Return slices [start_index:end_index] for both features and target.
    """
    return features[start_index:end_index], target[start_index:end_index]

def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: float  = 0.2,
    random_state: int = 42,
):
    """
    Split df into train/val/test on target stratification.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
