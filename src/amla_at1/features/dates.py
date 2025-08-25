from __future__ import annotations
import re
import numpy as np
import pandas as pd

def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(a, b, where=b!=0)
        out[~np.isfinite(out)] = 0.0
    return out

def parse_height_to_inches(ht_value: str) -> float:
    """
    Convert height strings to inches. Handles odd samples like '6-Jun' -> '6-10'.
    Accepts forms: 6-10, 6'10", 6'10, 6, '6ft 10in', numeric inches.
    """
    if ht_value is None or pd.isna(ht_value):
        return 0.0
    s = str(ht_value).strip().lower()
    s = re.sub(r"-\s*jun\b", "-10", s)  # heuristic for '-Jun'
    s = s.replace("ft", "'").replace("in", '"').replace(" ", "")

    m = re.match(r"^(\d+)[\'-](\d+)", s)
    if m:
        return float(int(m.group(1)) * 12 + int(m.group(2)))
    m2 = re.match(r"^(\d+)'?$", s)  # just feet like "6"
    if m2:
        return float(int(m2.group(1)) * 12)
    if re.match(r"^\d+(\.\d+)?$", s):
        return float(s)  # already inches
    return 0.0

def yr_to_ordinal(yr: str) -> int:
    """
    Fr=1, So=2, Jr=3, Sr=4; else 0.
    """
    mapping = {"fr":1, "so":2, "jr":3, "sr":4}
    if yr is None or pd.isna(yr):
        return 0
    return mapping.get(str(yr).strip().lower(), 0)

def normalize_type_flags(type_val: str):
    """
    Return (is_regular, is_post) from 'type'.
    """
    if type_val is None or pd.isna(type_val):
        return False, False
    t = str(type_val).strip().lower()
    is_reg  = t in {"r", "regular", "all"}
    is_post = t in {"p", "t", "ncaa", "post"}
    return is_reg, is_post

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for the NBA draft dataset.
    - ht -> inches
    - yr -> ordinal
    - type -> flags
    - shooting ratios (FTM/FTA, twoPM/twoPA, TPM/TPA)
    - per-minute rates: pts/mp, reb/mp, ast/mp
    """
    out = df.copy()

    if "ht" in out.columns:
        out["height_in"] = out["ht"].apply(parse_height_to_inches)

    if "yr" in out.columns:
        out["yr_ordinal"] = out["yr"].apply(yr_to_ordinal)

    if "type" in out.columns:
        flags = out["type"].apply(normalize_type_flags)
        out["is_regular"] = flags.apply(lambda x: x[0])
        out["is_post"]    = flags.apply(lambda x: x[1])

    # Ratios
    if {"FTM","FTA"}.issubset(out.columns):
        out["ft_ratio"] = _safe_div(out["FTM"], out["FTA"])
    if {"twoPM","twoPA"}.issubset(out.columns):
        out["twoP_ratio"] = _safe_div(out["twoPM"], out["twoPA"])
    if {"TPM","TPA"}.issubset(out.columns):
        out["tp_ratio"] = _safe_div(out["TPM"], out["TPA"])

    # Minutes-normalized
    if {"pts","mp"}.issubset(out.columns):
        out["pts_per_min"] = _safe_div(out["pts"], out["mp"])
    if {"treb","mp"}.issubset(out.columns):
        out["reb_per_min"] = _safe_div(out["treb"], out["mp"])
    if {"ast","mp"}.issubset(out.columns):
        out["ast_per_min"] = _safe_div(out["ast"], out["mp"])

    return out
