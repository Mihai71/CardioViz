# app/dataio.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Iterable

REQUIRED_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "num"]
VALID_NUM_VALUES = {0, 1, 2, 3, 4}

def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"[dataio] Dropped {before - after} rows with invalid critical values.")
    return df

def _validate_num_values(df: pd.DataFrame) -> None:
    bad = set(pd.unique(df["num"])) - VALID_NUM_VALUES
    if bad:
        raise ValueError(f"'num' contains invalid values: {sorted(bad)}; expected one of {sorted(VALID_NUM_VALUES)}")

def load_from_uci() -> pd.DataFrame:
    """Load UCI Heart Disease (ID=45) via ucimlrepo, validate & clean."""
    from ucimlrepo import fetch_ucirepo
    hd = fetch_ucirepo(id=45)
    X = hd.data.features
    y = hd.data.targets
    df = pd.concat([X, y], axis=1)

    _validate_columns(df)
    df = _coerce_numeric(df, REQUIRED_COLS)
    df = _drop_invalid_rows(df)
    _validate_num_values(df)
    return df

def load_from_csv(path: str | Path) -> pd.DataFrame:
    """Load local CSV (e.g., data/heart.csv), validate & clean."""
    df = pd.read_csv(path, na_values=["?", "NA", "na", "None", ""])
    _validate_columns(df)
    df = _coerce_numeric(df, REQUIRED_COLS)
    df = _drop_invalid_rows(df)
    _validate_num_values(df)
    return df
