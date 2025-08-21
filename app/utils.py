# app/utils.py
import pandas as pd

def add_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns used throughout the app (binary outcome + age bins)."""
    out = df.copy()
    out["has_disease"] = (out["num"] > 0).astype(int)
    out["age_bin"] = pd.cut(
        out["age"],
        bins=[0, 50, 60, 200],
        labels=["≤50", "51–60", ">60"],
        right=True,
        include_lowest=True
    )
    return out
