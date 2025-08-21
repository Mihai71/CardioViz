# app/utils.py
from __future__ import annotations
import pandas as pd

OUTCOME_LABELS = {
    0: "Healthy",
    1: "Stage 1",
    2: "Stage 2",
    3: "Stage 3",
    4: "Stage 4",
}

def add_helper_columns(df: pd.DataFrame, add_label: bool = True) -> pd.DataFrame:
    """
    Add columns used throughout the app:
      - has_disease: 1 if num>0 else 0
      - age_bin: ≤50, 51–60, >60 (categorică ordonată)
      - num_label: eticheta text pentru num (opțional)
    """
    out = df.copy()
    out["has_disease"] = (out["num"] > 0).astype(int)
    out["age_bin"] = pd.cut(
        out["age"],
        bins=[0, 50, 60, 200],
        labels=pd.Categorical(["≤50", "51–60", ">60"], ordered=True),
        right=True,
        include_lowest=True,
    )
    if add_label:
        out["num_label"] = out["num"].map(OUTCOME_LABELS).astype("category")
    return out

def outcome_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returnează un tabel cu distribuția lui `num` (count + procent),
    sortat crescător după severitate (0..4).
    """
    s = df["num"].value_counts(dropna=False).sort_index()
    total = int(s.sum())
    pct = (s / total * 100).round(2)
    out = pd.DataFrame({"count": s, "percent": pct})
    out.index.name = "num"
    out["label"] = out.index.map(OUTCOME_LABELS)
    cols = ["label", "count", "percent"]
    return out[cols].reset_index(drop=False)

def disease_rate_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """
    Rată boală (mean of has_disease) și count pe un subgrup (ex. 'age_bin' sau 'sex' dacă există).
    Returnează percent (0-100) rotunjit la 2 zecimale.
    """
    if by not in df.columns:
        raise KeyError(f"Column '{by}' not found in DataFrame.")
    g = df.groupby(by, dropna=False, observed=True)["has_disease"]
    out = pd.DataFrame({
        "count": g.size(),
        "disease_rate_percent": (g.mean() * 100).round(2),
    }).reset_index()
    return out
