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

def _sex_to_label(s: pd.Series) -> pd.Categorical:
    v = s.astype(str).str.strip().str.lower()
    female_mask = v.isin(["0", "f", "female"])
    male_mask   = v.isin(["1", "m", "male"])

    label = pd.Series(index=s.index, dtype="object")
    label[female_mask] = "Female"
    label[male_mask]   = "Male"
    label[~(female_mask | male_mask)] = "Unknown"
    return pd.Categorical(label, categories=["Female", "Male", "Unknown"], ordered=True)

def add_helper_columns(df: pd.DataFrame, add_label: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["has_disease"] = (out["num"] > 0).astype(int)
    out["age_bin"] = pd.cut(
        out["age"],
        bins=[0, 50, 60, 200],
        labels=pd.Categorical(["≤50", "51–60", ">60"], ordered=True),
        right=True,
        include_lowest=True,
    )

    # === NEW: sex_label dacă există coloana 'sex' ===
    if "sex" in out.columns:
        out["sex_label"] = _sex_to_label(out["sex"])

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
# --- simple "what-if" thresholds (no ML) ---

DEFAULT_THRESHOLDS = {
    "chol": (200.0, 240.0),     # normal <200, borderline 200–239, high >=240
    "trestbps": (130.0, 140.0), # normal <130, borderline 130–139, high >=140
    "oldpeak": (1.0, 2.0),      # normal <1, borderline 1–1.99, high >=2
    # poți extinde după nevoi
}

def bucket_by_thresholds(value: float, low: float, high: float) -> str:
    """Returnează 'normal' / 'borderline' / 'high' pe baza pragurilor."""
    if value < low: return "normal"
    if value < high: return "borderline"
    return "high"

def evaluate_patient_simple(values: dict, thresholds: dict | None = None) -> dict:
    """
    Evaluează un "pacient" (fără ML), doar pe baza pragurilor.
    values: ex. {"chol": 245, "trestbps": 138, "oldpeak": 1.2, "age": 54, "thalach": 150}
    thresholds: dict ca DEFAULT_THRESHOLDS (se pot override-ui pragurile).
    Returnează un dict cu status per variabilă + un rezumat.
    """
    th = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    results = {}
    flags_high = 0
    flags_border = 0

    for k, v in values.items():
        if k in th and v is not None:
            low, high = th[k]
            status = bucket_by_thresholds(float(v), float(low), float(high))
            results[k] = {"value": float(v), "thresholds": (float(low), float(high)), "status": status}
            if status == "high": flags_high += 1
            elif status == "borderline": flags_border += 1

    summary = {
        "flags_high": flags_high,
        "flags_borderline": flags_border,
        "note": "This is not a medical diagnosis. Thresholds are heuristic for visualization only."
    }
    return {"per_metric": results, "summary": summary}
