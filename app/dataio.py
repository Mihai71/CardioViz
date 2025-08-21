import pandas as pd

REQUIRED_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "num"]

def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def load_from_uci() -> pd.DataFrame:
    """Load UCI Heart Disease (ID=45) via ucimlrepo and validate."""
    from ucimlrepo import fetch_ucirepo
    hd = fetch_ucirepo(id=45)
    X = hd.data.features
    y = hd.data.targets
    df = pd.concat([X, y], axis=1)
    _validate_columns(df)
    return df

def load_from_csv(path: str) -> pd.DataFrame:
    """Load local CSV (e.g., data/heart.csv) and validate."""
    df = pd.read_csv(path)
    _validate_columns(df)
    return df