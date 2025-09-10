from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from app.dataio import load_from_uci
from app.utils import add_helper_columns

if __name__ == "__main__":
    df = load_from_uci()
    # selectăm coloanele cheie + sex; poți adăuga și altele dacă vrei
    keep = ["age","sex","trestbps","chol","thalach","oldpeak","num"]
    df = df[keep].copy()
    df = add_helper_columns(df, add_label=False)  # adaugă has_disease, age_bin

    out = Path("data/heart.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("[OK] Wrote:", out.resolve())
    print("Columns:", list(df.columns))
