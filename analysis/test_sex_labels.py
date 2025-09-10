from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from app.dataio import load_from_csv, load_from_uci
from app.utils import add_helper_columns, disease_rate_by

def get_df():
    p = Path("data/heart.csv")
    df = load_from_csv(p) if p.exists() else load_from_uci()
    return add_helper_columns(df)

if __name__ == "__main__":
    df = get_df()
    if "sex_label" in df.columns:
        print("[OK] sex_label present. Sample:")
        print(df[["sex", "sex_label"]].head())
        print("\n[COUNTS]")
        print(df["sex_label"].value_counts(dropna=False))

        print("\n[DISEASE RATE by sex_label]")
        print(disease_rate_by(df, "sex_label"))
    else:
        print("[INFO] 'sex' column not found in dataset → nothing to test here.")
