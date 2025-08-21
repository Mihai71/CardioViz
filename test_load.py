from app.dataio import load_from_uci, load_from_csv
from app.utils import add_helper_columns
import os

if __name__ == "__main__":
    # Test 1: UCI (necesită internet și ucimlrepo instalat)
    try:
        df = load_from_uci()
        df = add_helper_columns(df)
        print("[UCI] OK | rows:", len(df), "| cols:", len(df.columns))
        print(df[["age","chol","trestbps","thalach","oldpeak","num","has_disease","age_bin"]].head())
    except Exception as e:
        print("[UCI] FAILED:", e)

    # Test 2: local CSV (opțional)
    csv_path = os.path.join("data", "heart.csv")
    if os.path.exists(csv_path):
        try:
            df2 = load_from_csv(csv_path)
            df2 = add_helper_columns(df2)
            print("[CSV] OK | rows:", len(df2), "| cols:", len(df2.columns))
        except Exception as e:
            print("[CSV] FAILED:", e)
    else:
        print("[CSV] Skipped (data/heart.csv not found).")
