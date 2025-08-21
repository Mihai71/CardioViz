from app.dataio import load_from_uci, load_from_csv
from app.utils import add_helper_columns, outcome_summary, disease_rate_by
import os

CRIT = ["age","chol","trestbps","thalach","oldpeak","num"]

def print_types(df):
    print("\n[DTYPES]\n", df[CRIT].dtypes)

def print_summaries(df):
    print("\n[OUTCOME SUMMARY]\n", outcome_summary(df))
    try:
        print("\n[DISEASE RATE BY age_bin]\n", disease_rate_by(df, "age_bin"))
    except Exception as e:
        print("[age_bin] summary failed:", e)

if __name__ == "__main__":
    # Test 1: UCI (necesită internet și ucimlrepo instalat)
    try:
        df = load_from_uci()
        df = add_helper_columns(df)
        print("[UCI] OK | rows:", len(df), "| cols:", len(df.columns))
        print(df[["age","chol","trestbps","thalach","oldpeak","num","has_disease","age_bin"]].head())
        print_types(df)
        print_summaries(df)
    except Exception as e:
        print("[UCI] FAILED:", e)

    # Test 2: local CSV (opțional)
    csv_path = os.path.join("data", "heart.csv")
    if os.path.exists(csv_path):
        try:
            df2 = load_from_csv(csv_path)
            df2 = add_helper_columns(df2)
            print("\n[CSV] OK | rows:", len(df2), "| cols:", len(df2.columns))
            print_types(df2)
            print_summaries(df2)
        except Exception as e:
            print("[CSV] FAILED:", e)
    else:
        print("[CSV] Skipped (data/heart.csv not found).")
