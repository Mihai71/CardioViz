from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dataio import load_from_csv, load_from_uci
from app.utils import add_helper_columns, evaluate_patient_simple, DEFAULT_THRESHOLDS
from app.plots import plot_hist_overlay
from app.figutils import save_png

PATIENT = {"age": 54, "trestbps": 138, "chol": 245, "thalach": 150, "oldpeak": 1.2}
OUT = Path("static/plots/patient")

def get_df():
    csv = Path("data/heart.csv")
    df = load_from_csv(csv) if csv.exists() else load_from_uci()
    return add_helper_columns(df)

if __name__ == "__main__":
    df = get_df()

    # 1) evaluare simplă
    eval_res = evaluate_patient_simple(PATIENT, DEFAULT_THRESHOLDS)
    print("[EVAL]", eval_res)

    # 2) histograme cu marker "you"
    TH = {"chol": (200,240), "trestbps": (130,140), "oldpeak": (1.0,2.0)}
    OUT.mkdir(parents=True, exist_ok=True)
    for col in ["chol", "trestbps", "thalach", "oldpeak"]:
        fig, ax = plot_hist_overlay(
            df, col,
            normalize="percent",
            thresholds=TH.get(col),
            show_medians=True,
            patient_value=PATIENT.get(col),
        )
        save_png(fig, OUT / f"patient_{col}.png")
    print(f"[OK] Wrote patient charts to: {OUT.resolve()}")
