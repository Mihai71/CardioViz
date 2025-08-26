# analysis/generate_core_plots.py
from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]   # rădăcina repo-ului (CardioViz)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pathlib import Path
from app.dataio import load_from_csv, load_from_uci
from app.utils import add_helper_columns
from app.plots import plot_num_distribution, plot_hist_overlay, plot_box_by_outcome
from app.figutils import save_png

OUTDIR = Path("static/plots")

THRESHOLDS = {
    "chol": [200, 240],
    "trestbps": [130, 140],
    "oldpeak": [1.0, 2.0],
}

def get_df():
    csv = Path("data/heart.csv")
    df = load_from_csv(csv) if csv.exists() else load_from_uci()
    return add_helper_columns(df)

if __name__ == "__main__":
    df = get_df()

    # 1) Distribuția claselor
    fig, ax = plot_num_distribution(df)
    save_png(fig, OUTDIR / "dist_num.png")

    # 2) Histogramă + 3) Boxplot pentru variabilele cheie
    vars_to_plot = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    for col in vars_to_plot:
        fig, ax = plot_hist_overlay(df, col)
        save_png(fig, OUTDIR / f"hist_{col}.png")

        fig, ax = plot_box_by_outcome(df, col)
        save_png(fig, OUTDIR / f"box_{col}.png")

    print(f"[OK] Core plots generated in: {OUTDIR.resolve()}")
