# analysis/generate_corr_scatter.py
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dataio import load_from_csv, load_from_uci
from app.utils import add_helper_columns
from app.plots import plot_corr_heatmap, plot_scatter_with_trend
from app.figutils import save_png

# etichete prietenoase + unități pentru afișare
DISPLAY_LABELS = {
    "age": "Age (years)",
    "trestbps": "Resting BP (mmHg)",
    "chol": "Cholesterol (mg/dL)",
    "thalach": "Max HR (bpm)",
    "oldpeak": "ST depression",
}
def L(col: str) -> str:
    return DISPLAY_LABELS.get(col, col)

OUT = Path("static/plots")

def get_df():
    p = Path("data/heart.csv")
    df = load_from_csv(p) if p.exists() else load_from_uci()
    return add_helper_columns(df)

if __name__ == "__main__":
    df = get_df()
    n = len(df)

    # --- HEATMAP ---
    cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, ax = plot_corr_heatmap(df, cols)

    # suprascriem titlul cu N și etichetele cu cele “prietenoase”
    ax.set_title(f"Correlation Matrix (N={n})")
    ax.set_xticklabels([L(c) for c in cols], rotation=45, ha="right")
    ax.set_yticklabels([L(c) for c in cols])

    # (opțional) scriem r numeric în fiecare celulă
    corr = df[cols].corr(numeric_only=True)
    for j in range(len(cols)):          # rânduri (y)
        for i in range(len(cols)):      # coloane (x)
            val = float(corr.iloc[j, i])
            color = "white" if abs(val) >= 0.5 else "black"
            ax.text(i, j, f"{val:+.2f}", ha="center", va="center", fontsize=8, color=color)

    save_png(fig, OUT / "corr_heatmap_core.png")

    # --- SCATTER + TREND ---
    pairs = [("age", "thalach"), ("trestbps", "oldpeak"), ("age", "chol")]
    for x, y in pairs:
        fig, ax = plot_scatter_with_trend(
            df, x=x, y=y, color_by="has_disease", degree=1, show_r2=True
        )
        # etichete cu unități + N în titlu
        ax.set_xlabel(L(x))
        ax.set_ylabel(L(y))
        ax.set_title(f"Scatter with Trend: {L(x)} vs {L(y)} (N={n})")
        save_png(fig, OUT / f"scatter_{x}_vs_{y}.png")

    print(f"[OK] Corr & Scatter saved in: {OUT.resolve()}")
