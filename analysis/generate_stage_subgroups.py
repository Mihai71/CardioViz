# analysis/generate_stages_subgroups.py
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from app.dataio import load_from_csv, load_from_uci
from app.utils import add_helper_columns
from app.plots import (
    plot_stage_boxplots,
    plot_subgroup_rate_bars,
    plot_stage_distribution_by_group,
)
from app.figutils import save_png

OUT = Path("static/plots")

def get_df():
    p = Path("data/heart.csv")
    df = load_from_csv(p) if p.exists() else load_from_uci()
    return add_helper_columns(df)

if __name__ == "__main__":
    df = get_df()

    # 1) Stage-wise boxplots (exemple pe două variabile)
    for col in ["chol", "oldpeak"]:
        fig, ax = plot_stage_boxplots(df, col, include_healthy=True)
        save_png(fig, OUT / f"stage_box_{col}.png")

    # 2) Subgroup disease rate bars
    fig, ax = plot_subgroup_rate_bars(df, by="age_bin", outcome="has_disease")
    save_png(fig, OUT / "subgroup_rate_agebin.png")

    if "sex_label" in df.columns:
        fig, ax = plot_subgroup_rate_bars(df, by="sex_label", outcome="has_disease")
        save_png(fig, OUT / "subgroup_rate_sex.png")

    # 3) Stacked distribution of stages by subgroup
    fig, ax = plot_stage_distribution_by_group(df, by="age_bin")
    save_png(fig, OUT / "stage_dist_by_agebin.png")

    if "sex_label" in df.columns:
        fig, ax = plot_stage_distribution_by_group(df, by="sex_label")
        save_png(fig, OUT / "stage_dist_by_sex.png")

    print(f"[OK] Stages & Subgroups plots saved in: {OUT.resolve()}")
