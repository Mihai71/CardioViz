# app/plots.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sigur pentru server
import matplotlib.pyplot as plt
from .figutils import finalize_axes

# --- helpers interne ---
def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

def _require_two_groups(x0: np.ndarray, x1: np.ndarray, col: str) -> None:
    if len(x0) == 0 or len(x1) == 0:
        raise ValueError(f"Not enough data in both groups for column '{col}'.")

# --- Phase 3: CORE PLOTS ---

def plot_num_distribution(df: pd.DataFrame):
    """
    Bar chart pentru distribuția lui `num` (0..4).
    Returnează (fig, ax).
    """
    _require_cols(df, ["num"])
    counts = df["num"].value_counts().sort_index()
    labels = [str(i) for i in counts.index.tolist()]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values)

    # etichete cu valorile pe bare
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), str(int(v)),
                ha="center", va="bottom", fontsize=9)

    finalize_axes(
        ax,
        title="Class Distribution (num)",
        xlabel="num (0=healthy, 1–4=severity)",
        ylabel="Count",
    )
    return fig, ax


def plot_hist_overlay(df: pd.DataFrame, col: str):
    """
    Histogramă suprapusă pentru `col`, separat pe healthy vs disease (has_disease).
    Returnează (fig, ax).
    """
    _require_cols(df, [col, "has_disease"])
    x0 = df.loc[df["has_disease"] == 0, col].dropna().to_numpy()
    x1 = df.loc[df["has_disease"] == 1, col].dropna().to_numpy()
    _require_two_groups(x0, x1, col)

    # aceleași bin-uri pentru ambele grupuri
    allx = np.concatenate([x0, x1])
    bins = np.histogram_bin_edges(allx, bins="auto")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(x0, bins=bins, alpha=0.6, label="Healthy")
    ax.hist(x1, bins=bins, alpha=0.6, label="Has disease")
    ax.legend()

    finalize_axes(ax, title=f"Histogram Overlay — {col}", xlabel=col, ylabel="Frequency")
    return fig, ax


def plot_box_by_outcome(df: pd.DataFrame, col: str):
    """
    Boxplot pentru `col`, separat pe healthy vs disease (has_disease).
    Returnează (fig, ax).
    """
    _require_cols(df, [col, "has_disease"])
    g0 = df.loc[df["has_disease"] == 0, col].dropna().to_numpy()
    g1 = df.loc[df["has_disease"] == 1, col].dropna().to_numpy()
    _require_two_groups(g0, g1, col)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([g0, g1], labels=["Healthy", "Disease"], showfliers=False)
    finalize_axes(ax, title=f"Boxplot by Outcome — {col}", ylabel=col)
    return fig, ax
