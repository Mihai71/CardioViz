# app/plots.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sigur pentru server
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .figutils import finalize_axes

# --- stil/culori consistente pe tot proiectul ---
COLOR_HEALTHY = "#4C78A8"
COLOR_DISEASE = "#F58518"

# --- helpers interne ---
def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

def _require_two_groups(x0: np.ndarray, x1: np.ndarray, col: str) -> None:
    if len(x0) == 0 or len(x1) == 0:
        raise ValueError(f"Not enough data in both groups for column '{col}'.")

def _weights_percent(x: np.ndarray) -> np.ndarray:
    """Greutăți pentru ca histogramele să fie în % din grup."""
    if len(x) == 0:
        return np.array([])
    return np.ones_like(x, dtype=float) * (100.0 / len(x))

def _draw_thresholds(ax: plt.Axes, thresholds: list[float] | tuple[float, ...] | None, ymax_pad: float = 0.05):
    """Linii verticale de prag + etichete în partea de sus a plotului."""
    if not thresholds:
        return
    ylim = ax.get_ylim()
    y_top = ylim[1] * (1 - ymax_pad)
    for t in thresholds:
        ax.axvline(float(t), linestyle="--", linewidth=1)
        ax.text(float(t), y_top, str(t), rotation=90, va="top", ha="right", fontsize=8)

# --- Phase 3: CORE PLOTS ---

def plot_num_distribution(df: pd.DataFrame, normalize: bool = True, min_n_warn: int = 5):
    """
    Distribuția claselor `num` (0..4).
    - normalize=True: axe Y în procente (și etichetăm n și % pe bare)
    """
    _require_cols(df, ["num"])
    counts = df["num"].value_counts().sort_index()
    labels = [str(i) for i in counts.index.tolist()]
    total = int(counts.sum())
    perc = counts / total * 100.0

    height = perc.values if normalize else counts.values

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(labels, height)

    # etichete pe bare: n (p%)
    for lab, h, n, p, b in zip(labels, height, counts.values, perc.values, bars):
        txt = f"n={n} ({p:.1f}%)" if normalize else f"n={n}"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), txt,
                ha="center", va="bottom", fontsize=9)

    title = f"Class Distribution (num) — N={total}"
    ylabel = "Percent (%)" if normalize else "Count"
    finalize_axes(ax, title=title, xlabel="num (0=healthy, 1–4=severity)", ylabel=ylabel)

    # notă discretă dacă unele stadii au N mic
    small = [lab for lab, n in zip(labels, counts.values) if n < min_n_warn and lab != "0"]
    if small:
        ax.text(0.98, 0.02, f"Caution: low N in stages {', '.join(small)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, alpha=0.75)

    return fig, ax


def plot_hist_overlay(
    df: pd.DataFrame,
    col: str,
    normalize: str = "percent",  # "percent" sau "count"
    thresholds: list[float] | tuple[float, ...] | None = None,
    show_medians: bool = True,
    patient_value: float | None = None,   # <-- nou
):
    """
    Histogramă suprapusă pentru `col`, separat pe healthy vs disease (has_disease).
    - normalize="percent": Y în procente din fiecare grup (comparabil)
    - thresholds: valori verticale marcate (ex. chol 200/240; trestbps 130/140; oldpeak 1/2)
    - patient_value: dacă e dat, trasează linie verticală pentru valoarea "pacientului"
    """
    _require_cols(df, [col, "has_disease"])
    x0 = df.loc[df["has_disease"] == 0, col].dropna().to_numpy()
    x1 = df.loc[df["has_disease"] == 1, col].dropna().to_numpy()
    _require_two_groups(x0, x1, col)

    allx = np.concatenate([x0, x1])
    bins = np.histogram_bin_edges(allx, bins="auto")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if normalize == "percent":
        w0 = _weights_percent(x0)
        w1 = _weights_percent(x1)
        ax.hist(x0, bins=bins, alpha=0.6, label=f"Healthy (n={len(x0)})", weights=w0, color=COLOR_HEALTHY)
        ax.hist(x1, bins=bins, alpha=0.6, label=f"Disease (n={len(x1)})", weights=w1, color=COLOR_DISEASE)
        ylabel = "Percent of group (%)"
    else:
        ax.hist(x0, bins=bins, alpha=0.6, label=f"Healthy (n={len(x0)})", color=COLOR_HEALTHY)
        ax.hist(x1, bins=bins, alpha=0.6, label=f"Disease (n={len(x1)})", color=COLOR_DISEASE)
        ylabel = "Count"

    # mediane pe grupuri
    if show_medians:
        m0 = float(np.median(x0)); m1 = float(np.median(x1))
        ax.axvline(m0, color=COLOR_HEALTHY, linestyle="--", linewidth=1)
        ax.axvline(m1, color=COLOR_DISEASE, linestyle="--", linewidth=1)
        ytop = ax.get_ylim()[1] * 0.95
        ax.text(m0, ytop, f"med={m0:.1f}", color=COLOR_HEALTHY, fontsize=8, ha="right", va="top")
        ax.text(m1, ytop, f"med={m1:.1f}", color=COLOR_DISEASE, fontsize=8, ha="right", va="top")

    # praguri
    _draw_thresholds(ax, thresholds)

    # marker pentru "pacient"
    if patient_value is not None:
        ax.axvline(float(patient_value), color="black", linewidth=2.2)
        ax.text(float(patient_value), ax.get_ylim()[1]*0.85, f"you={patient_value:.1f}",
                rotation=90, ha="right", va="top", fontsize=9)

    ax.legend()
    finalize_axes(ax, title=f"Histogram Overlay — {col}", xlabel=col, ylabel=ylabel)
    return fig, ax



def plot_box_by_outcome(
    df: pd.DataFrame,
    col: str,
    stratify_by: str | None = None,  # ex: "age_bin" pentru perechi Healthy/Disease în fiecare bin
):
    """
    Boxplot pentru `col`:
      - simplu: Healthy vs Disease
      - stratificat: două boxuri per nivel al lui `stratify_by` (ex. age_bin)
    """
    _require_cols(df, [col, "has_disease"])

    # --- varianta simplă (două boxuri) ---
    if stratify_by is None:
        g0 = df.loc[df["has_disease"] == 0, col].dropna().to_numpy()
        g1 = df.loc[df["has_disease"] == 1, col].dropna().to_numpy()
        _require_two_groups(g0, g1, col)

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        bp = ax.boxplot([g0, g1], labels=["Healthy", "Disease"], showfliers=False, patch_artist=True)
        # culori
        for patch, c in zip(bp["boxes"], [COLOR_HEALTHY, COLOR_DISEASE]):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        legend_handles = [Patch(facecolor=COLOR_HEALTHY, alpha=0.6, label="Healthy"),
                          Patch(facecolor=COLOR_DISEASE, alpha=0.6, label="Disease")]
        ax.legend(handles=legend_handles, loc="best")
        finalize_axes(ax, title=f"Boxplot by Outcome — {col}", ylabel=col)
        return fig, ax

    # --- varianta stratificată ---
    if stratify_by not in df.columns:
        raise KeyError(f"Column '{stratify_by}' not found in DataFrame.")

    # Ordinea nivelurilor (dacă e categorică, respectăm ordinea)
    if pd.api.types.is_categorical_dtype(df[stratify_by]):
        levels = [lvl for lvl in df[stratify_by].cat.categories]
    else:
        levels = sorted(df[stratify_by].dropna().unique().tolist())

    data = []
    positions = []
    pos = 0
    width = 0.18  # distanța laterală între healthy/disease în același nivel
    xticks = []
    xticklabels = []
    skipped_levels = []

    for lvl in levels:
        g0 = df.loc[(df["has_disease"] == 0) & (df[stratify_by] == lvl), col].dropna().to_numpy()
        g1 = df.loc[(df["has_disease"] == 1) & (df[stratify_by] == lvl), col].dropna().to_numpy()

        # dacă ambele sunt goale, sărim total nivelul
        if len(g0) == 0 and len(g1) == 0:
            skipped_levels.append(str(lvl))
            continue

        # adăugăm (poate lipsi unul din ele; afișăm ce există)
        if len(g0) > 0:
            data.append(g0); positions.append(pos - width)
        if len(g1) > 0:
            data.append(g1); positions.append(pos + width)

        xticks.append(pos)
        xticklabels.append(str(lvl))
        pos += 1

    if not data:
        raise ValueError(f"No data available to plot '{col}' stratified by '{stratify_by}'.")

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bp = ax.boxplot(data, positions=positions, showfliers=False, patch_artist=True)

    # colorăm alternativ: healthy/disease repetitiv
    for i, patch in enumerate(bp["boxes"]):
        c = COLOR_HEALTHY if (i % 2 == 0) else COLOR_DISEASE
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    legend_handles = [Patch(facecolor=COLOR_HEALTHY, alpha=0.6, label="Healthy"),
                      Patch(facecolor=COLOR_DISEASE, alpha=0.6, label="Disease")]
    ax.legend(handles=legend_handles, loc="best")

    finalize_axes(ax, title=f"Boxplot by Outcome — {col} (stratified by {stratify_by})", ylabel=col)

    if skipped_levels:
        ax.text(0.98, 0.02, f"Skipped empty: {', '.join(skipped_levels)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, alpha=0.75)

    return fig, ax
def plot_corr_heatmap(df: pd.DataFrame, cols: list[str]):
    """
    Heatmap pentru matricea de corelație (Pearson) pe coloanele alese.
    - tratează non-numerice/NaN implicit prin corr(numeric_only=True)
    """
    _require_cols(df, cols)
    corr = df[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Pearson r", rotation=270, labelpad=10)
    finalize_axes(ax, title="Correlation Matrix")
    return fig, ax


def plot_scatter_with_trend(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_by: str = "has_disease",
    degree: int = 1,  # 1 = linie, 2 = polinom
    show_r2: bool = True,
):
    """
    Scatter x vs y + linie/polinom de trend (NumPy polyfit).
    - colorează după `color_by` (ex: has_disease) dacă există
    - afișează ecuația și R² pe datele curate (dropna)
    """
    need = [x, y]
    if color_by:
        need.append(color_by)
    _require_cols(df, need)

    dplot = df[[x, y] + ([color_by] if color_by else [])].dropna()
    if len(dplot) < 2:
        raise ValueError("Not enough data points after dropna for scatter.")

    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    # 1) puncte (colorate pe categorii dacă e cazul)
    if color_by:
        for val, sub in dplot.groupby(color_by):
            c = COLOR_HEALTHY if (val == 0 or str(val).lower() in {"0","healthy"}) else COLOR_DISEASE
            ax.scatter(sub[x].values, sub[y].values, alpha=0.75, label=f"{color_by}={val}", s=18, edgecolors="none", c=c)
        ax.legend()
    else:
        ax.scatter(dplot[x].values, dplot[y].values, alpha=0.75, s=18, edgecolors="none")

    # 2) linia/polinomul de trend pe tot setul curat
    X = dplot[x].values
    Y = dplot[y].values
    deg = max(1, min(3, int(degree)))
    coef = np.polyfit(X, Y, deg=deg)
    p = np.poly1d(coef)
    xs = np.linspace(X.min(), X.max(), 200)
    ax.plot(xs, p(xs), linewidth=2)

    # R^2
    if show_r2:
        yhat = p(X)
        ss_res = float(np.sum((Y - yhat) ** 2))
        ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        eq = " + ".join([f"{c:.3g}·x^{i}" for i, c in zip(range(deg, -1, -1), coef)])
        ax.text(0.02, 0.98, f"{eq}\nR² = {r2:.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9)

    finalize_axes(ax, title=f"Scatter with Trend: {x} vs {y}", xlabel=x, ylabel=y)
    return fig, ax
