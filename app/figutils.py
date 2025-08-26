# app/figutils.py
from __future__ import annotations
from io import BytesIO
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # backend non-interactiv pentru server
import matplotlib.pyplot as plt

def fig_to_png_bytes(fig, dpi: int = 150) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def save_png(fig, path: str | Path, dpi: int = 150) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def finalize_axes(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None):
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
