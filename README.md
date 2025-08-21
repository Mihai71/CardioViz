CardioViz Web — Exploratory & Clinical-Style Analytics for UCI Heart

Exploratory and clinical-style analytics for the UCI Heart Disease dataset, delivered as a lightweight web app using NumPy, Pandas, Matplotlib, Flask/FastAPI, HTML/CSS/JS. No advanced ML — just transparent, reproducible analytics.

Table of Contents

User-Facing Features

Data Loading

Exploratory Visualizations (Matplotlib on Web UI)

What-if (without ML)

Export

Out of Scope

Dependencies & Setup

Project Structure

Technical Backbone (per module)

dataio.py

utils.py

plots.py

report.py

server.py + routes.py

Chronological Steps (Estimated Hours)

User-Facing Features
Data Loading

Load dataset directly from UCI (via ucimlrepo) or from a local CSV (heart.csv).

Automatic validation of key columns: age, trestbps, chol, thalach, oldpeak, num.

Exploratory Visualizations (Matplotlib on Web UI)

Class distribution (num: 0 = healthy; 1–4 = severity levels).

Histogram + boxplot (healthy vs sick) for user-selected variables.

Correlation matrix for selected variables.

Scatter plots with trend line (NumPy polyfit).

Subgroups: bar charts by sex and by age groups (≤50, 51–60, >60).

“Pseudo-stages” (1–4): comparative plots by severity.

What-if (without ML)

Interactive sliders to adjust thresholds for risk categories (e.g., “high cholesterol”).

Live recalculation of distributions using only NumPy/Pandas (no ML).

Export

Multi-page PDF report (matplotlib.backends.backend_pdf.PdfPages) with clinical-style notes under each chart.

Export of standalone images (PNG).

Save/load session settings (JSON).

Out of Scope

No deep learning, no advanced ML pipelines.

No Streamlit or Seaborn.

Strictly NumPy, Pandas, Matplotlib, Flask/FastAPI, HTML/CSS/JS.

Dependencies & Setup

Python version: 3.10+

Libraries:

Core: numpy, pandas, matplotlib

Data: ucimlrepo (optional, for direct UCI fetch)

Backend: flask or fastapi, jinja2

Utilities: json, pathlib, argparse

(Optional) openpyxl for .xlsx import

Minimum versions (requirements.txt):

numpy>=1.26
pandas>=2.2
matplotlib>=3.8
flask>=3.0
ucimlrepo>=0.0.6
openpyxl>=3.1

Project Structure
cardioviz/
  app/
    __init__.py
    server.py        # Flask/FastAPI entrypoint
    routes.py        # routing logic (API + pages)
    controllers.py   # logic: load, validation, state, interactions
    plots.py         # all matplotlib plotting functions
    report.py        # PdfPages: build multi-page report
    dataio.py        # load UCI/local, save session (JSON)
    utils.py         # binning, summary tables, outcome mapping
  templates/         # HTML templates (Jinja2)
    index.html
    plot.html
  static/            # CSS, JS, generated PNGs
  analysis/
    notebooks/       # optional Jupyter for initial EDA
    figures/         # generated images
  data/
    heart.csv        # optional local copy
  reports/
    CardioViz_Report.pdf  # generated
  README.md
  requirements.txt

Technical Backbone (per module)
dataio.py

load_from_uci() → ucimlrepo.fetch_ucirepo(id=45) → combine X + y.

load_from_csv(path) → pandas.read_csv.

Validation of required columns.

Create helper columns: has_disease = (num > 0).astype(int) and age_bin.

utils.py

Functions for age_bin (≤50, 51–60, >60).

NumPy percentile calculations.

Summary tables for outcomes.

plots.py (each returns fig, ax, no global display)

plot_num_distribution(df)

plot_hist_overlay(df, col)

plot_box_by_outcome(df, col)

plot_corr_heatmap(df, cols) (Matplotlib imshow)

plot_scatter_with_trend(df, x, y, color_by='has_disease') (NumPy polyfit)

plot_stage_boxplots(df, col) (stages 1–4)

report.py

build_pdf(df, selections, path) → assemble multi-page report (title, graphs, clinical notes).

server.py + routes.py

Flask/FastAPI app.

Endpoints:

/       # Home
/plot   # Plot generation/selection
/export # PDF/image export


Templates rendered with Jinja2, charts embedded as PNGs.

Chronological Steps (Estimated Hours)

Phase 0 — Documentation (5h)

Create the project detailed documentation (LaTeX) and Readme file for the GIT repository.

Phase 1 — Setup (2–3h)

Repo + folder structure, virtualenv, requirements.txt.

Initial README.md.

Phase 2 — Data Loading & Validation (4–6h)

Implement dataio.py.

Add validation + preprocessing (utils.py).

Phase 3 — Core Plots (8–12h)

Implement distribution + histogram/boxplots.

Test in standalone scripts.

Phase 4 — Correlation & Scatter (6–8h)

Correlation heatmap.

Scatter with polynomial trend line.

Phase 5 — Stages & Subgroups (6–8h)

Stage-wise boxplots.

Subgroup bar charts (sex, age bins).

Phase 6 — Web UI (12–16h)

Flask app skeleton.

Integrate plots (Matplotlib → PNG → browser).

Add “What-if” sliders for thresholds.

Phase 7 — PDF Report (6–8h)

Generate multi-page report (8–12 pages).

Clinical-style annotations.

Export/download from UI.

Phase 8 — UX & Polish (6–10h)

Error messages, save/load session.

Responsive design (Bootstrap).

Final README.