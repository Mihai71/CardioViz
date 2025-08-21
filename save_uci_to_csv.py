# save_uci_to_csv.py
import pandas as pd
from ucimlrepo import fetch_ucirepo
from pathlib import Path

OUT = Path("data/heart.csv")

hd = fetch_ucirepo(id=45)  # UCI Heart Disease (Cleveland)
df = pd.concat([hd.data.features, hd.data.targets], axis=1)

# Păstrăm doar coloanele necesare aplicației
need = ["age", "trestbps", "chol", "thalach", "oldpeak", "num"]
df = df[need].copy()

# Curățăm tipurile (siguranță) și salvăm
for c in need:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=need).reset_index(drop=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Salvat: {OUT} | rows={len(df)} cols={len(df.columns)}")
print(df.head())
