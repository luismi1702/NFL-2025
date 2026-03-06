# box_vs_yards.py
# Gráfico de % de box cargados (X) vs yardas/carrera (Y) con logos NFL

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# === Config ===
SEASON   = 2025
LOGO_DIR = "logos"
FIGSIZE  = (10, 8)
DPI      = 200
BG       = "#0f1115"
FG       = "#EDEDED"
GRID     = "#2a2f3a"
OUT      = f"box_vs_yards_{SEASON}.png"

HARD_PENALTY = {"NYJ": 4.5}

TEAM_TO_ABBR = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LA","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
    "Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS",
}

# ==== Datos (hardcoded — box% no es columna estándar nflverse) ====
data = [
    ["Seattle Seahawks",56.21,3.52],
    ["Buffalo Bills",53.09,4.16],
    ["Baltimore Ravens",47.18,4.06],
    ["Detroit Lions",45.71,4.35],
    ["Los Angeles Chargers",45.14,4.00],
    ["San Francisco 49ers",44.72,2.01],
    ["Cleveland Browns",44.68,2.71],
    ["Arizona Cardinals",44.22,4.98],
    ["Houston Texans",42.86,3.07],
    ["Washington Commanders",41.35,4.95],
    ["New England Patriots",41.14,2.51],
    ["Los Angeles Rams",41.10,2.80],
    ["Indianapolis Colts",41.04,2.63],
    ["Green Bay Packers",38.31,2.46],
    ["Pittsburgh Steelers",37.90,2.94],
    ["Denver Broncos",36.90,3.35],
    ["Philadelphia Eagles",36.42,1.92],
    ["Carolina Panthers",35.47,4.08],
    ["Chicago Bears",35.19,2.58],
    ["Tennessee Titans",33.07,2.71],
    ["New York Jets",32.93,5.78],
    ["Minnesota Vikings",32.77,4.10],
    ["Tampa Bay Buccaneers",31.29,1.92],
    ["Dallas Cowboys",30.56,4.11],
    ["Miami Dolphins",30.51,3.08],
    ["Las Vegas Raiders",30.19,2.83],
    ["New Orleans Saints",30.12,3.14],
    ["Kansas City Chiefs",29.22,3.22],
    ["Atlanta Falcons",28.35,2.67],
    ["Cincinnati Bengals",27.78,1.17],
    ["Jacksonville Jaguars",26.88,5.02],
    ["New York Giants",24.86,3.09],
]
df = pd.DataFrame(data, columns=["Equipo", "Box%", "YdsCarrera"])


def load_logo(abbr, base_zoom=0.15):
    path = os.path.join(LOGO_DIR, f"{abbr}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        h, w = img.shape[:2]
        aspect = w / float(h) if h else 1.0
        if abbr in HARD_PENALTY:
            zoom = base_zoom / HARD_PENALTY[abbr]
        else:
            div = np.clip(1.0 + 0.6 * max(0.0, aspect - 1.3), 1.0, 2.2)
            zoom = base_zoom / div
        return OffsetImage(img, zoom=zoom, resample=True)
    except Exception:
        return None


# ==== Gráfico ====
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)
ax.tick_params(colors=FG)
ax.xaxis.label.set_color(FG)
ax.yaxis.label.set_color(FG)
ax.title.set_color(FG)

ax.scatter(df["Box%"], df["YdsCarrera"], s=40, alpha=0.0)  # invisible, solo para fijar límites

for _, row in df.iterrows():
    abbr = TEAM_TO_ABBR.get(row["Equipo"])
    if abbr:
        im = load_logo(abbr)
        if im is not None:
            ab = AnnotationBbox(im, (row["Box%"], row["YdsCarrera"]), frameon=False)
            ax.add_artist(ab)

# Ejes y límites
x_pad = (df["Box%"].max() - df["Box%"].min()) * 0.06
y_pad = (df["YdsCarrera"].max() - df["YdsCarrera"].min()) * 0.12
ax.set_xlim(df["Box%"].min() - x_pad, df["Box%"].max() + x_pad)
ax.set_ylim(df["YdsCarrera"].min() - y_pad, df["YdsCarrera"].max() + y_pad)

ax.set_title(f"Yardas por carrera vs % de box cargados (NFL {SEASON})", fontsize=15, weight="bold", pad=14, color=FG)
ax.set_xlabel("% de box cargados", fontsize=12, color=FG, labelpad=6)
ax.set_ylabel("Yardas por carrera", fontsize=12, color=FG, labelpad=6)
ax.grid(alpha=0.2, color=GRID, linestyle="--", lw=0.5)

# Lineas de promedio
ax.axvline(df["Box%"].mean(), color=GRID, linewidth=0.8, linestyle=":")
ax.axhline(df["YdsCarrera"].mean(), color=GRID, linewidth=0.8, linestyle=":")

# Fuente y firma
fig.text(0.01, 0.01, f"Datos: temporada {SEASON}  |  Box% hardcoded",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

plt.tight_layout()
plt.savefig(OUT, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"Guardado: {OUT}")
