# box_vs_yards.py
# Gráfico de % de box cargados (X) vs yardas/carrera (Y) con logos NFL

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ==== Config ====
LOGO_DIR = "logos"
FIGSIZE = (10, 8)
DPI = 300
BG = "#0f1115"

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

# ==== Datos ====
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
df = pd.DataFrame(data, columns=["Equipo","Box%","YdsCarrera"])

# ==== Gráfico ====
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.scatter(df["Box%"], df["YdsCarrera"], s=40, alpha=0.0)  # invisible, solo para fijar límites

# Añadir logos
def add_logo(team, x, y):
    abbr = TEAM_TO_ABBR.get(team)
    if not abbr:
        return
    path = os.path.join(LOGO_DIR, f"{abbr}.png")
    if not os.path.exists(path):
        return

    img = plt.imread(path)
    h, w = img.shape[0], img.shape[1]
    max_side = max(h, w)
    base_zoom = 0.15
    scale = min(1.0, 180.0 / max_side)
    zoom = base_zoom * scale
    if abbr == "NYJ":
        zoom *= 0.85

    oi = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(oi, (x, y), frameon=False)
    ax.add_artist(ab)

for _, row in df.iterrows():
    add_logo(row["Equipo"], row["Box%"], row["YdsCarrera"])

# Ejes y títulos
ax.set_title("Yardas por carrera vs % de box cargados (NFL 2025)", fontsize=15, weight="bold", pad=14)
ax.set_xlabel("% de box cargados", fontsize=12)
ax.set_ylabel("Yardas por carrera", fontsize=12)

ax.grid(alpha=0.2, color="white", linestyle="--", lw=0.5)

# Límites automáticos con margen
x_pad = (df["Box%"].max() - df["Box%"].min()) * 0.05
y_pad = (df["YdsCarrera"].max() - df["YdsCarrera"].min()) * 0.1
ax.set_xlim(df["Box%"].min()-x_pad, df["Box%"].max()+x_pad)
ax.set_ylim(df["YdsCarrera"].min()-y_pad, df["YdsCarrera"].max()+y_pad)

# Firma
ax.text(df["Box%"].max()+x_pad*0.8, df["YdsCarrera"].min()-y_pad*0.3, "@CuartayDato",
        ha="right", va="center", fontsize=10, color="#8f98ad", alpha=0.9)

plt.tight_layout()
plt.savefig("box_vs_yards.png", dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close()
print("✅ Gráfico generado: box_vs_yards.png")
