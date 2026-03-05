# coberturas_por_equipo_visual.py
# Barras apiladas + logos (modo oscuro) con orden fijo de segmentos:
# COVER 1 -> COVER 2 -> COVER 3 -> COVER 4 (si no está en el top3, vale 0%)

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ==== Config ====
LOGO_DIR = "logos"
FIGSIZE = (14, 16)
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

COLOR = {
    "COVER 1": "#ff6b6b",
    "COVER 2": "#f4a261",
    "COVER 3": "#2a9d8f",
    "COVER 4": "#457b9d",
}
ORDER = ["COVER 1","COVER 2","COVER 3","COVER 4"]

# ==== Datos ====
data = [
    ["Arizona Cardinals","COVER 4",29.28,"COVER 3",25.23,"COVER 1",17.57],
    ["Atlanta Falcons","COVER 3",39.52,"COVER 1",29.03,"COVER 4",19.35],
    ["Baltimore Ravens","COVER 1",29.44,"COVER 3",28.50,"COVER 2",12.62],
    ["Buffalo Bills","COVER 3",30.00,"COVER 2",21.88,"COVER 1",13.75],
    ["Carolina Panthers","COVER 3",41.32,"COVER 2",18.56,"COVER 4",16.17],
    ["Chicago Bears","COVER 2",26.98,"COVER 1",23.81,"COVER 3",15.87],
    ["Cincinnati Bengals","COVER 3",33.51,"COVER 1",20.62,"COVER 2",16.49],
    ["Cleveland Browns","COVER 1",36.65,"COVER 3",23.60,"COVER 2",20.50],
    ["Dallas Cowboys","COVER 3",34.83,"COVER 2",25.37,"COVER 4",15.92],
    ["Denver Broncos","COVER 1",38.07,"COVER 3",18.27,"COVER 2",13.20],
    ["Detroit Lions","COVER 3",31.18,"COVER 1",30.65,"COVER 4",19.35],
    ["Green Bay Packers","COVER 3",38.15,"COVER 2",21.39,"COVER 1",17.92],
    ["Houston Texans","COVER 3",34.10,"COVER 4",21.97,"COVER 1",17.92],
    ["Indianapolis Colts","COVER 3",30.30,"COVER 1",21.72,"COVER 2",18.69],
    ["Jacksonville Jaguars","COVER 3",34.26,"COVER 2",17.59,"COVER 1",15.28],
    ["Kansas City Chiefs","COVER 3",23.20,"COVER 2",22.10,"COVER 1",15.47],
    ["Las Vegas Raiders","COVER 3",47.74,"COVER 4",21.94,"COVER 2",10.97],
    ["Los Angeles Chargers","COVER 3",32.98,"COVER 4",25.13,"COVER 1",15.18],
    ["Los Angeles Rams","COVER 3",35.35,"COVER 2",18.69,"COVER 1",17.68],
    ["Miami Dolphins","COVER 2",27.44,"COVER 3",23.78,"COVER 1",19.51],
    ["Minnesota Vikings","COVER 2",32.50,"COVER 3",20.00,"COVER 1",11.88],
    ["New England Patriots","COVER 1",30.77,"COVER 3",26.04,"COVER 2",17.16],
    ["New Orleans Saints","COVER 3",43.68,"COVER 4",20.11,"COVER 1",13.22],
    ["New York Giants","COVER 1",37.68,"COVER 4",18.84,"COVER 3",17.39],
    ["New York Jets","COVER 1",27.33,"COVER 3",23.60,"COVER 4",19.25],
    ["Philadelphia Eagles","COVER 1",30.26,"COVER 3",25.64,"COVER 4",20.51],
    ["Pittsburgh Steelers","COVER 3",38.65,"COVER 1",26.99,"COVER 2",11.04],
    ["San Francisco 49ers","COVER 3",33.16,"COVER 4",29.95,"COVER 1",16.04],
    ["Seattle Seahawks","COVER 3",25.46,"COVER 2",20.83,"COVER 4",18.98],
    ["Tampa Bay Buccaneers","COVER 3",29.84,"COVER 4",19.37,"COVER 1",17.28],
    ["Tennessee Titans","COVER 3",25.14,"COVER 2",24.00,"COVER 4",20.00],
    ["Washington Commanders","COVER 3",27.53,"COVER 1",26.40,"COVER 2",16.85],
]
df = pd.DataFrame(data, columns=[
    "Equipo","Cov1","Pct1","Cov2","Pct2","Cov3","Pct3"
])

# Mapa por equipo -> {coverage: porcentaje} con orden fijo
def build_cover_map(row):
    m = {row["Cov1"]: row["Pct1"], row["Cov2"]: row["Pct2"], row["Cov3"]: row["Pct3"]}
    return {c: float(m.get(c, 0.0)) for c in ORDER}

cover_maps = {r["Equipo"]: build_cover_map(r) for _, r in df.iterrows()}

# Orden de equipos: mantenemos la lista original o si prefieres,
# ordénalos por la suma total (opcional).
orden = df["Equipo"].tolist()

# === Plot ===
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Límites y tamaños
left_margin_for_logos = -9.0
totales = [sum(cover_maps[t].values()) for t in orden]
right_limit = max(65, max(totales) + 12)
ax.set_xlim(left_margin_for_logos, right_limit)
ax.set_ylim(-1, len(orden))
bar_height = 0.6

# Barras apiladas con orden fijo COVER1->COVER4
for i, team in enumerate(orden):
    x_left = 0.0
    for cov in ORDER:
        pct = cover_maps[team][cov]
        color = COLOR.get(cov, "#888")
        ax.barh(i, pct, left=x_left, height=bar_height, color=color, alpha=0.95)
        if pct >= 8:
            ax.text(x_left + pct/2, i, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=9, color="#0f1115", fontweight="bold")
        x_left += pct

# Títulos y ejes
ax.set_title("Coberturas principales por equipo (NFL 2025)", fontsize=16, weight="bold", pad=14)
ax.set_xlabel("Porcentaje de snaps (%)", fontsize=12, labelpad=8)
ax.set_yticks(range(len(orden)))
ax.set_yticklabels([""]*len(orden))
ax.grid(False)

# Leyenda (en el mismo orden fijo)
legend_handles = [ax.barh(-10, 0, color=COLOR[c], label=c) for c in ORDER]
ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
          ncol=4, frameon=False, title="Cobertura", fontsize=10, title_fontsize=11)

# --- Logos (más grandes + fix NYJ) ---
def add_logo(team, y_pos):
    abbr = TEAM_TO_ABBR.get(team)
    if not abbr:
        return
    path = os.path.join(LOGO_DIR, f"{abbr}.png")
    if not os.path.exists(path):
        return

    img = plt.imread(path)
    h, w = img.shape[0], img.shape[1]
    max_side = max(h, w)
    base_zoom = 0.125
    scale = min(1.0, 180.0 / max_side)
    zoom = base_zoom * scale
    if abbr == "NYJ":
        zoom *= 0.85  # normaliza Jets si su canvas es grande

    oi = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(oi, (left_margin_for_logos + 2.2, y_pos), frameon=False)
    ax.add_artist(ab)

for i, team in enumerate(orden):
    add_logo(team, i)

# Firma
ax.text(right_limit - 0.8, -0.8, "@CuartayDato", ha="right", va="center",
        fontsize=10, color="#8f98ad", alpha=0.9)

plt.tight_layout()
plt.savefig("coberturas_por_equipo_visual.png", dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close()
print("✅ Gráfico generado: coberturas_por_equipo_visual.png")
