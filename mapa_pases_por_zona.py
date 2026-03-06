# mapa_pases_nextgen_ranges_y30_fix_sin_LOS_v3.py
# Ajustes:
#  - Leyenda un poco más pequeña.
#  - Bloque amarillo desplazado levemente a la izquierda.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# -----------------------------
# 1) DATOS
# -----------------------------
RAW = {
    ("Cortos", "Izquierda"):  {"Intentos":66, "Comp":52, "Comp%":78.8, "Yardas":434, "TD":2},
    ("Cortos", "Centro"):     {"Intentos":26, "Comp":21, "Comp%":80.8, "Yardas":208, "TD":2},
    ("Cortos", "Derecha"):    {"Intentos":81, "Comp":61, "Comp%":75.3, "Yardas":413, "TD":6},

    ("Medios", "Izquierda"):  {"Intentos":16, "Comp":9,  "Comp%":56.3, "Yardas":137, "TD":0},
    ("Medios", "Centro"):     {"Intentos":11, "Comp":8,  "Comp%":72.7, "Yardas":138, "TD":2},
    ("Medios", "Derecha"):    {"Intentos":18, "Comp":12, "Comp%":66.7, "Yardas":273, "TD":1},

    ("Profundos", "Izquierda"): {"Intentos":11, "Comp":4, "Comp%":36.4, "Yardas":149, "TD":0},
    ("Profundos", "Centro"):    {"Intentos":13, "Comp":6, "Comp%":46.2, "Yardas":182, "TD":0},
    ("Profundos", "Derecha"):   {"Intentos":8,  "Comp":4, "Comp%":50.0, "Yardas":137, "TD":0}
}

# -----------------------------
# 2) PARÁMETROS
# -----------------------------
SEASON       = 2025
METRICA      = "Comp%"
MARGEN       = 5.0
OUTFILE      = f"mapa_pases_{SEASON}.png"

FIGSIZE      = (12.0, 14.0)
DPI          = 200
BG           = "#0e1117"
CARD         = "#141a22"
COL_GREEN    = "#12c79a"
COL_YELLOW   = "#f4c542"
COL_RED      = "#e74c3c"
INK          = "#e8edf7"
INK_DIM      = "#9aa6bd"

FS_VAL   = 20
FS_SUB   = 11
FS_LEG   = 10.5   # reducido para hacer leyenda más pequeña
FS_FIRMA = 10.5
FS_YARD  = 11

Y_MAX = 30.0

RANGE_YDS = {
    "Cortos":     (0.0, 9.0),
    "Medios":     (10.0, 19.0),
    "Profundos":  (20.0, 30.0),
}

ZONAS   = ["Izquierda", "Centro", "Derecha"]
COL_X   = {"Izquierda": 0.27, "Centro": 0.50, "Derecha": 0.73}
W_BLOCK = 0.18

INNER_PAD_Y_DEFAULT = 0.004
INNER_PAD_Y_CORTOS  = 0.001

# -----------------------------
# 3) DATOS Y COLOR
# -----------------------------
df = pd.DataFrame(RAW).T.reset_index()
df.columns = ["Profundidad", "Zona"] + list(df.columns[2:])
media_prof = df.groupby("Profundidad")[METRICA].mean().to_dict()
df["MediaNFL"] = df["Profundidad"].map(media_prof)

def color_for(value, base, tol):
    if np.isnan(value) or np.isnan(base):
        return "#666666"
    if value >= base + tol:
        return COL_GREEN
    if value <= base - tol:
        return COL_RED
    return COL_YELLOW

# -----------------------------
# 4) LAYOUT
# -----------------------------
def yard_to_ax(yards: float, base_y_ax: float, top_y_ax: float) -> float:
    return base_y_ax + (top_y_ax - base_y_ax) * (yards / Y_MAX)

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis("off")

# Tarjeta de fondo
ax.add_patch(
    patches.FancyBboxPatch((0.05, 0.06), 0.90, 0.90,
                           boxstyle="round,pad=0.02,rounding_size=0.02",
                           linewidth=0, facecolor=CARD, transform=ax.transAxes)
)

# Vertical (SIN LOS)
TOP_Y   = 0.86
BASE_Y  = 0.195

# Regla 0–30
ruler_x = 0.12
ax.plot([ruler_x, ruler_x], [BASE_Y, TOP_Y], color=INK_DIM, lw=2, alpha=0.9, transform=ax.transAxes)
for yv in range(0, 31, 5):
    yy = yard_to_ax(yv, BASE_Y, TOP_Y)
    ax.plot([ruler_x-0.006, ruler_x], [yy, yy], color=INK_DIM, lw=2, transform=ax.transAxes)
    ax.text(ruler_x-0.010, yy, f"{yv}", ha="right", va="center",
            color=INK_DIM, fontsize=FS_YARD, transform=ax.transAxes)

# BLOQUES
for profundidad, (y0_yds, y1_yds) in RANGE_YDS.items():
    y0_ax = yard_to_ax(y0_yds, BASE_Y, TOP_Y)
    y1_ax = yard_to_ax(y1_yds, BASE_Y, TOP_Y)

    if y0_yds == 0.0:
        y0_ax += INNER_PAD_Y_CORTOS
        y1_ax -= INNER_PAD_Y_DEFAULT
    else:
        y0_ax += INNER_PAD_Y_DEFAULT
        y1_ax -= INNER_PAD_Y_DEFAULT

    H_block_ax = max(0.05, y1_ax - y0_ax)

    for zona in ZONAS:
        row = df[(df["Profundidad"]==profundidad) & (df["Zona"]==zona)].iloc[0]
        val   = float(row[METRICA])
        media = float(row["MediaNFL"])
        face  = color_for(val, media, MARGEN)

        cx    = COL_X[zona]
        x0    = cx - W_BLOCK/2
        y_mid = (y0_ax + y1_ax) / 2

        ax.add_patch(
            patches.FancyBboxPatch((x0, y0_ax), W_BLOCK, H_block_ax,
                                   boxstyle="round,pad=0.012,rounding_size=0.016",
                                   linewidth=0, facecolor=face, transform=ax.transAxes, alpha=0.98)
        )

        att = int(row["Intentos"]); cmp_ = int(row["Comp"]); td = int(row["TD"])
        ax.text(cx, y_mid + H_block_ax*0.20, f"{val:.1f}%",
                ha="center", va="center", color="white",
                fontsize=FS_VAL, fontweight="bold", transform=ax.transAxes)
        ax.text(cx, y_mid - H_block_ax*0.15, f"Cmp {cmp_}/{att} • TD {td}",
                ha="center", va="center", color="#0c1219",
                fontsize=FS_SUB, fontweight="bold", transform=ax.transAxes)

# -----------------------------
# LEYENDA más pequeña y ajustada
# -----------------------------
legend_y = 0.07
legend_w = 0.72    # antes 0.80 → 10% más pequeña
legend_h = 0.045   # antes 0.056
legend_x = 0.14

ax.add_patch(
    patches.FancyBboxPatch((legend_x, legend_y), legend_w, legend_h,
                           boxstyle="round,pad=0.008,rounding_size=0.012",
                           linewidth=0, facecolor="#0c1219", alpha=0.90, transform=ax.transAxes, zorder=6)
)

items = [("MEJOR QUE LA MEDIA", COL_GREEN),
         ("EN LA MEDIA",        COL_YELLOW),
         ("PEOR QUE LA MEDIA",  COL_RED)]

# Verde izquierda, amarillo un poco desplazado a la izquierda, rojo derecha
centers = [legend_x + 0.10, legend_x + legend_w/2 - 0.03, legend_x + legend_w - 0.20]

for (label, col), cx in zip(items, centers):
    ax.add_patch(patches.FancyBboxPatch((cx-0.012, legend_y + legend_h/2 - 0.010),
                                        0.026, 0.020,
                                        boxstyle="round,pad=0.003,rounding_size=0.006",
                                        linewidth=0, facecolor=col, transform=ax.transAxes, zorder=7))
    ax.text(cx + 0.020, legend_y + legend_h/2, label,
            ha="left", va="center", color=INK, fontsize=FS_LEG, fontweight="bold",
            transform=ax.transAxes, zorder=7)

# Firma
ax.text(0.90, 0.045, "@CuartayDato", ha="right", va="center",
        color=INK_DIM, fontsize=FS_FIRMA, alpha=0.9, transform=ax.transAxes)

plt.savefig(OUTFILE, dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Guardado: {OUTFILE}")
