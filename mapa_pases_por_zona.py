# mapa_pases_por_zona.py
# Mapa de pases por zona para cualquier QB — datos nflverse PBP.
# Pide el nombre del QB por teclado y genera el mapa automáticamente.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEASON   = 2025
MIN_WEEK = 1
MAX_WEEK = 18

URL = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"

FIGSIZE   = (12.0, 14.5)
DPI       = 200
BG        = "#0e1117"
CARD      = "#141a22"
COL_GREEN  = "#12c79a"
COL_YELLOW = "#f4c542"
COL_RED    = "#e74c3c"
INK        = "#e8edf7"
INK_DIM    = "#9aa6bd"

FS_VAL   = 20
FS_SUB   = 11
FS_LEG   = 10.5
FS_FIRMA = 10.5
FS_YARD  = 11

METRICA = "Comp%"
MARGEN  = 5.0
Y_MAX   = 30.0

RANGE_YDS = {
    "Cortos":    (0.0,  9.0),
    "Medios":    (10.0, 19.0),
    "Profundos": (20.0, 30.0),
}
ZONAS   = ["Izquierda", "Centro", "Derecha"]
COL_X   = {"Izquierda": 0.27, "Centro": 0.50, "Derecha": 0.73}
W_BLOCK = 0.18

INNER_PAD_Y_DEFAULT = 0.004
INNER_PAD_Y_CORTOS  = 0.001

# ── INPUT ─────────────────────────────────────────────────────────────────────
QB_INPUT = input("Nombre del QB (ej: Mahomes, Lamar, Allen): ").strip()

# ── CARGA DE DATOS ────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
df = pd.read_csv(URL, low_memory=False, compression="infer")
df["week"]          = pd.to_numeric(df["week"],          errors="coerce")
df["air_yards"]     = pd.to_numeric(df["air_yards"],     errors="coerce")
df["complete_pass"] = pd.to_numeric(df["complete_pass"], errors="coerce")
df["touchdown"]     = pd.to_numeric(df["touchdown"],     errors="coerce")
df["yards_gained"]  = pd.to_numeric(df["yards_gained"],  errors="coerce")
df["sack"]          = pd.to_numeric(df.get("sack"),      errors="coerce").fillna(0)
df["qb_spike"]      = pd.to_numeric(df.get("qb_spike"),  errors="coerce").fillna(0)
print(f"Filas: {len(df):,}")

# ── FILTRO: PASES VÁLIDOS CON DATOS DE ZONA ───────────────────────────────────
passer_col = "passer_player_name" if "passer_player_name" in df.columns else "passer"

pass_plays = df[
    df["week"].between(MIN_WEEK, MAX_WEEK) &
    (df["play_type"] == "pass") &
    (df["sack"] == 0) &
    (df["qb_spike"] == 0) &
    df[passer_col].notna() &
    df["pass_location"].notna() &
    df["air_yards"].notna()
].copy()

# ── BUSCAR QB ─────────────────────────────────────────────────────────────────
mask = pass_plays[passer_col].str.contains(QB_INPUT, case=False, na=False)
matches = pass_plays[mask][passer_col].value_counts()

if matches.empty:
    raise SystemExit(f"No se encontró ningún QB con '{QB_INPUT}'. Prueba con apellido o iniciales.")

QB_NAME = matches.index[0]
print(f"QB encontrado: {QB_NAME}  ({matches.iloc[0]} jugadas)")

qb_plays = pass_plays[pass_plays[passer_col] == QB_NAME].copy()

# ── CLASIFICAR PROFUNDIDAD Y ZONA ─────────────────────────────────────────────
def classify_depth(ay):
    if ay <= 9:
        return "Cortos"
    elif ay <= 19:
        return "Medios"
    else:
        return "Profundos"

LOC_MAP = {"left": "Izquierda", "middle": "Centro", "right": "Derecha"}

qb_plays["depth"] = qb_plays["air_yards"].apply(classify_depth)
qb_plays["zone"]  = qb_plays["pass_location"].str.lower().map(LOC_MAP)
qb_plays = qb_plays[qb_plays["zone"].notna()]

# ── PROMEDIOS LIGA (para comparación de color) ────────────────────────────────
pass_plays["depth"] = pass_plays["air_yards"].apply(classify_depth)
pass_plays["zone"]  = pass_plays["pass_location"].str.lower().map(LOC_MAP)
league_avg = (
    pass_plays[pass_plays["zone"].notna()]
    .groupby("depth")["complete_pass"]
    .mean()
    .mul(100)
    .to_dict()
)
print("Media liga Comp% por profundidad:", {k: f"{v:.1f}%" for k, v in league_avg.items()})

# ── CONSTRUIR RAW ─────────────────────────────────────────────────────────────
RAW = {}
for prof in ["Cortos", "Medios", "Profundos"]:
    for zona in ZONAS:
        sub = qb_plays[(qb_plays["depth"] == prof) & (qb_plays["zone"] == zona)]
        intentos = len(sub)
        if intentos == 0:
            RAW[(prof, zona)] = {"Intentos": 0, "Comp": 0, "Comp%": 0.0, "Yardas": 0, "TD": 0}
            continue
        comp     = int(sub["complete_pass"].sum())
        comp_pct = round(comp / intentos * 100, 1)
        yardas   = int(sub.loc[sub["complete_pass"] == 1, "yards_gained"].sum())
        tds      = int(sub["touchdown"].sum())
        RAW[(prof, zona)] = {"Intentos": intentos, "Comp": comp, "Comp%": comp_pct,
                             "Yardas": yardas, "TD": tds}

# Resumen consola
print(f"\n{'-'*60}")
print(f"  {QB_NAME} | Mapa de pases | NFL {SEASON}  S{MIN_WEEK}-S{MAX_WEEK}")
print(f"{'-'*60}")
print(f"{'':22} {'Izquierda':>12} {'Centro':>12} {'Derecha':>12}")
for prof in ["Profundos", "Medios", "Cortos"]:
    row = "  ".join(
        f"{RAW[(prof,z)]['Comp%']:>5.1f}% ({RAW[(prof,z)]['Intentos']}att)"
        for z in ZONAS
    )
    print(f"  {prof:<20} {row}")
print()

# ── COLOR POR ZONA ────────────────────────────────────────────────────────────
df_stats = pd.DataFrame(RAW).T.reset_index()
df_stats.columns = ["Profundidad", "Zona"] + list(df_stats.columns[2:])
df_stats["MediaLiga"] = df_stats["Profundidad"].map(league_avg)

def color_for(value, base, tol):
    if np.isnan(value) or np.isnan(base):
        return "#666666"
    if value >= base + tol:
        return COL_GREEN
    if value <= base - tol:
        return COL_RED
    return COL_YELLOW

# ── FIGURA ────────────────────────────────────────────────────────────────────
def yard_to_ax(yards, base_y_ax, top_y_ax):
    return base_y_ax + (top_y_ax - base_y_ax) * (yards / Y_MAX)

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis("off")

# Tarjeta de fondo
ax.add_patch(
    patches.FancyBboxPatch((0.05, 0.06), 0.90, 0.91,
                           boxstyle="round,pad=0.02,rounding_size=0.02",
                           linewidth=0, facecolor=CARD, transform=ax.transAxes)
)

TOP_Y  = 0.84
BASE_Y = 0.185

# Título y subtítulo dentro de la tarjeta
week_label = f"Semanas {MIN_WEEK}-{MAX_WEEK}" if MAX_WEEK < 18 else "Temporada completa"
ax.text(0.50, 0.956, QB_NAME,
        ha="center", va="top", color=INK,
        fontsize=19, fontweight="bold", transform=ax.transAxes)
ax.text(0.50, 0.921, f"Mapa de pases por zona  ·  NFL {SEASON}  ·  {week_label}",
        ha="center", va="top", color=INK_DIM,
        fontsize=10.5, transform=ax.transAxes)

# Regla 0–30
ruler_x = 0.12
ax.plot([ruler_x, ruler_x], [BASE_Y, TOP_Y], color=INK_DIM, lw=2, alpha=0.9, transform=ax.transAxes)
for yv in range(0, 31, 5):
    yy = yard_to_ax(yv, BASE_Y, TOP_Y)
    ax.plot([ruler_x - 0.006, ruler_x], [yy, yy], color=INK_DIM, lw=2, transform=ax.transAxes)
    ax.text(ruler_x - 0.010, yy, f"{yv}", ha="right", va="center",
            color=INK_DIM, fontsize=FS_YARD, transform=ax.transAxes)

# Cabeceras de zona
for zona in ZONAS:
    ax.text(COL_X[zona], TOP_Y + 0.018, zona.upper(),
            ha="center", va="bottom", color=INK_DIM,
            fontsize=10, fontweight="bold", transform=ax.transAxes)

# Bloques
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
        row   = df_stats[(df_stats["Profundidad"] == profundidad) & (df_stats["Zona"] == zona)].iloc[0]
        val   = float(row[METRICA])
        media = float(row["MediaLiga"])
        face  = color_for(val, media, MARGEN)

        cx = COL_X[zona]
        x0 = cx - W_BLOCK / 2

        ax.add_patch(
            patches.FancyBboxPatch((x0, y0_ax), W_BLOCK, H_block_ax,
                                   boxstyle="round,pad=0.012,rounding_size=0.016",
                                   linewidth=0, facecolor=face, transform=ax.transAxes, alpha=0.98)
        )

        att  = int(row["Intentos"])
        cmp_ = int(row["Comp"])
        td   = int(row["TD"])
        y_mid = (y0_ax + y1_ax) / 2

        ax.text(cx, y_mid + H_block_ax * 0.20, f"{val:.1f}%",
                ha="center", va="center", color="white",
                fontsize=FS_VAL, fontweight="bold", transform=ax.transAxes)
        ax.text(cx, y_mid - H_block_ax * 0.15, f"Cmp {cmp_}/{att} • TD {td}",
                ha="center", va="center", color="#0c1219",
                fontsize=FS_SUB, fontweight="bold", transform=ax.transAxes)

# Leyenda
legend_y = 0.07
legend_w = 0.72
legend_h = 0.045
legend_x = 0.14

ax.add_patch(
    patches.FancyBboxPatch((legend_x, legend_y), legend_w, legend_h,
                           boxstyle="round,pad=0.008,rounding_size=0.012",
                           linewidth=0, facecolor="#0c1219", alpha=0.90,
                           transform=ax.transAxes, zorder=6)
)

items   = [("MEJOR QUE LA MEDIA", COL_GREEN),
           ("EN LA MEDIA",        COL_YELLOW),
           ("PEOR QUE LA MEDIA",  COL_RED)]
centers = [legend_x + 0.10, legend_x + legend_w / 2 - 0.03, legend_x + legend_w - 0.20]

for (label, col), cx in zip(items, centers):
    ax.add_patch(patches.FancyBboxPatch(
        (cx - 0.012, legend_y + legend_h / 2 - 0.010), 0.026, 0.020,
        boxstyle="round,pad=0.003,rounding_size=0.006",
        linewidth=0, facecolor=col, transform=ax.transAxes, zorder=7))
    ax.text(cx + 0.020, legend_y + legend_h / 2, label,
            ha="left", va="center", color=INK, fontsize=FS_LEG, fontweight="bold",
            transform=ax.transAxes, zorder=7)

ax.text(0.50, legend_y - 0.018,
        f"Comparado con la media liga por profundidad  ·  Fuente: nflverse PBP {SEASON}",
        ha="center", va="top", color="#555555", fontsize=8, fontstyle="italic",
        transform=ax.transAxes)

ax.text(0.90, 0.042, "@CuartayDato", ha="right", va="center",
        color=INK_DIM, fontsize=FS_FIRMA, alpha=0.9, transform=ax.transAxes)

# ── GUARDAR ───────────────────────────────────────────────────────────────────
slug    = QB_NAME.replace(".", "").replace(" ", "_")
outfile = f"mapa_pases_{slug}_{SEASON}.png"
plt.savefig(outfile, dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Guardado: {outfile}")
