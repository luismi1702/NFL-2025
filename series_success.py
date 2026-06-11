"""
series_success.py
Heatmap de eficiencia de drive para los 32 equipos:
% de series que acaban en TD, FG, 1st Down, Punt o Turnover.
Una "serie" = posesión desde que el equipo recibe el balón.
nflfastR registra el resultado de cada serie en series_result.
NFL 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON   = 2025
URL_PBP  = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
BG       = "#0f1115"
FG       = "#EDEDED"
GRID     = "#2a2f3a"
DPI      = 170
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}

# Resultados de serie (series_result en nflfastR)
# nflfastR usa: "Touchdown", "Field goal", "First down", "Punt", "Turnover",
#               "Turnover on downs", "End of half", "End of game", "Opp touchdown" (safety)
CONV_RESULTS  = ["Touchdown", "Field goal", "First down"]   # serie convertida
PUNT_RESULTS  = ["Punt"]
TO_RESULTS    = ["Turnover", "Turnover on downs", "Opp touchdown"]

# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_logo(team, base_zoom=0.038):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        h, w = img.shape[:2]
        aspect = w / float(h) if h else 1.0
        if team in HARD_PENALTY:
            zoom = base_zoom / HARD_PENALTY[team]
        else:
            div = np.clip(1.0 + 0.6 * max(0.0, aspect - 1.3), 1.0, 2.2)
            zoom = base_zoom / div
        return OffsetImage(img, zoom=zoom, resample=True)
    except Exception:
        return None


# ── CARGA DE DATOS ─────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
pbp = pd.read_csv(
    URL_PBP, low_memory=False, compression="infer",
    usecols=["game_id", "posteam", "series", "series_result", "play_type", "week"]
)
print(f"PBP filas: {len(pbp):,}")

# ── FILTRAR ────────────────────────────────────────────────────────────────────
# Quedarse con la última jugada de cada serie (la que tiene el series_result)
plays = pbp[
    pbp["posteam"].notna() &
    pbp["series"].notna() &
    pbp["series_result"].notna()
].copy()

# Una fila por serie: última jugada de cada serie por equipo en cada partido
series_df = (
    plays.sort_values(["game_id", "posteam", "series"])
    .drop_duplicates(subset=["game_id", "posteam", "series"], keep="last")
)

print(f"Series únicas: {len(series_df):,}")

# ── CLASIFICAR RESULTADO ────────────────────────────────────────────────────────
def classify_result(r):
    if r in CONV_RESULTS:
        return r         # "Touchdown", "Field goal", "First down"
    if r in PUNT_RESULTS:
        return "Punt"
    if r in TO_RESULTS:
        return "Turnover"
    return None          # End of half / End of game → no contar

series_df = series_df.copy()
series_df["result_cat"] = series_df["series_result"].apply(classify_result)
series_df = series_df[series_df["result_cat"].notna()].copy()

print(f"Series con resultado clasificable: {len(series_df):,}")

# ── AGREGAR POR EQUIPO ─────────────────────────────────────────────────────────
CATS = ["Touchdown", "Field goal", "First down", "Punt", "Turnover"]

agg = (
    series_df.groupby(["posteam", "result_cat"])
    .size()
    .reset_index(name="n")
)

piv = agg.pivot(index="posteam", columns="result_cat", values="n").fillna(0)
for cat in CATS:
    if cat not in piv.columns:
        piv[cat] = 0
piv = piv[CATS]

piv["total"] = piv[CATS].sum(axis=1)
for cat in CATS:
    piv[f"pct_{cat}"] = piv[cat] / piv["total"] * 100

# Conv% = TD + FG + 1st Down
piv["conv_pct"] = (piv["Touchdown"] + piv["Field goal"] + piv["First down"]) / piv["total"] * 100

# Ordenar por conv_pct descendente
piv = piv.sort_values("conv_pct", ascending=False)

teams   = piv.index.tolist()
n_teams = len(teams)

# ── CONSOLA ────────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  Series Success | NFL {SEASON}  (ordenado por Conv%)")
print(f"{'='*80}")
print(f"{'Equipo':<6} {'Series':>7}  {'Conv%':>6}  {'TD%':>5}  {'FG%':>5}  {'1st%':>5}  {'Punt%':>6}  {'TO%':>5}")
print("-" * 60)
for tm in teams:
    row = piv.loc[tm]
    print(f"{tm:<6} {int(row['total']):>7}  {row['conv_pct']:>5.1f}%  "
          f"{row['pct_Touchdown']:>4.1f}%  {row['pct_Field goal']:>4.1f}%  "
          f"{row['pct_First down']:>4.1f}%  {row['pct_Punt']:>5.1f}%  {row['pct_Turnover']:>4.1f}%")
print()

# ── FIGURA ─────────────────────────────────────────────────────────────────────
# Columnas: Conv% · TD% · FG% · 1st% · Punt% · TO%
COL_DEFS = [
    ("conv_pct",         "Conv%\n(TD+FG+1st)",  "higher_better"),
    ("pct_Touchdown",    "TD%",                  "higher_better"),
    ("pct_Field goal",   "FG%",                  "neutral"),
    ("pct_First down",   "1st%",                 "higher_better"),
    ("pct_Punt",         "Punt%",                "lower_better"),
    ("pct_Turnover",     "TO%",                  "lower_better"),
]
COL_KEYS   = [c[0] for c in COL_DEFS]
COL_LABELS = [c[1] for c in COL_DEFS]
COL_DIR    = [c[2] for c in COL_DEFS]

n_cols  = len(COL_DEFS)
cell_w  = 1.5
cell_h  = 0.52
logo_w  = 1.2
fig_w   = logo_w + n_cols * cell_w + 1.4
fig_h   = max(8, n_teams * cell_h + 2.5)

fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
ax.set_facecolor(BG)
ax.axis("off")
ax.set_xlim(-logo_w, n_cols)
ax.set_ylim(-1, n_teams + 0.8)

# Colormaps por columna
GREEN_RED = plt.cm.RdYlGn      # higher = verde
RED_GREEN = plt.cm.RdYlGn_r    # lower = verde
NEUTRAL   = LinearSegmentedColormap.from_list("neutral", ["#1e2430", "#2d6cdf"])

def get_cmap(direction):
    if direction == "higher_better": return GREEN_RED
    if direction == "lower_better":  return RED_GREEN
    return NEUTRAL

# Normalizar cada columna independientemente
norms = {}
for key in COL_KEYS:
    vals = piv[key].values
    vmin, vmax = vals.min(), vals.max()
    norms[key] = Normalize(vmin=vmin, vmax=vmax if vmax != vmin else vmin + 1)

# ── CELDAS ────────────────────────────────────────────────────────────────────
for row_i, team in enumerate(teams):
    y = n_teams - row_i - 1
    for col_j, (key, label, direction) in enumerate(COL_DEFS):
        val  = piv.loc[team, key]
        x    = col_j
        cmap = get_cmap(direction)
        norm = norms[key]

        bg_color = cmap(norm(val))
        rect = plt.Rectangle((x, y), 1, 1, color=bg_color,
                              linewidth=0.4, edgecolor=BG, zorder=1)
        ax.add_patch(rect)

        txt_color = "#0a0e13" if 0.25 < norm(val) < 0.75 else FG
        ax.text(x + 0.5, y + 0.5, f"{val:.1f}%",
                ha="center", va="center",
                color=txt_color, fontsize=8, fontweight="bold", zorder=2)

# ── LOGOS ──────────────────────────────────────────────────────────────────────
for row_i, team in enumerate(teams):
    y = n_teams - row_i - 1
    img = load_logo(team, base_zoom=0.036)
    if img is not None:
        ab = AnnotationBbox(img, (-logo_w / 2, y + 0.5),
                            frameon=False, zorder=3,
                            box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        ax.text(-logo_w / 2, y + 0.5, team,
                ha="center", va="center",
                color=FG, fontsize=7.5, fontweight="bold")

# ── CABECERAS ─────────────────────────────────────────────────────────────────
for col_j, (key, label, direction) in enumerate(COL_DEFS):
    ax.text(col_j + 0.5, n_teams + 0.35, label,
            ha="center", va="center",
            color=FG, fontsize=8, fontweight="bold", linespacing=1.3)

ax.axhline(n_teams, color=GRID, linewidth=0.8, zorder=3)

# Separador visual tras Conv%
ax.axvline(1, color=GRID, linewidth=1.2, ymin=0, ymax=1, zorder=3)

# ── TÍTULOS ───────────────────────────────────────────────────────────────────
fig.text(0.5, 0.99,
         f"Series Success — Eficiencia de drive | NFL {SEASON}",
         ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
fig.text(0.5, 0.975,
         "% de posesiones por resultado  |  Ordenado por Conv% (TD+FG+1st Down)  |  Excluye final de mitad y final de partido",
         ha="center", va="top", fontsize=8.5, color="#888888", fontstyle="italic")
fig.text(0.01, 0.005, f"Fuente: nflverse-data  |  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.90, 0.005, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

outfile = f"series_success_{SEASON}.png"
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
