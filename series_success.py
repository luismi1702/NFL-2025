"""
series_success.py
Heatmap de eficiencia de series para los 32 equipos:
% de series que acaban en TD, FG, 1st Down, Punt o Fallo (TO/downs/FGx/safety).
Una "serie" = una cadena de downs: cada 1er down conseguido abre una serie
nueva dentro del mismo drive (no confundir con posesion/drive).
nflfastR registra el resultado de cada serie en series_result.
NFL 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from pbp_loader import cargar_pbp, salida, season_cli, sello
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON   = season_cli()   # None = auto-detectar última temporada
BG       = "#0f1115"
FG       = "#EDEDED"
GRID     = "#2a2f3a"
DPI      = 170
LOGOS_DIR    = "logos"

# Resultados de serie (series_result en nflfastR)
# nflfastR usa: "Touchdown", "Field goal", "First down", "Punt", "Turnover",
#               "Turnover on downs", "Missed field goal", "Safety",
#               "Opp touchdown" (pick-six/fumble-six), "QB kneel",
#               "End of half", "End of game"
# "Fallo" agrupa todas las series que acaban regalando el balon sin puntos:
# turnover, turnover on downs, FG fallado, safety y TD defensivo del rival.
CONV_RESULTS  = ["Touchdown", "Field goal", "First down"]   # serie convertida
PUNT_RESULTS  = ["Punt"]
TO_RESULTS    = ["Turnover", "Turnover on downs", "Opp touchdown",
                 "Missed field goal", "Safety"]

# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_logo(team, base_zoom=0.038):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        # Recorta margenes transparentes: algunos archivos traen mucho aire
        # (NYJ: tinta 3768x1186 en lienzo 4096x4096) y sin recorte salen enanos
        if img.ndim == 3 and img.shape[2] == 4:
            ys, xs = np.where(img[:, :, 3] > 0.02)
            if len(ys):
                img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = img.shape[:2]
        # Normaliza por el area de tinta real; los wordmarks apaisados
        # pueden ensancharse hasta 1.8x para compensar su poca altura
        zoom = base_zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * zoom > 900.0 * (base_zoom):
            zoom = 900.0 * (base_zoom) / w
        return OffsetImage(img, zoom=zoom, resample=True)
    except Exception:
        return None


# ── CARGA DE DATOS ─────────────────────────────────────────────────────────────
pbp, SEASON = cargar_pbp(SEASON, columns=["game_id", "posteam", "series",
                                          "series_result", "play_type", "week"])
print(f"PBP {SEASON}: {len(pbp):,} jugadas REG")

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
    return None          # End of half / End of game / QB kneel → no contar

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
print(f"{'Equipo':<6} {'Series':>7}  {'Conv%':>6}  {'TD%':>5}  {'FG%':>5}  {'1st%':>5}  {'Punt%':>6}  {'Fallo%':>6}")
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
    ("pct_Turnover",     "Fallo%\n(TO+FGx+Sfty)", "lower_better"),
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
fig.subplots_adjust(left=0.02, right=0.98, top=0.955, bottom=0.015)
ax.set_facecolor(BG)
ax.axis("off")
ax.set_xlim(-logo_w, n_cols)
ax.set_ylim(-0.35, n_teams + 0.9)

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

        _r, _g, _b = bg_color[0], bg_color[1], bg_color[2]
        txt_color = "#0a0e13" if (0.299*_r + 0.587*_g + 0.114*_b) > 0.45 else FG
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

# Separador visual tras Conv% (solo a lo alto de la tabla)
ax.plot([1, 1], [0, n_teams], color=GRID, linewidth=1.2, zorder=3)

# ── TÍTULOS ───────────────────────────────────────────────────────────────────
fig.text(0.5, 0.99,
         f"Series Success — Eficiencia de series | NFL {SEASON}",
         ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
fig.text(0.5, 0.975,
         "% de series ofensivas por resultado (cada 1er down abre una serie nueva)  |  Ordenado por Conv% (TD+FG+1st Down)  |  "
         "Fallo = TO, downs, FG fallado, safety  |  Excluye final de mitad/partido y kneels",
         ha="center", va="top", fontsize=8.5, color="#888888", fontstyle="italic")
fig.text(0.01, 0.005, f"Fuente: nflverse-data  |  {sello(SEASON)}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.90, 0.005, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

outfile = salida(f"series_success_{SEASON}.png", SEASON)
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
