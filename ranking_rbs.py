"""
ranking_rbs.py
RBs — EPA/acarreo en Red Zone (X) vs 3er down (Y)
Tamaño del logo proporcional al nº de acarreos totales.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbp_loader import cargar_pbp, cargar_stats
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── Config ────────────────────────────────────────────────────────────────────
SEASON       = None   # None = auto-detectar última temporada
LOGOS_DIR    = "logos"
BG           = "#0f1115"
FG           = "#EDEDED"
GRID         = "#2a2f3a"
DPI          = 170
HARD_PENALTY = {"NYJ": 4.5}

MIN_RZ  = 8    # mínimo acarreos en zona roja
MIN_3RD = 10   # mínimo acarreos en 3er down

ZOOM_MIN = 0.018
ZOOM_MAX = 0.050

# ── Helpers ───────────────────────────────────────────────────────────────────
def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def pick_col(df, *cands):
    for c in cands:
        if c and c in df.columns:
            return c
    return None

def short_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    parts = name.replace("-", " ").split()
    if len(parts) == 1:
        return parts[0][:14]
    return (parts[0][:1] + ". " + parts[-1])[:16]

def load_logo(team, zoom=0.030):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        h, w = img.shape[:2]
        aspect = w / float(h) if h else 1.0
        if team in HARD_PENALTY:
            z = zoom / HARD_PENALTY[team]
        else:
            div = np.clip(1.0 + 0.6 * max(0.0, aspect - 1.3), 1.0, 2.2)
            z = zoom / div
        return OffsetImage(img, zoom=z, resample=True)
    except Exception:
        return None

def volume_zoom(n, n_min, n_max):
    if n_max == n_min:
        return (ZOOM_MIN + ZOOM_MAX) / 2
    t = (n - n_min) / (n_max - n_min)
    return ZOOM_MIN + t * (ZOOM_MAX - ZOOM_MIN)

# ── Datos ─────────────────────────────────────────────────────────────────────
df, SEASON = cargar_pbp(SEASON)
to_num(df, ["epa", "yardline_100", "down"])
print(f"PBP {SEASON}: {len(df):,} jugadas REG")

df_stats, _ = cargar_stats(SEASON)
OUT = f"scatter_RB_RZ_vs_3rd_{SEASON}.png"

name_col_s = pick_col(df_stats, "player_name", "player_display_name")
disp_col_s = pick_col(df_stats, "player_display_name", "player_name")
pos_col_s  = pick_col(df_stats, "position", "pos")

rb_names    = set()
display_map = {}
if name_col_s and pos_col_s:
    for _, row in df_stats[df_stats[pos_col_s].isin({"RB", "FB"})].iterrows():
        sn = row[name_col_s]
        if pd.isna(sn):
            continue
        rb_names.add(sn)
        display_map[sn] = row[disp_col_s] if disp_col_s else sn

rusher_col = pick_col(df, "rusher_player_name", "rusher")
if rusher_col is None:
    raise SystemExit("No se encontró columna de corredor en el PBP.")

rush_df = df[
    (df["play_type"] == "run") &
    df["epa"].notna() &
    df[rusher_col].notna()
].copy()

# ── Métricas por RB ───────────────────────────────────────────────────────────
sub = rush_df[rush_df[rusher_col].isin(rb_names)].copy()

rz  = sub[sub["yardline_100"] <= 20].groupby(rusher_col).agg(
    rz_epa=("epa", "mean"), rz_n=("epa", "count"))
d3  = sub[sub["down"] == 3].groupby(rusher_col).agg(
    d3_epa=("epa", "mean"), d3_n=("epa", "count"))
vol = sub.groupby(rusher_col).agg(total_n=("epa", "count"))

stats = rz.join(d3, how="inner").join(vol, how="inner")
stats = stats[(stats["rz_n"] >= MIN_RZ) & (stats["d3_n"] >= MIN_3RD)].copy()

teams = (
    sub.dropna(subset=[rusher_col, "posteam"])
    .groupby(rusher_col)["posteam"]
    .agg(lambda x: x.mode().iloc[0])
)
stats["team"]  = teams
stats["label"] = [short_name(display_map.get(n, n)) for n in stats.index]

print(f"\nRBs en el scatter: {len(stats)}")

avg_rz = stats["rz_epa"].mean()
avg_d3 = stats["d3_epa"].mean()
n_min  = stats["total_n"].min()
n_max  = stats["total_n"].max()

# ── Gráfico ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 9), dpi=DPI)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)
ax.tick_params(colors=FG, labelsize=9)
plt.setp(ax.get_xticklabels(), color=FG)
plt.setp(ax.get_yticklabels(), color=FG)

x_vals = stats["rz_epa"].values
y_vals = stats["d3_epa"].values
x_pad  = (x_vals.max() - x_vals.min()) * 0.14
y_pad  = (y_vals.max() - y_vals.min()) * 0.14
ax.set_xlim(x_vals.min() - x_pad, x_vals.max() + x_pad)
ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)

ax.axhline(avg_d3, color=GRID, linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)
ax.axvline(avg_rz, color=GRID, linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)
ax.grid(True, linestyle="--", alpha=0.12, color=GRID, zorder=0)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

x_lo, x_hi = ax.get_xlim()
y_lo, y_hi = ax.get_ylim()
xm = (x_hi - x_lo) * 0.03
ym = (y_hi - y_lo) * 0.03
q_kw = dict(fontsize=8.5, alpha=0.30, color=FG, fontstyle="italic")
ax.text(x_hi - xm, y_hi - ym, "Élite en ambas",            ha="right", va="top",    **q_kw)
ax.text(x_lo + xm, y_hi - ym, "Bueno en 3ro / Malo RZ",    ha="left",  va="top",    **q_kw)
ax.text(x_hi - xm, y_lo + ym, "Bueno en RZ / Malo 3ro",    ha="right", va="bottom", **q_kw)
ax.text(x_lo + xm, y_lo + ym, "Peor en ambas",             ha="left",  va="bottom", **q_kw)

y_range      = y_hi - y_lo
label_offset = y_range * 0.038

for rb_key, row in stats.iterrows():
    x    = row["rz_epa"]
    y    = row["d3_epa"]
    team = row["team"]
    name = row["label"]
    zoom = volume_zoom(row["total_n"], n_min, n_max)

    logo = load_logo(str(team) if not pd.isna(team) else "", zoom=zoom)
    if logo:
        ab = AnnotationBbox(logo, (x, y), frameon=False, zorder=3,
                            box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        ax.scatter(x, y, s=60 + 120 * (zoom - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN),
                   color="#888888", zorder=3, alpha=0.8)

    ax.text(x, y - label_offset, name,
            ha="center", va="top", fontsize=7.5, color=FG, alpha=0.88, zorder=4)

ax.set_xlabel("EPA/acarreo en Red Zone", fontsize=11, color=FG, labelpad=7)
ax.set_ylabel("EPA/acarreo en 3er down", fontsize=11, color=FG, labelpad=7)
ax.set_title(f"RBs NFL {SEASON} — Red Zone vs 3er down",
             fontsize=15, pad=12, color=FG, fontweight="bold")

for label, frac in [("Poco volumen", 0.0), ("Volumen medio", 0.5), ("Alto volumen", 1.0)]:
    ax.scatter([], [], s=30 + 120 * frac, color="#555555", label=label, alpha=0.7)
ax.legend(loc="lower right", fontsize=7.5, framealpha=0.2,
          facecolor="#151924", edgecolor=GRID, labelcolor=FG,
          title="Tamaño = nº acarreos", title_fontsize=7)

fig.text(0.5, 0.01,
         f"Fuente: nflverse-data  ·  mín. {MIN_RZ} acarreos en RZ y {MIN_3RD} en 3er down  ·  Líneas = media de la muestra",
         ha="center", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
ax.text(0.99, 0.02, "@CuartayDato", fontsize=9, color="#888888",
        ha="right", va="bottom", transform=ax.transAxes, alpha=0.85, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Guardado: {OUT}")
