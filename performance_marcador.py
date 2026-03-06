"""
performance_marcador.py
EPA/play por equipo según estado del marcador.
Scatter: EPA ganando vs EPA perdiendo. Tamaño = EPA igualado.
NFL 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── Style constants ─────────────────────────────────────────────────────────
SEASON       = 2025
URL          = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
BG           = "#0f1115"
FG           = "#EDEDED"
GRID         = "#2a2f3a"
DPI          = 200
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}
RYG          = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

MIN_PLAYS = 40

# ── Helpers ──────────────────────────────────────────────────────────────────
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


def load_logo(team, base_zoom=0.055):
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


# ── Load data ────────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
df = pd.read_csv(URL, low_memory=False, compression="infer")
to_num(df, ["epa", "score_differential"])

# ── Filter ───────────────────────────────────────────────────────────────────
score_col = pick_col(df, "score_differential")
if score_col is None:
    raise SystemExit("No se encontró la columna 'score_differential' en los datos.")

df = df[
    df["play_type"].isin(["pass", "run"])
    & df["posteam"].notna()
    & df["epa"].notna()
    & df[score_col].notna()
].copy()

# ── Assign buckets ────────────────────────────────────────────────────────────
conditions = [
    df[score_col] >= 8,
    (df[score_col] >= -7) & (df[score_col] <= 7),
    df[score_col] <= -8,
]
bucket_names = ["Ganando", "Igualado", "Perdiendo"]
df["bucket"] = np.select(conditions, bucket_names, default=None)
df = df[df["bucket"].notna()].copy()

# ── Aggregate ─────────────────────────────────────────────────────────────────
grp = (
    df.groupby(["posteam", "bucket"], group_keys=False)["epa"]
    .agg(epa_play="mean", n_plays="count")
    .reset_index()
)

# Filter buckets with enough plays
grp = grp[grp["n_plays"] >= MIN_PLAYS].copy()

# Pivot: rows = posteam, cols = bucket
pivot = grp.pivot(index="posteam", columns="bucket", values="epa_play")
pivot = pivot.reset_index()

# Keep only teams with all 3 buckets
required_buckets = ["Ganando", "Igualado", "Perdiendo"]
pivot = pivot.dropna(subset=required_buckets).copy()
pivot = pivot.reset_index(drop=True)

if pivot.empty:
    raise SystemExit("No hay equipos con suficientes jugadas en los 3 escenarios de marcador.")

# ── Normalize sizes (based on EPA_Igualado) ──────────────────────────────────
iq_min  = pivot["Igualado"].min()
iq_max  = pivot["Igualado"].max()
iq_range = iq_max - iq_min if iq_max != iq_min else 1
sizes   = 60 + 140 * (pivot["Igualado"] - iq_min) / iq_range

# ── Colors: by EPA_Igualado ───────────────────────────────────────────────────
norm_iq = Normalize(vmin=iq_min, vmax=iq_max)
colors  = [RYG(norm_iq(v)) for v in pivot["Igualado"]]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8.5), facecolor=BG)
ax.set_facecolor(BG)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)
ax.tick_params(colors=FG)
ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5, zorder=0)

# Reference lines
ax.axhline(0, color=FG, linewidth=0.5, alpha=0.3, zorder=1)
ax.axvline(0, color=FG, linewidth=0.5, alpha=0.3, zorder=1)

# Diagonal reference (x = y)
xy_lo = min(pivot["Ganando"].min(), pivot["Perdiendo"].min()) - 0.02
xy_hi = max(pivot["Ganando"].max(), pivot["Perdiendo"].max()) + 0.02
ax.plot([xy_lo, xy_hi], [xy_lo, xy_hi], color=GRID, linewidth=1.0,
        linestyle="--", alpha=0.7, zorder=1)

# Scatter (invisible, sized for reference)
ax.scatter(
    pivot["Ganando"],
    pivot["Perdiendo"],
    s=sizes,
    c=colors,
    edgecolors="none",
    alpha=0.0,   # invisible — replaced by logos
    zorder=2,
)

# Logos or text as markers
for i in range(len(pivot)):
    row  = pivot.iloc[i]
    team = row["posteam"]
    x    = row["Ganando"]
    y    = row["Perdiendo"]
    img  = load_logo(team, base_zoom=0.030)
    if img is not None:
        ab = AnnotationBbox(
            img,
            (x, y),
            frameon=False,
            xycoords="data",
            box_alignment=(0.5, 0.5),
        )
        ax.add_artist(ab)
    else:
        ax.text(
            x, y, team,
            ha="center", va="center",
            color=FG, fontsize=8, fontweight="bold", zorder=4,
        )

# Quadrant labels
x_lo, x_hi = ax.get_xlim()
y_lo, y_hi = ax.get_ylim()
pad_x = (x_hi - x_lo) * 0.02
pad_y = (y_hi - y_lo) * 0.02
quad_kw = dict(fontsize=8.5, color="#888888", alpha=0.75, fontstyle="italic")
ax.text(x_hi - pad_x, y_hi - pad_y, "Buenos en todo",                    ha="right", va="top",    **quad_kw)
ax.text(x_hi - pad_x, y_lo + pad_y, "Solo cuando van ganando",           ha="right", va="bottom", **quad_kw)
ax.text(x_lo + pad_x, y_hi - pad_y, "Remontan / juegan mejor perdiendo", ha="left",  va="top",    **quad_kw)
ax.text(x_lo + pad_x, y_lo + pad_y, "Malos en todo",                     ha="left",  va="bottom", **quad_kw)

ax.set_xlabel("EPA/jugada cuando VAN GANANDO (+8 o más)", color=FG, fontsize=11)
ax.set_ylabel("EPA/jugada cuando VAN PERDIENDO (-8 o más)", color=FG, fontsize=11)

# Colorbar for EPA igualado
sm = plt.cm.ScalarMappable(cmap=RYG, norm=norm_iq)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
cbar.set_label("EPA en partidos igualados", color=FG, fontsize=9)
cbar.ax.yaxis.set_tick_params(color=FG)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FG, fontsize=8)
cbar.outline.set_edgecolor(GRID)

# Titles & credits
fig.text(0.5,  0.97, f"Rendimiento según el marcador — NFL {SEASON}",
         ha="center", va="top", fontsize=16, fontweight="bold", color=FG)
fig.text(0.5,  0.92,
         "Tamaño = EPA en partidos igualados (-7 a +7) | Cada punto = un equipo",
         ha="center", va="top", fontsize=10, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data  ·  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

fig.tight_layout(rect=[0, 0.03, 1, 0.91])
out = f"performance_marcador_{SEASON}.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Guardado: {out}")
