"""
air_yards_yac.py
Scatter: Air yards promedio (X) vs YAC promedio (Y) por receptor. Toda la temporada.
NFL 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
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

MIN_TARGETS  = 30

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


def short_name(full_name):
    """Convert 'Justin Jefferson' → 'J. Jefferson'."""
    if not isinstance(full_name, str):
        return str(full_name)
    parts = full_name.strip().split()
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


# ── Load data ────────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
df = pd.read_csv(URL, low_memory=False, compression="infer")
to_num(df, ["epa", "air_yards", "yards_after_catch", "week"])

# ── Filter ───────────────────────────────────────────────────────────────────
mask = (
    (df["play_type"] == "pass")
    & df["epa"].notna()
    & df["air_yards"].notna()
    & df["yards_after_catch"].notna()
)
df = df[mask].copy()

receiver_col = pick_col(df, "receiver", "receiver_player_name")
if receiver_col is None:
    raise SystemExit("No se encontró columna de receptor en los datos.")

df = df[df[receiver_col].notna()].copy()

# ── Aggregate ─────────────────────────────────────────────────────────────────
grp = (
    df.groupby(receiver_col, group_keys=False)
    .agg(
        avg_air=("air_yards", "mean"),
        avg_yac=("yards_after_catch", "mean"),
        total_epa=("epa", "sum"),
        epa_per_rec=("epa", "mean"),
        n=("epa", "count"),
    )
    .reset_index()
)
grp = grp[grp["n"] >= MIN_TARGETS].copy()
grp["label"] = grp[receiver_col].apply(short_name)

# ── Normalize sizes & colors ──────────────────────────────────────────────────
n_min, n_max = grp["n"].min(), grp["n"].max()
size_range = n_max - n_min if n_max != n_min else 1
sizes = 40 + 160 * (grp["n"] - n_min) / size_range

vmin = grp["epa_per_rec"].min()
vmax = grp["epa_per_rec"].max()
norm = Normalize(vmin=vmin, vmax=vmax)
colors = RYG(norm(grp["epa_per_rec"].values))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=BG)
ax.set_facecolor(BG)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)

ax.scatter(
    grp["avg_air"],
    grp["avg_yac"],
    s=sizes,
    c=colors,
    edgecolors="none",
    alpha=0.88,
    zorder=3,
)

# Labels with stroke
for _, row in grp.iterrows():
    ax.text(
        row["avg_air"],
        row["avg_yac"] + 0.15,
        row["label"],
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=FG,
        zorder=4,
        path_effects=[
            pe.withStroke(linewidth=2, foreground=BG),
        ],
    )

# Reference lines
mean_air = grp["avg_air"].mean()
mean_yac  = grp["avg_yac"].mean()
ax.axhline(mean_yac,  color=GRID, linewidth=1.0, linestyle="--", zorder=1)
ax.axvline(mean_air, color=GRID, linewidth=1.0, linestyle="--", zorder=1)
ax.axhline(0, color=FG, linewidth=0.4, alpha=0.3, zorder=1)
ax.axvline(0, color=FG, linewidth=0.4, alpha=0.3, zorder=1)

# Grid
ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6, zorder=0)

# Quadrant labels
x_lo, x_hi = ax.get_xlim()
y_lo, y_hi = ax.get_ylim()
quad_kw = dict(fontsize=8, color="#888888", alpha=0.7, fontstyle="italic", zorder=2)
ax.text(x_hi - 0.2, y_hi - 0.2, "Receptores de profundidad",  ha="right", va="top",    **quad_kw)
ax.text(x_lo + 0.2, y_hi - 0.2, "Receptores YAC",              ha="left",  va="top",    **quad_kw)
ax.text(x_lo + 0.2, y_lo + 0.2, "Bajo rendimiento",            ha="left",  va="bottom", **quad_kw)
ax.text(x_hi - 0.2, y_lo + 0.2, "Receptores equilibrados",     ha="right", va="bottom", **quad_kw)

# Axes labels & ticks
ax.set_xlabel("Air yards promedio por recepción", color=FG, fontsize=11)
ax.set_ylabel("YAC promedio por recepción",       color=FG, fontsize=11)
ax.tick_params(colors=FG)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=RYG, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
cbar.set_label("EPA por recepción", color=FG, fontsize=9)
cbar.ax.yaxis.set_tick_params(color=FG)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FG, fontsize=8)
cbar.outline.set_edgecolor(GRID)

# Titles & credits
fig.text(0.5,  0.97, f"Air Yards vs YAC por receptor — NFL {SEASON}",
         ha="center", va="top", fontsize=16, fontweight="bold", color=FG)
fig.text(0.5,  0.92,
         f"min. {MIN_TARGETS} recepciones con datos de air yards | Tamaño = nº recepciones | Color = EPA/recepción",
         ha="center", va="top", fontsize=10, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data  ·  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

fig.tight_layout(rect=[0, 0.03, 1, 0.91])
out = f"air_yards_yac_{SEASON}.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Guardado: {out}")
