"""
qb_presion.py
QB performance: pocket limpio vs bajo presion.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON            = 2025
MIN_WEEK          = 1
MAX_WEEK          = 18
MIN_SNAPS_CLEAN   = 50
MIN_SNAPS_PRESSURE= 20
DPI               = 200
BG                = "#0f1115"
FG                = "#EDEDED"
GRID              = "#2a2f3a"
RYG               = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}

URL = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"

# ── HELPERS ────────────────────────────────────────────────────────────────────
def pick_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def short_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return name
    parts = name.strip().split()
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


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

# ── DATA ───────────────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
df = pd.read_csv(URL, low_memory=False, compression="infer")

print(f"Filas descargadas: {len(df):,}")

# Filter pass plays in week range
mask = (
    df["play_type"].isin(["pass"]) &
    df["epa"].notna() &
    df["week"].between(MIN_WEEK, MAX_WEEK)
)
df = df[mask].copy()

# Pick ID and name columns
id_col   = pick_col(df, "passer_player_id", "passer_id")
name_col = pick_col(df, "passer", "passer_player_name")

if id_col is None:
    print("No se encontro columna de ID de QB, abortando.")
    sys.exit(1)

df = df[df[id_col].notna()].copy()

# Pressure detection
pressure_source = "unknown"
press_col = pick_col(df, "was_pressure")
if press_col is not None:
    df["pressured"] = pd.to_numeric(df[press_col], errors="coerce").fillna(0).astype(bool)
    pressure_source = "was_pressure"
else:
    hit_col  = pick_col(df, "qb_hit")
    sack_col = pick_col(df, "sack")
    cols_used = []
    df["pressured"] = False
    if hit_col is not None:
        df["pressured"] = df["pressured"] | (pd.to_numeric(df[hit_col], errors="coerce").fillna(0) == 1)
        cols_used.append("qb_hit")
    if sack_col is not None:
        df["pressured"] = df["pressured"] | (pd.to_numeric(df[sack_col], errors="coerce").fillna(0) == 1)
        cols_used.append("sack")
    pressure_source = f"proxy ({'+'.join(cols_used)})" if cols_used else "no disponible"
    print(f"Columna 'was_pressure' no encontrada. Usando proxy: {pressure_source}")

print(f"Fuente de presion: {pressure_source}")

# Groups
clean_grp = (
    df[~df["pressured"]]
    .groupby(id_col)
    .agg(epa_clean=("epa", "mean"), snaps_clean=("epa", "count"))
    .reset_index()
)
press_grp = (
    df[df["pressured"]]
    .groupby(id_col)
    .agg(epa_press=("epa", "mean"), snaps_press=("epa", "count"))
    .reset_index()
)

merged = clean_grp.merge(press_grp, on=id_col, how="inner")
merged = merged[
    (merged["snaps_clean"] >= MIN_SNAPS_CLEAN) &
    (merged["snaps_press"] >= MIN_SNAPS_PRESSURE)
].copy()

# Map ID to most-frequent name
if name_col is not None:
    name_map = (
        df.dropna(subset=[id_col, name_col])
        .groupby(id_col)[name_col]
        .agg(lambda s: s.value_counts().index[0] if len(s) else "")
    )
    merged["qb_name"] = merged[id_col].map(name_map).fillna(merged[id_col])
else:
    merged["qb_name"] = merged[id_col]

merged["qb_label"] = merged["qb_name"].apply(short_name)

print(f"QBs incluidos: {len(merged)}")

# ── PLOT ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
ax.set_facecolor(BG)

# Normalize epa_press for color
norm  = Normalize(vmin=merged["epa_press"].min(), vmax=merged["epa_press"].max())
cmap  = RYG

# Size proportional to snaps_press
size_raw = merged["snaps_press"].values.astype(float)
size_min, size_max = 40, 200
size_norm = (size_raw - size_raw.min()) / (size_raw.max() - size_raw.min() + 1e-9)
sizes = size_min + size_norm * (size_max - size_min)

sc = ax.scatter(
    merged["epa_clean"], merged["epa_press"],
    s=sizes,
    c=merged["epa_press"],
    cmap=cmap, norm=norm,
    edgecolors="#ffffff", linewidths=0.4,
    alpha=0.85, zorder=3,
)

# QB labels
for _, row in merged.iterrows():
    ax.text(
        row["epa_clean"], row["epa_press"], row["qb_label"],
        ha="center", va="center", fontsize=8.5, color=FG, fontweight="bold",
        path_effects=[pe.Stroke(linewidth=2, foreground=BG), pe.Normal()],
        zorder=4,
    )

# Reference lines
ax.axhline(0, color=FG, linewidth=0.7, linestyle="--", alpha=0.35, zorder=2)
ax.axvline(0, color=FG, linewidth=0.7, linestyle="--", alpha=0.35, zorder=2)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
diag_min = max(xmin, ymin)
diag_max = min(xmax, ymax)
ax.plot([diag_min, diag_max], [diag_min, diag_max],
        color="#888888", linewidth=1.0, linestyle=":", alpha=0.5, zorder=2)

# Quadrant labels
pad = 0.03
ax.text(xmax - pad, ymax - pad, "Elite\n(bueno en todo)",
        ha="right", va="top", color="#aaaaaa", fontsize=7.5, alpha=0.7)
ax.text(xmin + pad, ymax - pad, "Resistente a\nla presion",
        ha="left", va="top", color="#aaaaaa", fontsize=7.5, alpha=0.7)
ax.text(xmax - pad, ymin + pad, "Solo funciona sin\npresion",
        ha="right", va="bottom", color="#aaaaaa", fontsize=7.5, alpha=0.7)
ax.text(xmin + pad, ymin + pad, "Problemas en\ntodo",
        ha="left", va="bottom", color="#aaaaaa", fontsize=7.5, alpha=0.7)

# Colorbar
cb = fig.colorbar(sc, ax=ax, pad=0.01)
cb.set_label("EPA/jugada bajo presion", color=FG, fontsize=8)
cb.ax.yaxis.set_tick_params(color=FG)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG, fontsize=7)
cb.outline.set_edgecolor(GRID)

# Axes styling
ax.set_xlabel("EPA/jugada — Pocket limpio", color=FG, fontsize=10)
ax.set_ylabel("EPA/jugada — Bajo presion", color=FG, fontsize=10)
ax.tick_params(colors=FG)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)
ax.grid(color=GRID, linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)
ax.xaxis.label.set_color(FG)
ax.yaxis.label.set_color(FG)
plt.setp(ax.get_xticklabels(), color=FG, fontsize=8)
plt.setp(ax.get_yticklabels(), color=FG, fontsize=8)

# Texts
subtitle_press = pressure_source
fig.text(0.5, 0.97, f"QB performance: Pocket limpio vs Bajo presion — NFL {SEASON}",
         ha="center", va="top", color=FG, fontsize=13, fontweight="bold")
fig.text(0.5, 0.92,
         f"Presion: {subtitle_press} | Min {MIN_SNAPS_CLEAN} snaps limpios, {MIN_SNAPS_PRESSURE} bajo presion | Color = EPA bajo presion",
         ha="center", va="top", color="#aaaaaa", fontsize=8.5)
fig.text(0.01, 0.01, "Fuente: nflverse PBP",
         ha="left", va="bottom", color="#666666", fontsize=7)
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", color="#666666", fontsize=7)

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

out = f"qb_presion_{SEASON}.png"
fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {out}")
