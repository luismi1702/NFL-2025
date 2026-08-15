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
from pbp_loader import cargar_pbp, cargar_participation, salida, season_cli
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON            = season_cli()   # None = auto-detectar última temporada
MIN_WEEK          = 1
MAX_WEEK          = 18
MIN_SNAPS_CLEAN   = 50
MIN_SNAPS_PRESSURE= 20
DPI               = 170
BG                = "#0f1115"
FG                = "#EDEDED"
GRID              = "#2a2f3a"
RYG               = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

LOGOS_DIR    = "logos"

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


def separar_etiquetas(xs, lys, x_tol, y_tol):
    """Separa verticalmente etiquetas que caerian casi encima (los puntos
    no se mueven; solo el texto baja). lys = y iniciales de las etiquetas."""
    idx = sorted(range(len(xs)), key=lambda i: -lys[i])
    out = list(lys)
    for pos, i in enumerate(idx):
        for j in idx[:pos]:
            if abs(xs[i] - xs[j]) < x_tol and abs(out[i] - out[j]) < y_tol:
                out[i] = min(out[i], out[j] - y_tol)
    return out


def load_logo(team, base_zoom=0.055):
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

# ── DATA ───────────────────────────────────────────────────────────────────────
df, SEASON = cargar_pbp(SEASON)
print(f"PBP {SEASON}: {len(df):,} jugadas REG")

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

# Presion real FTN (was_pressure) via participation — incluye hurries,
# no solo qb_hit/sack. Si no esta disponible, cae al proxy de siempre.
try:
    part, _ = cargar_participation(SEASON)
    part = part[["nflverse_game_id", "play_id", "was_pressure"]].rename(
        columns={"nflverse_game_id": "game_id"})
    part["play_id"] = pd.to_numeric(part["play_id"], errors="coerce")
    df["play_id"]   = pd.to_numeric(df["play_id"],   errors="coerce")
    df = df.merge(part, on=["game_id", "play_id"], how="left")
except Exception as e:
    print(f"  Aviso: participacion FTN no disponible ({e})")

# Pressure detection
pressure_source = "unknown"
press_col = pick_col(df, "was_pressure")
if press_col is not None:
    df["pressured"] = pd.to_numeric(df[press_col], errors="coerce").fillna(0).astype(bool)
    if "sack" in df.columns:   # un sack siempre es presion (por si FTN no lo marca)
        df["pressured"] |= pd.to_numeric(df["sack"], errors="coerce").fillna(0).eq(1)
    pressure_source = "was_pressure (FTN)"
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

# Color = % de dropbacks bajo presion (info nueva; antes duplicaba el eje Y)
merged["press_pct"] = (merged["snaps_press"] /
                       (merged["snaps_press"] + merged["snaps_clean"]) * 100)
norm  = Normalize(vmin=merged["press_pct"].min(), vmax=merged["press_pct"].max())
cmap  = LinearSegmentedColormap.from_list("ryg_inv", ["#06d6a0", "#ffd166", "#d84a4a"])

# Size proportional to snaps_press
size_raw = merged["snaps_press"].values.astype(float)
size_min, size_max = 40, 200
size_norm = (size_raw - size_raw.min()) / (size_raw.max() - size_raw.min() + 1e-9)
sizes = size_min + size_norm * (size_max - size_min)

sc = ax.scatter(
    merged["epa_clean"], merged["epa_press"],
    s=sizes,
    c=merged["press_pct"],
    cmap=cmap, norm=norm,
    edgecolors="#ffffff", linewidths=0.4,
    alpha=0.85, zorder=3,
)

# QB labels — debajo del punto y sin solaparse entre QBs cercanos
xr_rng = merged["epa_clean"].max() - merged["epa_clean"].min()
yr_rng = merged["epa_press"].max() - merged["epa_press"].min()
label_off = yr_rng * 0.05
lys = separar_etiquetas(
    merged["epa_clean"].tolist(),
    [v - label_off for v in merged["epa_press"]],
    xr_rng * 0.10, yr_rng * 0.045)
for (_, row), ly in zip(merged.iterrows(), lys):
    ax.text(
        row["epa_clean"], ly, row["qb_label"],
        ha="center", va="top", fontsize=8.5, color=FG, fontweight="bold",
        path_effects=[pe.Stroke(linewidth=2, foreground=BG), pe.Normal()],
        zorder=4,
    )

# Reference lines
ax.axhline(0, color=FG, linewidth=0.7, linestyle="--", alpha=0.35, zorder=2)
ax.axvline(0, color=FG, linewidth=0.7, linestyle="--", alpha=0.35, zorder=2)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

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
cb.set_label("% de dropbacks bajo presion (rojo = sufre mas presion)", color=FG, fontsize=8)
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
         f"Presion: {subtitle_press} | Min {MIN_SNAPS_CLEAN} snaps limpios, {MIN_SNAPS_PRESSURE} bajo presion | "
         f"Color = % de presion sufrida | Tamaño = snaps bajo presion",
         ha="center", va="top", color="#aaaaaa", fontsize=8.5)
fig.text(0.01, 0.01, "Fuente: nflverse PBP",
         ha="left", va="bottom", color="#666666", fontsize=7)
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", color="#666666", fontsize=7)

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

out = salida(f"qb_presion_{SEASON}.png", SEASON)
fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {out}")
