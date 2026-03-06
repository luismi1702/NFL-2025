"""
target_share.py
Para un equipo elegido por el usuario: distribución de blancos y air yards
por receptor. Barras horizontales lado a lado.
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
    if not isinstance(full_name, str):
        return str(full_name)
    parts = full_name.strip().split()
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


# ── User input ───────────────────────────────────────────────────────────────
team = input("Equipo (siglas, p.ej. KC): ").strip().upper()

# ── Load data ────────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
df = pd.read_csv(URL, low_memory=False, compression="infer")
to_num(df, ["epa", "air_yards"])

# ── Filter ───────────────────────────────────────────────────────────────────
df = df[(df["play_type"] == "pass") & (df["posteam"] == team)].copy()

if df.empty:
    raise SystemExit(f"No se encontraron jugadas de pase para el equipo {team}.")

receiver_col = pick_col(df, "receiver", "receiver_player_name")
if receiver_col is None:
    raise SystemExit("No se encontró columna de receptor en los datos.")

# ── Totals ───────────────────────────────────────────────────────────────────
total_targets = len(df[df[receiver_col].notna()])
total_air     = df["air_yards"].sum() if "air_yards" in df.columns else 0

# ── Aggregate per receiver ────────────────────────────────────────────────────
targeted = df[df[receiver_col].notna()].copy()
grp = (
    targeted.groupby(receiver_col, group_keys=False)
    .agg(
        targets=("epa", "count"),
        air_yards_sum=("air_yards", lambda x: x.sum()),
    )
    .reset_index()
)
grp["target_share"]    = grp["targets"] / total_targets * 100
grp["air_yards_share"] = grp["air_yards_sum"] / total_air * 100 if total_air else 0.0
grp = grp.sort_values("targets", ascending=False).reset_index(drop=True)

# ── Top-8 + Otros ─────────────────────────────────────────────────────────────
TOP = 8
if len(grp) > TOP:
    top8  = grp.iloc[:TOP].copy()
    otros = grp.iloc[TOP:].copy()
    otros_row = pd.DataFrame([{
        receiver_col:      "Otros",
        "targets":         otros["targets"].sum(),
        "air_yards_sum":   otros["air_yards_sum"].sum(),
        "target_share":    otros["target_share"].sum(),
        "air_yards_share": otros["air_yards_share"].sum(),
    }])
    grp_plot = pd.concat([top8, otros_row], ignore_index=True)
else:
    grp_plot = grp.copy()

grp_plot["label"] = grp_plot[receiver_col].apply(short_name)

# ── Print to console ──────────────────────────────────────────────────────────
print(f"\n{'Receptor':<22} {'Targets':>7}  {'Target %':>9}  {'Air Yds %':>10}")
print("-" * 55)
for _, row in grp_plot.iterrows():
    print(f"{row['label']:<22} {row['targets']:>7.0f}  {row['target_share']:>9.1f}%  {row['air_yards_share']:>9.1f}%")

# ── Colors ────────────────────────────────────────────────────────────────────
def bar_colors(series, cmap):
    vmin, vmax = series.min(), series.max()
    norm = Normalize(vmin=vmin, vmax=vmax if vmax != vmin else vmin + 1)
    return [cmap(norm(v)) for v in series]

ts_colors  = bar_colors(grp_plot["target_share"],    RYG)
ay_colors  = bar_colors(grp_plot["air_yards_share"], RYG)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7), facecolor=BG)

n_rows = len(grp_plot)
y_pos  = np.arange(n_rows)

for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=FG, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.6, zorder=0)

# Left panel — target share
bars1 = ax1.barh(y_pos, grp_plot["target_share"], color=ts_colors, height=0.6, zorder=3)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(grp_plot["label"], color=FG, fontsize=10)
ax1.invert_yaxis()
ax1.set_xlabel("Target share (%)", color=FG, fontsize=10)
ax1.set_title("Target Share", color=FG, fontsize=12, pad=6)
for bar, val in zip(bars1, grp_plot["target_share"]):
    ax1.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%",
        va="center", ha="left", color=FG, fontsize=9, zorder=4,
    )

# Right panel — air yards share
bars2 = ax2.barh(y_pos, grp_plot["air_yards_share"], color=ay_colors, height=0.6, zorder=3)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([""] * n_rows)
ax2.invert_yaxis()
ax2.set_xlabel("Air Yards share (%)", color=FG, fontsize=10)
ax2.set_title("Air Yards Share", color=FG, fontsize=12, pad=6)
for bar, val in zip(bars2, grp_plot["air_yards_share"]):
    ax2.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%",
        va="center", ha="left", color=FG, fontsize=9, zorder=4,
    )

# Titles & credits
fig.text(0.5,  0.97, f"Distribución de blancos — {team} — NFL {SEASON}",
         ha="center", va="top", fontsize=16, fontweight="bold", color=FG)
fig.text(0.5,  0.92,
         "Target share (%) y Air Yards share (%) por receptor | Toda la temporada",
         ha="center", va="top", fontsize=10, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data  ·  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

fig.tight_layout(rect=[0, 0.03, 1, 0.91])
out = f"target_share_{team}_{SEASON}.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"\nGuardado: {out}")
