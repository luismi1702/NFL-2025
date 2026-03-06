"""
season_arc.py
Curva de temporada NFL — EPA/jugada por semana, rolling 3-week average.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON      = 2025
ROLL        = 3
MIN_PLAYS   = 15
DPI         = 200
BG          = "#0f1115"
FG          = "#EDEDED"
GRID        = "#2a2f3a"
RYG         = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

HIGHLIGHT_COLORS = [
    "#06d6a0", "#ffd166", "#ef476f",
    "#4e9af1", "#ff9f1c", "#c77dff",
]

URL = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"

# ── DATA ───────────────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
df = pd.read_csv(URL, low_memory=False, compression="infer")

print(f"Filas descargadas: {len(df):,}")

# Filter pass + run plays with valid EPA
mask = (
    df["play_type"].isin(["pass", "run"]) &
    df["epa"].notna() &
    df["posteam"].notna() &
    df["week"].notna()
)
df = df[mask].copy()

# Group by team + week
weekly = (
    df.groupby(["posteam", "week"])
    .agg(epa_mean=("epa", "mean"), n=("epa", "count"))
    .reset_index()
)
weekly = weekly[weekly["n"] >= MIN_PLAYS]

# Pivot
pivot = weekly.pivot(index="week", columns="posteam", values="epa_mean").sort_index()

# Rolling mean
rolled = pivot.rolling(ROLL, min_periods=1).mean()

weeks  = rolled.index.tolist()
teams  = rolled.columns.tolist()

# ── TEAM SELECTION ─────────────────────────────────────────────────────────────
last_valid = rolled.ffill().iloc[-1].dropna()

auto_top    = last_valid.nlargest(3).index.tolist()
auto_bottom = last_valid.nsmallest(3).index.tolist()

raw = input(
    "Equipos a destacar (siglas separadas por coma, p.ej. KC,SF,DAL - Enter para auto top/bottom 3): "
).strip()

if raw == "":
    highlight = auto_top + auto_bottom
    teams_str = f"Auto top 3 + bottom 3 semana final"
else:
    highlight = [t.strip().upper() for t in raw.split(",") if t.strip()]
    teams_str = ", ".join(highlight)

# Keep only valid teams
highlight = [t for t in highlight if t in teams]
if not highlight:
    print("Ningun equipo valido encontrado, usando auto seleccion.")
    highlight = auto_top + auto_bottom
    teams_str = "Auto top 3 + bottom 3 semana final"

# ── PLOT ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
ax.set_facecolor(BG)

# Gray background lines
for team in teams:
    series = rolled[team].dropna()
    if series.empty:
        continue
    ax.plot(
        series.index, series.values,
        color="#3a3f4a", linewidth=0.8, alpha=0.6, zorder=1,
    )

# Highlighted teams
color_cycle = HIGHLIGHT_COLORS[:len(highlight)]
for team, color in zip(highlight, color_cycle):
    if team not in rolled.columns:
        continue
    series = rolled[team].dropna()
    if series.empty:
        continue
    ax.plot(
        series.index, series.values,
        color=color, linewidth=2.2, alpha=0.95, zorder=3,
        path_effects=[pe.Stroke(linewidth=3.5, foreground=BG), pe.Normal()],
    )
    # Label at end of line
    last_week = series.index[-1]
    last_val  = series.values[-1]
    ax.text(
        last_week + 0.15, last_val, team,
        color=color, fontsize=9, fontweight="bold", va="center", zorder=4,
        path_effects=[pe.Stroke(linewidth=2.5, foreground=BG), pe.Normal()],
    )

# Reference line
ax.axhline(0, color=FG, linewidth=0.8, linestyle="--", alpha=0.4, zorder=2)

# Axes styling
ax.set_xticks(weeks)
ax.set_xticklabels([str(int(w)) for w in weeks], color=FG, fontsize=8)
ax.set_yticklabels([f"{v:.2f}" for v in ax.get_yticks()], color=FG, fontsize=8)
ax.tick_params(colors=FG)

xmin, xmax = ax.get_xlim()
ax.set_xlim(xmin, xmax + 1.5)

ax.set_xlabel("Semana", color=FG, fontsize=10)
ax.set_ylabel("EPA / jugada (rolling)", color=FG, fontsize=10)

for spine in ax.spines.values():
    spine.set_edgecolor(GRID)
ax.tick_params(axis="both", colors=FG)

ax.yaxis.label.set_color(FG)
ax.xaxis.label.set_color(FG)

ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

# Texts
fig.text(0.5, 0.97, f"Curva de temporada NFL {SEASON} — EPA/jugada semanal",
         ha="center", va="top", color=FG, fontsize=14, fontweight="bold")
fig.text(0.5, 0.92,
         f"Rolling {ROLL} semanas | Jugadas de pase y carrera | Destacados: {teams_str}",
         ha="center", va="top", color="#aaaaaa", fontsize=9)
fig.text(0.01, 0.01, "Fuente: nflverse PBP",
         ha="left", va="bottom", color="#666666", fontsize=7)
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", color="#666666", fontsize=7)

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

out = f"season_arc_{SEASON}.png"
fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {out}")
