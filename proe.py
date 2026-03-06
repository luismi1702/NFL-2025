"""
proe.py
Pass Rate Over Expectation (PROE) by team — horizontal bar chart.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON     = 2025
MIN_PLAYS  = 200
DPI        = 200
BG         = "#0f1115"
FG         = "#EDEDED"
GRID       = "#2a2f3a"
RYG        = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}

URL = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"

# ── HELPERS ────────────────────────────────────────────────────────────────────
def pick_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
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

# ── DATA ───────────────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
df = pd.read_csv(URL, low_memory=False, compression="infer")

print(f"Filas descargadas: {len(df):,}")

# Filter pass + run plays
mask = (
    df["play_type"].isin(["pass", "run"]) &
    df["posteam"].notna()
)
df = df[mask].copy()

# Check xpass column
if "xpass" not in df.columns:
    print("La columna 'xpass' no esta disponible en estos datos. Abortando.")
    sys.exit(1)

df["xpass"] = pd.to_numeric(df["xpass"], errors="coerce")
df = df.dropna(subset=["xpass"]).copy()

df["actual_pass"] = (df["play_type"] == "pass").astype(float)

# PROE by team
def team_proe(grp):
    n              = len(grp)
    actual_rate    = grp["actual_pass"].mean() * 100
    expected_rate  = grp["xpass"].mean() * 100
    proe           = actual_rate - expected_rate
    return pd.Series({
        "n_plays":        n,
        "actual_pass_pct":  actual_rate,
        "expected_pass_pct": expected_rate,
        "proe":           proe,
    })

stats = (
    df.groupby("posteam")
    .apply(team_proe)
    .reset_index()
)
stats = stats[stats["n_plays"] >= MIN_PLAYS].sort_values("proe", ascending=False).reset_index(drop=True)
print(f"Equipos incluidos: {len(stats)}")

# ── PLOT ───────────────────────────────────────────────────────────────────────
n_teams = len(stats)
fig_h   = max(9, n_teams * 0.32)
fig, ax = plt.subplots(figsize=(12, fig_h), facecolor=BG)
ax.set_facecolor(BG)

# Color: green if positive, red if negative, yellow if near 0
def proe_color(v):
    abs_max = max(abs(stats["proe"].max()), abs(stats["proe"].min()), 1e-6)
    norm_v  = (v + abs_max) / (2 * abs_max)   # 0..1, center = neutral
    return RYG(norm_v)

bar_colors = [proe_color(v) for v in stats["proe"]]
y_pos      = np.arange(n_teams)

ax.barh(y_pos, stats["proe"], color=bar_colors, height=0.65, zorder=2)

# Xlim
abs_max = max(abs(stats["proe"].min()), abs(stats["proe"].max()), 1.0)
ax.set_xlim(
    -(abs_max * 1.45),
    abs_max * 1.45,
)
xlim_left, xlim_right = ax.get_xlim()

base_zoom = 0.040

# Logos and value labels
for i, row in stats.iterrows():
    team = row["posteam"]
    y    = y_pos[i]  # type: ignore[index]
    v    = row["proe"]

    logo_x = xlim_left + (xlim_right - xlim_left) * 0.012
    logo = load_logo(team, base_zoom=base_zoom)
    if logo is not None:
        ab = AnnotationBbox(
            logo, (logo_x, y),
            frameon=False, zorder=4,
            box_alignment=(0.0, 0.5),
        )
        ax.add_artist(ab)
    else:
        ax.text(logo_x, y, team,
                ha="left", va="center", fontsize=7, color=FG, fontweight="bold", zorder=4)

    # Value text to the right of bar
    text_x = v + (xlim_right - xlim_left) * 0.008 if v >= 0 else v - (xlim_right - xlim_left) * 0.008
    ha     = "left" if v >= 0 else "right"
    ax.text(text_x, y, f"{v:+.1f}%",
            ha=ha, va="center", fontsize=7.5, color=FG, zorder=4)

# Reference line at 0
ax.axvline(0, color=FG, linewidth=0.8, linestyle="--", alpha=0.4, zorder=3)

# Y ticks
ax.set_yticks(y_pos)
ax.set_yticklabels([""] * n_teams)
ax.invert_yaxis()

# Axes styling
ax.set_xlabel("PROE (%)", color=FG, fontsize=10)
ax.tick_params(colors=FG)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)
ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)
plt.setp(ax.get_xticklabels(), color=FG, fontsize=8)

# Texts
fig.text(0.5, 0.97, f"Pass Rate Over Expectation (PROE) — NFL {SEASON}",
         ha="center", va="top", color=FG, fontsize=14, fontweight="bold")
fig.text(0.5, 0.92,
         "Tasa de pase real menos esperada (ajustada por marcador, down, distancia y tiempo) | + = pasa mas de lo esperado",
         ha="center", va="top", color="#aaaaaa", fontsize=9)
fig.text(0.01, 0.01, "Fuente: nflverse PBP",
         ha="left", va="bottom", color="#666666", fontsize=7)
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", color="#666666", fontsize=7)

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

out = f"proe_{SEASON}.png"
fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {out}")
