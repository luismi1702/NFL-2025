"""
play_action.py
Efectividad desde shotgun vs bajo centro (under center) — horizontal bar chart.
Nota: play_action no está en nflverse PBP; se usa shotgun como proxy de formación.
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
SEASON = 2025
DPI    = 200
BG     = "#0f1115"
FG     = "#EDEDED"
GRID   = "#2a2f3a"
RYG    = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

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

# Filter pass plays
mask = (
    df["play_type"].isin(["pass"]) &
    df["posteam"].notna() &
    df["epa"].notna() &
    df["shotgun"].notna()
)
df = df[mask].copy()
df["shotgun"] = pd.to_numeric(df["shotgun"], errors="coerce").fillna(0).astype(int)

# Group by team
MIN_SG    = 25   # min jugadas desde shotgun
MIN_NO_SG = 15   # min jugadas desde bajo centro

results = []
for team, grp in df.groupby("posteam"):
    sg_plays    = grp[grp["shotgun"] == 1]
    no_sg_plays = grp[grp["shotgun"] == 0]

    if len(sg_plays) < MIN_SG or len(no_sg_plays) < MIN_NO_SG:
        continue

    epa_sg    = sg_plays["epa"].mean()
    epa_no_sg = no_sg_plays["epa"].mean()
    sg_rate   = (grp["shotgun"] == 1).mean() * 100
    boost     = epa_sg - epa_no_sg   # positivo = shotgun más efectivo

    results.append({
        "team":      team,
        "epa_sg":    epa_sg,
        "epa_no_sg": epa_no_sg,
        "sg_rate":   sg_rate,
        "boost":     boost,
        "n_sg":      len(sg_plays),
        "n_no_sg":   len(no_sg_plays),
    })

stats = pd.DataFrame(results).sort_values("boost", ascending=False).reset_index(drop=True)
print(f"Equipos incluidos: {len(stats)}")

# ── PLOT ───────────────────────────────────────────────────────────────────────
n_teams  = len(stats)
fig_h    = max(9, n_teams * 0.32)
fig, ax  = plt.subplots(figsize=(12, fig_h), facecolor=BG)
ax.set_facecolor(BG)

norm  = Normalize(vmin=stats["boost"].min(), vmax=stats["boost"].max())
y_pos = np.arange(n_teams)

base_zoom = 0.040
logo_offset = 0.02  # in data coords, will adjust after xlim set

# Draw bars
bar_colors = [RYG(norm(v)) for v in stats["boost"]]
bars = ax.barh(y_pos, stats["boost"], color=bar_colors, height=0.65, zorder=2)

# Get xlim for logo placement
ax.set_xlim(
    stats["boost"].min() - abs(stats["boost"].min()) * 0.4 - 0.05,
    stats["boost"].max() + abs(stats["boost"].max()) * 0.4 + 0.15,
)
xlim_left, xlim_right = ax.get_xlim()

# Place logos to the left of bars
for i, row in stats.iterrows():
    team = row["team"]
    y    = y_pos[i]  # type: ignore[index]

    logo_x = xlim_left + (xlim_right - xlim_left) * 0.015
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

    # Value to right of bar
    boost_val = row["boost"]
    text_x    = boost_val + (xlim_right - xlim_left) * 0.008 if boost_val >= 0 else boost_val - (xlim_right - xlim_left) * 0.008
    ha        = "left" if boost_val >= 0 else "right"
    ax.text(text_x, y, f"{boost_val:+.3f}",
            ha=ha, va="center", fontsize=7.5, color=FG, zorder=4)

    # Shotgun rate in small text
    pa_text_x = xlim_right - (xlim_right - xlim_left) * 0.01
    ax.text(pa_text_x, y, f"SG rate: {row['sg_rate']:.0f}%",
            ha="right", va="center", fontsize=6.5, color="#aaaaaa", zorder=4)

# Reference line at 0
ax.axvline(0, color=FG, linewidth=0.8, linestyle="--", alpha=0.4, zorder=3)

# Y ticks
ax.set_yticks(y_pos)
ax.set_yticklabels([""] * n_teams)
ax.invert_yaxis()

# Axes styling
ax.set_xlabel("Diferencia de EPA/pase (Shotgun - Bajo centro)", color=FG, fontsize=10)
ax.tick_params(colors=FG)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)
ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)
plt.setp(ax.get_xticklabels(), color=FG, fontsize=8)

# Texts
fig.text(0.5, 0.97, f"Shotgun vs bajo centro — EPA/pase — NFL {SEASON}",
         ha="center", va="top", color=FG, fontsize=14, fontweight="bold")
fig.text(0.5, 0.92,
         "EPA/pase desde shotgun menos EPA/pase desde bajo centro | + = shotgun mas efectivo | SG rate = % de pases desde shotgun",
         ha="center", va="top", color="#aaaaaa", fontsize=9)
fig.text(0.01, 0.01, "Fuente: nflverse PBP",
         ha="left", va="bottom", color="#666666", fontsize=7)
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", color="#666666", fontsize=7)

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

out = f"shotgun_vs_uc_{SEASON}.png"
fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {out}")
