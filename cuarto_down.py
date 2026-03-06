"""
cuarto_down.py
Decisiones en 4 down: go-for-it rate vs conversion rate, con logos de equipos.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON       = 2025
DPI          = 200
BG           = "#0f1115"
FG           = "#EDEDED"
GRID         = "#2a2f3a"
RYG          = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

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

# Filter 4th down scenarios
EXCL = ["no_play", "qb_kneel", "qb_spike"]
mask = (
    (df["down"] == 4) &
    (df["ydstogo"] <= 5) &
    (df["yardline_100"] <= 50) &
    df["posteam"].notna() &
    (~df["play_type"].isin(EXCL)) &
    df["play_type"].notna()
)
df4 = df[mask].copy()

print(f"Jugadas 4to down filtradas: {len(df4):,}")

# Went for it flag
df4["went_for_it"] = df4["play_type"].isin(["pass", "run"])

# Success: first down or yards_gained >= ydstogo
fd_col = pick_col(df4, "first_down", "first_down_rush", "first_down_pass")
if fd_col is not None:
    df4["success"] = pd.to_numeric(df4[fd_col], errors="coerce").fillna(0) == 1
else:
    yg_col = pick_col(df4, "yards_gained")
    if yg_col is not None:
        df4["success"] = pd.to_numeric(df4[yg_col], errors="coerce").fillna(0) >= df4["ydstogo"]
    else:
        print("No se encontro columna de exito en 4to down, usando yards_gained >= ydstogo fallback.")
        df4["success"] = False

# Group by team
def team_stats(group):
    opp   = len(group)
    go    = group["went_for_it"].mean() * 100
    went  = group[group["went_for_it"]]
    sr    = went["success"].mean() * 100 if len(went) > 0 else np.nan
    return pd.Series({"opportunities": opp, "go_rate": go, "success_rate": sr, "n_went": len(went)})

stats = df4.groupby("posteam").apply(team_stats).reset_index()
stats = stats[stats["opportunities"] >= 5].dropna(subset=["success_rate"])
print(f"Equipos incluidos: {len(stats)}")

# League averages
avg_go      = stats["go_rate"].mean()
avg_success = stats["success_rate"].mean()

# Normalize for size
size_raw  = stats["opportunities"].values.astype(float)
size_norm = (size_raw - size_raw.min()) / (size_raw.max() - size_raw.min() + 1e-9)
sizes     = 60 + size_norm * (300 - 60)

norm = Normalize(vmin=stats["go_rate"].min(), vmax=stats["go_rate"].max())

# ── PLOT ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8), facecolor=BG)
ax.set_facecolor(BG)

# Scatter background (for colorbar reference)
sc = ax.scatter(
    stats["go_rate"], stats["success_rate"],
    s=sizes,
    c=stats["go_rate"],
    cmap=RYG, norm=norm,
    alpha=0.0,  # invisible — logos will be plotted on top
    zorder=1,
)

# Team logos or text
base_zoom = 0.032
for _, row in stats.iterrows():
    team = row["posteam"]
    x, y = row["go_rate"], row["success_rate"]
    logo = load_logo(team, base_zoom=base_zoom)
    if logo is not None:
        ab = AnnotationBbox(
            logo, (x, y),
            frameon=False, zorder=3,
            box_alignment=(0.5, 0.5),
        )
        ax.add_artist(ab)
    else:
        color = RYG(norm(x))
        ax.text(x, y, team,
                ha="center", va="center", fontsize=7.5,
                color=color, fontweight="bold", zorder=3)

# League avg lines
ax.axhline(avg_success, color="#888888", linewidth=1.0, linestyle="--", alpha=0.6, zorder=2)
ax.axvline(avg_go,      color="#888888", linewidth=1.0, linestyle="--", alpha=0.6, zorder=2)

# Liga avg labels
ax.text(ax.get_xlim()[0] + 0.3, avg_success + 0.5,
        f"Liga avg ({avg_success:.1f}%)", color="#aaaaaa", fontsize=7.5,
        ha="left", va="bottom")
ax.text(avg_go + 0.3, ax.get_ylim()[0] + 0.5,
        f"Liga avg ({avg_go:.1f}%)", color="#aaaaaa", fontsize=7.5,
        ha="left", va="bottom")

# Colorbar
cb = fig.colorbar(sc, ax=ax, pad=0.01)
cb.set_label("% de intentos (agresividad)", color=FG, fontsize=8)
cb.ax.yaxis.set_tick_params(color=FG)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG, fontsize=7)
cb.outline.set_edgecolor(GRID)

# Axes styling
ax.set_xlabel("% de intentos en 4 down y corto (<=5 yds, mitad rival)", color=FG, fontsize=10)
ax.set_ylabel("% de conversion cuando van a por ello", color=FG, fontsize=10)
ax.tick_params(colors=FG)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)
ax.grid(color=GRID, linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)
plt.setp(ax.get_xticklabels(), color=FG, fontsize=8)
plt.setp(ax.get_yticklabels(), color=FG, fontsize=8)

# Texts
fig.text(0.5, 0.97, f"Decisiones en 4 down — NFL {SEASON}",
         ha="center", va="top", color=FG, fontsize=14, fontweight="bold")
fig.text(0.5, 0.92,
         "4 down y <=5 yardas en mitad del campo rival | Tamano = numero de oportunidades",
         ha="center", va="top", color="#aaaaaa", fontsize=9)
fig.text(0.01, 0.01, "Fuente: nflverse PBP",
         ha="left", va="bottom", color="#666666", fontsize=7)
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", color="#666666", fontsize=7)

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

out = f"cuarto_down_{SEASON}.png"
fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {out}")
