"""
ajuste_medio_tiempo.py
EPA/play 1ª mitad vs 2ª mitad por equipo.
Panel A: scatter. Panel B: horizontal bar chart del delta.
NFL 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

MIN_PLAYS_PER_HALF = 80

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
to_num(df, ["epa", "qtr"])

# ── Filter ───────────────────────────────────────────────────────────────────
df = df[
    df["play_type"].isin(["pass", "run"])
    & df["posteam"].notna()
    & df["epa"].notna()
].copy()

# ── Detect half ───────────────────────────────────────────────────────────────
if "game_half" in df.columns:
    half1_mask = df["game_half"] == "Half1"
    half2_mask = df["game_half"] == "Half2"
else:
    half1_mask = df["qtr"] <= 2
    half2_mask = (df["qtr"] == 3) | (df["qtr"] == 4)

df_h1 = df[half1_mask].copy()
df_h2 = df[half2_mask].copy()

# ── Aggregate ─────────────────────────────────────────────────────────────────
def agg_half(sub):
    return sub.groupby("posteam", group_keys=False)["epa"].agg(
        epa_play="mean",
        n_plays="count",
    ).reset_index()

h1 = agg_half(df_h1).rename(columns={"epa_play": "epa_h1", "n_plays": "n_h1"})
h2 = agg_half(df_h2).rename(columns={"epa_play": "epa_h2", "n_plays": "n_h2"})

merged = h1.merge(h2, on="posteam", how="inner")
merged = merged[
    (merged["n_h1"] >= MIN_PLAYS_PER_HALF) &
    (merged["n_h2"] >= MIN_PLAYS_PER_HALF)
].copy()

merged["delta"] = merged["epa_h2"] - merged["epa_h1"]
merged = merged.sort_values("delta", ascending=False).reset_index(drop=True)

# ── Normalize colors ──────────────────────────────────────────────────────────
d_min, d_max  = merged["delta"].min(), merged["delta"].max()
d_range       = d_max - d_min if d_max != d_min else 1
norm_delta     = Normalize(vmin=d_min, vmax=d_max)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor=BG)
gs  = gridspec.GridSpec(1, 2, width_ratios=[4, 6], figure=fig)
ax_sc  = fig.add_subplot(gs[0])   # Panel A — scatter
ax_bar = fig.add_subplot(gs[1])   # Panel B — horizontal bars

for ax in (ax_sc, ax_bar):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=FG)

# ══════════════════════════════════════════════════════════════════
# Panel A — Scatter
# ══════════════════════════════════════════════════════════════════
scatter_colors = [RYG(norm_delta(v)) for v in merged["delta"]]
ax_sc.scatter(
    merged["epa_h1"],
    merged["epa_h2"],
    s=80,
    c=scatter_colors,
    edgecolors="none",
    alpha=0.90,
    zorder=3,
)

# Diagonal reference (x = y)
lim_lo = min(merged["epa_h1"].min(), merged["epa_h2"].min()) - 0.02
lim_hi = max(merged["epa_h1"].max(), merged["epa_h2"].max()) + 0.02
ax_sc.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=GRID, linewidth=1.0,
           linestyle="--", zorder=1, label="x = y")
ax_sc.axhline(0, color=FG, linewidth=0.4, alpha=0.3, zorder=1)
ax_sc.axvline(0, color=FG, linewidth=0.4, alpha=0.3, zorder=1)
ax_sc.grid(True, color=GRID, linewidth=0.5, alpha=0.5, zorder=0)

# Logos on scatter
merged_sc = merged.reset_index(drop=True)
for i in range(len(merged_sc)):
    row  = merged_sc.iloc[i]
    team = row["posteam"]
    img  = load_logo(team, base_zoom=0.030)
    if img is not None:
        ab = AnnotationBbox(
            img,
            (row["epa_h1"], row["epa_h2"]),
            frameon=False,
            xycoords="data",
            box_alignment=(0.5, 0.5),
        )
        ax_sc.add_artist(ab)
    else:
        ax_sc.text(
            row["epa_h1"], row["epa_h2"], team,
            ha="center", va="center", color=FG, fontsize=7,
        )

# Quadrant labels
x_lo_s, x_hi_s = ax_sc.get_xlim()
y_lo_s, y_hi_s = ax_sc.get_ylim()
quad_kw = dict(fontsize=7.5, color="#888888", alpha=0.7, fontstyle="italic")
ax_sc.text(x_lo_s + 0.01, y_hi_s - 0.01, "Mejoran (2H > 1H)", ha="left", va="top",    **quad_kw)
ax_sc.text(x_hi_s - 0.01, y_lo_s + 0.01, "Peores en 2ª mitad", ha="right", va="bottom", **quad_kw)
ax_sc.text(
    (x_lo_s + x_hi_s) / 2,
    (y_lo_s + y_hi_s) / 2 + 0.03,
    "Constantes ↗",
    ha="center", va="center",
    fontsize=7.5, color="#888888", alpha=0.6, fontstyle="italic", rotation=35,
)

ax_sc.set_xlabel("EPA/jugada — 1ª mitad", color=FG, fontsize=10)
ax_sc.set_ylabel("EPA/jugada — 2ª mitad", color=FG, fontsize=10)

# ══════════════════════════════════════════════════════════════════
# Panel B — Horizontal bar chart
# ══════════════════════════════════════════════════════════════════
n_teams   = len(merged)
y_pos     = np.arange(n_teams)
bar_colors = [RYG(norm_delta(v)) for v in merged["delta"]]

bars = ax_bar.barh(y_pos, merged["delta"], color=bar_colors, height=0.65, zorder=3)
ax_bar.invert_yaxis()   # best improvers at top
ax_bar.axvline(0, color=FG, linewidth=1.0, alpha=0.6, zorder=4)
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels([""] * n_teams)
ax_bar.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.6, zorder=0)
ax_bar.set_xlabel("ΔEPA (2ª mitad − 1ª mitad)", color=FG, fontsize=10)

# Value labels
for bar, val in zip(bars, merged["delta"]):
    x_off = 0.003 if val >= 0 else -0.003
    ha    = "left" if val >= 0 else "right"
    ax_bar.text(
        bar.get_width() + x_off,
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.3f}",
        va="center", ha=ha, color=FG, fontsize=8, zorder=5,
    )

# Logos to the left
for i in range(len(merged)):
    row  = merged.iloc[i]
    team = row["posteam"]
    img  = load_logo(team, base_zoom=0.040)
    if img is not None:
        ab = AnnotationBbox(
            img,
            (merged["delta"].min() - 0.01, y_pos[i]),
            xycoords="data",
            frameon=False,
            box_alignment=(1.0, 0.5),
            xybox=(-4, 0),
            boxcoords="offset points",
        )
        ax_bar.add_artist(ab)
    else:
        ax_bar.text(
            merged["delta"].min() - 0.01, y_pos[i], team,
            ha="right", va="center", color=FG, fontsize=8, fontweight="bold",
        )

# Give room for logos on the left in bar chart
x_lo_b, x_hi_b = ax_bar.get_xlim()
ax_bar.set_xlim(x_lo_b - 0.05, x_hi_b + 0.04)

# ── Titles & credits ──────────────────────────────────────────────────────────
fig.text(0.5,  0.97, f"Ajuste de medio tiempo — NFL {SEASON}",
         ha="center", va="top", fontsize=16, fontweight="bold", color=FG)
fig.text(0.5,  0.92,
         "ΔEPA = EPA/jugada 2ª mitad menos 1ª mitad | + = mejora en 2ª mitad",
         ha="center", va="top", fontsize=10, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data  ·  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

fig.subplots_adjust(left=0.05, right=0.97, wspace=0.35, top=0.90, bottom=0.05)
out = f"ajuste_medio_tiempo_{SEASON}.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Guardado: {out}")
