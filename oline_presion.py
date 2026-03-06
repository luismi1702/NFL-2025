"""
oline_presion.py
Ranking de líneas ofensivas por tasa de presión permitida.
Horizontal bar chart con logos.
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

MIN_PLAYS = 100

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
to_num(df, ["epa", "qb_hit", "sack"])

# ── Filter ───────────────────────────────────────────────────────────────────
df = df[(df["play_type"] == "pass") & df["posteam"].notna()].copy()

# ── Build pressured column ────────────────────────────────────────────────────
qb_hit_col = pick_col(df, "qb_hit")
sack_col   = pick_col(df, "sack")

pressured = pd.Series(False, index=df.index)
if qb_hit_col:
    pressured |= df[qb_hit_col].fillna(0) == 1
if sack_col:
    pressured |= df[sack_col].fillna(0) == 1
df["pressured"] = pressured

# ── Aggregate ─────────────────────────────────────────────────────────────────
def agg_team(g):
    result = {
        "pressure_rate": g["pressured"].mean() * 100,
        "n_plays":       len(g),
    }
    if sack_col:
        result["sack_rate"] = (g[sack_col].fillna(0) == 1).mean() * 100
    return pd.Series(result)

grp = (
    df.groupby("posteam", group_keys=False)
    .apply(agg_team)
    .reset_index()
)
grp = grp[grp["n_plays"] >= MIN_PLAYS].copy()
grp = grp.sort_values("pressure_rate", ascending=True).reset_index(drop=True)

league_avg = grp["pressure_rate"].mean()

# ── Colors: invert RYG so low pressure = green ────────────────────────────────
RYG_inv  = LinearSegmentedColormap.from_list("ryg_inv", ["#06d6a0", "#ffd166", "#d84a4a"])
pr_min   = grp["pressure_rate"].min()
pr_max   = grp["pressure_rate"].max()
pr_range = pr_max - pr_min if pr_max != pr_min else 1
norm_pr  = Normalize(vmin=pr_min, vmax=pr_max)
bar_colors = [RYG_inv(norm_pr(v)) for v in grp["pressure_rate"]]

# ── Plot ──────────────────────────────────────────────────────────────────────
n_teams = len(grp)
fig_h   = max(8, n_teams * 0.42)
fig, ax = plt.subplots(figsize=(12, fig_h), facecolor=BG)
ax.set_facecolor(BG)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)

y_pos = np.arange(n_teams)
bars  = ax.barh(y_pos, grp["pressure_rate"], color=bar_colors, height=0.65, zorder=3)

ax.set_yticks(y_pos)
ax.set_yticklabels([""] * n_teams)
ax.invert_yaxis()   # best (lowest pressure) at top
ax.tick_params(colors=FG)
ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.6, zorder=0)
ax.set_xlabel("% de jugadas con presión al QB (proxy: qb_hit + sack)", color=FG, fontsize=11)

# League avg line
ax.axvline(league_avg, color=FG, linewidth=1.2, linestyle="--", alpha=0.7, zorder=4)
ax.text(
    league_avg + 0.2, -0.6, f"Liga avg\n{league_avg:.1f}%",
    color=FG, fontsize=8, va="top", alpha=0.75,
)

# Value labels + sack rate text
has_sack = "sack_rate" in grp.columns
for i, (bar, row) in enumerate(zip(bars, grp.itertuples())):
    x_right = bar.get_width()
    label   = f"{x_right:.1f}%"
    if has_sack:
        label += f"  (sacks: {row.sack_rate:.1f}%)"
    ax.text(
        x_right + 0.3,
        bar.get_y() + bar.get_height() / 2,
        label,
        va="center", ha="left", color=FG, fontsize=8.5, zorder=4,
    )

# Logos to the left of each bar
grp_reset = grp.reset_index(drop=True)
for i in range(len(grp_reset)):
    row  = grp_reset.iloc[i]
    team = row["posteam"]
    img  = load_logo(team, base_zoom=0.040)
    if img is not None:
        ab = AnnotationBbox(
            img,
            (-0.3, y_pos[i]),
            xycoords=("data", "data"),
            frameon=False,
            box_alignment=(1.0, 0.5),
            xybox=(-6, 0),
            boxcoords="offset points",
        )
        ax.add_artist(ab)
    else:
        ax.text(
            -0.3, y_pos[i], team,
            ha="right", va="center", color=FG, fontsize=8, fontweight="bold",
        )

# Give room for logos on the left
x_lo, x_hi = ax.get_xlim()
ax.set_xlim(x_lo - 2, x_hi + 8)

# Titles & credits
fig.text(0.5,  0.97, f"Tasa de presión permitida — Líneas ofensivas — NFL {SEASON}",
         ha="center", va="top", fontsize=16, fontweight="bold", color=FG)
fig.text(0.5,  0.92,
         "Menor % = mejor línea ofensiva | Presión = qb_hit=1 o sack=1",
         ha="center", va="top", fontsize=10, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data  ·  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

fig.tight_layout(rect=[0, 0.03, 1, 0.91])
out = f"oline_presion_{SEASON}.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Guardado: {out}")
