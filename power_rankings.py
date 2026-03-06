"""
power_rankings.py
Power Rankings semanales composites con logos.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON = 2025
URL    = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
BG     = "#0f1115"
FG     = "#EDEDED"
GRID   = "#2a2f3a"
DPI    = 200
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}
RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

# ── HELPERS ────────────────────────────────────────────────────────────────────
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


def safe_norm(series):
    """Normalize a series 0-1; if all equal return 0.5."""
    mn = series.min()
    mx = series.max()
    rng = mx - mn
    if rng == 0:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / rng

# ── INPUT ──────────────────────────────────────────────────────────────────────
week = int(input("Semana (numero): ").strip())

# ── DATA ───────────────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
df = pd.read_csv(URL, low_memory=False, compression="infer")
print(f"Filas descargadas: {len(df):,}")

to_num(df, ["epa", "week"])

# Filter: up to input_week, pass+run plays, posteam and epa present
mask = (
    df["week"].notna() &
    (df["week"] <= week) &
    df["play_type"].isin(["pass", "run"]) &
    df["posteam"].notna() &
    df["epa"].notna()
)
plays = df[mask].copy()
print(f"Jugadas filtradas: {len(plays):,}")

# ── METRIC A: OFF EPA (mean EPA/play as posteam) ───────────────────────────────
off_epa = plays.groupby("posteam")["epa"].mean().rename("off_epa")

# ── METRIC B: DEF EPA (mean EPA/play as defteam, lower = better) ──────────────
def_mask = (
    df["week"].notna() &
    (df["week"] <= week) &
    df["play_type"].isin(["pass", "run"]) &
    df["defteam"].notna() &
    df["epa"].notna()
)
def_plays = df[def_mask].copy()
def_epa = def_plays.groupby("defteam")["epa"].mean().rename("def_epa")

# ── METRIC C: TRENDING (last 3 weeks vs previous) ─────────────────────────────
last3_weeks = sorted([w for w in [week - 2, week - 1, week] if w >= 1])
min_last3   = min(last3_weeks) if last3_weeks else week

off_last3 = (
    plays[plays["week"].isin(last3_weeks)]
    .groupby("posteam")["epa"]
    .mean()
    .rename("off_last3")
)

prev_plays = plays[plays["week"] < min_last3]
if len(prev_plays) > 0:
    off_prev = prev_plays.groupby("posteam")["epa"].mean().rename("off_prev")
else:
    off_prev = pd.Series(dtype=float, name="off_prev")

# ── COMBINE METRICS ────────────────────────────────────────────────────────────
stats = pd.DataFrame({"off_epa": off_epa, "def_epa": def_epa})
stats = stats.join(off_last3, how="left").join(off_prev, how="left")

if len(off_prev) > 0:
    stats["trending"] = stats["off_last3"].fillna(stats["off_epa"]) - stats["off_prev"].fillna(stats["off_epa"])
else:
    stats["trending"] = 0.0

stats["trending"] = stats["trending"].fillna(0.0)
stats = stats.dropna(subset=["off_epa", "def_epa"]).copy()

# ── NORMALIZE ─────────────────────────────────────────────────────────────────
stats["norm_off"]   = safe_norm(stats["off_epa"])
stats["norm_def"]   = 1.0 - safe_norm(stats["def_epa"])   # inverted: lower def_epa = better
stats["norm_trend"] = safe_norm(stats["trending"])

# ── COMPOSITE ─────────────────────────────────────────────────────────────────
stats["composite"] = (
    0.40 * stats["norm_off"] +
    0.40 * stats["norm_def"] +
    0.20 * stats["norm_trend"]
)

stats = stats.sort_values("composite", ascending=False).reset_index()
stats = stats.rename(columns={"posteam": "team", "index": "team"})

# Handle index column name (it may be "posteam" or "index")
if "posteam" in stats.columns:
    stats = stats.rename(columns={"posteam": "team"})
elif "index" in stats.columns:
    stats = stats.rename(columns={"index": "team"})

# Ensure team column exists and is clean
team_col_candidates = [c for c in stats.columns if c not in
                       ["off_epa","def_epa","off_last3","off_prev","trending",
                        "norm_off","norm_def","norm_trend","composite"]]
if team_col_candidates:
    stats = stats.rename(columns={team_col_candidates[0]: "team"})

stats = stats.reset_index(drop=True)
print(f"Equipos en rankings: {len(stats)}")

# ── PLOT ───────────────────────────────────────────────────────────────────────
n_teams = len(stats)
fig, ax = plt.subplots(figsize=(12, 11), facecolor=BG)
ax.set_facecolor(BG)

norm_color = Normalize(vmin=0.0, vmax=1.0)
y_pos = np.arange(n_teams)

# Draw bars
for idx in range(n_teams):
    comp = stats.loc[idx, "composite"]
    color = RYG(norm_color(comp))
    ax.barh(y_pos[idx], comp, color=color, height=0.70,
            edgecolor="#1e2430", linewidth=0.5, zorder=2)

# Midpoint reference
ax.axvline(0.5, color=GRID, linewidth=0.8, linestyle=":", zorder=3)

# ── LOGOS, RANK, VALUE ANNOTATIONS ────────────────────────────────────────────
stats = stats.reset_index(drop=True)
y_pos = np.arange(len(stats))

x_min, x_max = 0.0, 1.05
x_logo = -0.065
x_rank = -0.115

for idx in range(len(stats)):
    team = stats.loc[idx, "team"]
    y    = y_pos[idx]
    comp = stats.loc[idx, "composite"]
    off  = stats.loc[idx, "off_epa"]
    deff = stats.loc[idx, "def_epa"]
    rank = idx + 1

    # Rank number
    ax.text(x_rank, y, f"#{rank}", ha="center", va="center",
            fontsize=8.5, color="#aaaaaa", fontweight="bold", zorder=5)

    # Logo
    logo = load_logo(team, base_zoom=0.040)
    if logo is not None:
        ab = AnnotationBbox(logo, (x_logo, y), frameon=False, zorder=4)
        ax.add_artist(ab)
    else:
        ax.text(x_logo, y, team, ha="center", va="center", fontsize=7, color=FG)

    # Composite value
    ax.text(comp + 0.012, y, f"{comp:.3f}", ha="left", va="center",
            fontsize=8, color=FG, zorder=5)

    # Off/Def EPA inside or near bar
    bar_label = f"OF:{off:+.3f} DF:{deff:+.3f}"
    text_x = max(comp * 0.5, 0.05)
    ax.text(text_x, y, bar_label, ha="center", va="center",
            fontsize=6.5, color="#cccccc", alpha=0.85, zorder=5)

# ── AXES STYLING ───────────────────────────────────────────────────────────────
ax.set_yticks(y_pos)
ax.set_yticklabels([""] * n_teams)
ax.invert_yaxis()
ax.set_xlim(-0.14, 1.10)
ax.set_xlabel("Score compuesto (0-1)", color=FG, fontsize=11)
ax.tick_params(colors=FG, labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)
ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)
ax.xaxis.label.set_color(FG)

# Colorbar
sm = ScalarMappable(cmap=RYG, norm=norm_color)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.5)
cb.set_label("Score compuesto", color=FG, fontsize=8)
cb.ax.yaxis.set_tick_params(color=FG)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG, fontsize=7)
cb.outline.set_edgecolor(GRID)

# ── TITLES ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97, f"Power Rankings NFL {SEASON} \u2014 Semana {week}",
         ha="center", va="top", fontsize=16, fontweight="bold", color=FG)
fig.text(0.5, 0.92,
         "40% EPA ofensivo + 40% EPA defensivo + 20% tendencia \u00faltimas 3 semanas",
         ha="center", va="top", fontsize=10, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data  \u00b7  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 0.96, 0.91])

outfile = f"power_rankings_week{week}_{SEASON}.png"
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
