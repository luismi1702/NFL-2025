"""
valor_turnovers.py
Balance de EPA de turnovers (intercepciones + fumbles) por equipo.
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

# ── DATA ───────────────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
df = pd.read_csv(URL, low_memory=False, compression="infer")
print(f"Filas descargadas: {len(df):,}")

to_num(df, ["epa", "interception", "fumble_lost", "sack"])

interception_col = pick_col(df, "interception")
fumble_lost_col  = pick_col(df, "fumble_lost")

if interception_col is None and fumble_lost_col is None:
    raise SystemExit("No se encontraron columnas de turnovers (interception, fumble_lost).")

# ── OFFENSIVE COST ─────────────────────────────────────────────────────────────
# EPA lost by the offense when they turn the ball over (values will be negative)
off_int = pd.Series(dtype=float)
off_fum = pd.Series(dtype=float)

if interception_col is not None:
    mask_int = (df[interception_col] == 1) & df["posteam"].notna() & df["epa"].notna()
    off_int = df[mask_int].groupby("posteam")["epa"].sum()

if fumble_lost_col is not None:
    mask_fum = (df[fumble_lost_col] == 1) & df["posteam"].notna() & df["epa"].notna()
    off_fum = df[mask_fum].groupby("posteam")["epa"].sum()

off_cost = off_int.add(off_fum, fill_value=0)

# ── DEFENSIVE GAIN ─────────────────────────────────────────────────────────────
# EPA impact for the defense when they force a turnover (flip sign: good for defense)
def_int = pd.Series(dtype=float)
def_fum = pd.Series(dtype=float)

if interception_col is not None:
    mask_int_d = (df[interception_col] == 1) & df["defteam"].notna() & df["epa"].notna()
    def_int = df[mask_int_d].groupby("defteam")["epa"].sum() * -1

if fumble_lost_col is not None:
    mask_fum_d = (df[fumble_lost_col] == 1) & df["defteam"].notna() & df["epa"].notna()
    def_fum = df[mask_fum_d].groupby("defteam")["epa"].sum() * -1

def_gain = def_int.add(def_fum, fill_value=0)

# ── COMBINE ────────────────────────────────────────────────────────────────────
turnovers = pd.DataFrame({"off_cost": off_cost, "def_gain": def_gain}).fillna(0)
# off_cost is already negative; def_gain is positive; net = sum
turnovers["net"] = turnovers["def_gain"] + turnovers["off_cost"]
turnovers = turnovers.reset_index().rename(columns={"index": "team", "posteam": "team"})

# Handle the case where the index name might be "posteam" or "defteam"
if "posteam" in turnovers.columns:
    turnovers = turnovers.rename(columns={"posteam": "team"})
if "defteam" in turnovers.columns:
    turnovers = turnovers.rename(columns={"defteam": "team"})

# Drop rows where team looks like an index integer
turnovers = turnovers[turnovers["team"].apply(lambda x: isinstance(x, str) and len(x) <= 5)]
turnovers = turnovers.sort_values("net", ascending=False).reset_index(drop=True)

print(f"Equipos con datos de turnovers: {len(turnovers)}")

# ── PLOT ───────────────────────────────────────────────────────────────────────
n_teams = len(turnovers)
fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)
ax.set_facecolor(BG)

# Normalize net for RYG colormap
net_vals = turnovers["net"].values.astype(float)
net_min = net_vals.min()
net_max = net_vals.max()
net_range = net_max - net_min if (net_max - net_min) != 0 else 1.0
norm = Normalize(vmin=net_min, vmax=net_max)

y_positions = np.arange(n_teams)

# Draw horizontal bars
for i in range(n_teams):
    net_val = turnovers.loc[i, "net"]
    color   = RYG(norm(net_val))
    ax.barh(y_positions[i], net_val, color=color, height=0.65,
            edgecolor="#1e2430", linewidth=0.5, zorder=2)

# Zero line
ax.axvline(0, color=FG, linewidth=1.0, linestyle="-", alpha=0.6, zorder=3)

# ── LOGOS & ANNOTATIONS ────────────────────────────────────────────────────────
turnovers = turnovers.reset_index(drop=True)
y_pos = np.arange(len(turnovers))

x_lim_min, x_lim_max = ax.get_xlim()
x_logo = x_lim_min - (x_lim_max - x_lim_min) * 0.075

for idx in range(len(turnovers)):
    team     = turnovers.loc[idx, "team"]
    y        = y_pos[idx]
    net_val  = turnovers.loc[idx, "net"]
    def_g    = turnovers.loc[idx, "def_gain"]
    off_c    = turnovers.loc[idx, "off_cost"]

    # Logo
    logo = load_logo(team, base_zoom=0.040)
    if logo is not None:
        ab = AnnotationBbox(logo, (x_logo, y), frameon=False, zorder=4)
        ax.add_artist(ab)
    else:
        ax.text(x_logo, y, team, ha="center", va="center", fontsize=7, color=FG)

    # Annotation: gain / cost text
    offset = (x_lim_max - x_lim_min) * 0.012
    text_x = net_val + offset if net_val >= 0 else net_val - offset
    ha_val = "left" if net_val >= 0 else "right"
    label  = f"  \u2191{def_g:+.0f}  \u2193{off_c:.0f}"
    ax.text(text_x, y, label, va="center", ha=ha_val,
            fontsize=7.5, color="#aaaaaa", zorder=5)

# ── AXES STYLING ───────────────────────────────────────────────────────────────
ax.set_yticks(y_pos)
ax.set_yticklabels([""] * n_teams)  # logos handle labels
ax.invert_yaxis()

ax.set_xlabel("EPA neto de turnovers (ganados \u2212 perdidos)", color=FG, fontsize=11)
ax.tick_params(colors=FG, labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)
ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)
ax.xaxis.label.set_color(FG)

# Colorbar
sm = ScalarMappable(cmap=RYG, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.6)
cb.set_label("EPA neto", color=FG, fontsize=8)
cb.ax.yaxis.set_tick_params(color=FG)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG, fontsize=7)
cb.outline.set_edgecolor(GRID)

# Adjust x limits to make room for logos
cur_xlim = ax.get_xlim()
ax.set_xlim(cur_xlim[0] - (cur_xlim[1] - cur_xlim[0]) * 0.10, cur_xlim[1])

# ── TITLES ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97, f"Balance de turnovers en EPA \u2014 NFL {SEASON}",
         ha="center", va="top", fontsize=16, fontweight="bold", color=FG)
fig.text(0.5, 0.92,
         "EPA ganado por turnovers defensivos menos EPA perdido por turnovers ofensivos | + = mejor balance",
         ha="center", va="top", fontsize=10, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data  \u00b7  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 0.96, 0.91])

outfile = f"valor_turnovers_{SEASON}.png"
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
