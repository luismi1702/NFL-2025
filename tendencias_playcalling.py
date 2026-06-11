"""
tendencias_playcalling.py
Tendencias de ataque de un equipo por down × distancia.
Muestra pass%, EPA y diferencial vs media NFL por situación.
NFL 2025
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

SEASON   = 2025
URL_PBP  = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"

BG    = "#0f1115"
CARD  = "#151924"
CARD2 = "#1a2030"
FG    = "#EDEDED"
GRID  = "#2a2f3a"
DPI   = 170
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}

MIN_PLAYS = 10

RYG   = LinearSegmentedColormap.from_list("ryg",  ["#c0392b", "#e8b84b", "#27ae60"])
RYG_r = RYG.reversed()
# Azul a rojo: azul = más carrera (< media pass), rojo = más pase (> media pass)
BR    = LinearSegmentedColormap.from_list("br", ["#2d6cdf", "#888888", "#d84a4a"])

DIST_BINS   = [1, 4, 7, 99]
DIST_LABELS = ["Corta\n1-3 yds", "Media\n4-6 yds", "Larga\n7+ yds"]
DOWNS       = [1, 2, 3]
DOWN_LABELS = ["1er Down", "2º Down", "3er Down"]


def load_logo(team, base_zoom=0.10):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        h, w = img.shape[:2]
        aspect = w / float(h) if h else 1.0
        zoom = base_zoom / HARD_PENALTY[team] if team in HARD_PENALTY else \
               base_zoom / np.clip(1.0 + 0.6 * max(0.0, aspect - 1.3), 1.0, 2.2)
        return OffsetImage(img, zoom=zoom, resample=True)
    except Exception:
        return None


def dist_bucket(y):
    if y <= 3: return 0
    if y <= 6: return 1
    return 2


# ── INPUT ──────────────────────────────────────────────────────────────────────
team = input("Equipo (siglas, p.ej. SF): ").strip().upper()

# ── CARGA ─────────────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
pbp = pd.read_csv(URL_PBP, low_memory=False, compression="infer")
pbp["epa"]     = pd.to_numeric(pbp["epa"],     errors="coerce")
pbp["play_id"] = pd.to_numeric(pbp["play_id"], errors="coerce")

plays = pbp[
    pbp["play_type"].isin(["pass", "run"]) &
    pbp["epa"].notna() &
    pbp["down"].isin([1, 2, 3]) &
    pbp["ydstogo"].between(1, 30)
].copy()

plays["dist_idx"] = plays["ydstogo"].apply(dist_bucket)
plays["pass_att"] = (plays["play_type"] == "pass").astype(int)

off = plays[plays["posteam"] == team].copy()
if off.empty:
    raise SystemExit(f"No hay jugadas para {team}.")

print(f"{team}: {len(off):,} jugadas ofensivas")

# ── CALCULAR GRIDS ────────────────────────────────────────────────────────────
def build_grid(df, stat_col):
    grid = np.full((3, 3), np.nan)
    cnt  = np.zeros((3, 3), dtype=int)
    for d_idx, dn in enumerate(DOWNS):
        for dist_i in range(3):
            sub = df[(df["down"] == dn) & (df["dist_idx"] == dist_i)]
            if len(sub) >= MIN_PLAYS:
                grid[d_idx, dist_i] = sub[stat_col].mean()
                cnt[d_idx, dist_i]  = len(sub)
    return grid, cnt


epa_grid,  cnt_grid  = build_grid(off, "epa")
pass_grid, _         = build_grid(off, "pass_att")

# Medias de liga para cada celda
lg_epa_grid  = np.full((3, 3), np.nan)
lg_pass_grid = np.full((3, 3), np.nan)
for d_idx, dn in enumerate(DOWNS):
    for dist_i in range(3):
        sub = plays[(plays["down"] == dn) & (plays["dist_idx"] == dist_i)]
        if len(sub) >= MIN_PLAYS:
            lg_epa_grid[d_idx, dist_i]  = sub["epa"].mean()
            lg_pass_grid[d_idx, dist_i] = sub["pass_att"].mean()

epa_dev  = epa_grid  - lg_epa_grid
pass_dev = pass_grid - lg_pass_grid


# ── FIGURA ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor=BG)
gs  = gridspec.GridSpec(2, 2,
                         figure=fig,
                         hspace=0.45, wspace=0.35,
                         left=0.07, right=0.97,
                         top=0.87, bottom=0.06,
                         height_ratios=[1, 1],
                         width_ratios=[1, 1])

ax_epa  = fig.add_subplot(gs[0, 0])   # EPA / jugada por celda
ax_pct  = fig.add_subplot(gs[0, 1])   # Pass% por celda
ax_depa = fig.add_subplot(gs[1, 0])   # Delta EPA vs liga
ax_dpass= fig.add_subplot(gs[1, 1])   # Delta pass% vs liga


def draw_heatmap(ax, data, cnt, cmap, title, fmt_fn,
                 vmin=None, vmax=None, xlabel="Distancia", ylabel="Down"):
    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        ax.axis("off")
        return
    vmin = vmin if vmin is not None else np.nanmin(data)
    vmax = vmax if vmax is not None else np.nanmax(data)
    v_abs = max(abs(vmin), abs(vmax), 0.01)
    norm  = Normalize(vmin=-v_abs, vmax=v_abs)

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")

    for d_idx in range(3):
        for dist_i in range(3):
            val = data[d_idx, dist_i]
            n   = cnt[d_idx, dist_i] if cnt is not None else 0
            if np.isnan(val):
                ax.text(dist_i, d_idx, "n/d", ha="center", va="center",
                        color="#555", fontsize=8)
                continue
            txt = fmt_fn(val)
            n_txt = f"\n(n={n})" if n > 0 else ""
            ax.text(dist_i, d_idx, txt + n_txt,
                    ha="center", va="center",
                    color="#0a0e13", fontsize=9, fontweight="bold",
                    linespacing=1.3)

    ax.set_xticks(range(3))
    ax.set_xticklabels(DIST_LABELS, color=FG, fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(DOWN_LABELS, color=FG, fontsize=8)
    ax.set_title(title, color=FG, fontsize=9.5, pad=6, fontweight="bold", loc="left")
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(length=0)


# ── Panel EPA ────────────────────────────────────────────────────────────────
draw_heatmap(ax_epa, epa_grid, cnt_grid, RYG,
             f"EPA / jugada  —  {team}",
             lambda v: f"{'+'if v>=0 else ''}{v:.3f}")

# ── Panel Pass% ─────────────────────────────────────────────────────────────
draw_heatmap(ax_pct, pass_grid, cnt_grid, BR,
             f"Pass%  —  {team}",
             lambda v: f"{v*100:.0f}%",
             vmin=-0.5, vmax=0.5)

# ── Panel Delta EPA vs liga ───────────────────────────────────────────────────
draw_heatmap(ax_depa, epa_dev, cnt_grid, RYG,
             f"Delta EPA vs liga  —  {team}",
             lambda v: f"{'+'if v>=0 else ''}{v:.3f}")

# ── Panel Delta pass% vs liga ─────────────────────────────────────────────────
draw_heatmap(ax_dpass, pass_dev, cnt_grid, BR,
             f"Delta pass% vs liga  —  {team}",
             lambda v: f"{'+'if v>=0 else ''}{v*100:.0f}pp")

# ── Escala de colores de referencia ──────────────────────────────────────────
fig.text(0.5, 0.895,
         "Heatmap EPA: verde = EPA más alto  |  Heatmap Pass%: rojo = más pase que la liga, azul = más carrera",
         ha="center", fontsize=7, color="#888", fontstyle="italic")

# ── Logo + título ─────────────────────────────────────────────────────────────
logo = load_logo(team, base_zoom=0.095)
if logo:
    lax = fig.add_axes([0.03, 0.912, 0.055, 0.075])
    lax.imshow(logo.get_data())
    lax.axis("off")

fig.text(0.5, 0.975,
         f"{team}  |  Tendencias de Playcalling por Down × Distancia  |  NFL {SEASON}",
         ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
fig.text(0.01, 0.008, f"Fuente: nflverse PBP  |  NFL {SEASON}  |  Mín {MIN_PLAYS} jugadas/celda",
         ha="left", va="bottom", fontsize=7, color="#555", fontstyle="italic")
fig.text(0.99, 0.008, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888", alpha=0.8, fontstyle="italic")

outfile = f"tendencias_playcalling_{team}_{SEASON}.png"
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
