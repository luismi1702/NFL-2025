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
from pbp_loader import cargar_pbp, salida, season_cli, sello
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

SEASON   = season_cli()   # None = auto-detectar última temporada

BG    = "#0f1115"
CARD  = "#151924"
CARD2 = "#1a2030"
FG    = "#EDEDED"
GRID  = "#2a2f3a"
DPI   = 170
LOGOS_DIR    = "logos"

MIN_PLAYS = 25   # con menos, la celda sale "n/d" (n=10-16 destacaba igual que n=456)

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
        # Recorta margenes transparentes: algunos archivos traen mucho aire
        # (NYJ: tinta 3768x1186 en lienzo 4096x4096) y sin recorte salen enanos
        if img.ndim == 3 and img.shape[2] == 4:
            ys, xs = np.where(img[:, :, 3] > 0.02)
            if len(ys):
                img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = img.shape[:2]
        # Normaliza por el area de tinta real; los wordmarks apaisados
        # pueden ensancharse hasta 1.8x para compensar su poca altura
        zoom = base_zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * zoom > 900.0 * (base_zoom):
            zoom = 900.0 * (base_zoom) / w
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
pbp, SEASON = cargar_pbp(SEASON)
print(f"PBP {SEASON}: {len(pbp):,} jugadas REG")
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
# 2 paneles: valor del equipo + Δ vs liga en la MISMA celda (el color es el Δ).
# Antes había 4 paneles y los de "Delta" repetían visualmente los de arriba.
fig = plt.figure(figsize=(14, 6.6), facecolor=BG)
gs  = gridspec.GridSpec(1, 2,
                         figure=fig,
                         wspace=0.32,
                         left=0.07, right=0.97,
                         top=0.78, bottom=0.10)

ax_epa = fig.add_subplot(gs[0, 0])   # EPA por celda (color = Δ vs liga)
ax_pct = fig.add_subplot(gs[0, 1])   # Pass% por celda (color = Δ vs liga)


def draw_heatmap(ax, data, dev, cnt, cmap, title, fmt_fn, dev_fmt_fn,
                 xlabel="Distancia", ylabel="Down"):
    """Celda: valor del equipo (grande) + Δ vs liga y n (pequeño).
    El color de la celda representa el Δ vs liga (simétrico en 0)."""
    valid = dev[~np.isnan(dev)]
    if len(valid) == 0:
        ax.axis("off")
        return
    v_abs = max(abs(np.nanmin(dev)), abs(np.nanmax(dev)), 0.01)
    norm  = Normalize(vmin=-v_abs, vmax=v_abs)

    ax.imshow(dev, cmap=cmap, norm=norm, aspect="auto",
              interpolation="nearest")

    for d_idx in range(3):
        for dist_i in range(3):
            val = data[d_idx, dist_i]
            dv  = dev[d_idx, dist_i]
            n   = cnt[d_idx, dist_i] if cnt is not None else 0
            if np.isnan(val) or np.isnan(dv):
                ax.text(dist_i, d_idx, "n/d", ha="center", va="center",
                        color="#555", fontsize=8)
                continue
            bg = cmap(norm(dv))
            lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            tcol = "#0a0e13" if lum > 0.45 else FG
            ax.text(dist_i, d_idx,
                    f"{fmt_fn(val)}\n{dev_fmt_fn(dv)} · n={n}",
                    ha="center", va="center",
                    color=tcol, fontsize=9.5, fontweight="bold",
                    linespacing=1.45)

    ax.set_xticks(range(3))
    ax.set_xticklabels(DIST_LABELS, color=FG, fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(DOWN_LABELS, color=FG, fontsize=8)
    ax.set_title(title, color=FG, fontsize=10, pad=6, fontweight="bold", loc="left")
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(length=0)


# ── Panel EPA ────────────────────────────────────────────────────────────────
draw_heatmap(ax_epa, epa_grid, epa_dev, cnt_grid, RYG,
             f"EPA / jugada  —  {team}",
             lambda v: f"{'+'if v>=0 else ''}{v:.3f}",
             lambda d: f"Δ liga {'+'if d>=0 else ''}{d:.3f}")

# ── Panel Pass% ─────────────────────────────────────────────────────────────
draw_heatmap(ax_pct, pass_grid, pass_dev, cnt_grid, BR,
             f"Pass%  —  {team}",
             lambda v: f"{v*100:.0f}% pase",
             lambda d: f"Δ liga {'+'if d>=0 else ''}{d*100:.0f}pp")

# ── Escala de colores de referencia ──────────────────────────────────────────
fig.text(0.5, 0.855,
         "Distancia = yardas POR AVANZAR para el 1er down (no profundidad del pase)  |  "
         "Color = diferencia vs media de liga  |  EPA: verde = mejor que la liga  |  "
         "Pass%: rojo = más pase de lo normal, azul = más carrera",
         ha="center", fontsize=7.5, color="#888", fontstyle="italic")

# ── Logo + título ─────────────────────────────────────────────────────────────
logo = load_logo(team, base_zoom=0.095)
if logo:
    lax = fig.add_axes([0.03, 0.87, 0.075, 0.11])
    lax.imshow(logo.get_data())
    lax.axis("off")

fig.text(0.5, 0.965,
         f"{team}  |  Tendencias de Playcalling por Down × Distancia  |  NFL {SEASON}",
         ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
fig.text(0.01, 0.008, f"Fuente: nflverse PBP  |  {sello(SEASON)}  |  Mín {MIN_PLAYS} jugadas/celda",
         ha="left", va="bottom", fontsize=7, color="#555", fontstyle="italic")
fig.text(0.99, 0.008, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888", alpha=0.8, fontstyle="italic")

outfile = salida(f"tendencias_playcalling_{team}_{SEASON}.png", SEASON)
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
