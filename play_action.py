"""
play_action.py
Efectividad del play-action por equipo: EPA/pase con PA vs sin PA.
Dato real is_play_action de FTN charting (nflverse, solo 2022+).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbp_loader import cargar_pbp, cargar_ftn, salida, season_cli
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON = season_cli()   # None = auto-detectar última temporada
DPI    = 170
BG     = "#0f1115"
FG     = "#EDEDED"
GRID   = "#2a2f3a"
RYG    = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

LOGOS_DIR    = "logos"

MIN_PA    = 25   # min pases con play-action
MIN_NO_PA = 50   # min pases sin play-action

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


def to_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in ("true", "1", "t", "yes")

# ── DATA ───────────────────────────────────────────────────────────────────────
df, SEASON = cargar_pbp(SEASON)
print(f"PBP {SEASON}: {len(df):,} jugadas REG")

# Filter pass plays (incluye sacks: el PA que acaba en sack tambien cuenta)
mask = (
    df["play_type"].isin(["pass"]) &
    df["posteam"].notna() &
    df["epa"].notna()
)
df = df[mask].copy()

# FTN charting: is_play_action (solo 2022+)
try:
    ftn, _ = cargar_ftn(SEASON)
except Exception as e:
    sys.exit(f"No se pudo cargar FTN charting ({e}). "
             f"is_play_action solo existe desde 2022.")

ftn = ftn[["nflverse_game_id", "nflverse_play_id", "is_play_action"]].rename(
    columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
ftn["play_id"] = pd.to_numeric(ftn["play_id"], errors="coerce")
df["play_id"]  = pd.to_numeric(df["play_id"],  errors="coerce")
df = df.merge(ftn, on=["game_id", "play_id"], how="left")
df = df[df["is_play_action"].notna()].copy()
if df.empty:
    sys.exit(f"Sin datos de play-action para {SEASON} (FTN charting es 2022+).")
df["pa"] = df["is_play_action"].map(to_bool)
print(f"Pases charteados: {len(df):,}  |  con play-action: {df['pa'].sum():,}")

# Group by team
results = []
for team, grp in df.groupby("posteam"):
    pa_plays    = grp[grp["pa"]]
    no_pa_plays = grp[~grp["pa"]]

    if len(pa_plays) < MIN_PA or len(no_pa_plays) < MIN_NO_PA:
        continue

    epa_pa    = pa_plays["epa"].mean()
    epa_no_pa = no_pa_plays["epa"].mean()
    pa_rate   = grp["pa"].mean() * 100
    boost     = epa_pa - epa_no_pa   # positivo = play-action más efectivo

    results.append({
        "team":      team,
        "epa_pa":    epa_pa,
        "epa_no_pa": epa_no_pa,
        "pa_rate":   pa_rate,
        "boost":     boost,
        "n_pa":      len(pa_plays),
        "n_no_pa":   len(no_pa_plays),
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

    # Componentes: EPA con y sin PA + tasa de uso (el boost solo es la diferencia)
    pa_text_x = xlim_right - (xlim_right - xlim_left) * 0.01
    ax.text(pa_text_x, y,
            f"PA {row['epa_pa']:+.2f} · sin {row['epa_no_pa']:+.2f} · uso {row['pa_rate']:.0f}%",
            ha="right", va="center", fontsize=6.5, color="#aaaaaa", zorder=4)

# Reference line at 0
ax.axvline(0, color=FG, linewidth=0.8, linestyle="--", alpha=0.4, zorder=3)

# Y ticks
ax.set_yticks([])   # sin ticks: los logos hacen de etiqueta
ax.invert_yaxis()

# Axes styling
ax.set_xlabel("Diferencia de EPA/pase (Play-action - Sin play-action)", color=FG, fontsize=10)
ax.tick_params(colors=FG)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)
ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)
plt.setp(ax.get_xticklabels(), color=FG, fontsize=8)

# Texts
fig.text(0.5, 0.97, f"Play-action vs sin play-action — EPA/pase — NFL {SEASON}",
         ha="center", va="top", color=FG, fontsize=14, fontweight="bold")
fig.text(0.5, 0.92,
         "EPA/pase con play-action menos EPA/pase sin play-action | + = el PA anade valor | PA rate = % de pases con play-action",
         ha="center", va="top", color="#aaaaaa", fontsize=9)
fig.text(0.01, 0.01, "Fuente: nflverse PBP + FTN charting (is_play_action)",
         ha="left", va="bottom", color="#666666", fontsize=7)
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", color="#666666", fontsize=7)

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

out = salida(f"play_action_{SEASON}.png", SEASON)
fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {out}")
