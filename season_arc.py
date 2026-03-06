"""
season_arc.py
Curva de temporada NFL — EPA/jugada por semana, rolling 3-week average.
Escribe siglas separadas por coma (ej: KC,SF,DAL) o Enter para auto top/bottom 3.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON    = 2025
ROLL      = 3
MIN_PLAYS = 15
DPI       = 200
BG        = "#0f1115"
FG        = "#EDEDED"
GRID      = "#2a2f3a"

LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}

HIGHLIGHT_COLORS = [
    "#06d6a0", "#ffd166", "#ef476f",
    "#4e9af1", "#ff9f1c", "#c77dff",
]

URL = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"

# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_logo(team, base_zoom=0.028):
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

# ── SELECTOR ATAQUE / DEFENSA ──────────────────────────────────────────────────
mode_raw = input("Ataque o Defensa? (A/D  —  Enter = Ataque): ").strip().upper()
MODE = "D" if mode_raw == "D" else "A"

if MODE == "A":
    team_col   = "posteam"
    mode_label = "Ataque"
    ylabel_txt = "EPA / jugada ofensiva (rolling)"
    # Auto: mejor ataque = EPA más alto
    auto_best_fn  = lambda s: s.nlargest(3).index.tolist()
    auto_worst_fn = lambda s: s.nsmallest(3).index.tolist()
else:
    team_col   = "defteam"
    mode_label = "Defensa"
    ylabel_txt = "EPA permitido / jugada (rolling)"
    # Auto: mejor defensa = EPA más bajo (menos puntos permitidos)
    auto_best_fn  = lambda s: s.nsmallest(3).index.tolist()
    auto_worst_fn = lambda s: s.nlargest(3).index.tolist()

print(f"Modo: {mode_label}")

# ── DATA ───────────────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
df = pd.read_csv(URL, low_memory=False, compression="infer")
print(f"Filas: {len(df):,}")

mask = (
    df["play_type"].isin(["pass", "run"]) &
    df["epa"].notna() &
    df[team_col].notna() &
    df["week"].notna()
)
df = df[mask].copy()

weekly = (
    df.groupby([team_col, "week"])
    .agg(epa_mean=("epa", "mean"), n=("epa", "count"))
    .reset_index()
    .rename(columns={team_col: "team"})
)
weekly = weekly[weekly["n"] >= MIN_PLAYS]

pivot  = weekly.pivot(index="week", columns="team", values="epa_mean").sort_index()
rolled = pivot.rolling(ROLL, min_periods=1).mean()

weeks = rolled.index.tolist()
teams = rolled.columns.tolist()

# ── SELECCIÓN DE EQUIPOS ───────────────────────────────────────────────────────
last_valid  = rolled.ffill().iloc[-1].dropna()
auto_top    = auto_best_fn(last_valid)
auto_bottom = auto_worst_fn(last_valid)

raw = input(
    "Equipos a destacar (ej: KC,SF,DAL  —  Enter para auto top/bottom 3): "
).strip()

if raw == "":
    highlight  = auto_top + auto_bottom
    teams_str  = "Auto top 3 + bottom 3 semana final"
else:
    highlight  = [t.strip().upper() for t in raw.split(",") if t.strip()]
    teams_str  = ", ".join(highlight)

highlight = [t for t in highlight if t in teams]
if not highlight:
    print("Ningún equipo válido, usando auto selección.")
    highlight = auto_top + auto_bottom
    teams_str = "Auto top 3 + bottom 3 semana final"

print(f"Destacados: {highlight}")

# ── PLOT ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 8), facecolor=BG)
ax.set_facecolor(BG)

# Líneas de fondo (todos los equipos)
for team in teams:
    series = rolled[team].dropna()
    if series.empty:
        continue
    ax.plot(series.index, series.values,
            color="#2e3340", linewidth=0.9, alpha=0.7, zorder=1)

# Ampliar X para dejar espacio a logos
xmin, xmax = rolled.index.min(), rolled.index.max()
ax.set_xlim(xmin - 0.5, xmax + 3.0)

# Línea de referencia en 0
ax.axhline(0, color=FG, linewidth=0.8, linestyle="--", alpha=0.35, zorder=2)

# ── LÍNEAS DESTACADAS ──────────────────────────────────────────────────────────
color_map = {team: HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
             for i, team in enumerate(highlight)}

endpoints = []   # (y_data, team, color) para ajuste de etiquetas
for team in highlight:
    if team not in rolled.columns:
        continue
    series = rolled[team].dropna()
    if series.empty:
        continue
    color = color_map[team]
    ax.plot(series.index, series.values,
            color=color, linewidth=2.4, alpha=0.95, zorder=3,
            path_effects=[pe.Stroke(linewidth=4.0, foreground=BG), pe.Normal()])

    # Punto final
    lw, lv = series.index[-1], series.values[-1]
    ax.scatter(lw, lv, color=color, s=50, zorder=5, linewidths=0)
    endpoints.append((lv, lw, team, color))

# ── ETIQUETAS SIN SOLAPAMIENTO ────────────────────────────────────────────────
# Ordenar por valor final y separar posiciones con demasiada proximidad
endpoints.sort(key=lambda x: x[0])
y_range  = ax.get_ylim()[1] - ax.get_ylim()[0]
min_gap  = y_range * 0.06   # 6% del rango Y como espacio mínimo entre etiquetas

adj_y = [ep[0] for ep in endpoints]
for i in range(1, len(adj_y)):
    if adj_y[i] - adj_y[i - 1] < min_gap:
        adj_y[i] = adj_y[i - 1] + min_gap

for (orig_y, lw, team, color), new_y in zip(endpoints, adj_y):
    x_label = xmax + 0.5

    # Línea conectora si el label se desplazó
    if abs(new_y - orig_y) > y_range * 0.01:
        ax.plot([lw, x_label - 0.1], [orig_y, new_y],
                color=color, linewidth=0.6, alpha=0.4, zorder=3)

    # Logo
    logo = load_logo(team, base_zoom=0.028)
    if logo:
        ab = AnnotationBbox(logo, (x_label + 0.6, new_y),
                            frameon=False, zorder=6, box_alignment=(0.0, 0.5))
        ax.add_artist(ab)
    else:
        ax.text(x_label, new_y, team,
                color=color, fontsize=9, fontweight="bold",
                va="center", zorder=6,
                path_effects=[pe.Stroke(linewidth=2, foreground=BG), pe.Normal()])

# ── EJES ───────────────────────────────────────────────────────────────────────
ax.set_xticks(weeks)
ax.set_xticklabels([str(int(w)) for w in weeks], color=FG, fontsize=8)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.tick_params(colors=FG, labelsize=8)
plt.setp(ax.get_yticklabels(), color=FG)

ax.set_xlabel("Semana", color=FG, fontsize=10, labelpad=6)
ax.set_ylabel(ylabel_txt, color=FG, fontsize=10, labelpad=6)

for spine in ax.spines.values():
    spine.set_edgecolor(GRID)

ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

# ── TÍTULOS ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97,
         f"Curva de temporada NFL {SEASON} — {mode_label}  |  EPA/jugada semanal",
         ha="center", va="top", color=FG, fontsize=14, fontweight="bold")
fig.text(0.5, 0.925,
         f"Media movil {ROLL} semanas  |  Jugadas de pase y carrera  |  Destacados: {teams_str}",
         ha="center", va="top", color="#aaaaaa", fontsize=9)
fig.text(0.01, 0.01, f"Fuente: nflverse PBP {SEASON}",
         ha="left", va="bottom", color="#666666", fontsize=7.5)
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", color="#888888", fontsize=9, alpha=0.85, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.90])

out = f"season_arc_{mode_label.lower()}_{SEASON}.png"
fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {out}")
