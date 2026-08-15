"""
comparador_safeties.py
Radar chart comparando dos Safeties (FS/SS) en 6 dimensiones.

Métricas:
  1. Tackles totales (solo + asistencias) — run support
  2. Pases defendidos
  3. Intercepciones
  4. Blitz (QB hits + sacks)
  5. TFL
  6. Impacto por partido: (tackles + pases defendidos) / PJ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbp_loader import cargar_stats, salida, season_cli, sello
from matplotlib.colors import LinearSegmentedColormap

import sys
# Consolas Windows (cp1252) no soportan caracteres como '≥'
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON            = season_cli()   # None = auto-detectar última temporada
BG   = "#0f1115"
FG   = "#EDEDED"
GRID = "#2a2f3a"
DPI  = 170
RYG  = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

S_POSITIONS = {"FS", "SS", "DB", "S"}
MIN_GAMES   = 6

P1_COLOR = "#06d6a0"
P2_COLOR = "#ffd166"

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


def safe_norm_series(series):
    mn, mx = series.min(), series.max()
    rng = mx - mn
    if rng == 0:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / rng


def find_player(query, s_stats, nm_col):
    matches = s_stats[s_stats[nm_col].str.lower().str.contains(query.lower(), na=False)]
    if matches.empty:
        raise SystemExit(f"Safety '{query}' no encontrado. Prueba con otra parte del nombre.")
    row = matches.sort_values("def_tackles_total", ascending=False).iloc[0]
    if len(matches) > 1:
        names = ", ".join(matches[nm_col].tolist())
        print(f"  [!] '{query}' coincide con: {names} >> seleccionado: {row[nm_col]}")
    return row[nm_col], row


def compute_s_metrics(player_name, stats_row):
    tackles = (
        float(stats_row.get("def_tackles", 0) or 0) +
        float(stats_row.get("def_tackle_assists", 0) or 0)
    )
    pds   = float(stats_row.get("def_pass_defended", float("nan")))
    ints  = float(stats_row.get("def_interceptions", float("nan")))
    blitz = (
        float(stats_row.get("def_qb_hits", 0) or 0) +
        float(stats_row.get("def_sacks", 0) or 0)
    )
    tfl   = float(stats_row.get("def_tackles_for_loss", float("nan")))

    games = float(stats_row.get("games", 0) or 0)
    impacto_pj = (tackles + pds) / games if games > 0 else float("nan")

    return {
        "Tackles totales":       tackles,
        "Pases defendidos":      pds,
        "Intercepciones":        ints,
        "Blitz (QB hits+sacks)": blitz,
        "TFL":                   tfl,
        "Impacto/PJ":            impacto_pj,
    }

# ── INPUT ──────────────────────────────────────────────────────────────────────
p1_input = input("Safety 1 (apellido o nombre parcial, p.ej. Adams): ").strip()
p2_input = input("Safety 2 (apellido o nombre parcial, p.ej. Byard): ").strip()

# ── DATA — STATS PLAYER ────────────────────────────────────────────────────────
df_stats, SEASON = cargar_stats(SEASON)
to_num(df_stats, ["def_tackles", "def_tackle_assists", "def_tackles_for_loss",
                  "def_sacks", "def_qb_hits", "def_pass_defended",
                  "def_interceptions", "games"])

df_stats["def_tackles_total"] = (
    df_stats.get("def_tackles", pd.Series(0, index=df_stats.index)).fillna(0) +
    df_stats.get("def_tackle_assists", pd.Series(0, index=df_stats.index)).fillna(0)
)

s_stats = df_stats[
    df_stats["position"].isin(S_POSITIONS) &
    (df_stats["games"] >= MIN_GAMES)
].copy()
print(f"Safeties con ≥{MIN_GAMES} partidos: {len(s_stats)}")

_id_cols = [c for c in df_stats.columns if "id" in c.lower() or "gsis" in c.lower()]
print(f"  [info] Columnas ID en stats_player: {_id_cols}")

# ── FIND PLAYERS ───────────────────────────────────────────────────────────────
nm_col = pick_col(s_stats, "player_name", "player_display_name")
p1_name, p1_stats_row = find_player(p1_input, s_stats, nm_col)
p2_name, p2_stats_row = find_player(p2_input, s_stats, nm_col)
print(f"Comparando: {p1_name} vs {p2_name}")
team1 = str(p1_stats_row.get("recent_team", "") or "")
team2 = str(p2_stats_row.get("recent_team", "") or "")

# ── COMPUTE METRICS ────────────────────────────────────────────────────────────
METRIC_KEYS = [
    "Tackles totales",
    "Pases defendidos",
    "Intercepciones",
    "Blitz (QB hits+sacks)",
    "TFL",
    "Impacto/PJ",
]

qualified_names = s_stats[nm_col].tolist()
for name in [p1_name, p2_name]:
    if name not in qualified_names:
        qualified_names.append(name)

all_raw = {}
for name in qualified_names:
    row = s_stats[s_stats[nm_col] == name]
    if row.empty:
        continue
    all_raw[name] = compute_s_metrics(name, row.iloc[0])


norm_df     = pd.DataFrame(all_raw).T
norm_scaled = pd.DataFrame(index=norm_df.index)
for col in METRIC_KEYS:
    col_vals = pd.to_numeric(norm_df[col], errors="coerce")
    norm_scaled[col] = safe_norm_series(col_vals)
norm_scaled = norm_scaled.fillna(0.5)

p1_raw  = all_raw[p1_name]
p2_raw  = all_raw[p2_name]
p1_norm = norm_scaled.loc[p1_name]
p2_norm = norm_scaled.loc[p2_name]

# ── CONSOLE TABLE ──────────────────────────────────────────────────────────────
print()
header = f"{'Metrica':<26} {p1_name:<20} {p2_name:<20}"
print(header)
print("-" * len(header))
for metric in METRIC_KEYS:
    v1, v2 = p1_raw[metric], p2_raw[metric]
    s1 = f"{v1:+.3f}" if not pd.isna(v1) else "  N/D "
    s2 = f"{v2:+.3f}" if not pd.isna(v2) else "  N/D "
    print(f"{metric:<26} {s1:<20} {s2:<20}")
print()

# ── RADAR CHART ────────────────────────────────────────────────────────────────
N = len(METRIC_KEYS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

def make_radar_values(norm_row):
    vals = [float(norm_row[m]) for m in METRIC_KEYS]
    vals += vals[:1]
    return vals

v1_radar = make_radar_values(p1_norm)
v2_radar = make_radar_values(p2_norm)

fig = plt.figure(figsize=(8, 8), facecolor=BG)
fig.patch.set_facecolor(BG)
ax = fig.add_subplot(111, polar=True)
ax.set_facecolor("#151924")

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["", "", "", ""], color=FG, fontsize=7)
ax.yaxis.grid(color=GRID, linewidth=0.8, alpha=0.5, linestyle="--")
ax.spines["polar"].set_color(GRID)
ax.set_xticks(angles[:-1])

xticklabels = []
for metric in METRIC_KEYS:
    v1_actual, v2_actual = p1_raw[metric], p2_raw[metric]
    s1 = f"{v1_actual:+.2f}" if not pd.isna(v1_actual) else "N/D"
    s2 = f"{v2_actual:+.2f}" if not pd.isna(v2_actual) else "N/D"
    label = f"{metric}\n{p1_name.split('.')[-1].strip()}: {s1}\n{p2_name.split('.')[-1].strip()}: {s2}"
    xticklabels.append(label)

ax.set_xticklabels(xticklabels, color=FG, fontsize=7.5, ha="center")

ax.plot(angles, v1_radar, color=P1_COLOR, linewidth=2.2, zorder=4, label=f"{p1_name} ({team1})")
ax.fill(angles, v1_radar, color=P1_COLOR, alpha=0.20, zorder=3)
ax.plot(angles, v2_radar, color=P2_COLOR, linewidth=2.2, zorder=4, label=f"{p2_name} ({team2})")
ax.fill(angles, v2_radar, color=P2_COLOR, alpha=0.20, zorder=3)

ref_ring = [0.5] * (N + 1)
ax.plot(angles, ref_ring, color=GRID, linewidth=0.8, linestyle=":", alpha=0.6, zorder=2)
ax.scatter(angles[:-1], v1_radar[:-1], color=P1_COLOR, s=30, zorder=5)
ax.scatter(angles[:-1], v2_radar[:-1], color=P2_COLOR, s=30, zorder=5)

ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
          framealpha=0.25, facecolor="#151924", edgecolor=GRID,
          fontsize=9, labelcolor=FG)
for label in ax.get_xticklabels():
    label.set_color(FG)

safe_p1 = p1_name.split(".")[-1].strip().replace(" ", "_")
safe_p2 = p2_name.split(".")[-1].strip().replace(" ", "_")

fig.text(0.5, 0.97, f"{p1_name} vs {p2_name}",
         ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
fig.text(0.5, 0.92,
         f"Comparación radar — 6 dimensiones | Normalizadas entre Safeties con ≥{MIN_GAMES} partidos",
         ha="center", va="top", fontsize=9, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data · stats_player  ·  {sello(SEASON)}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

outfile = salida(f"comparador_{safe_p1}_{safe_p2}_{SEASON}.png", SEASON)
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
