"""
comparador_edges.py
Radar chart comparando dos Edge Rushers (DE/OLB) en 6 dimensiones.
5 métricas de stats_player + EPA en sacks del PBP.

Métrica 6: Disrupción por partido = (sacks + QB hits + TFL) / PJ.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbp_loader import cargar_pbp, cargar_stats, cargar_pfr, salida, season_cli, sello
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import sys
# Consolas Windows (cp1252) no soportan caracteres como '≥'
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON            = season_cli()   # None = auto-detectar última temporada
BG        = "#0f1115"
FG        = "#EDEDED"
GRID      = "#2a2f3a"
DPI       = 170
LOGOS_DIR    = "logos"
RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

EDGE_POSITIONS = {"DE", "OLB"}   # DE en 4-3, OLB en 3-4
MIN_SACKS = 1                    # mínimo sacks para entrar en la cohorte

E1_COLOR = "#06d6a0"
E2_COLOR = "#ffd166"

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


def find_edge(query, edge_stats, df_stats):
    """Return player_name of the Edge Rusher matching query (most sacks if multiple)."""
    nm_col = pick_col(edge_stats, "player_name", "player_display_name")
    if nm_col is None:
        raise SystemExit("No se encontró columna de nombre en stats_player.")
    matches = edge_stats[edge_stats[nm_col].str.lower().str.contains(query.lower(), na=False)]
    if matches.empty:
        raise SystemExit(f"Edge rusher '{query}' no encontrado. Prueba con otra parte del nombre.")
    row = matches.sort_values("def_sacks", ascending=False).iloc[0]
    if len(matches) > 1:
        names = ", ".join(matches[nm_col].tolist())
        print(f"  [!] '{query}' coincide con: {names} >> seleccionado: {row[nm_col]}")
    all_matches = df_stats[
        df_stats[nm_col].str.lower().str.contains(query.lower(), na=False) &
        df_stats["position"].isin(EDGE_POSITIONS)
    ]
    excluded = all_matches[~all_matches[nm_col].isin(matches[nm_col])]
    for _, exc in excluded.iterrows():
        print(f"  [i] {exc[nm_col]} excluido de la cohorte "
              f"({exc.get('def_sacks', 0):.0f} sacks < mínimo {MIN_SACKS})")
    return row[nm_col], row


def compute_edge_metrics(player_name, stats_row, sack_df, sack_col):
    # 1. Sacks
    sacks = float(stats_row.get("def_sacks", float("nan")))

    # 2. QB hits
    qb_hits = float(stats_row.get("def_qb_hits", float("nan")))

    # 3. TFL
    tfl = float(stats_row.get("def_tackles_for_loss", float("nan")))

    # 4. Fumbles forzados
    ff = float(stats_row.get("def_fumbles_forced", float("nan")))

    # 5. EPA en sacks (invertido)
    if sack_col:
        player_sacks = sack_df[sack_df[sack_col] == player_name]["epa"]
        epa_sacks = -player_sacks.mean() if len(player_sacks) >= 3 else float("nan")
    else:
        epa_sacks = float("nan")

    # 6. EPA on/off
    games = float(stats_row.get("games", 0) or 0)
    disrupcion_pj = (sacks + qb_hits + tfl) / games if games > 0 else float("nan")

    return {
        "Sacks":            sacks,
        "QB hits":          qb_hits,
        "TFL":              tfl,
        "Fumbles forzados": ff,
        "EPA en sacks":     epa_sacks,
        "Presiones/PJ":     PRESIONES.get(_clave(player_name), float("nan")),
    }

# ── INPUT ──────────────────────────────────────────────────────────────────────
e1_input = input("Edge 1 (apellido o nombre parcial, p.ej. Parsons): ").strip()
e2_input = input("Edge 2 (apellido o nombre parcial, p.ej. Thibodeaux): ").strip()

# ── DATA — STATS PLAYER ────────────────────────────────────────────────────────
df_stats, SEASON = cargar_stats(SEASON)
to_num(df_stats, ["def_sacks", "def_qb_hits", "def_tackles_for_loss",
                  "def_fumbles_forced", "def_pass_defended", "games"])

edge_stats = df_stats[
    df_stats["position"].isin(EDGE_POSITIONS) &
    (df_stats["def_sacks"] >= MIN_SACKS)
].copy()
print(f"Edge rushers con ≥{MIN_SACKS} sack: {len(edge_stats)}")

# Diagnóstico: columnas de ID en stats_player
_id_cols = [c for c in df_stats.columns if "id" in c.lower() or "gsis" in c.lower()]
print(f"  [info] Columnas ID en stats_player: {_id_cols}")

# ── DATA — PBP ─────────────────────────────────────────────────────────────────
df_pbp, SEASON = cargar_pbp(SEASON)
print(f"PBP {SEASON}: {len(df_pbp):,} jugadas REG")
to_num(df_pbp, ["epa", "sack"])

sack_col = pick_col(df_pbp, "sack_player_name")
sack_df  = df_pbp[
    (df_pbp["sack"] == 1) &
    df_pbp["epa"].notna() &
    (df_pbp[sack_col].notna() if sack_col else pd.Series(False, index=df_pbp.index))
].copy() if sack_col else pd.DataFrame()
print(f"Jugadas de sack con EPA: {len(sack_df):,}")

# ── FIND EDGE RUSHERS ──────────────────────────────────────────────────────────
e1_name, e1_stats_row = find_edge(e1_input, edge_stats, df_stats)
e2_name, e2_stats_row = find_edge(e2_input, edge_stats, df_stats)
print(f"Comparando: {e1_name} vs {e2_name}")
team1 = str(e1_stats_row.get("recent_team", "") or "")
team2 = str(e2_stats_row.get("recent_team", "") or "")

# ── COMPUTE METRICS FOR ALL EDGES (for normalization) ──────────────────────────

# Presiones reales por jugador (Pro Football Reference). Sustituyen a
# "Disrupción/PJ", que era un compuesto de sacks+hits+TFL y correlacionaba 0.933
# con los sacks que ya estaban en el radar: ocupaba un eje sin aportar nada.
def _clave(n):
    """Clave para cruzar fuentes con formatos de nombre distintos.

    stats_player abrevia ("M.Garrett") y PFR no ("Myles Garrett"), asi que la
    clave es inicial del nombre + apellido: ambos caen en ("m", "garrett").
    """
    n = str(n).lower().replace(".", " ").replace("'", "").replace("-", " ")
    partes = [p for p in n.split() if p not in
              ("jr", "sr", "ii", "iii", "iv", "v")]
    if not partes:
        return ""
    if len(partes) == 1:
        return partes[0]
    return partes[0][0] + " " + partes[-1]


PRESIONES = {}
try:
    _pfr, _ = cargar_pfr("def", SEASON)
    for _, _r in _pfr.iterrows():
        _g = float(_r.get("g", 0) or 0)
        if _g > 0:
            PRESIONES[_clave(_r["player"])] = float(_r.get("prss", 0) or 0) / _g
    print(f"PFR: presiones de {len(PRESIONES)} defensores")
except Exception as e:
    print(f"  Aviso: sin datos de presiones PFR ({e}) — ese eje saldra N/D")

METRIC_KEYS = [
    "Sacks",
    "QB hits",
    "TFL",
    "Fumbles forzados",
    "EPA en sacks",
    "Presiones/PJ",
]

nm_col = pick_col(edge_stats, "player_name", "player_display_name")

qualified_names = edge_stats[nm_col].tolist()
for name in [e1_name, e2_name]:
    if name not in qualified_names:
        qualified_names.append(name)

all_raw = {}
for name in qualified_names:
    row = edge_stats[edge_stats[nm_col] == name]
    if row.empty:
        continue
    all_raw[name] = compute_edge_metrics(
        name, row.iloc[0], sack_df, sack_col
    )


norm_df     = pd.DataFrame(all_raw).T
norm_scaled = pd.DataFrame(index=norm_df.index)
for col in METRIC_KEYS:
    col_vals = pd.to_numeric(norm_df[col], errors="coerce")
    norm_scaled[col] = safe_norm_series(col_vals)
norm_scaled = norm_scaled.fillna(0.5)

e1_raw  = all_raw[e1_name]
e2_raw  = all_raw[e2_name]
e1_norm = norm_scaled.loc[e1_name]
e2_norm = norm_scaled.loc[e2_name]

# ── CONSOLE TABLE ──────────────────────────────────────────────────────────────
print()
header = f"{'Metrica':<22} {e1_name:<20} {e2_name:<20}"
print(header)
print("-" * len(header))
for metric in METRIC_KEYS:
    v1, v2 = e1_raw[metric], e2_raw[metric]
    s1 = f"{v1:+.3f}" if not pd.isna(v1) else "  N/D "
    s2 = f"{v2:+.3f}" if not pd.isna(v2) else "  N/D "
    print(f"{metric:<22} {s1:<20} {s2:<20}")
print()

# ── RADAR CHART ────────────────────────────────────────────────────────────────
N = len(METRIC_KEYS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

def make_radar_values(norm_row):
    vals = [float(norm_row[m]) for m in METRIC_KEYS]
    vals += vals[:1]
    return vals

v1_radar = make_radar_values(e1_norm)
v2_radar = make_radar_values(e2_norm)

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
    v1_actual, v2_actual = e1_raw[metric], e2_raw[metric]
    s1 = f"{v1_actual:+.2f}" if not pd.isna(v1_actual) else "N/D"
    s2 = f"{v2_actual:+.2f}" if not pd.isna(v2_actual) else "N/D"
    label = f"{metric}\n{e1_name.split('.')[-1].strip()}: {s1}\n{e2_name.split('.')[-1].strip()}: {s2}"
    xticklabels.append(label)

ax.set_xticklabels(xticklabels, color=FG, fontsize=7.5, ha="center")

ax.plot(angles, v1_radar, color=E1_COLOR, linewidth=2.2, zorder=4, label=f"{e1_name} ({team1})")
ax.fill(angles, v1_radar, color=E1_COLOR, alpha=0.20, zorder=3)
ax.plot(angles, v2_radar, color=E2_COLOR, linewidth=2.2, zorder=4, label=f"{e2_name} ({team2})")
ax.fill(angles, v2_radar, color=E2_COLOR, alpha=0.20, zorder=3)

ref_ring = [0.5] * (N + 1)
ax.plot(angles, ref_ring, color=GRID, linewidth=0.8, linestyle=":", alpha=0.6, zorder=2)
ax.scatter(angles[:-1], v1_radar[:-1], color=E1_COLOR, s=30, zorder=5)
ax.scatter(angles[:-1], v2_radar[:-1], color=E2_COLOR, s=30, zorder=5)

ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
          framealpha=0.25, facecolor="#151924", edgecolor=GRID,
          fontsize=9, labelcolor=FG)
for label in ax.get_xticklabels():
    label.set_color(FG)

safe_e1 = e1_name.split(".")[-1].strip().replace(" ", "_")
safe_e2 = e2_name.split(".")[-1].strip().replace(" ", "_")

fig.text(0.5, 0.97, f"{e1_name} vs {e2_name}",
         ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
fig.text(0.5, 0.92,
         f"Comparación radar — 6 dimensiones | Normalizadas entre DE/OLB con ≥{MIN_SACKS} sack",
         ha="center", va="top", fontsize=9, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data · stats_player + PBP  ·  {sello(SEASON)}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

outfile = salida(f"comparador_{safe_e1}_{safe_e2}_{SEASON}.png", SEASON)
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
