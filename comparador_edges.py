"""
comparador_edges.py
Radar chart comparando dos Edge Rushers (DE/OLB) en 6 dimensiones.
5 métricas de stats_player + EPA on/off del PBP + participación.

Métrica 6: EPA on/off (pbp_participation)
  = EPA_off (sin jugador en campo) - EPA_on (con jugador en campo)
  Positivo = el equipo permite más EPA sin él → buen defensor.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON            = 2025
PBP_URL           = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
STATS_URL         = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_reg_{SEASON}.csv.gz"
PARTICIPATION_URL = f"https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{SEASON}.csv"
BG        = "#0f1115"
FG        = "#EDEDED"
GRID      = "#2a2f3a"
DPI       = 170
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}
RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

EDGE_POSITIONS = {"DE", "OLB"}   # DE en 4-3, OLB en 3-4
MIN_SACKS = 1                    # mínimo sacks para entrar en la cohorte
MIN_SNAPS = 5                    # mínimo jugadas on/off para calcular EPA

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


def get_gsis_id(stats_row):
    """Busca GSIS ID en varias columnas posibles del stats_row."""
    for col in ("player_id", "gsis_id", "player_gsis_id", "nflverse_id"):
        val = stats_row.get(col)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            return str(val).strip()
    return None


def get_player_team(stats_row):
    for col in ("recent_team", "team", "team_abbr"):
        val = stats_row.get(col)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            return str(val).strip()
    return None


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


# ── PARTICIPACIÓN — JOIN ROBUSTO ────────────────────────────────────────────────
_PART_GAME_COL = None   # se detecta una vez

def _detect_part_columns(part_df, pbp_df):
    """Detecta los nombres correctos de columnas en participation y diagnostica."""
    global _PART_GAME_COL

    print("\n--- Diagnóstico pbp_participation ---")
    print(f"  Columnas: {list(part_df.columns)}")

    # Detectar columna game_id en participation
    for cand in ("game_id", "nflverse_game_id", "old_game_id"):
        if cand in part_df.columns:
            _PART_GAME_COL = cand
            break

    if _PART_GAME_COL is None:
        print("  [ERROR] No se encontró columna game_id en participation")
        return False

    # Comparar formato game_id
    pbp_gid  = str(pbp_df["game_id"].dropna().iloc[0]) if "game_id" in pbp_df.columns else "???"
    part_gid = str(part_df[_PART_GAME_COL].dropna().iloc[0])
    print(f"  PBP game_id (ejemplo): {pbp_gid}")
    print(f"  Participation {_PART_GAME_COL} (ejemplo): {part_gid}")
    print(f"  Formatos {'COINCIDEN' if pbp_gid[:4] == part_gid[:4] else 'DISTINTOS — JOIN FALLARÁ'}")

    # Comparar play_id tipos
    print(f"  PBP play_id dtype: {pbp_df['play_id'].dtype if 'play_id' in pbp_df.columns else 'NO EXISTE'}")
    print(f"  Participation play_id dtype: {part_df['play_id'].dtype if 'play_id' in part_df.columns else 'NO EXISTE'}")

    # Muestra defense_players
    dp_sample = str(part_df["defense_players"].dropna().iloc[0]) if "defense_players" in part_df.columns else "NO EXISTE"
    print(f"  defense_players (ejemplo): {dp_sample[:80]}")
    print("-----------------------------------\n")
    return True


def get_on_off_epa(gsis_id, team, pbp_df, part_df):
    """
    EPA on/off usando pbp_participation.
    Positivo = el equipo permite más EPA sin el jugador → buen defensor.
    """
    if gsis_id is None or part_df is None or team is None:
        return float("nan")

    game_col = _PART_GAME_COL or "game_id"

    # Jugadas defensivas del equipo (run + pass)
    def_plays = pbp_df[
        (pbp_df["defteam"] == team) &
        pbp_df["epa"].notna() &
        pbp_df["play_type"].isin(["run", "pass"])
    ][["play_id", "game_id", "epa"]].copy()

    if def_plays.empty:
        return float("nan")

    # Preparar participation con nombre correcto de game_id
    part_sub = part_df[["play_id", game_col, "defense_players"]].copy()
    if game_col != "game_id":
        part_sub = part_sub.rename(columns={game_col: "game_id"})

    merged = def_plays.merge(part_sub, on=["play_id", "game_id"], how="inner")

    if merged.empty:
        return float("nan")

    merged["on_field"] = merged["defense_players"].fillna("").str.contains(
        gsis_id, regex=False
    )

    on_plays  = merged[merged["on_field"]]["epa"]
    off_plays = merged[~merged["on_field"]]["epa"]

    if len(on_plays) < MIN_SNAPS or len(off_plays) < MIN_SNAPS:
        return float("nan")

    return off_plays.mean() - on_plays.mean()


def compute_edge_metrics(player_name, stats_row, sack_df, sack_col, pbp_df, part_df):
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
    gsis_id = get_gsis_id(stats_row)
    team    = get_player_team(stats_row)
    epa_on_off = get_on_off_epa(gsis_id, team, pbp_df, part_df)

    return {
        "Sacks":            sacks,
        "QB hits":          qb_hits,
        "TFL":              tfl,
        "Fumbles forzados": ff,
        "EPA en sacks":     epa_sacks,
        "EPA on/off":       epa_on_off,
    }

# ── INPUT ──────────────────────────────────────────────────────────────────────
e1_input = input("Edge 1 (apellido o nombre parcial, p.ej. Parsons): ").strip()
e2_input = input("Edge 2 (apellido o nombre parcial, p.ej. Thibodeaux): ").strip()

# ── DATA — STATS PLAYER ────────────────────────────────────────────────────────
print(f"Descargando stats_player {SEASON}...")
df_stats = pd.read_csv(STATS_URL, low_memory=False, compression="infer")
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
print(f"Descargando PBP {SEASON}...")
df_pbp = pd.read_csv(PBP_URL, low_memory=False, compression="infer")
print(f"Filas descargadas: {len(df_pbp):,}")
to_num(df_pbp, ["epa", "sack"])

sack_col = pick_col(df_pbp, "sack_player_name")
sack_df  = df_pbp[
    (df_pbp["sack"] == 1) &
    df_pbp["epa"].notna() &
    (df_pbp[sack_col].notna() if sack_col else pd.Series(False, index=df_pbp.index))
].copy() if sack_col else pd.DataFrame()
print(f"Jugadas de sack con EPA: {len(sack_df):,}")

# ── DATA — PARTICIPACIÓN ────────────────────────────────────────────────────────
print(f"Descargando pbp_participation {SEASON}...")
try:
    part_df = pd.read_csv(PARTICIPATION_URL, low_memory=False, compression="infer")
    # Normalizar tipos para el join
    if "play_id" in part_df.columns:
        part_df["play_id"] = pd.to_numeric(part_df["play_id"], errors="coerce")
    if "play_id" in df_pbp.columns:
        df_pbp["play_id"] = pd.to_numeric(df_pbp["play_id"], errors="coerce")
    # game_id como string
    for gid_cand in ("game_id", "nflverse_game_id"):
        if gid_cand in part_df.columns:
            part_df[gid_cand] = part_df[gid_cand].astype(str)
    if "game_id" in df_pbp.columns:
        df_pbp["game_id"] = df_pbp["game_id"].astype(str)
    print(f"Participación: {len(part_df):,} filas")
    _detect_part_columns(part_df, df_pbp)
except Exception as e:
    print(f"  [!] No se pudo cargar participación: {e}")
    print("  [i] EPA on/off mostrará N/D para todos los jugadores.")
    part_df = None

# ── FIND EDGE RUSHERS ──────────────────────────────────────────────────────────
e1_name, e1_stats_row = find_edge(e1_input, edge_stats, df_stats)
e2_name, e2_stats_row = find_edge(e2_input, edge_stats, df_stats)
print(f"Comparando: {e1_name} vs {e2_name}")
print(f"  [info] GSIS IDs: {get_gsis_id(e1_stats_row)!r}, {get_gsis_id(e2_stats_row)!r}")
print(f"  [info] Teams: {get_player_team(e1_stats_row)!r}, {get_player_team(e2_stats_row)!r}")

# ── COMPUTE METRICS FOR ALL EDGES (for normalization) ──────────────────────────
METRIC_KEYS = [
    "Sacks",
    "QB hits",
    "TFL",
    "Fumbles forzados",
    "EPA en sacks",
    "EPA on/off",
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
        name, row.iloc[0], sack_df, sack_col, df_pbp, part_df
    )

# Diagnóstico EPA on/off para los dos jugadores elegidos
nd_count = sum(1 for v in all_raw.values() if pd.isna(v.get("EPA on/off", float("nan"))))
print(f"  [info] EPA on/off: {len(all_raw)-nd_count}/{len(all_raw)} jugadores tienen valor (resto N/D)")

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

ax.plot(angles, v1_radar, color=E1_COLOR, linewidth=2.2, zorder=4, label=e1_name)
ax.fill(angles, v1_radar, color=E1_COLOR, alpha=0.20, zorder=3)
ax.plot(angles, v2_radar, color=E2_COLOR, linewidth=2.2, zorder=4, label=e2_name)
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
fig.text(0.01, 0.01, f"Fuente: nflverse-data + stats_player + pbp_participation  ·  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

outfile = f"comparador_{safe_e1}_{safe_e2}_{SEASON}.png"
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
