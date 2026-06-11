"""
comparador_cbs.py
Radar chart comparando dos Cornerbacks en 6 dimensiones.

Métricas:
  1. Pases defendidos
  2. Intercepciones
  3. Tackles totales
  4. TFL
  5. PD + INT (cobertura combinada)
  6. EPA on/off (pbp_participation)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON            = 2025
PBP_URL           = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
STATS_URL         = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_reg_{SEASON}.csv.gz"
PARTICIPATION_URL = f"https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{SEASON}.csv"
BG   = "#0f1115"
FG   = "#EDEDED"
GRID = "#2a2f3a"
DPI  = 170
RYG  = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

CB_POSITIONS = {"CB"}
MIN_GAMES    = 6
MIN_SNAPS    = 5

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


def get_gsis_id(stats_row):
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


def find_player(query, cb_stats, nm_col):
    matches = cb_stats[cb_stats[nm_col].str.lower().str.contains(query.lower(), na=False)]
    if matches.empty:
        raise SystemExit(f"CB '{query}' no encontrado. Prueba con otra parte del nombre.")
    row = matches.sort_values("def_pass_defended", ascending=False).iloc[0]
    if len(matches) > 1:
        names = ", ".join(matches[nm_col].tolist())
        print(f"  [!] '{query}' coincide con: {names} >> seleccionado: {row[nm_col]}")
    return row[nm_col], row


# ── PARTICIPACIÓN ───────────────────────────────────────────────────────────────
_PART_GAME_COL = None

def _detect_part_columns(part_df, pbp_df):
    global _PART_GAME_COL
    print("\n--- Diagnóstico pbp_participation ---")
    print(f"  Columnas: {list(part_df.columns)}")
    for cand in ("game_id", "nflverse_game_id", "old_game_id"):
        if cand in part_df.columns:
            _PART_GAME_COL = cand
            break
    if _PART_GAME_COL is None:
        print("  [ERROR] No se encontró columna game_id en participation")
        return
    pbp_gid  = str(pbp_df["game_id"].dropna().iloc[0]) if "game_id" in pbp_df.columns else "???"
    part_gid = str(part_df[_PART_GAME_COL].dropna().iloc[0])
    print(f"  PBP game_id (ejemplo): {pbp_gid}")
    print(f"  Participation {_PART_GAME_COL} (ejemplo): {part_gid}")
    print(f"  Formatos {'COINCIDEN' if pbp_gid[:4] == part_gid[:4] else 'DISTINTOS'}")
    dp_sample = str(part_df["defense_players"].dropna().iloc[0]) if "defense_players" in part_df.columns else "NO EXISTE"
    print(f"  defense_players (ejemplo): {dp_sample[:80]}")
    print("-----------------------------------\n")


def get_on_off_epa(gsis_id, team, pbp_df, part_df):
    if gsis_id is None or part_df is None or team is None:
        return float("nan")

    game_col = _PART_GAME_COL or "game_id"

    def_plays = pbp_df[
        (pbp_df["defteam"] == team) &
        pbp_df["epa"].notna() &
        pbp_df["play_type"].isin(["run", "pass"])
    ][["play_id", "game_id", "epa"]].copy()

    if def_plays.empty:
        return float("nan")

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


def compute_cb_metrics(player_name, stats_row, pbp_df, part_df):
    pds     = float(stats_row.get("def_pass_defended", float("nan")))
    ints    = float(stats_row.get("def_interceptions", float("nan")))
    tackles = (
        float(stats_row.get("def_tackles", 0) or 0) +
        float(stats_row.get("def_tackle_assists", 0) or 0)
    )
    tfl     = float(stats_row.get("def_tackles_for_loss", float("nan")))
    gsis_id    = get_gsis_id(stats_row)
    team       = get_player_team(stats_row)
    epa_on_off = get_on_off_epa(gsis_id, team, pbp_df, part_df)

    ff = float(stats_row.get("def_fumbles_forced", float("nan")))

    return {
        "Pases defendidos":  pds,
        "Intercepciones":    ints,
        "Tackles":           tackles,
        "TFL":               tfl,
        "Fumbles forzados":  ff,
        "EPA on/off":        epa_on_off,
    }

# ── INPUT ──────────────────────────────────────────────────────────────────────
p1_input = input("CB 1 (apellido o nombre parcial, p.ej. McDuffie): ").strip()
p2_input = input("CB 2 (apellido o nombre parcial, p.ej. Sauce): ").strip()

# ── DATA — STATS PLAYER ────────────────────────────────────────────────────────
print(f"Descargando stats_player {SEASON}...")
df_stats = pd.read_csv(STATS_URL, low_memory=False, compression="infer")
to_num(df_stats, ["def_pass_defended", "def_interceptions", "def_tackles",
                  "def_tackle_assists", "def_tackles_for_loss", "games"])

cb_stats = df_stats[
    df_stats["position"].isin(CB_POSITIONS) &
    (df_stats["games"] >= MIN_GAMES)
].copy()
print(f"CBs con ≥{MIN_GAMES} partidos: {len(cb_stats)}")

_id_cols = [c for c in df_stats.columns if "id" in c.lower() or "gsis" in c.lower()]
print(f"  [info] Columnas ID en stats_player: {_id_cols}")

# ── DATA — PBP ─────────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
df_pbp = pd.read_csv(PBP_URL, low_memory=False, compression="infer")
print(f"Filas descargadas: {len(df_pbp):,}")
to_num(df_pbp, ["epa"])

# ── DATA — PARTICIPACIÓN ────────────────────────────────────────────────────────
print(f"Descargando pbp_participation {SEASON}...")
try:
    part_df = pd.read_csv(PARTICIPATION_URL, low_memory=False, compression="infer")
    if "play_id" in part_df.columns:
        part_df["play_id"] = pd.to_numeric(part_df["play_id"], errors="coerce")
    if "play_id" in df_pbp.columns:
        df_pbp["play_id"] = pd.to_numeric(df_pbp["play_id"], errors="coerce")
    for gid_cand in ("game_id", "nflverse_game_id"):
        if gid_cand in part_df.columns:
            part_df[gid_cand] = part_df[gid_cand].astype(str)
    if "game_id" in df_pbp.columns:
        df_pbp["game_id"] = df_pbp["game_id"].astype(str)
    print(f"Participación: {len(part_df):,} filas")
    _detect_part_columns(part_df, df_pbp)
except Exception as e:
    print(f"  [!] No se pudo cargar participación: {e}")
    part_df = None

# ── FIND PLAYERS ───────────────────────────────────────────────────────────────
nm_col = pick_col(cb_stats, "player_name", "player_display_name")
p1_name, p1_stats_row = find_player(p1_input, cb_stats, nm_col)
p2_name, p2_stats_row = find_player(p2_input, cb_stats, nm_col)
print(f"Comparando: {p1_name} vs {p2_name}")
print(f"  [info] GSIS IDs: {get_gsis_id(p1_stats_row)!r}, {get_gsis_id(p2_stats_row)!r}")
print(f"  [info] Teams: {get_player_team(p1_stats_row)!r}, {get_player_team(p2_stats_row)!r}")

# ── COMPUTE METRICS ────────────────────────────────────────────────────────────
METRIC_KEYS = [
    "Pases defendidos",
    "Intercepciones",
    "Tackles",
    "TFL",
    "Fumbles forzados",
    "EPA on/off",
]

qualified_names = cb_stats[nm_col].tolist()
for name in [p1_name, p2_name]:
    if name not in qualified_names:
        qualified_names.append(name)

all_raw = {}
for name in qualified_names:
    row = cb_stats[cb_stats[nm_col] == name]
    if row.empty:
        continue
    all_raw[name] = compute_cb_metrics(name, row.iloc[0], df_pbp, part_df)

nd_count = sum(1 for v in all_raw.values() if pd.isna(v.get("EPA on/off", float("nan"))))
print(f"  [info] EPA on/off: {len(all_raw)-nd_count}/{len(all_raw)} jugadores tienen valor")

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
header = f"{'Metrica':<22} {p1_name:<20} {p2_name:<20}"
print(header)
print("-" * len(header))
for metric in METRIC_KEYS:
    v1, v2 = p1_raw[metric], p2_raw[metric]
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

ax.plot(angles, v1_radar, color=P1_COLOR, linewidth=2.2, zorder=4, label=p1_name)
ax.fill(angles, v1_radar, color=P1_COLOR, alpha=0.20, zorder=3)
ax.plot(angles, v2_radar, color=P2_COLOR, linewidth=2.2, zorder=4, label=p2_name)
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
         f"Comparación radar — 6 dimensiones | Normalizadas entre CBs con ≥{MIN_GAMES} partidos",
         ha="center", va="top", fontsize=9, color="#888888", fontstyle="italic")
fig.text(0.01, 0.01, f"Fuente: nflverse-data + stats_player + pbp_participation  ·  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.91])

outfile = f"comparador_{safe_p1}_{safe_p2}_{SEASON}.png"
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
