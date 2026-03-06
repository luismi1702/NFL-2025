"""
resumen_partido.py
Resumen automatico de partido: jugadas clave + lideres EPA.
Salida por consola + PNG opcional.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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


def epa_color(val):
    """Return green or red depending on sign of EPA value."""
    if pd.isna(val):
        return "#888888"
    return "#06d6a0" if val >= 0 else "#ef476f"


def top_player_epa(game_df, play_types, player_col):
    """Return (name, epa_sum) for the player with highest summed EPA."""
    if player_col is None:
        return ("N/D", float("nan"))
    sub = game_df[
        game_df["play_type"].isin(play_types) &
        game_df[player_col].notna() &
        game_df["epa"].notna()
    ]
    if sub.empty:
        return ("N/D", float("nan"))
    grp = sub.groupby(player_col)["epa"].sum()
    best = grp.idxmax()
    return (best, grp[best])

# ── INPUT ──────────────────────────────────────────────────────────────────────
team_a = input("Equipo local/visitante A (siglas): ").strip().upper()
team_b = input("Equipo B (siglas): ").strip().upper()
week   = int(input("Semana: ").strip())

# ── DATA ───────────────────────────────────────────────────────────────────────
print(f"Descargando PBP {SEASON}...")
df = pd.read_csv(URL, low_memory=False, compression="infer")
print(f"Filas descargadas: {len(df):,}")

to_num(df, ["epa", "wpa", "week"])

# Filter to requested week
week_df = df[df["week"] == week].copy()
if week_df.empty:
    raise SystemExit(f"No hay datos para la semana {week}.")

# Find the game
home_col    = pick_col(week_df, "home_team")
away_col    = pick_col(week_df, "away_team")
game_id_col = pick_col(week_df, "game_id")

game_df = pd.DataFrame()

if home_col and away_col and game_id_col:
    team_set = {team_a, team_b}
    mask_game = (
        week_df[home_col].isin(team_set) | week_df[away_col].isin(team_set)
    )
    candidate = week_df[mask_game]
    # Find game_ids where both teams appear
    if not candidate.empty:
        for gid, gdf in candidate.groupby(game_id_col):
            teams_in_game = set(gdf[home_col].dropna().tolist() + gdf[away_col].dropna().tolist())
            if team_a in teams_in_game and team_b in teams_in_game:
                game_df = gdf.copy()
                break

if game_df.empty:
    # Fallback: filter by posteam / defteam
    print("Buscando por posteam/defteam como alternativa...")
    team_set = {team_a, team_b}
    mask_fb = (
        week_df["posteam"].isin(team_set) & week_df["defteam"].isin(team_set)
    )
    game_df = week_df[mask_fb].copy()

if game_df.empty:
    raise SystemExit(f"No se encontro el partido {team_a} vs {team_b} en semana {week}.")

print(f"Jugadas del partido encontradas: {len(game_df):,}")

# ── PLAYER COLUMNS ─────────────────────────────────────────────────────────────
passer_col   = pick_col(game_df, "passer", "passer_player_name")
rusher_col   = pick_col(game_df, "rusher", "rusher_player_name")
receiver_col = pick_col(game_df, "receiver", "receiver_player_name")
desc_col     = pick_col(game_df, "desc", "play_description")
wpa_col      = pick_col(game_df, "wpa")
qtr_col      = pick_col(game_df, "qtr")

# ── STATS PER TEAM ─────────────────────────────────────────────────────────────
team_stats = {}

for team in [team_a, team_b]:
    off_plays = game_df[
        (game_df["posteam"] == team) &
        game_df["play_type"].isin(["pass", "run"]) &
        game_df["epa"].notna()
    ]
    off_epa_play  = off_plays["epa"].mean() if not off_plays.empty else float("nan")
    off_epa_total = off_plays["epa"].sum()  if not off_plays.empty else float("nan")
    n_plays       = len(off_plays)

    qb_name, qb_epa   = top_player_epa(game_df[game_df["posteam"] == team],
                                        ["pass"], passer_col)
    rb_name, rb_epa   = top_player_epa(game_df[game_df["posteam"] == team],
                                        ["run"], rusher_col)
    wr_name, wr_epa   = top_player_epa(game_df[game_df["posteam"] == team],
                                        ["pass"], receiver_col)

    team_stats[team] = {
        "off_epa_play":  off_epa_play,
        "off_epa_total": off_epa_total,
        "n_plays":       n_plays,
        "qb_name":       qb_name,
        "qb_epa":        qb_epa,
        "rb_name":       rb_name,
        "rb_epa":        rb_epa,
        "wr_name":       wr_name,
        "wr_epa":        wr_epa,
    }

# ── TOP 3 PLAYS BY |WPA| ───────────────────────────────────────────────────────
top3_plays = []
if wpa_col:
    real_plays = game_df[
        game_df["play_type"].isin(["pass", "run", "field_goal"]) &
        game_df[wpa_col].notna()
    ]
    if not real_plays.empty:
        top3 = real_plays.nlargest(3, wpa_col)
        for _, row in top3.iterrows():
            qtr_val  = row[qtr_col]  if qtr_col  else "?"
            posteam  = row.get("posteam", "?")
            defteam  = row.get("defteam", "?")
            wpa_val  = row[wpa_col]
            epa_val  = row["epa"] if "epa" in row.index else float("nan")
            desc_raw = str(row[desc_col])[:80] if desc_col else "(sin descripcion)"
            top3_plays.append({
                "qtr":     qtr_val,
                "posteam": posteam,
                "defteam": defteam,
                "wpa":     wpa_val,
                "epa":     epa_val,
                "desc":    desc_raw,
            })

# ── CONSOLE OUTPUT ─────────────────────────────────────────────────────────────
sep = "=" * 70
print(f"\n{sep}")
print(f"  RESUMEN: {team_a} vs {team_b} | Semana {week} NFL {SEASON}")
print(f"{sep}\n")

for team in [team_a, team_b]:
    s = team_stats[team]
    epa_sign = "+" if not pd.isna(s["off_epa_play"]) and s["off_epa_play"] >= 0 else ""
    tot_sign = "+" if not pd.isna(s["off_epa_total"]) and s["off_epa_total"] >= 0 else ""
    print(f"ATAQUE {team}: EPA/jugada {epa_sign}{s['off_epa_play']:.3f} | "
          f"Total EPA {tot_sign}{s['off_epa_total']:.1f} | {s['n_plays']} jugadas")

    qb_e = f"{s['qb_epa']:+.3f}" if not pd.isna(s['qb_epa']) else "N/D"
    rb_e = f"{s['rb_epa']:+.3f}" if not pd.isna(s['rb_epa']) else "N/D"
    wr_e = f"{s['wr_epa']:+.3f}" if not pd.isna(s['wr_epa']) else "N/D"
    print(f"  QB lider: {s['qb_name']} ({qb_e})  |  "
          f"RB: {s['rb_name']} ({rb_e})  |  "
          f"WR/TE: {s['wr_name']} ({wr_e})")
    print()

if top3_plays:
    print("TOP 3 JUGADAS (WPA):")
    for rank, play in enumerate(top3_plays, 1):
        epa_str = f"{play['epa']:+.3f}" if not pd.isna(play['epa']) else "N/D"
        print(f"  {rank}) Q{play['qtr']} {play['posteam']} vs {play['defteam']}: "
              f"{play['desc']} -> WPA {play['wpa']:+.3f} | EPA {epa_str}")
    print()

# ── PNG INFOGRAPHIC ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
ax.set_facecolor(BG)
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)

# Title bar
ax.add_patch(plt.Rectangle((0, 6.1), 10, 0.9, color="#151924", zorder=0))
ax.text(5.0, 6.55, f"{team_a}  vs  {team_b}",
        ha="center", va="center", fontsize=18, fontweight="bold", color=FG, zorder=1)
ax.text(5.0, 6.18, f"Semana {week} | NFL {SEASON}",
        ha="center", va="center", fontsize=10, color="#888888", fontstyle="italic", zorder=1)

# Column backgrounds
ax.add_patch(plt.Rectangle((0.15, 0.55), 4.40, 5.40, color="#151924", zorder=0,
                            linewidth=0, alpha=0.6))
ax.add_patch(plt.Rectangle((5.45, 0.55), 4.40, 5.40, color="#151924", zorder=0,
                            linewidth=0, alpha=0.6))

col_x = {"left": 2.35, "right": 7.65}
teams_order = [team_a, team_b]
col_sides   = ["left", "right"]
col_anchor  = [0.55, 5.75]   # left edge x for text columns

for team, side, anchor_x in zip(teams_order, col_sides, col_anchor):
    cx = col_x[side]
    s  = team_stats[team]

    # Logo
    logo = load_logo(team, base_zoom=0.12)
    if logo is not None:
        ab = AnnotationBbox(logo, (cx, 5.35), frameon=False, zorder=3)
        ax.add_artist(ab)
    else:
        ax.text(cx, 5.35, team, ha="center", va="center",
                fontsize=22, fontweight="bold", color=FG, zorder=3)

    # Team name label
    ax.text(cx, 4.75, team, ha="center", va="center",
            fontsize=14, fontweight="bold", color=FG, zorder=3)

    # EPA/jugada
    epa_c = epa_color(s["off_epa_play"])
    sign  = "+" if not pd.isna(s["off_epa_play"]) and s["off_epa_play"] >= 0 else ""
    ax.text(cx, 4.35, "EPA / jugada", ha="center", va="center",
            fontsize=8, color="#888888", zorder=3)
    ax.text(cx, 4.05, f"{sign}{s['off_epa_play']:.3f}" if not pd.isna(s["off_epa_play"]) else "N/D",
            ha="center", va="center", fontsize=14, fontweight="bold", color=epa_c, zorder=3)

    # Total EPA
    epa_tot_c = epa_color(s["off_epa_total"])
    sign_t    = "+" if not pd.isna(s["off_epa_total"]) and s["off_epa_total"] >= 0 else ""
    ax.text(cx, 3.65, "EPA total", ha="center", va="center",
            fontsize=8, color="#888888", zorder=3)
    ax.text(cx, 3.38, f"{sign_t}{s['off_epa_total']:.1f}" if not pd.isna(s["off_epa_total"]) else "N/D",
            ha="center", va="center", fontsize=12, fontweight="bold", color=epa_tot_c, zorder=3)
    ax.text(cx, 3.12, f"{s['n_plays']} jugadas", ha="center", va="center",
            fontsize=8, color="#666666", zorder=3)

    # Top players
    ax.text(anchor_x + 0.05, 2.80, "QB:", ha="left", va="center",
            fontsize=8, color="#888888", zorder=3)
    qb_e_c = epa_color(s["qb_epa"])
    qb_e_s = f"{s['qb_epa']:+.3f}" if not pd.isna(s["qb_epa"]) else ""
    ax.text(anchor_x + 0.65, 2.80, f"{s['qb_name']}", ha="left", va="center",
            fontsize=8.5, color=FG, fontweight="bold", zorder=3)
    ax.text(anchor_x + 0.05, 2.57, qb_e_s, ha="left", va="center",
            fontsize=7.5, color=qb_e_c, zorder=3)

    ax.text(anchor_x + 0.05, 2.28, "RB:", ha="left", va="center",
            fontsize=8, color="#888888", zorder=3)
    rb_e_c = epa_color(s["rb_epa"])
    rb_e_s = f"{s['rb_epa']:+.3f}" if not pd.isna(s["rb_epa"]) else ""
    ax.text(anchor_x + 0.65, 2.28, f"{s['rb_name']}", ha="left", va="center",
            fontsize=8.5, color=FG, fontweight="bold", zorder=3)
    ax.text(anchor_x + 0.05, 2.05, rb_e_s, ha="left", va="center",
            fontsize=7.5, color=rb_e_c, zorder=3)

    ax.text(anchor_x + 0.05, 1.78, "WR/TE:", ha="left", va="center",
            fontsize=8, color="#888888", zorder=3)
    wr_e_c = epa_color(s["wr_epa"])
    wr_e_s = f"{s['wr_epa']:+.3f}" if not pd.isna(s["wr_epa"]) else ""
    ax.text(anchor_x + 0.65, 1.78, f"{s['wr_name']}", ha="left", va="center",
            fontsize=8.5, color=FG, fontweight="bold", zorder=3)
    ax.text(anchor_x + 0.05, 1.55, wr_e_s, ha="left", va="center",
            fontsize=7.5, color=wr_e_c, zorder=3)

# ── TOP 3 PLAYS SECTION ────────────────────────────────────────────────────────
ax.add_patch(plt.Rectangle((0.15, 0.0), 9.70, 0.52, color="#0d1117", zorder=0))
ax.text(5.0, 0.44, "TOP 3 JUGADAS POR WPA", ha="center", va="center",
        fontsize=7, color="#888888", fontweight="bold", zorder=2)

for rank, play in enumerate(top3_plays[:3]):
    y_txt = 0.30 - rank * 0.13
    epa_str = f"EPA {play['epa']:+.3f}" if not pd.isna(play["epa"]) else ""
    line = (f"#{rank+1} Q{play['qtr']} {play['posteam']}: "
            f"{play['desc'][:65]}  WPA {play['wpa']:+.3f}  {epa_str}")
    ax.text(0.25, y_txt, line, ha="left", va="center",
            fontsize=6.0, color="#cccccc", zorder=2)

# ── FOOTER ────────────────────────────────────────────────────────────────────
fig.text(0.01, 0.01, f"Fuente: nflverse-data  \u00b7  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.01, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

plt.tight_layout(pad=0.5)

outfile = f"resumen_{team_a}_vs_{team_b}_week{week}_{SEASON}.png"
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
