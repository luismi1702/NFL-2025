"""
resumen_partido.py
Resumen automatico de partido: jugadas clave + lideres EPA.
Salida por consola + PNG opcional.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbp_loader import (cargar_pbp, salida, season_cli, week_cli, sello,
                        orden_partido)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON = season_cli()   # None = auto-detectar última temporada
BG     = "#0f1115"
FG     = "#EDEDED"
GRID   = "#2a2f3a"
DPI    = 170
LOGOS_DIR    = "logos"
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


def epa_color(val):
    """Return green or red depending on sign of EPA value."""
    if pd.isna(val):
        return "#888888"
    return "#06d6a0" if val >= 0 else "#ef476f"


def top_player_epa(game_df, play_types, player_col, excluir=None):
    """Return (name, epa_sum) for the player with highest summed EPA.
    excluir: nombres a descartar (ej. el QB en el apartado RB, cuyos
    scrambles cuentan como carrera)."""
    if player_col is None:
        return ("N/D", float("nan"))
    sub = game_df[
        game_df["play_type"].isin(play_types) &
        game_df[player_col].notna() &
        game_df["epa"].notna()
    ]
    if excluir:
        sub = sub[~sub[player_col].isin(excluir)]
    if sub.empty:
        return ("N/D", float("nan"))
    grp = sub.groupby(player_col)["epa"].sum()
    best = grp.idxmax()
    return (best, grp[best])

# ── INPUT ──────────────────────────────────────────────────────────────────────
# Siglas habituales que nflverse escribe distinto (Rams = "LA", no "LAR")
ALIAS = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "LVR": "LV", "OAK": "LV",
         "SD": "LAC", "STL": "LA", "GNB": "GB", "KAN": "KC", "NWE": "NE",
         "NOR": "NO", "SFO": "SF", "TAM": "TB"}

def leer_equipo(prompt):
    # lstrip del BOM: PowerShell lo antepone al redirigir texto por stdin
    sigla = input(prompt).strip().lstrip("﻿").upper()
    return ALIAS.get(sigla, sigla)

team_a = leer_equipo("Equipo local/visitante A (siglas): ")
team_b = leer_equipo("Equipo B (siglas): ")
week   = week_cli() or int(input("Semana: ").strip())

# ── DATA ───────────────────────────────────────────────────────────────────────
# solo_reg=False: un resumen puede ser de un partido de playoffs (semanas 19+)
df, SEASON = cargar_pbp(SEASON, solo_reg=False)
print(f"PBP {SEASON}: {len(df):,} jugadas")

to_num(df, ["epa", "wpa", "week", "yards_gained", "complete_pass",
            "incomplete_pass", "interception", "sack", "pass_touchdown",
            "rush_touchdown", "fumble_lost", "third_down_converted",
            "third_down_failed", "yardline_100", "fixed_drive",
            "game_seconds_remaining", "home_wp", "air_yards",
            "vegas_home_wp", "vegas_wpa"])

# Presión real FTN (was_pressure) para la faceta de presión de las claves
try:
    from pbp_loader import cargar_participation
    part, _ = cargar_participation(SEASON)
    part = part[["nflverse_game_id", "play_id", "was_pressure"]].rename(
        columns={"nflverse_game_id": "game_id"})
    part["play_id"] = pd.to_numeric(part["play_id"], errors="coerce")
    df["play_id"]   = pd.to_numeric(df["play_id"],   errors="coerce")
    df = df.merge(part, on=["game_id", "play_id"], how="left")
except Exception as e:
    print(f"  Aviso: participación FTN no disponible ({e}) — sin faceta de presión")

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
    jugados = sorted({f"{a} @ {h}" for a, h in
                      week_df[["away_team", "home_team"]].dropna().itertuples(index=False)})
    raise SystemExit(f"No se encontro el partido {team_a} vs {team_b} en semana {week}.\n"
                     f"Partidos disponibles: {', '.join(jugados)}")

print(f"Jugadas del partido encontradas: {len(game_df):,}")

# ── PLAYER COLUMNS ─────────────────────────────────────────────────────────────
passer_col   = pick_col(game_df, "passer", "passer_player_name")
rusher_col   = pick_col(game_df, "rusher", "rusher_player_name")
receiver_col = pick_col(game_df, "receiver", "receiver_player_name")
desc_col     = pick_col(game_df, "desc", "play_description")
wpa_col      = pick_col(game_df, "vegas_wpa", "wpa")
qtr_col      = pick_col(game_df, "qtr")

# ── STATS PER TEAM ─────────────────────────────────────────────────────────────
team_stats = {}

for team in [team_a, team_b]:
    off_all   = game_df[game_df["posteam"] == team]
    off_plays = off_all[
        off_all["play_type"].isin(["pass", "run"]) &
        off_all["epa"].notna()
    ]
    off_epa_play  = off_plays["epa"].mean() if not off_plays.empty else float("nan")
    off_epa_total = off_plays["epa"].sum()  if not off_plays.empty else float("nan")
    n_plays       = len(off_plays)

    # ── Boxscore ──────────────────────────────────────────────────────────
    pass_p = off_all[off_all["play_type"] == "pass"]
    run_p  = off_all[off_all["play_type"] == "run"]

    comp     = int(pass_p["complete_pass"].fillna(0).sum())
    inc      = int(pass_p["incomplete_pass"].fillna(0).sum())
    ints     = int(pass_p["interception"].fillna(0).sum())
    att      = comp + inc + ints
    pass_yds = int(pass_p.loc[pass_p["complete_pass"] == 1, "yards_gained"].fillna(0).sum())
    pass_td  = int(pass_p["pass_touchdown"].fillna(0).sum())
    sacks    = int(pass_p["sack"].fillna(0).sum())

    rush_att = len(run_p)
    rush_yds = int(run_p["yards_gained"].fillna(0).sum())
    ypc      = rush_yds / rush_att if rush_att else float("nan")
    rush_td  = int(run_p["rush_touchdown"].fillna(0).sum())

    total_yds = int(off_plays["yards_gained"].fillna(0).sum())
    epa_pass  = pass_p["epa"].mean() if len(pass_p) else float("nan")
    epa_run   = run_p["epa"].mean()  if len(run_p)  else float("nan")

    tdc = int(off_all["third_down_converted"].fillna(0).sum())
    tdf = int(off_all["third_down_failed"].fillna(0).sum())

    turnovers  = int(off_all["interception"].fillna(0).sum() +
                     off_all["fumble_lost"].fillna(0).sum())
    explosivas = int((pass_p["yards_gained"].fillna(0) >= 20).sum() +
                     (run_p["yards_gained"].fillna(0) >= 10).sum())

    # Red zone por drive (fixed_drive es único dentro de un partido)
    rz_trips = rz_tds = 0
    drv = off_all[off_all["fixed_drive"].notna()]
    if not drv.empty and "td_team" in drv.columns:
        for _, g in drv.groupby("fixed_drive"):
            if g["yardline_100"].min() <= 20:
                rz_trips += 1
                if (g["td_team"].astype(str) == team).any():
                    rz_tds += 1

    qb_name, qb_epa   = top_player_epa(game_df[game_df["posteam"] == team],
                                        ["pass"], passer_col)
    # Excluir a los QBs del apartado RB (sus scrambles son play_type "run")
    qbs_equipo = set(game_df.loc[game_df["posteam"] == team, passer_col].dropna()) \
                 if passer_col else set()
    rb_name, rb_epa   = top_player_epa(game_df[game_df["posteam"] == team],
                                        ["run"], rusher_col, excluir=qbs_equipo)
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
        "total_yds":     total_yds,
        "comp": comp, "att": att, "pass_yds": pass_yds,
        "pass_td": pass_td, "ints": ints, "sacks": sacks,
        "rush_att": rush_att, "rush_yds": rush_yds, "ypc": ypc, "rush_td": rush_td,
        "epa_pass": epa_pass, "epa_run": epa_run,
        "tdc": tdc, "tdf": tdf,
        "turnovers": turnovers, "explosivas": explosivas,
        "rz_trips": rz_trips, "rz_tds": rz_tds,
    }

# ── TOP 3 PLAYS BY |WPA| ───────────────────────────────────────────────────────
# vegas_home_wp incorpora la linea pregame: un favorito de -7 ARRANCA en ~70%
# en vez de en 50%, asi que una remontada del underdog se ve como lo que es.
# Verificado sobre 2025: difiere >10 puntos de home_wp en el 36,5% de las
# jugadas. Fallback a la neutral para temporadas sin linea.
wp_col = pick_col(game_df, "vegas_home_wp", "home_wp", "wp")

top3_plays = []
if wpa_col:
    real_plays = game_df[
        game_df["play_type"].isin(["pass", "run", "field_goal"]) &
        game_df[wpa_col].notna()
    ].copy()
    if not real_plays.empty:
        # |WPA|: una jugada desastrosa (fumble, pick-six) también es Top-3
        real_plays["_abs_wpa"] = real_plays[wpa_col].abs()
        top3 = real_plays.nlargest(3, "_abs_wpa")
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
                "gsr":     float(row.get("game_seconds_remaining", float("nan"))),
                "wp":      float(row.get(wp_col, float("nan"))) if wp_col else float("nan"),
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
fig, ax = plt.subplots(figsize=(10, 10.6), facecolor=BG)
fig.subplots_adjust(left=0.02, right=0.98, top=0.995, bottom=0.005)
ax.set_facecolor(BG)
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10.6)

# Marcador final (home_score/away_score son el resultado final en nflverse)
try:
    _hs = int(pd.to_numeric(game_df["home_score"], errors="coerce").max())
    _as = int(pd.to_numeric(game_df["away_score"], errors="coerce").max())
    _home = game_df[home_col].dropna().iloc[0] if home_col else None
    score_a = _hs if team_a == _home else _as
    score_b = _as if team_a == _home else _hs
    titulo = f"{team_a}  {score_a} - {score_b}  {team_b}"
except Exception:
    titulo = f"{team_a}  vs  {team_b}"

# Title bar
ax.add_patch(plt.Rectangle((0, 9.6), 10, 1.0, color="#151924", zorder=0))
ax.text(5.0, 10.15, titulo,
        ha="center", va="center", fontsize=18, fontweight="bold", color=FG, zorder=1)
ax.text(5.0, 9.78, f"Semana {week} | NFL {SEASON}",
        ha="center", va="center", fontsize=10, color="#888888", fontstyle="italic", zorder=1)

# Column backgrounds
ax.add_patch(plt.Rectangle((0.15, 3.62), 4.40, 5.75, color="#151924", zorder=0,
                            linewidth=0, alpha=0.6))
ax.add_patch(plt.Rectangle((5.45, 3.62), 4.40, 5.75, color="#151924", zorder=0,
                            linewidth=0, alpha=0.6))

col_x = {"left": 2.35, "right": 7.65}
teams_order = [team_a, team_b]
col_sides   = ["left", "right"]
col_anchor  = [0.55, 5.75]   # left edge x for text columns

for team, side, anchor_x in zip(teams_order, col_sides, col_anchor):
    cx = col_x[side]
    s  = team_stats[team]

    # Logo
    logo = load_logo(team, base_zoom=0.11)
    if logo is not None:
        ab = AnnotationBbox(logo, (cx, 8.85), frameon=False, zorder=3)
        ax.add_artist(ab)
    else:
        ax.text(cx, 8.85, team, ha="center", va="center",
                fontsize=22, fontweight="bold", color=FG, zorder=3)

    # Team name label
    ax.text(cx, 8.28, team, ha="center", va="center",
            fontsize=14, fontweight="bold", color=FG, zorder=3)

    # EPA/jugada
    epa_c = epa_color(s["off_epa_play"])
    sign  = "+" if not pd.isna(s["off_epa_play"]) and s["off_epa_play"] >= 0 else ""
    ax.text(cx, 7.94, "EPA / jugada", ha="center", va="center",
            fontsize=8, color="#888888", zorder=3)
    ax.text(cx, 7.64, f"{sign}{s['off_epa_play']:.3f}" if not pd.isna(s["off_epa_play"]) else "N/D",
            ha="center", va="center", fontsize=15, fontweight="bold", color=epa_c, zorder=3)

    # Total EPA + jugadas (una línea)
    epa_tot_c = epa_color(s["off_epa_total"])
    sign_t    = "+" if not pd.isna(s["off_epa_total"]) and s["off_epa_total"] >= 0 else ""
    tot_s     = f"{sign_t}{s['off_epa_total']:.1f}" if not pd.isna(s["off_epa_total"]) else "N/D"
    ax.text(cx, 7.28, f"EPA total {tot_s}  ·  {s['n_plays']} jugadas",
            ha="center", va="center", fontsize=8.5, color="#aaaaaa", zorder=3)

    # ── Boxscore ─────────────────────────────────────────────────────────
    ax.plot([anchor_x + 0.05, anchor_x + 3.85], [7.04, 7.04],
            color=GRID, linewidth=0.7, alpha=0.7, zorder=2)

    ypc_s = f"{s['ypc']:.1f}" if not pd.isna(s["ypc"]) else "—"
    ep_s  = f"{s['epa_pass']:+.2f}" if not pd.isna(s["epa_pass"]) else "—"
    er_s  = f"{s['epa_run']:+.2f}"  if not pd.isna(s["epa_run"])  else "—"
    filas = [
        ("Yardas totales",  f"{s['total_yds']}"),
        ("Pase",            f"{s['comp']}/{s['att']} · {s['pass_yds']} yds · {s['pass_td']} TD · {s['ints']} INT"),
        ("Carrera",         f"{s['rush_att']} att · {s['rush_yds']} yds · {ypc_s} ypc · {s['rush_td']} TD"),
        ("EPA pase / carr.", f"{ep_s} / {er_s}"),
        ("3er down",        f"{s['tdc']}/{s['tdc'] + s['tdf']}"),
        ("Red Zone",        f"{s['rz_tds']} TD / {s['rz_trips']} viajes"),
        ("Turnovers · Sacks", f"{s['turnovers']} · {s['sacks']}"),
        ("Explosivas",      f"{s['explosivas']}"),
    ]
    for i, (lbl, val) in enumerate(filas):
        fy = 6.78 - i * 0.315
        ax.text(anchor_x + 0.05, fy, lbl, ha="left", va="center",
                fontsize=7.5, color="#888888", zorder=3)
        ax.text(anchor_x + 3.85, fy, val, ha="right", va="center",
                fontsize=7.5, color=FG, fontweight="bold", zorder=3)

    # ── Líderes EPA ──────────────────────────────────────────────────────
    ax.plot([anchor_x + 0.05, anchor_x + 3.85], [4.42, 4.42],
            color=GRID, linewidth=0.7, alpha=0.7, zorder=2)

    def _lider(nombre, epa):
        e = f" {epa:+.1f}" if not pd.isna(epa) else ""
        return f"{nombre}{e}"

    ax.text(anchor_x + 0.05, 4.16,
            f"QB {_lider(s['qb_name'], s['qb_epa'])}   ·   RB {_lider(s['rb_name'], s['rb_epa'])}",
            ha="left", va="center", fontsize=7.3, color=FG, zorder=3)
    ax.text(anchor_x + 0.05, 3.88,
            f"WR/TE {_lider(s['wr_name'], s['wr_epa'])}   (EPA sumado del partido)",
            ha="left", va="center", fontsize=7.3, color="#aaaaaa", zorder=3)

# ── WIN PROBABILITY ─────────────────────────────────────────────────────
home_g = game_df[home_col].dropna().iloc[0] if home_col else team_a
away_g = team_b if home_g == team_a else team_a

_wp_nota = "con la línea de apuestas" if wp_col == "vegas_home_wp" else "sin línea pregame"
ax.text(5.0, 3.42, f"WIN PROBABILITY {_wp_nota}  ·  curva = {home_g}  ·  ● = Top-3 jugadas por |WPA|",
        ha="center", va="center", fontsize=8, color="#888888",
        fontweight="bold", zorder=2)

wp_ax = fig.add_axes([0.075, 0.128, 0.85, 0.170], facecolor="#10141c")
if wp_col and game_df[wp_col].notna().any():
    wpd = (game_df.dropna(subset=[wp_col, "game_seconds_remaining"])
           .sort_values("game_seconds_remaining", ascending=False))
    xw = 3600 - wpd["game_seconds_remaining"]
    yw = wpd[wp_col] * 100
    x_max = max(3600.0, float(xw.max()))

    wp_ax.fill_between(xw, 50, yw, where=(yw >= 50), color="#4e9af1", alpha=0.16, zorder=2)
    wp_ax.fill_between(xw, 50, yw, where=(yw < 50),  color="#d84a4a", alpha=0.16, zorder=2)
    wp_ax.plot(xw, yw, color="#4e9af1", linewidth=1.6, zorder=3)
    wp_ax.axhline(50, color=GRID, linewidth=0.8, linestyle="--", alpha=0.8, zorder=1)
    for q in (900, 1800, 2700, 3600):
        if q < x_max:
            wp_ax.axvline(q, color=GRID, linewidth=0.6, alpha=0.5, zorder=1)
    for qi, q_lbl in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        wp_ax.text(450 + qi * 900, 4, q_lbl, ha="center", va="bottom",
                   color="#555555", fontsize=6.5, zorder=2)
    if x_max > 3600:
        wp_ax.text((3600 + x_max) / 2, 4, "OT", ha="center", va="bottom",
                   color="#555555", fontsize=6.5, zorder=2)

    # Top-3 jugadas sobre la curva
    for i, p in enumerate(top3_plays):
        if not (pd.isna(p["gsr"]) or pd.isna(p["wp"])):
            px_, py_ = 3600 - p["gsr"], p["wp"] * 100
            wp_ax.scatter([px_], [py_], s=48, color="#ffd166",
                          edgecolors=BG, linewidths=1.0, zorder=5)
            wp_ax.annotate(f"{i+1}", (px_, py_), ha="center", va="center",
                           fontsize=5.5, fontweight="bold", color="#0a0e13", zorder=6)

    wp_ax.set_xlim(0, x_max)
    wp_ax.set_ylim(0, 100)
    wp_ax.set_yticks([0, 50, 100])
    wp_ax.set_yticklabels([f"{away_g} 100%", "50%", f"{home_g} 100%"],
                          fontsize=6.5, color="#888888")
    wp_ax.set_xticks([])
    wp_ax.tick_params(length=0)
    for sp in wp_ax.spines.values():
        sp.set_edgecolor(GRID)
else:
    wp_ax.axis("off")

# ── TOP 3 PLAYS (texto) ───────────────────────────────────────────────
for rank, play in enumerate(top3_plays[:3]):
    y_txt = 0.94 - rank * 0.27
    epa_str = f"EPA {play['epa']:+.3f}" if not pd.isna(play["epa"]) else ""
    line = (f"#{rank+1}  Q{play['qtr']} {play['posteam']}: "
            f"{play['desc'][:70]}  ·  WPA {play['wpa']:+.3f}  ·  {epa_str}")
    ax.text(0.30, y_txt, line, ha="left", va="center",
            fontsize=6.3, color="#cccccc", zorder=2)

# ── FOOTER ─────────────────────────────────────────────────────
fig.text(0.01, 0.006, f"Fuente: nflverse-data  ·  {sello(SEASON)}",
         ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig.text(0.99, 0.006, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

# El nombre lleva delante el orden de kickoff (01 = partido inaugural) y los
# equipos como visitante_vs_local, para que la carpeta de la semana se lea como
# se jugo la jornada y no segun el orden en que se tecleen las siglas.
_orden = orden_partido(SEASON, week, team_a, team_b)
if _orden:
    _idx, _visit, _local = _orden
    MOTE = f"{_idx:02d}_resumen_{_visit}_vs_{_local}"
else:
    MOTE = f"resumen_{team_a}_vs_{team_b}"

outfile = salida(f"{MOTE}_{SEASON}.png", SEASON, week)
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")


# ══════════════════════════════════════════════════════════════════════════════
# CLAVES DEL PARTIDO — desviaciones vs la norma de temporada (2º PNG)
# Qué hizo cada equipo distinto a lo que venía haciendo, y contra qué solía
# permitir el rival. Selecciona automáticamente las 4 mayores desviaciones.
# ══════════════════════════════════════════════════════════════════════════════

FACETAS_CFG = {
    # faceta: (min_n_partido, escala_para_ranking, mas_alto_mejor, formato)
    "EPA / pase":                       (10, 0.12, True,  "epa"),
    "EPA / carrera":                    (8,  0.12, True,  "epa"),
    "Carrera exterior (end/tackle)":    (5,  0.15, True,  "epa"),
    "Carrera interior (guard/centro)":  (5,  0.15, True,  "epa"),
    "3er down convertido":              (6,  7.0,  True,  "pct"),
    "Presión sufrida":                  (10, 6.0,  False, "pct"),
    "Jugadas explosivas":               (20, 4.0,  True,  "pct"),
    "Pase profundo (15+ air)":          (4,  0.25, True,  "epa"),
    "EPA en Red Zone":                  (5,  0.20, True,  "epa"),
}


def facetas(sub):
    """Diccionario faceta -> (valor, n) sobre un conjunto de jugadas ofensivas."""
    out = {}
    pr  = sub[sub["play_type"].isin(["pass", "run"]) & sub["epa"].notna()]
    pas = pr[pr["play_type"] == "pass"]
    run = pr[pr["play_type"] == "run"]

    out["EPA / pase"]    = (pas["epa"].mean() if len(pas) else float("nan"), len(pas))
    out["EPA / carrera"] = (run["epa"].mean() if len(run) else float("nan"), len(run))

    if "run_gap" in run.columns:
        ext  = run[run["run_gap"].isin(["end", "tackle"])]
        inte = run[(run["run_gap"] == "guard") | (run["run_location"] == "middle")]
        out["Carrera exterior (end/tackle)"]   = (ext["epa"].mean()  if len(ext)  else float("nan"), len(ext))
        out["Carrera interior (guard/centro)"] = (inte["epa"].mean() if len(inte) else float("nan"), len(inte))

    tdc = sub["third_down_converted"].fillna(0).sum()
    tdf = sub["third_down_failed"].fillna(0).sum()
    out["3er down convertido"] = (tdc / (tdc + tdf) * 100 if (tdc + tdf) > 0 else float("nan"),
                                  int(tdc + tdf))

    if "was_pressure" in sub.columns and pas["was_pressure"].notna().any():
        wpres = pd.to_numeric(pas["was_pressure"], errors="coerce").fillna(0)
        out["Presión sufrida"] = (wpres.mean() * 100, len(pas))

    expl = int((pas["yards_gained"].fillna(0) >= 20).sum() +
               (run["yards_gained"].fillna(0) >= 10).sum())
    out["Jugadas explosivas"] = (expl / len(pr) * 100 if len(pr) else float("nan"), len(pr))

    deep = pas[pas["air_yards"].fillna(-99) >= 15]
    out["Pase profundo (15+ air)"] = (deep["epa"].mean() if len(deep) else float("nan"), len(deep))

    rz = pr[pr["yardline_100"] <= 20]
    out["EPA en Red Zone"] = (rz["epa"].mean() if len(rz) else float("nan"), len(rz))
    return out


gid_actual = game_df[game_id_col].iloc[0] if game_id_col else None

# Norma = resto de partidos de la temporada. Con menos de MIN_PARTIDOS_NORMA
# (semanas 1-3) no hay muestra: se usa la temporada regular anterior entera y
# el PNG lo rotula, porque entre años cambian plantillas y entrenadores.
MIN_PARTIDOS_NORMA = 3
_df_prev = None


def _temporada_previa():
    global _df_prev
    if _df_prev is None:
        _df_prev, _ = cargar_pbp(SEASON - 1)
        to_num(_df_prev, ["epa", "yards_gained", "third_down_converted",
                          "third_down_failed", "yardline_100", "air_yards"])
    return _df_prev


def norma(col, team):
    """(jugadas de la norma, etiqueta) para posteam/defteam == team."""
    sub = df[df[col] == team]
    if gid_actual is not None:
        sub = sub[sub[game_id_col] != gid_actual]
    if sub[game_id_col].nunique() >= MIN_PARTIDOS_NORMA:
        return sub, "temporada"
    prev = _temporada_previa()
    return prev[prev[col] == team], str(SEASON - 1)


def claves_equipo(team, rival):
    """Top-4 desviaciones del partido vs la norma del equipo."""
    g_fac = facetas(game_df[game_df["posteam"] == team])
    resto, t_lab = norma("posteam", team)
    rdefn, _     = norma("defteam", rival)
    normas_usadas.add(t_lab)
    t_fac = facetas(resto)
    r_fac = facetas(rdefn)

    out = []
    for fac, (min_n, escala, hb, fmt) in FACETAS_CFG.items():
        if fac not in g_fac or fac not in t_fac:
            continue
        gv, gn = g_fac[fac]
        tv, tn = t_fac[fac]
        rv     = r_fac.get(fac, (float("nan"), 0))[0]
        if gn < min_n or tn < 25 or pd.isna(gv) or pd.isna(tv):
            continue
        delta  = gv - tv
        out.append(dict(faceta=fac, gv=gv, tv=tv, rv=rv, n=gn, delta=delta, lab=t_lab,
                        score=abs(delta) / escala,
                        mejora=(delta * (1 if hb else -1)) > 0, fmt=fmt))
    out.sort(key=lambda d: -d["score"])
    return out[:4]


def _fv(v, fmt):
    if pd.isna(v):
        return "—"
    return f"{v:+.2f}" if fmt == "epa" else f"{v:.0f}%"


def _fd(d, fmt):
    return f"{d:+.2f} EPA" if fmt == "epa" else f"{d:+.0f} pp"


normas_usadas = set()
claves = {team_a: claves_equipo(team_a, team_b),
          team_b: claves_equipo(team_b, team_a)}
if normas_usadas == {"temporada"}:
    sub_norma  = "su norma de temporada"
    pie_norma  = "norma = resto de partidos de la temporada"
else:
    sub_norma  = f"su temporada {SEASON - 1}"
    pie_norma  = (f"norma = temporada regular {SEASON - 1} (aun no hay "
                  f"{MIN_PARTIDOS_NORMA} partidos en {SEASON}; plantillas cambian)")

# ── PNG CLAVES ────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 7.0), facecolor=BG)
fig2.subplots_adjust(left=0.02, right=0.98, top=0.995, bottom=0.005)
ax2.set_facecolor(BG)
ax2.axis("off")
ax2.set_xlim(0, 10)
ax2.set_ylim(0.55, 7.5)

ax2.add_patch(plt.Rectangle((0, 6.55), 10, 0.95, color="#151924", zorder=0))
ax2.text(5.0, 7.10, f"CLAVES DEL PARTIDO  ·  {titulo}",
         ha="center", va="center", fontsize=15, fontweight="bold", color=FG, zorder=1)
ax2.text(5.0, 6.76,
         f"Semana {week} | NFL {SEASON}  ·  las mayores desviaciones de cada equipo vs {sub_norma}",
         ha="center", va="center", fontsize=9, color="#888888", fontstyle="italic", zorder=1)

ax2.add_patch(plt.Rectangle((0.15, 0.95), 4.40, 5.35, color="#151924",
                             zorder=0, linewidth=0, alpha=0.6))
ax2.add_patch(plt.Rectangle((5.45, 0.95), 4.40, 5.35, color="#151924",
                             zorder=0, linewidth=0, alpha=0.6))

for team, rival, anchor_x, cx in [(team_a, team_b, 0.55, 2.35),
                                   (team_b, team_a, 5.75, 7.65)]:
    logo = load_logo(team, base_zoom=0.055)
    if logo is not None:
        ab = AnnotationBbox(logo, (anchor_x + 0.35, 5.90), frameon=False, zorder=3)
        ax2.add_artist(ab)
    ax2.text(anchor_x + 0.80, 5.90, team, ha="left", va="center",
             fontsize=15, fontweight="bold", color=FG, zorder=3)

    lista = claves[team]
    if not lista:
        ax2.text(cx, 3.6, "Sin desviaciones\ncon muestra suficiente",
                 ha="center", va="center", fontsize=9, color="#666666", zorder=3)
        continue

    for i, c in enumerate(lista):
        y0 = 5.30 - i * 1.15
        col = "#06d6a0" if c["mejora"] else "#d84a4a"

        ax2.text(anchor_x + 0.05, y0, f"▸ {c['faceta']}",
                 ha="left", va="center", fontsize=9, fontweight="bold",
                 color=FG, zorder=3)

        # Minibarra del delta (longitud = magnitud, capada)
        bar_len = min(c["score"], 2.5) / 2.5 * 1.55
        ax2.add_patch(plt.Rectangle((anchor_x + 0.10, y0 - 0.36), 1.55, 0.13,
                                     color="#252c3b", zorder=2))
        ax2.add_patch(plt.Rectangle((anchor_x + 0.10, y0 - 0.36), bar_len, 0.13,
                                     color=col, zorder=3))
        vs_txt = "vs su norma" if c["lab"] == "temporada" else f"vs su {c['lab']}"
        ax2.text(anchor_x + 1.78, y0 - 0.295, f"{_fd(c['delta'], c['fmt'])} {vs_txt}",
                 ha="left", va="center", fontsize=8.5, fontweight="bold",
                 color=col, zorder=3)

        verbo = "generar" if c["faceta"] == "Presión sufrida" else "ceder"
        ax2.text(anchor_x + 0.10, y0 - 0.66,
                 f"partido {_fv(c['gv'], c['fmt'])} (n={c['n']})  ·  "
                 f"{c['lab']} {_fv(c['tv'], c['fmt'])}  ·  "
                 f"{rival} solía {verbo} {_fv(c['rv'], c['fmt'])}",
                 ha="left", va="center", fontsize=7.0, color="#9aa3b5", zorder=3)

fig2.text(0.01, 0.006,
          f"Fuente: nflverse-data  ·  {sello(SEASON)}  ·  {pie_norma}",
          ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
fig2.text(0.99, 0.006, "@CuartayDato",
          ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85,
          fontstyle="italic")

outfile2 = salida(f"{MOTE.replace('resumen_', 'resumen_claves_', 1)}_{SEASON}.png",
                  SEASON, week)
fig2.savefig(outfile2, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig2)
print(f"Guardado: {outfile2}")
