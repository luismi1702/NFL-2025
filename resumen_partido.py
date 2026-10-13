"""
resumen_partido.py
Resumen automatico de partido: jugadas clave + lideres EPA.
Salida por consola + PNG opcional.
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")   # consola cp1252: el − del titular la rompe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbp_loader import (cargar_pbp, salida, season_cli, week_cli, sello,
                        orden_partido)
from estilo import *   # noqa: F401,F403  (estilo comun, docs/estilo-visual.md)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON = season_cli()   # None = auto-detectar última temporada
BG     = "#0f1115"
FG     = "#EDEDED"
GRID   = "#2a2f3a"
DPI    = 170
LOGOS_DIR    = "logos"
RYG_CMAP = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

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

# Segundos jugados desde el kickoff. game_seconds_remaining vuelve a empezar en
# la prorroga (600 en temporada regular, 900 en playoffs): con 3600 - gsr la
# prorroga se pintaba ENCIMA del ultimo cuarto y el titular la contaba como
# "segunda mitad" (13-oct-2026, BUF 37-34 LA)
_gsr = pd.to_numeric(game_df["game_seconds_remaining"], errors="coerce")
_q = (pd.to_numeric(game_df[qtr_col], errors="coerce") if qtr_col
      else pd.Series(1, index=game_df.index))
game_df["_prorroga"] = _q >= 5
_dur_pr = 900 if _gsr[game_df["_prorroga"]].max() > 600 else 600
game_df["_seg_jugados"] = np.where(game_df["_prorroga"],
                                   3600 + (_q - 5) * _dur_pr + (_dur_pr - _gsr),
                                   3600 - _gsr)

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

    # El fumble perdido es de quien lo suelta, no de quien tenia la posesion:
    # un punt que se le escapa al retornador contaba como perdida del equipo
    # que pateaba (SF @ SEA, sem. 5 de 2026: salia SEA 5 - SF 2 y fue 4 - 3)
    if "fumbled_1_team" in game_df.columns:
        fum_perd = int(((game_df["fumble_lost"] == 1) &
                        (game_df["fumbled_1_team"] == team)).sum())
    else:
        fum_perd = int(off_all["fumble_lost"].fillna(0).sum())
    turnovers  = int(off_all["interception"].fillna(0).sum()) + fum_perd
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

    # QB = quien mas dropbacks jugo, no quien mas EPA sumo: un pase de truco
    # de un receptor (Z.Flowers, BAL @ ATL sem. 5 de 2026) le quitaba la linea
    # al quarterback de verdad
    _dp = game_df[(game_df["posteam"] == team) & (game_df["play_type"] == "pass")
                  & game_df["epa"].notna()]
    if passer_col and _dp[passer_col].notna().any():
        qb_name = _dp[passer_col].value_counts().idxmax()
        qb_epa = _dp.loc[_dp[passer_col] == qb_name, "epa"].sum()
    else:
        qb_name, qb_epa = "N/D", float("nan")
    # Excluir a los QBs del apartado RB (sus scrambles son play_type "run")
    qbs_equipo = set(game_df.loc[game_df["posteam"] == team, passer_col].dropna()) \
                 if passer_col else set()
    # RB = quien mas acarreos tuvo (misma razon: un end-around suelto ponia a
    # J.Jefferson o a J.Smith-Njigba de RB)
    _dr = game_df[(game_df["posteam"] == team) & (game_df["play_type"] == "run")
                  & game_df["epa"].notna()]
    if rusher_col:
        _dr = _dr[_dr[rusher_col].notna() & ~_dr[rusher_col].isin(qbs_equipo)]
    if rusher_col and not _dr.empty:
        rb_name = _dr[rusher_col].value_counts().idxmax()
        rb_epa = _dr.loc[_dr[rusher_col] == rb_name, "epa"].sum()
    else:
        rb_name, rb_epa = "N/D", float("nan")
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
                "gsr":     float(row.get("_seg_jugados", float("nan"))),
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

# ── PNG RESUMEN (estilo docs/estilo-visual.md) ─────────────────────────────────
home_g = game_df[home_col].dropna().iloc[0] if home_col else team_a
away_g = team_b if home_g == team_a else team_a
try:
    _hs = int(pd.to_numeric(game_df["home_score"], errors="coerce").max())
    _as = int(pd.to_numeric(game_df["away_score"], errors="coerce").max())
except Exception:
    _hs = _as = None
marcador = {home_g: _hs, away_g: _as}
titulo = (f"{team_a}  {marcador[team_a]} - {marcador[team_b]}  {team_b}"
          if _hs is not None else f"{team_a}  vs  {team_b}")


def nombre_equipo(team):
    """Ciudad para el titular; los equipos que comparten ciudad, por su apodo."""
    try:
        eq = pd.read_parquet("pbp_cache/teams.parquet").set_index("team_abbr")
        if team in ("LA", "LAC", "NYG", "NYJ"):
            return "los " + eq.loc[team, "team_nick"]
        return eq.loc[team, "team_name"].replace(" " + eq.loc[team, "team_nick"], "")
    except Exception:
        return team


def _cap(s):
    return s[0].upper() + s[1:]


def titular_resumen():
    """Plantillas que solo afirman hechos del partido; None si no hay historia."""
    if _hs is None or _hs == _as:
        return None
    W = home_g if _hs > _as else away_g
    Lz = away_g if W == home_g else home_g
    sw, sl = team_stats[W], team_stats[Lz]
    nW, nL = nombre_equipo(W), nombre_equipo(Lz)
    g = game_df.sort_values("play_id") if "play_id" in game_df.columns else game_df
    gsr = pd.to_numeric(g["game_seconds_remaining"], errors="coerce")
    hs_ = pd.to_numeric(g["total_home_score"], errors="coerce").ffill().fillna(0)
    as_ = pd.to_numeric(g["total_away_score"], errors="coerce").ffill().fillna(0)
    dif_w = (hs_ - as_) if W == home_g else (as_ - hs_)      # marcador del ganador tras cada jugada
    # 1. Remontada: SOLO si iba perdiendo en la segunda mitad y su probabilidad
    # bajo del 25% en esa segunda mitad (02-oct-2026: el titular dijo que
    # Cleveland "remonto" por un 20% del primer cuarto en un partido que fue
    # ganando casi entero)
    prorroga = g["_prorroga"]
    segunda = (gsr <= 1800) & ~prorroga
    if wp_col and g[wp_col].notna().any() and (dif_w[segunda] < 0).any():
        wp_home = pd.to_numeric(g[wp_col], errors="coerce")
        wp_w = (wp_home if W == home_g else 1 - wp_home)[segunda].dropna()
        if len(wp_w) and float(wp_w.min()) < 0.25:
            return _cap(f"{nW} {v(nW, 'remontó', 'remontaron')}: en la segunda mitad "
                        f"{v(nW, 'llegó', 'llegaron')} a tener solo un {float(wp_w.min()) * 100:.0f}% "
                        f"de opciones de ganar.")
    # 2. Decidido al final: el ganador se pone por delante (desde el empate o
    # por detras) en los dos ultimos minutos. Sin segundos exactos: el reloj
    # del PBP es el del snap, no el del final de la jugada
    # En la prorroga el reloj vuelve a empezar: "los dos ultimos minutos" serian
    # los de la prorroga, asi que ese partido se cuenta como lo que fue
    previo = dif_w.shift(1).fillna(0)
    if prorroga.any():
        clave = g[prorroga & (previo <= 0) & (dif_w > 0)]
    else:
        clave = g[(gsr <= 120) & (previo <= 0) & (dif_w > 0) & (g["posteam"] == W)]
    if not clave.empty:
        j = clave.iloc[-1]
        if prorroga.any():
            cuando = "en la prórroga"
        else:
            cuando = "en el último minuto" if j["game_seconds_remaining"] <= 60 else "en los dos últimos minutos"
        if str(j.get("field_goal_result", "")) == "made" and pd.notna(j.get("kick_distance")):
            como = f"con un field goal de {int(j['kick_distance'])} yardas"
        elif j.get("touchdown", 0) == 1:
            como = "con un touchdown"
        else:
            como = ""
        return _cap(f"{nW} lo {v(nW, 'ganó', 'ganaron')} {cuando}{(' ' + como) if como else ''}.")
    # 3. Gano con peor EPA por jugada
    if (pd.notna(sw["off_epa_play"]) and pd.notna(sl["off_epa_play"])
            and sw["off_epa_play"] < sl["off_epa_play"] - 0.05):
        return _cap(f"{nW} {v(nW, 'ganó', 'ganaron')} con peor EPA por jugada que {nL}: "
                    f"{num(sw['off_epa_play'])} contra {num(sl['off_epa_play'])}.")
    # 4. Gano con el ataque en negativo (05-oct-2026: IND 30-13 WAS salia como
    # "Indianapolis le gano por 17 puntos a Washington", que solo repite el
    # marcador de debajo; esa plantilla de "paliza" se quito: si no hay
    # conclusion, sin titular)
    if pd.notna(sw["off_epa_play"]) and sw["off_epa_play"] < 0:
        return _cap(f"{nW} {v(nW, 'ganó', 'ganaron')} por {abs(_hs - _as)} con el ataque en negativo: "
                    f"{num(sw['off_epa_play'])} de EPA por jugada.")
    # 5. Las perdidas
    if sl["turnovers"] - sw["turnovers"] >= 2:
        return _cap(f"{nL} lo {v(nL, 'perdió', 'perdieron')} con {sl['turnovers']} pérdidas de balón; {nW} {v(nW, 'cometió', 'cometieron')} "
                    f"{sw['turnovers']}.")
    return None


COLP = colores_pareja(team_a, team_b)
fig = lienzo()
antetitulo(fig, f"Final · semana {week}")
tit = titular_resumen()
print(f"Titular: {tit or '(sin titular)'}")
y = (titular(fig, tit, y=0.925) - 0.025) if tit else 0.905

# Marcador: el ganador en FG, el perdedor en gris
y_top = y
# Con titular de tres lineas el contenido (0,721 de alto) pisaba el pie: lo que
# falte se quita primero a la grafica y despues al paso de las filas
_falta = max(0.0, 0.085 - (y_top - 0.721))
_menos_wp = min(_falta, 0.025)
PASO_FILA = 0.029 - (_falta - _menos_wp) / 9
yc = y - 0.06
for x, t, ha in ((L + 0.03, team_a, "left"), (R - 0.03, team_b, "right")):
    gana = _hs is not None and marcador[t] == max(_hs, _as) and _hs != _as
    poner_logo(fig, t, x + (0.02 if ha == "left" else -0.02), yc, zoom=0.03)
    xs = x + 0.10 if ha == "left" else x - 0.10
    fig.text(xs, yc + 0.012, t, family=TXT, weight="bold", size=10, color=COLP[t],
             va="center", ha=ha)
    fig.text(xs, yc - 0.016, "local" if t == home_g else "visitante", family=TXT, size=7.5,
             color=GRIS1, va="center", ha=ha)
    if _hs is not None:
        numero(fig, str(marcador[t]), 0.44 if ha == "left" else 0.56, yc, size=40,
               color=FG if gana else GRIS2, ha="right" if ha == "left" else "left")
fig.text(0.5, yc, "–", family=DISP, weight="bold", size=24, color=GRIS2, ha="center",
         va="center")

# Boxscore: misma etiqueta en el centro, cada equipo a su lado; el que gana la
# fila en FG negrita, el otro en gris (aqui no hay rangos: no hay verde ni rojo)
y = yc - 0.062
linea_h(fig, y + 0.02)
sa, sb = team_stats[team_a], team_stats[team_b]


def fila(lbl, va, vb, txt_a, txt_b, mayor_mejor=True):
    global y
    mejor_a = (va > vb) if mayor_mejor else (va < vb)
    mejor_b = (vb > va) if mayor_mejor else (vb < va)
    if pd.isna(va) or pd.isna(vb):
        mejor_a = mejor_b = False
    fig.text(0.5, y, lbl, family=TXT, size=7.5, color=GRIS1, ha="center", va="center")
    for xv, txt_v, mejor, ha in ((L + 0.01, txt_a, mejor_a, "left"), (R - 0.01, txt_b, mejor_b, "right")):
        fig.text(xv, y, txt_v, family=TXT, size=9, weight="bold" if mejor else "normal",
                 color=FG if mejor else GRIS1, ha=ha, va="center")
    y -= PASO_FILA


def _tdpct(s):
    tot = s["tdc"] + s["tdf"]
    return s["tdc"] / tot if tot else float("nan")


fila("EPA por jugada", sa["off_epa_play"], sb["off_epa_play"],
     num(sa["off_epa_play"]) if pd.notna(sa["off_epa_play"]) else "–",
     num(sb["off_epa_play"]) if pd.notna(sb["off_epa_play"]) else "–")
fila("Yardas", sa["total_yds"], sb["total_yds"], str(sa["total_yds"]), str(sb["total_yds"]))
fila("Pase", sa["pass_yds"], sb["pass_yds"],
     f"{sa['comp']}/{sa['att']} · {sa['pass_yds']} yds", f"{sb['comp']}/{sb['att']} · {sb['pass_yds']} yds")
fila("Carrera", sa["rush_yds"], sb["rush_yds"],
     f"{sa['rush_att']} · {sa['rush_yds']} yds", f"{sb['rush_att']} · {sb['rush_yds']} yds")
fila("3er down", _tdpct(sa), _tdpct(sb), f"{sa['tdc']}/{sa['tdc'] + sa['tdf']}",
     f"{sb['tdc']}/{sb['tdc'] + sb['tdf']}")
fila("Red zone (TD)", sa["rz_tds"] / sa["rz_trips"] if sa["rz_trips"] else float("nan"),
     sb["rz_tds"] / sb["rz_trips"] if sb["rz_trips"] else float("nan"),
     f"{sa['rz_tds']}/{sa['rz_trips']}", f"{sb['rz_tds']}/{sb['rz_trips']}")
fila("Pérdidas", sa["turnovers"], sb["turnovers"], str(sa["turnovers"]), str(sb["turnovers"]),
     mayor_mejor=False)
fila("Sacks recibidos", sa["sacks"], sb["sacks"], str(sa["sacks"]), str(sb["sacks"]),
     mayor_mejor=False)
fila("Explosivas", sa["explosivas"], sb["explosivas"], str(sa["explosivas"]),
     str(sb["explosivas"]))

# Lideres (EPA sumado): una linea por equipo
linea_h(fig, y + 0.012)
y -= 0.012
etiqueta(fig, "Líderes · EPA sumado", 0.5, y, ha="center")
y -= 0.03


def _lider(pos, nombre, epa):
    if not nombre or pd.isna(epa):
        return None
    return f"{pos} {nombre} {num(epa, 1)}"


for i, (pos, k) in enumerate((("QB", "qb"), ("RB", "rb"), ("WR/TE", "wr"))):
    for xv, s, ha in ((L + 0.01, sa, "left"), (R - 0.01, sb, "right")):
        t = _lider(pos, s[f"{k}_name"], s[f"{k}_epa"])
        if t:
            fig.text(xv, y, t, family=TXT, size=7.8, color=FG if i == 0 else GRIS1, ha=ha,
                     va="center")
    y -= 0.022

# Probabilidad de victoria del local, con las tres jugadas de mas |WPA|
y -= 0.008
linea_h(fig, y + 0.012)
y -= 0.012
_wp_nota = "con la línea de apuestas" if wp_col == "vegas_home_wp" else "sin línea previa"
etiqueta(fig, f"Probabilidad de victoria de {home_g} ({_wp_nota})", L + 0.01, y)
alto_wp = 0.095 - _menos_wp
wp_ax = estilo_ejes(fig.add_axes([L + 0.06, y - 0.02 - alto_wp, R - L - 0.07, alto_wp]), x=False)
if wp_col and game_df[wp_col].notna().any():
    wpd = game_df.dropna(subset=[wp_col, "_seg_jugados"]).sort_values("_seg_jugados")
    xw = wpd["_seg_jugados"]
    yw = wpd[wp_col] * 100
    x_max = max(3600.0, float(xw.max()))
    ch, ca = COLP[home_g], COLP[away_g]
    wp_ax.fill_between(xw, 50, yw, where=(yw >= 50), color=ch, alpha=0.18, lw=0)
    wp_ax.fill_between(xw, 50, yw, where=(yw < 50), color=ca, alpha=0.18, lw=0)
    wp_ax.plot(xw, yw, color=FG, lw=1.1)
    wp_ax.axhline(50, color=GRID, lw=0.8)
    for q in (900, 1800, 2700):
        wp_ax.axvline(q, color=GRID, lw=0.6)
    for qi, q_lbl in enumerate(["1º", "2º", "3º", "4º"]):
        wp_ax.text(450 + qi * 900, 3, q_lbl, ha="center", va="bottom", color=GRIS2, fontsize=6.5)
    if x_max > 3600:
        wp_ax.axvline(3600, color=GRID, lw=0.6)
        wp_ax.text((3600 + x_max) / 2, 3, "PR", ha="center", va="bottom", color=GRIS2, fontsize=6.5)
    for i, p in enumerate(top3_plays):
        if not (pd.isna(p["gsr"]) or pd.isna(p["wp"])):
            px_, py_ = p["gsr"], p["wp"] * 100
            wp_ax.scatter([px_], [py_], s=60, color=FG, edgecolors=CARD, linewidths=1.2, zorder=5,
                          clip_on=False)
            wp_ax.annotate(str(i + 1), (px_, py_), ha="center", va="center", fontsize=5.8,
                           fontweight="bold", color=CARD, zorder=6, annotation_clip=False)
    wp_ax.set_xlim(0, x_max)
    wp_ax.set_ylim(0, 100)
    wp_ax.text(-40, 92, home_g, ha="right", va="center", color=ch, fontsize=6.5, weight="bold")
    wp_ax.text(-40, 8, away_g, ha="right", va="center", color=ca, fontsize=6.5, weight="bold")
y = y - 0.02 - alto_wp - 0.02
for i, p in enumerate(top3_plays[:3]):
    desc = str(p["desc"])
    while desc.startswith("("):            # quita "(8:40)", "(Shotgun)", "(No Huddle...)"
        desc = desc[desc.find(")") + 1:].strip()
    try:
        cuarto = f"{int(float(p['qtr']))}º cuarto" if float(p["qtr"]) <= 4 else "prórroga"
    except (TypeError, ValueError):
        cuarto = "prórroga"
    fig.text(L + 0.01, y, f"{i + 1}", family=TXT, weight="bold", size=7.5, color=FG, va="center")
    fig.text(L + 0.04, y, f"{cuarto} · {desc[:72]}", family=TXT, size=6.8,
             color=GRIS1, va="center")
    y -= 0.021

tarjeta(fig, y - 0.012, y_top)
pie(fig, "Explosiva = pase de 20+ o carrera de 10+ · Puntos = jugadas de mayor impacto en la probabilidad",
    f"NFL {SEASON} · semana {week} · nflverse PBP")

# El nombre lleva delante el orden de kickoff (01 = partido inaugural) y los
# equipos como visitante_vs_local, para que la carpeta de la semana se lea como
# se jugo la jornada y no segun el orden en que se tecleen las siglas.
_orden = orden_partido(SEASON, week, team_a, team_b)
if _orden:
    _idx, _visit, _local = _orden
    # 1_resumen, 2_ficha, 3_resumen_claves: el orden en que Luis quiere ver
    # cada partido en la carpeta (12-oct-2026)
    MOTE = f"{_idx:02d}_1_resumen_{_visit}_vs_{_local}"
else:
    MOTE = f"resumen_{team_a}_vs_{team_b}"

outfile = salida(f"{MOTE}_{SEASON}.png", SEASON, week)
guardar(fig, outfile)


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
    pie_norma  = (f"norma = su temporada {SEASON - 1} (aún no hay "
                  f"{MIN_PARTIDOS_NORMA} partidos en {SEASON})")

# ── PNG CLAVES (estilo docs/estilo-visual.md) ─────────────────────────────────
FRASE_FACETA = {
    "EPA / pase": "El pase de {T}",
    "EPA / carrera": "La carrera de {T}",
    "Carrera exterior (end/tackle)": "La carrera por fuera de {T}",
    "Carrera interior (guard/centro)": "La carrera por dentro de {T}",
    "3er down convertido": "{T} en 3er down",
    "Presión sufrida": "La presión que sufrió {T}",
    "Jugadas explosivas": "Las jugadas explosivas de {T}",
    "Pase profundo (15+ air)": "El pase profundo de {T}",
    "EPA en Red Zone": "{T} en la red zone",
}


NOMBRE_FACETA = {
    "EPA / pase": "EPA por pase", "EPA / carrera": "EPA por carrera",
    "Carrera exterior (end/tackle)": "Carrera por fuera",
    "Carrera interior (guard/centro)": "Carrera por dentro",
    "3er down convertido": "3er down convertido", "Presión sufrida": "Presión sufrida",
    "Jugadas explosivas": "Jugadas explosivas",
    "Pase profundo (15+ air)": "Pase profundo (15+ yardas en el aire)",
    "EPA en Red Zone": "EPA en red zone",
}


def _fv2(v, fmt):
    if pd.isna(v):
        return "–"
    return num(v) if fmt == "epa" else f"{v:.0f}%"


def _fd2(d, fmt):
    return f"{num(d)} EPA" if fmt == "epa" else f"{'+' if d > 0 else '−'}{abs(d):.0f} puntos"


def titular_claves():
    todas = [(c, t) for t in (team_a, team_b) for c in claves[t]]
    if not todas:
        return None
    c, t = max(todas, key=lambda x: x[0]["score"])
    if c["score"] < 1.0:
        return None
    frase = FRASE_FACETA.get(c["faceta"], c["faceta"] + " de {T}").format(T=nombre_equipo(t))
    donde = "su norma" if c["lab"] == "temporada" else f"su {c['lab']}"
    return _cap(f"{frase}: {_fv2(c['gv'], c['fmt'])} en el partido; {_fv2(c['tv'], c['fmt'])} en {donde}.")


fig2 = lienzo()
antetitulo(fig2, f"Claves · semana {week} · {titulo.replace('  ', ' ')}")
tit2 = titular_claves()
y = (titular(fig2, tit2, y=0.925) - 0.025) if tit2 else 0.905
y_top = y
y -= 0.035
for team, rival in ((team_a, team_b), (team_b, team_a)):
    poner_logo(fig2, team, L + 0.035, y, zoom=0.02)
    fig2.text(L + 0.08, y, team, family=TXT, weight="bold", size=10, color=COLP[team],
              va="center")
    fig2.text(R - 0.01, y, f"contra {rival}", family=TXT, size=7.5, color=GRIS1, va="center",
              ha="right")
    y -= 0.045
    lista = claves[team]
    if not lista:
        nota(fig2, "Sin desviaciones con muestra suficiente", L + 0.01, y)
        y -= 0.05
    for c in lista:
        col = RYG[2] if c["mejora"] else RYG[0]
        fig2.text(L + 0.01, y, NOMBRE_FACETA.get(c["faceta"], c["faceta"]), family=TXT, weight="semibold", size=8.8, color=FG,
                  va="center")
        fig2.text(R - 0.01, y, _fd2(c["delta"], c["fmt"]), family=TXT, weight="bold", size=8.8,
                  color=col, va="center", ha="right")
        # barra de la desviacion (longitud = magnitud, capada)
        bw = 0.30
        x0b = R - 0.01 - bw - 0.11
        fig2.add_artist(Rectangle((x0b, y - 0.004), bw, 0.008, transform=fig2.transFigure,
                                  color=GRID, lw=0))
        fig2.add_artist(Rectangle((x0b, y - 0.004), min(c["score"], 2.5) / 2.5 * bw, 0.008,
                                  transform=fig2.transFigure, color=col, lw=0))
        donde = "su norma" if c["lab"] == "temporada" else f"su {c['lab']}"
        verbo = "generar" if c["faceta"] == "Presión sufrida" else "ceder"
        nota(fig2, f"Partido {_fv2(c['gv'], c['fmt'])} ({c['n']} jugadas) · {donde} "
                   f"{_fv2(c['tv'], c['fmt'])} · {rival} solía {verbo} {_fv2(c['rv'], c['fmt'])}",
             L + 0.01, y - 0.025)
        y -= 0.068
    linea_h(fig2, y + 0.02)
    y -= 0.02

tarjeta(fig2, y + 0.005, y_top)
pie(fig2, "Mayores desviaciones sobre la norma · " + pie_norma,
    f"NFL {SEASON} · semana {week} · nflverse PBP")

_mote2 = (MOTE.replace("1_resumen_", "3_resumen_claves_", 1) if _orden
          else MOTE.replace("resumen_", "resumen_claves_", 1))
outfile2 = salida(f"{_mote2}_{SEASON}.png",
                  SEASON, week)
guardar(fig2, outfile2)
