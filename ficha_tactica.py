"""
ficha_tactica.py
Ficha tactica de un partido: 11 metricas de calidad con su PERCENTIL contra
los partidos de equipo de la temporada anterior, mas 3 de identidad sin juicio.

Por que existe: el segundo PNG de resumen_partido cuenta desviaciones (que hizo
distinto cada equipo). Esta ficha cuenta NIVEL (como de bueno fue, comparado con
la liga). Las dos lecturas son distintas y se complementan.

Todas las metricas salen de columnas del PBP de nflverse — ninguna necesita
charting de pago. Las de identidad (bajo centro, scrambles) van sin percentil
ni color: ser el que mas corre bajo centro no es bueno ni malo, es una forma de
jugar (decision de jul-2026, ver docs/decisiones.md).

Uso:
  python ficha_tactica.py --season 2026 --week 1     # pide los dos equipos
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from pbp_loader import (cargar_pbp, salida, season_cli, week_cli, sello,
                        orden_partido)
from estilo import *   # noqa: F401,F403  (estilo comun, docs/estilo-visual.md)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON = season_cli()
BG     = "#0f1115"
CARD   = "#151924"
FG     = "#EDEDED"
GRID   = "#2a2f3a"
VERDE  = "#06d6a0"
AMBAR  = "#ffd166"
ROJO   = "#d84a4a"
NEUTRO = "#7f8796"
LOGOS_DIR = "logos"
DPI    = 200

ALIAS = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "LVR": "LV", "OAK": "LV",
         "SD": "LAC", "STL": "LA", "GNB": "GB", "KAN": "KC", "NWE": "NE",
         "NOR": "NO", "SFO": "SF", "TAM": "TB"}

# (etiqueta, clave, formato, mas_alto_mejor, seccion)
METRICAS = [
    ("EPA / jugada",          "epa",      "epa", True,  "GLOBAL"),
    ("Jugadas exitosas",      "sr",       "pct", True,  "GLOBAL"),
    ("Jugadas explosivas",    "exp",      "pct", True,  "GLOBAL"),
    ("EPA / dropback",        "epa_db",   "epa", True,  "PASE"),
    ("Dropbacks exitosos",    "sr_db",    "pct", True,  "PASE"),
    ("Sacks encajados",       "sack",     "pct", False, "PASE"),
    ("EPA / carrera",         "epa_ru",   "epa", True,  "CARRERA"),
    ("Carreras exitosas",     "sr_ru",    "pct", True,  "CARRERA"),
    ("EPA en 1er y 2o down",  "epa_early","epa", True,  "SITUACIONAL"),
    ("Exito en 3er y 4o down","sr_late",  "pct", True,  "SITUACIONAL"),
    ("EPA en Red Zone",       "epa_rz",   "epa", True,  "SITUACIONAL"),
    ("Jugadas bajo centro",   "uc",       "pct", None,  "IDENTIDAD"),
    ("Scrambles del QB",      "scr",      "pct", None,  "IDENTIDAD"),
    ("EPA / scramble",        "epa_scr",  "epa", None,  "IDENTIDAD"),
]


# ── HELPERS ────────────────────────────────────────────────────────────────────
def leer_equipo(prompt):
    sigla = input(prompt).strip().lstrip("﻿").upper()
    return ALIAS.get(sigla, sigla)


def load_logo(team, zoom=0.055):
    """Logo normalizado por area de tinta real (ver CLAUDE.md)."""
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        if img.ndim == 3 and img.shape[2] == 4:
            ys, xs = np.where(img[:, :, 3] > 0.02)
            if len(ys):
                img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = img.shape[:2]
        z = zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * z > 900.0 * zoom:
            z = 900.0 * zoom / w
        return OffsetImage(img, zoom=z, resample=True)
    except Exception:
        return None


def _m(serie):
    return serie.mean() if len(serie) else np.nan


def metricas(o):
    """Las 14 casillas de un ataque, a partir de sus jugadas del partido."""
    o = o[o["play_type"].isin(["pass", "run"]) & o["epa"].notna()]
    for c in ("qb_kneel", "qb_spike"):
        if c in o.columns:
            o = o[o[c] != 1]
    if len(o) < 5:
        return None

    es_db = (o["pass_attempt"].fillna(0) > 0) | (o["sack"].fillna(0) > 0) | \
            (o["qb_scramble"].fillna(0) == 1)
    db  = o[es_db]
    ru  = o[(o["rush_attempt"] == 1) & (o["qb_scramble"] != 1)]
    pas = o[o["pass_attempt"] == 1]
    scr = o[o["qb_scramble"] == 1]
    rz  = o[o["yardline_100"] <= 20]
    uc  = o[o["shotgun"] == 0] if "shotgun" in o.columns else o.iloc[0:0]

    expl = (pas["yards_gained"].fillna(0) >= 20).sum() + \
           (ru["yards_gained"].fillna(0) >= 10).sum()

    return {
        "epa":       _m(o["epa"]),
        "sr":        _m(o["success"]) * 100,
        "exp":       expl / len(o) * 100,
        "epa_db":    _m(db["epa"]),
        "sr_db":     _m(db["success"]) * 100,
        "sack":      db["sack"].fillna(0).sum() / len(db) * 100 if len(db) else np.nan,
        "epa_ru":    _m(ru["epa"]),
        "sr_ru":     _m(ru["success"]) * 100,
        "epa_early": _m(o[o["down"] <= 2]["epa"]),
        "sr_late":   _m(o[o["down"] >= 3]["success"]) * 100,
        "epa_rz":    _m(rz["epa"]),
        "uc":        len(uc) / len(o) * 100,
        "scr":       len(scr) / len(db) * 100 if len(db) else np.nan,
        "epa_scr":   _m(scr["epa"]),
        "n":         len(o),
    }


def fmt(v, f):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.2f}" if f == "epa" else f"{v:.0f}%"




# ── INPUT ──────────────────────────────────────────────────────────────────────
team_a = leer_equipo("Equipo A (siglas): ")
team_b = leer_equipo("Equipo B (siglas): ")
week   = week_cli() or int(input("Semana: ").strip())

# ── DATOS ──────────────────────────────────────────────────────────────────────
df, SEASON = cargar_pbp(SEASON, solo_reg=False, avisar=False)
w = df[df["week"] == week]
gid = None
for g, sub in w.groupby("game_id"):
    equipos = set(sub["home_team"].dropna()) | set(sub["away_team"].dropna())
    if {team_a, team_b} <= equipos:
        gid, juego = g, sub
        break
if gid is None:
    jugados = sorted({f"{a} @ {h}" for a, h in
                      w[["away_team", "home_team"]].dropna().itertuples(index=False)})
    raise SystemExit(f"No se encontro {team_a} vs {team_b} en la semana {week}.\n"
                     f"Partidos disponibles: {', '.join(jugados)}")

marcador = {t: juego[juego["posteam"] == t]["total_home_score"].max()
            for t in (team_a, team_b)}
home = juego["home_team"].iloc[0]
away = juego["away_team"].iloc[0]
pts = {home: float(juego["total_home_score"].max()),
       away: float(juego["total_away_score"].max())}

vals = {t: metricas(juego[juego["posteam"] == t]) for t in (team_a, team_b)}
if any(v is None for v in vals.values()):
    raise SystemExit("Muestra insuficiente en el partido.")

def mejor(clave, mas_alto_mejor):
    """Que equipo gana esa fila, o None si empatan o es metrica de identidad.

    Sin percentiles, la unica comparacion honesta es la del propio partido:
    un equipo contra el otro. Las metricas de identidad no se comparan, que
    jugar mas bajo centro no es ganar nada.
    """
    if mas_alto_mejor is None:
        return None
    a, b = vals[team_a][clave], vals[team_b][clave]
    if a is None or b is None or np.isnan(a) or np.isnan(b) or a == b:
        return None
    return team_a if (a > b) == mas_alto_mejor else team_b


# ── CONSOLA ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  FICHA TACTICA: {away} {pts[away]:.0f} - {pts[home]:.0f} {home} | "
      f"semana {week} NFL {SEASON}")
print(f"{'='*70}")
for etq, clave, f, hb, _sec in METRICAS:
    gana = mejor(clave, hb)
    fila = f"  {etq:24}"
    for t in (team_a, team_b):
        marca = " <" if gana == t else "  "
        fila += f" | {t} {fmt(vals[t][clave], f):>7}{marca}"
    print(fila)

# ── PNG (estilo docs/estilo-visual.md) ─────────────────────────────────────────
def nombre_equipo(team):
    """Ciudad para el titular; los equipos que comparten ciudad, por su apodo."""
    try:
        eq = pd.read_parquet("pbp_cache/teams.parquet").set_index("team_abbr")
        if team in ("LA", "LAC", "NYG", "NYJ"):
            return "los " + eq.loc[team, "team_nick"]
        return eq.loc[team, "team_name"].replace(" " + eq.loc[team, "team_nick"], "")
    except Exception:
        return team


def titular_ficha():
    """Facetas ganadas CONTADAS por el script (nunca a ojo: ver CLAUDE.md, punto 4)."""
    con_ganador = [m for m in METRICAS if m[3] is not None]
    n = len(con_ganador)
    ganadas = {t: sum(mejor(m[1], m[3]) == t for m in con_ganador) for t in (team_a, team_b)}
    if pts[home] == pts[away]:
        return None
    gan_partido = home if pts[home] > pts[away] else away
    per_partido = away if gan_partido == home else home
    if ganadas[per_partido] > ganadas[gan_partido]:
        np_ = nombre_equipo(per_partido)
        s = f"{np_} {v(np_, 'ganó', 'ganaron')} {ganadas[per_partido]} de las {n} facetas y {v(np_, 'perdió', 'perdieron')} el partido."
    else:
        t = max(ganadas, key=ganadas.get)
        if ganadas[t] < 8:
            return None
        nt_ = nombre_equipo(t)
        s = f"{nt_} {v(nt_, 'ganó', 'ganaron')} {ganadas[t]} de las {n} facetas del partido."
    return s[0].upper() + s[1:]


COLP = colores_pareja(team_a, team_b)
fig = lienzo()
antetitulo(fig, f"Ficha táctica · semana {week} · {away} {pts[away]:.0f} - {pts[home]:.0f} {home}")
tit = titular_ficha()
y = (titular(fig, tit, y=0.925) - 0.025) if tit else 0.905
y_top = y

# Cabecera: cada equipo en su lado
yc = y - 0.04
for t, x, ha in ((team_a, L + 0.03, "left"), (team_b, R - 0.03, "right")):
    poner_logo(fig, t, x + (0.015 if ha == "left" else -0.015), yc, zoom=0.02)
    fig.text(x + (0.055 if ha == "left" else -0.055), yc, t, family=TXT, weight="bold",
             size=10, color=COLP[t], va="center", ha=ha)
fig.text(0.5, yc, "en blanco, quien gana cada faceta", family=TXT, size=7, color=GRIS2,
         ha="center", va="center")

y = yc - 0.045
sec_actual = None
CENTRO, HUECO, BAR_MAX = 0.5, 0.135, 0.17      # etiqueta en el centro, barras a los lados
for etiq, clave, f, hb, sec in METRICAS:
    if sec != sec_actual:
        sec_actual = sec
        linea_h(fig, y + 0.012)
        etiqueta(fig, sec.capitalize() if sec != "IDENTIDAD" else "Identidad (sin ganador)",
                 CENTRO, y - 0.006, ha="center")
        y -= 0.034
    nombre = (etiq.replace(" / ", " por ").replace("2o", "2º").replace("4o", "4º")
              .replace("Exito", "Éxito").replace("Red Zone", "red zone"))
    fig.text(CENTRO, y, nombre, family=TXT, size=7.6, color=GRIS1, ha="center", va="center")
    gana = mejor(clave, hb)
    for t, signo in ((team_a, -1), (team_b, 1)):
        v = vals[t][clave]
        vacio = v is None or (isinstance(v, float) and np.isnan(v))
        neg = (not vacio) and f == "epa" and v < 0
        if not vacio:
            frac = abs(v) / 100.0 if f == "pct" else min(abs(v), 0.8) / 0.8
            largo = BAR_MAX * min(frac, 1.0)
            x0 = CENTRO + signo * HUECO
            col = NEUTRO if (neg or hb is None) else COLP[t]
            if largo > 0.002:
                fig.add_artist(Rectangle((x0 - largo if signo < 0 else x0, y - 0.0065), largo, 0.013,
                                         transform=fig.transFigure, color=col, lw=0))
        xt = CENTRO + signo * (HUECO + BAR_MAX + 0.015)
        txt = "–" if vacio else (num(v) if f == "epa" else f"{v:.0f}%")
        fig.text(xt, y, txt, family=TXT, size=9, weight="bold" if gana == t else "normal",
                 color=FG if gana == t else GRIS1, ha="right" if signo < 0 else "left",
                 va="center")
    y -= 0.03

tarjeta(fig, y + 0.005, y_top)
pie(fig, f"{vals[team_a]['n']} y {vals[team_b]['n']} jugadas · Éxito = EPA > 0 · "
         "Explosiva = pase de 20+ o carrera de 10+",
    f"NFL {SEASON} · semana {week} · nflverse PBP")

orden = orden_partido(SEASON, week, team_a, team_b)
if orden:
    idx, vis, loc = orden
    nombre = f"{idx:02d}_2_ficha_{vis}_vs_{loc}_{SEASON}.png"
else:
    nombre = f"ficha_{team_a}_vs_{team_b}_{SEASON}.png"

out = salida(nombre, SEASON, week)
guardar(fig, out)
