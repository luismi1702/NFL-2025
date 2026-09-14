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
for etiqueta, clave, f, hb, _sec in METRICAS:
    gana = mejor(clave, hb)
    fila = f"  {etiqueta:24}"
    for t in (team_a, team_b):
        marca = " <" if gana == t else "  "
        fila += f" | {t} {fmt(vals[t][clave], f):>7}{marca}"
    print(fila)

# ── PNG ────────────────────────────────────────────────────────────────────────
secciones = []
for _, _, _, _, sec in METRICAS:
    if sec not in secciones:
        secciones.append(sec)
filas_total = len(METRICAS) + len(secciones)

fig, ax = plt.subplots(figsize=(11.5, 0.62 * filas_total + 3.1), facecolor=BG)
ax.set_facecolor(BG)
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, filas_total + 3.0)

techo = filas_total + 3.0

# Cabecera
ax.add_patch(plt.Rectangle((0, techo - 1.85), 10, 1.85, color=CARD, zorder=0))
ax.text(5.0, techo - 0.62, f"{away}  {pts[away]:.0f} - {pts[home]:.0f}  {home}",
        ha="center", va="center", fontsize=19, fontweight="bold", color=FG, zorder=2)
ax.text(5.0, techo - 1.12, f"FICHA TACTICA  ·  semana {week} | NFL {SEASON}",
        ha="center", va="center", fontsize=10, color="#9aa3b5",
        fontstyle="italic", zorder=2)
ax.text(5.0, techo - 1.55,
        "en verde, el equipo que gana cada faceta",
        ha="center", va="center", fontsize=8.5, color="#6b7280", zorder=2)

for t, x in ((team_a, 1.15), (team_b, 8.85)):
    logo = load_logo(t, zoom=0.052)
    if logo is not None:
        ax.add_artist(AnnotationBbox(logo, (x, techo - 0.95), frameon=False, zorder=3))
    ax.text(x, techo - 1.62, t, ha="center", va="center", fontsize=12,
            fontweight="bold", color=FG, zorder=3)

# Filas
y = techo - 2.25
sec_actual = None
BAR_MAX = 2.05          # largo maximo de la barra de percentil
for etiqueta, clave, f, hb, sec in METRICAS:
    if sec != sec_actual:
        sec_actual = sec
        ax.text(5.0, y, sec, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="#5c6473", zorder=3)
        ax.plot([0.35, 3.6], [y, y], color=GRID, linewidth=0.8, zorder=1)
        ax.plot([6.4, 9.65], [y, y], color=GRID, linewidth=0.8, zorder=1)
        y -= 1.0

    ax.text(5.0, y, etiqueta, ha="center", va="center", fontsize=9.5,
            color="#cfd6e4", zorder=3)

    gana = mejor(clave, hb)
    for t, signo in ((team_a, -1), (team_b, 1)):
        v = vals[t][clave]
        # El largo es la magnitud y el COLOR el signo: sin esto, el -0.80 en
        # Red Zone de un equipo salia como la barra mas larga de la ficha, que
        # es justo su peor casilla. Un EPA negativo va en rojo aunque gane la
        # fila (puede ganarla siendo menos malo que el rival).
        negativo = (f == "epa" and hb is not None and v is not None
                    and not (isinstance(v, float) and np.isnan(v)) and v < 0)
        col = ROJO if negativo else (VERDE if gana == t else NEUTRO)
        borde = 3.75 if signo < 0 else 6.25
        # El largo de la barra es el VALOR, no un percentil: los porcentajes
        # sobre su escala natural (0-100) y el EPA sobre +-0.8, que cubre casi
        # todo lo que se ve en un partido. Asi la barra no esconde un dato que
        # el lector no puede ver.
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            frac = abs(v) / 100.0 if f == "pct" else min(abs(v), 0.8) / 0.8
            largo = BAR_MAX * min(frac, 1.0)
            if largo > 0.02:
                ax.add_patch(plt.Rectangle(
                    (borde - largo if signo < 0 else borde, y - 0.16),
                    largo, 0.3, color=col, alpha=0.85, zorder=2))
        ax.text(borde - BAR_MAX - 0.12 if signo < 0 else borde + BAR_MAX + 0.12, y,
                fmt(v, f), ha="right" if signo < 0 else "left", va="center",
                fontsize=12, fontweight="bold", color=col, zorder=3)
    y -= 1.0

ax.text(0.35, 0.35, f"Fuente: nflverse-data  ·  {sello(SEASON)}  ·  "
                    f"{vals[team_a]['n']} y {vals[team_b]['n']} jugadas  ·  "
                    f"identidad sin ganador: no es mejor ni peor",
        ha="left", va="center", fontsize=7.5, color="#555555", fontstyle="italic")
ax.text(9.65, 0.35, "@CuartayDato", ha="right", va="center", fontsize=9,
        color="#888888", alpha=0.8, fontstyle="italic")

orden = orden_partido(SEASON, week, team_a, team_b)
if orden:
    idx, vis, loc = orden
    nombre = f"{idx:02d}_ficha_{vis}_vs_{loc}_{SEASON}.png"
else:
    nombre = f"ficha_{team_a}_vs_{team_b}_{SEASON}.png"

out = salida(nombre, SEASON, week)
fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nGuardado: {out}")
