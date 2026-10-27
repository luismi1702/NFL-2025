"""
manning_picks.py — los picks de Manning Bot de una jornada en un PNG (01-oct-2026).

Lee lo que ya ha escrito el bot, sin volver a predecir nada: asi el numero
del PNG y el del post son el mismo por construccion.

  picks   salidas/{año}/w{N-1}/textos/bot_picks.txt (lo deja el batch del
          martes en la carpeta de la ultima jornada jugada)
  orden   salidas/{año}/w{N}/textos/previas_numeros_{año}_w{N}.txt, que va
          por kickoff (TNF primero); si aun no existe, el del calendario

Una fila por partido: visitante a la izquierda, local a la derecha y la barra
partida con la probabilidad de cada uno; el pick en naranja (el color de los
LED del bot) y el otro lado apagado. Sin titulo (regla de la casa): la
mascota de portada va pequeña abajo, con el sello y la marca de agua.

    python manning_picks.py --week 4
"""
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from pbp_loader import salida, sello, season_cli, week_cli, temporada_actual, proxima_semana
from estilo import *   # noqa: F401,F403  (estilo comun, docs/estilo-visual.md)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON = season_cli() or temporada_actual()
WEEK   = week_cli() or proxima_semana()
BG, CARD, FG, GRID = "#0f1115", "#151924", "#EDEDED", "#2a2f3a"
LED    = "#ff8a1f"            # el naranja de Manning Bot
APAGADO = "#3a4050"
RAIZ   = os.path.dirname(os.path.abspath(__file__))
LOGOS_DIR = os.path.join(RAIZ, "logos")
MASCOTA = os.path.join(RAIZ, "marca", "manning_bot", "manning_bot_portada.png")
DIAS = {"Thursday": "JUE", "Friday": "VIE", "Saturday": "SÁB", "Sunday": "DOM",
        "Monday": "LUN", "Tuesday": "MAR", "Wednesday": "MIÉ"}


def carpeta(w):
    return os.path.join(RAIZ, "salidas", str(SEASON), f"w{int(w):02d}", "textos")


def load_logo(team, zoom=0.030, alpha=1.0):
    """ranking_wrs.load_logo: recorte por tinta real y zoom por su area."""
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    img = plt.imread(path).copy()
    if img.ndim == 3 and img.shape[2] == 4:
        ys, xs = np.where(img[:, :, 3] > 0.02)
        if len(ys):
            img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        img[:, :, 3] *= alpha
    h, w = img.shape[:2]
    z = zoom * 500.0 / max((h * w) ** 0.5, 1.0)
    if w * z > 900.0 * zoom:
        z = 900.0 * zoom / w
    return OffsetImage(img, zoom=z, resample=True)


# ── DATOS ──────────────────────────────────────────────────────────────────────
ruta_picks = os.path.join(carpeta(WEEK - 1), "bot_picks.txt")
if not os.path.exists(ruta_picks):
    sys.exit(f"No esta {ruta_picks}: el bot de la semana {WEEK} sale en el batch del martes")
texto = open(ruta_picks, encoding="utf-8").read()
m = re.search(r"MANNING BOT .*?Semana (\d+)", texto)
if not m or int(m.group(1)) != WEEK:
    sys.exit(f"{ruta_picks} no es de la semana {WEEK}")
picks = {}
for away, home, pick, p_home, p_away in re.findall(
        r"^\s+([A-Z]{2,3}) @ ([A-Z]{2,3})\s+([A-Z]{2,3})\s+([\d.]+)%\s+([\d.]+)%", texto, re.M):
    picks[(away, home)] = dict(pick=pick, p_home=float(p_home), p_away=float(p_away), dia="")
print(f"Picks de la semana {WEEK}: {len(picks)} partidos ({os.path.relpath(ruta_picks, RAIZ)})")

orden = list(picks)
ruta_orden = os.path.join(carpeta(WEEK), f"previas_numeros_{SEASON}_w{WEEK:02d}.txt")
if os.path.exists(ruta_orden):
    kick = re.findall(r"^== \d+ ([A-Z]{2,3}) @ ([A-Z]{2,3}) — (\w+)", open(ruta_orden, encoding="utf-8").read(), re.M)
    por_kick = [(a, h) for a, h, _ in kick if (a, h) in picks]
    for a, h, d in kick:
        if (a, h) in picks:
            picks[(a, h)]["dia"] = DIAS.get(d, "")
    orden = por_kick + [k for k in orden if k not in por_kick]
else:
    # El jueves aun no hay previas (salen el sabado): el kickoff, del calendario
    try:
        import pandas as pd
        from pbp_loader import cargar_calendario
        sch, _ = cargar_calendario(SEASON)
        sch = sch[(sch["game_type"] == "REG") & (sch["week"] == WEEK)]
        sch = sch.sort_values(["gameday", "gametime"], kind="stable")
        por_kick = []
        for f in sch.itertuples(index=False):
            if (f.away_team, f.home_team) in picks:
                por_kick.append((f.away_team, f.home_team))
                dia = pd.Timestamp(f.gameday).day_name()
                picks[(f.away_team, f.home_team)]["dia"] = DIAS.get(dia, "")
        orden = por_kick + [k for k in orden if k not in por_kick]
        print("  (sin previas_numeros: orden de kickoff del calendario)")
    except Exception as e:
        print(f"  (sin previas_numeros ni calendario: orden del bot, sin dia — {e})")

# ── FIGURA (estilo docs/estilo-visual.md; el naranja LED es la marca del bot) ──
def nombre_equipo(team):
    try:
        import pandas as pd
        eq = pd.read_parquet("pbp_cache/teams.parquet").set_index("team_abbr")
        if team in ("LA", "LAC", "NYG", "NYJ"):
            return "los " + eq.loc[team, "team_nick"]
        return eq.loc[team, "team_name"].replace(" " + eq.loc[team, "team_nick"], "")
    except Exception:
        return team


n = len(orden)
seguro = max(orden, key=lambda k: max(picks[k]["p_home"], picks[k]["p_away"]))
r0 = picks[seguro]
p0 = max(r0["p_home"], r0["p_away"])
tit = f"Su pick más claro: {nombre_equipo(r0['pick'])}, con un {p0:.0f}% de opciones."

fig = lienzo()
antetitulo(fig, f"Manning Bot · picks de la semana {WEEK}")
y_top = titular(fig, tit, y=0.925, ancho=R - L - 0.17) - 0.03
if os.path.exists(MASCOTA):
    axm = fig.add_axes([R - 0.15, y_top + 0.005, 0.16, 0.945 - y_top])
    axm.imshow(plt.imread(MASCOTA))
    axm.axis("off")

alto = (y_top - 0.16) / n
X0, X1 = 0.36, 0.72                       # la barra de probabilidad
etiqueta(fig, "Visitante", X0, y_top - 0.02)
etiqueta(fig, "Local", X1, y_top - 0.02, ha="right")
for i, (a, h) in enumerate(orden):
    y = y_top - 0.05 - i * alto
    r = picks[(a, h)]
    gana_local = r["pick"] == h
    fig.text(L + 0.01, y, r["dia"], family=TXT, weight="semibold", size=7, color=GRIS2,
             va="center")
    for t, xl, xt, ha, es_pick in ((a, L + 0.085, L + 0.12, "left", not gana_local),
                                   (h, R - 0.035, R - 0.07, "right", gana_local)):
        lo = load_logo(t, 0.0115, 1.0 if es_pick else 0.35)
        if lo:
            fig.add_artist(AnnotationBbox(lo, (xl, y), xycoords="figure fraction", frameon=False))
        fig.text(xt, y, t, family=TXT, weight="bold" if es_pick else "normal", size=8.5,
                 color=FG if es_pick else GRIS2, ha=ha, va="center")
    corte = X0 + (X1 - X0) * r["p_away"] / 100
    for x_ini, x_fin, encendido in ((X0, corte, not gana_local), (corte, X1, gana_local)):
        fig.add_artist(Rectangle((x_ini, y - 0.009), x_fin - x_ini, 0.018,
                                 transform=fig.transFigure,
                                 color=LED if encendido else APAGADO, lw=0))
    fig.add_artist(Rectangle((corte - 0.001, y - 0.012), 0.002, 0.024,
                             transform=fig.transFigure, color=CARD, lw=0))
    # un tramo de menos del 15% no tiene sitio para su numero: va por fuera
    for p, x_in, x_out, ha_in, ha_out, encendido in (
            (r["p_away"], X0 + 0.008, X0 - 0.008, "left", "right", not gana_local),
            (r["p_home"], X1 - 0.008, X1 + 0.008, "right", "left", gana_local)):
        fuera = p < 15
        fig.text(x_out if fuera else x_in, y, f"{p:.0f}%", family=TXT, weight="bold", size=7.5,
                 color=GRIS1 if fuera else (BG if encendido else FG),
                 ha=ha_out if fuera else ha_in, va="center")

tarjeta(fig, y_top - 0.05 - (n - 1) * alto - alto / 2 - 0.005, y_top)
pie(fig, "Probabilidad de victoria según el modelo; en naranja, el pick",
    f"NFL {SEASON} · datos hasta sem. {WEEK - 1} · picks de la semana {WEEK}")

out = salida(f"manning_picks_{SEASON}.png", SEASON, week=WEEK)
guardar(fig, out)
