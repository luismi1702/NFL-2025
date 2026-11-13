"""
ngs_partidos_en_juego.py
Los partidos que mas se juegan de la jornada: para cada partido que queda,
la probabilidad de playoffs de cada equipo si gana y si pierde (modelo de
NFL Next Gen Stats) y el "leverage" (la diferencia). Se ordenan por la suma
de los dos: la web lo da equipo a equipo; el orden por partido es nuestro.
Al pie, la carrera por el nº 1 del draft (misma fuente).

Datos de datos_web/ngs_trends_playoffs_* (`python copiar_ngs_trends.py`, que
la portada trae ya con la jornada en curso). Para el jueves o el viernes.
13-nov-2026.
"""

import glob
import os

import pandas as pd
from matplotlib.patches import Rectangle

from pbp_loader import salida, season_cli
from estilo import *

SEASON = season_cli() or 2026
RAIZ = os.path.dirname(os.path.abspath(__file__))


def ultimo(patron):
    fs = sorted(glob.glob(os.path.join(RAIZ, "datos_web", patron)))
    if not fs:
        raise SystemExit(f"Falta datos_web/{patron}: lanza `python copiar_ngs_trends.py`")
    return fs[-1]


ruta = ultimo(f"ngs_trends_playoffs_{SEASON}_w*.csv")
p = pd.read_csv(ruta)
SEM = int(p["semana"].max())
copiado = str(p["copiado"].iloc[0])[:10]
v = p[p["proximo_fecha"].notna() & (p["proximo_fecha"] >= copiado)
      & p["playoffProbabilityIfWin"].notna()].set_index("equipo")
partidos, vistos = [], set()
for t, r in v.iterrows():
    o = r["proximo_rival"]
    if t in vistos or o not in v.index:
        continue
    vistos |= {t, o}
    a, b = (t, o) if r["proximo_campo"] == "AWAY" else (o, t)      # visitante, local
    partidos.append({"vis": a, "loc": b, "fecha": r["proximo_fecha"],
                     "lev": v.loc[a, "playoffLeverage"] + v.loc[b, "playoffLeverage"]})
g = pd.DataFrame(partidos).sort_values("lev", ascending=False).reset_index(drop=True)
print(g.to_string())
if g.empty:
    raise SystemExit("No quedan partidos con probabilidades (¿copia vieja?)")

cap = lambda s: s[0].upper() + s[1:]
top = g.iloc[0]
qui = max([top["vis"], top["loc"]], key=lambda t: v.loc[t, "playoffLeverage"])
tit = (f"{cap(nombre_equipo(top['vis']))}-{nombre_equipo(top['loc'])}, el partido que más se juega:"
       f" {nombre_equipo(qui)}, del {v.loc[qui, 'playoffProbabilityIfLose'] * 100:.0f}% al"
       f" {v.loc[qui, 'playoffProbabilityIfWin'] * 100:.0f}% de playoffs")

fig = lienzo()
antetitulo(fig, f"Lo que se juega cada uno · semana {SEM} · NFL {SEASON}")
y = titular(fig, tit, y=0.93, size=18)
y = contexto(fig, [("Probabilidad de playoffs si gana y si pierde", "fuerte"),
                   ("(modelo de Next Gen Stats). La barra es lo que se juega: la diferencia"
                    " entre las dos. Visitante a la izquierda.", None)], y=y - 0.01)

top_y = y - 0.035
fila = min(0.05, (top_y - 0.12) / len(g))
tarjeta(fig, 0.10, top_y + 0.02)
XC, ESC = 0.5, 0.33 / max(0.35, v["playoffLeverage"].max())   # ancho por punto de prob.
for i, r in g.iterrows():
    yy = top_y - (i + 0.5) * fila
    for t, lado in ((r["vis"], -1), (r["loc"], 1)):
        lev = v.loc[t, "playoffLeverage"]
        w = lev * ESC
        x = XC - w if lado < 0 else XC
        fig.add_artist(Rectangle((x, yy - fila * 0.28), w, fila * 0.56, transform=fig.transFigure,
                                 fc=color_equipo(t), ec="none", alpha=0.9))
        xl = XC + lado * (w + 0.012)
        ha = "right" if lado < 0 else "left"
        fig.text(xl, yy + fila * 0.13, t, family=DISP, size=8, color=FG, weight="bold",
                 ha=ha, va="center")
        fig.text(xl, yy - fila * 0.2,
                 f"{v.loc[t, 'playoffProbabilityIfLose'] * 100:.0f}% / "
                 f"{v.loc[t, 'playoffProbabilityIfWin'] * 100:.0f}%",
                 family=TXT, size=5.8, color=GRIS1, ha=ha, va="center")
        poner_logo(fig, t, xl + lado * 0.088, yy, zoom=0.0105)
fig.add_artist(Rectangle((XC - 0.0007, top_y - len(g) * fila), 0.0014, len(g) * fila,
                         transform=fig.transFigure, fc=GRIS2, ec="none"))
fig.text(L, 0.112, "Debajo de cada equipo: playoffs si pierde / si gana.", family=TXT, size=6,
         color=GRIS1)

top3 = p.sort_values("firstPickProbability", ascending=False).head(3)
carrera = ", ".join(f"{t} {x * 100:.0f}%" for t, x in zip(top3["equipo"], top3["firstPickProbability"]))
pie(fig, f"Carrera por el nº 1 del draft: {carrera}. Probabilidades del modelo de NGS.",
    f"Fuente: NFL Next Gen Trends · NFL {SEASON} · antes de la semana {SEM} (copiado {copiado})")
guardar(fig, salida(f"ngs_en_juego_{SEASON}.png", SEASON, week=SEM))
