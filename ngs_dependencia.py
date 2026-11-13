"""
ngs_dependencia.py
De que vive cada ataque: diferencia de EPA por jugada CON y SIN cada recurso
(motion, play action en early downs, quick passing y pase pedido frente a
carrera pedida) segun NFL Next Gen Trends, con el % de uso y el puesto.

La web da el con/sin equipo a equipo; el ranking de 32 es nuestro. Todo es
EPA del modelo de NGS (no se mezcla con nflverse). Datos de
datos_web/ngs_trends_equipos_* (`python copiar_ngs_trends.py`). 13-nov-2026.
"""

import glob
import os

import pandas as pd
from matplotlib.patches import FancyBboxPatch

from pbp_loader import salida, season_cli
from estilo import *

SEASON = season_cli() or 2026
RAIZ = os.path.dirname(os.path.abspath(__file__))
COLS = [("earlyDownPlayAction", "Play action"), ("motion", "Motion"),
        ("quickPassing", "Quick passing"), ("calledPassRate", "Pase vs carrera")]


def ultimo(patron):
    fs = sorted(glob.glob(os.path.join(RAIZ, "datos_web", patron)))
    if not fs:
        raise SystemExit(f"Falta datos_web/{patron}: lanza `python copiar_ngs_trends.py`")
    return fs[-1]


e = pd.read_csv(ultimo(f"ngs_trends_equipos_{SEASON}_w*.csv"))
SEM = int(e["semana"].mode()[0])
e = e[e["metrica"].isin(dict(COLS)) & (e["lado"] == "ataque")].copy()
e["dif"] = e["epa"] - e["sin_epa"]
g = pd.DataFrame(index=sorted(e["equipo"].unique()))
for k, _ in COLS:
    x = e[e["metrica"] == k].set_index("equipo")
    g[k], g["uso_" + k], g["rk_" + k] = x["dif"], x["uso"], x["dif"].rank(ascending=False, method="min")
    g["n_" + k], g["nsin_" + k] = x["jugadas"], x["sin_jugadas"]
liga = {k: (e[e["metrica"] == k]["dif"].median()) for k, _ in COLS}
g = g.sort_values("earlyDownPlayAction", ascending=False)
print(g.round(3).to_string())
print("mediana liga:", {k: round(v, 3) for k, v in liga.items()})

pa = g["earlyDownPlayAction"].idxmax()
mo = g["motion"].idxmax()
cap = lambda s: s[0].upper() + s[1:]
tit = (f"{cap(nombre_equipo(pa))} gana {num(g.loc[pa, 'earlyDownPlayAction'])} por jugada con"
       f" play action. {cap(nombre_equipo(mo))}, {num(g.loc[mo, 'motion'])} con motion")

fig = lienzo()
antetitulo(fig, f"De qué vive cada ataque · NFL {SEASON}")
y = titular(fig, tit, y=0.93, size=19)
y = contexto(fig, [("EPA por jugada con el recurso menos sin él", "fuerte"),
                   ("(Next Gen Stats) y su puesto. En pequeño, % de uso. Pase vs carrera:"
                    " pases pedidos frente a carreras pedidas.", None)], y=y - 0.01)

top = y - 0.04
X_LOGO, X_NOM = L + 0.015, L + 0.045
XC = [0.40, 0.55, 0.70, 0.85]
WC = 0.135
fila = (top - 0.105) / len(g)
tarjeta(fig, 0.085, top + 0.03)
for (k, nom), xc in zip(COLS, XC):
    etiqueta(fig, nom, xc, top + 0.012, ha="center")

for i, (t, r) in enumerate(g.iterrows()):
    yy = top - (i + 0.5) * fila
    poner_logo(fig, t, X_LOGO, yy, zoom=0.0105)
    fig.text(X_NOM, yy, t, family=TXT, size=7.6, color=FG, va="center")
    for (k, _), xc in zip(COLS, XC):
        v, rk = r[k], r["rk_" + k]
        if pd.isna(v):
            continue
        col = color_rango(int(rk), 32)
        col = col if col != GRIS1 else NEUTRO
        tc = BG if col in RYG else FG
        fig.add_artist(FancyBboxPatch((xc - WC / 2, yy - fila * 0.40), WC, fila * 0.80,
                                      boxstyle="round,pad=0,rounding_size=0.004",
                                      transform=fig.transFigure, fc=col, ec="none"))
        fig.text(xc - 0.012, yy, f"{num(v)} · {int(rk)}º", family=TXT, size=6.1, color=tc,
                 va="center", ha="center", weight="semibold" if col in RYG else "normal")
        fig.text(xc + WC / 2 - 0.006, yy, f"{r['uso_' + k] * 100:.0f}%", family=TXT, size=5,
                 color=tc, va="center", ha="right", alpha=0.75)

pie(fig, "Mediana de la liga: " + ", ".join(f"{n.lower()} {num(liga[k])}" for k, n in COLS)
         + ". EPA de NGS.",
    f"Fuente: NFL Next Gen Trends · NFL {SEASON} · hasta la semana {SEM}")
guardar(fig, salida(f"ngs_dependencia_{SEASON}.png", SEASON))
