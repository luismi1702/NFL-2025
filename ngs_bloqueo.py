"""
ngs_bloqueo.py
Como corre cada equipo: % de sus carreras con cada esquema de bloqueo
(outside zone, inside zone, gap, man) segun NFL Next Gen Trends, y en que
esquema le va mejor (EPA por carrera de NGS, min. MIN_JUG carreras).

Dato que nflverse NO trae en temporada. Sale de datos_web/ngs_trends_equipos_*
(lanzar antes `python copiar_ngs_trends.py`). Todo el grafico es EPA de NGS:
no se mezcla con el de nflverse. 13-nov-2026.
"""

import glob
import os

import pandas as pd
from matplotlib.patches import Rectangle

from pbp_loader import salida, season_cli
from estilo import *

SEASON = season_cli() or 2026
MIN_JUG = 25
RAIZ = os.path.dirname(os.path.abspath(__file__))
ESQ = [("outsideZoneBlocking", "Outside zone", ACCENT),
       ("insideZoneBlocking", "Inside zone", aclarar(ACCENT, 0.45)),
       ("gapBlocking", "Gap", GRIS1),
       ("manBlocking", "Man", NEUTRO)]
NOMBRE = {k: n for k, n, _ in ESQ}


def ultimo(patron):
    fs = sorted(glob.glob(os.path.join(RAIZ, "datos_web", patron)))
    if not fs:
        raise SystemExit(f"Falta datos_web/{patron}: lanza `python copiar_ngs_trends.py`")
    return fs[-1]


ruta = ultimo(f"ngs_trends_equipos_{SEASON}_w*.csv")
e = pd.read_csv(ruta)
SEM = int(e["semana"].mode()[0])
b = e[e["metrica"].isin(NOMBRE)]
uso = b.pivot(index="equipo", columns="metrica", values="uso")
epa = b.pivot(index="equipo", columns="metrica", values="epa")
jug = b.pivot(index="equipo", columns="metrica", values="jugadas")
liga = b.groupby("metrica")["uso_liga"].first()

g = pd.DataFrame(index=uso.index)
g["zona"] = uso["outsideZoneBlocking"] + uso["insideZoneBlocking"]
g["fuerza"] = uso["gapBlocking"] + uso["manBlocking"]
validos = epa.where(jug >= MIN_JUG)
g["mejor"] = validos.idxmax(axis=1)
g["mejor_epa"] = validos.max(axis=1)
g = g.sort_values("zona", ascending=False)
print(pd.concat([uso.round(3), g], axis=1).sort_values("zona", ascending=False).to_string())
print("liga:", liga.round(3).to_dict(), f"· {os.path.basename(ruta)}")

oz = uso["outsideZoneBlocking"].idxmax()
pod = uso[["gapBlocking", "manBlocking"]].stack().idxmax()      # (equipo, esquema)
tit = (f"{nombre_equipo(oz)} corre outside zone en el {uso.loc[oz, 'outsideZoneBlocking'] * 100:.0f}%"
       f" de sus carreras. {nombre_equipo(pod[0])[0].upper() + nombre_equipo(pod[0])[1:]}, {NOMBRE[pod[1]].lower()} en el"
       f" {uso.loc[pod] * 100:.0f}%")

fig = lienzo()
antetitulo(fig, f"Cómo corre cada equipo · NFL {SEASON}")
y = titular(fig, tit, y=0.93, size=19)
y = contexto(fig, [("% de carreras con cada esquema de bloqueo", "fuerte"),
                   (f"(Next Gen Stats). A la derecha, el esquema en el que más EPA saca"
                    f" por carrera (mín. {MIN_JUG}).", None)], y=y - 0.01)

top = y - 0.05
X_LOGO, X_BAR, W_BAR = L + 0.015, L + 0.05, 0.58
X_MEJOR = X_BAR + W_BAR + 0.03
fila = (top - 0.11) / len(g)
tarjeta(fig, 0.085, top + 0.035)
# leyenda
xl = X_BAR
for k, n, c in ESQ:
    fig.add_artist(Rectangle((xl, top + 0.012), 0.012, 0.009, transform=fig.transFigure,
                             fc=c, ec="none"))
    fig.text(xl + 0.016, top + 0.0165, n, family=TXT, size=6.5, color=GRIS1, va="center")
    xl += 0.025 + len(n) * 0.0095
etiqueta(fig, "Su mejor esquema", X_MEJOR, top + 0.012)

for i, (t, r) in enumerate(g.iterrows()):
    yy = top - (i + 0.5) * fila
    poner_logo(fig, t, X_LOGO, yy, zoom=0.0105)
    x = X_BAR
    for k, n, c in ESQ:
        w = uso.loc[t, k] * W_BAR
        fig.add_artist(Rectangle((x, yy - fila * 0.36), w, fila * 0.72,
                                 transform=fig.transFigure, fc=c, ec=CARD, lw=0.4))
        if uso.loc[t, k] >= 0.12:
            fig.text(x + w / 2, yy, f"{uso.loc[t, k] * 100:.0f}", family=TXT, size=5.6,
                     color=BG if c != NEUTRO else FG, ha="center", va="center")
        x += w
    if pd.notna(r["mejor"]):
        fig.text(X_MEJOR, yy, f"{NOMBRE[r['mejor']]}  {num(r['mejor_epa'])}", family=TXT,
                 size=6.4, color=FG if r["mejor_epa"] > 0 else GRIS1, va="center")
# media de la liga: marca de la zona
xz = X_BAR + (liga["outsideZoneBlocking"] + liga["insideZoneBlocking"]) * W_BAR
fig.add_artist(Rectangle((xz - 0.0006, top - len(g) * fila), 0.0012, len(g) * fila,
                         transform=fig.transFigure, fc=FG, ec="none", alpha=0.5))
fig.text(xz, top - len(g) * fila - 0.012, "zona, media liga", family=TXT, size=6,
         color=GRIS1, ha="center")

pie(fig, f"Liga: outside zone {pct(liga['outsideZoneBlocking'])}, inside zone"
         f" {pct(liga['insideZoneBlocking'])}, gap {pct(liga['gapBlocking'])}, man"
         f" {pct(liga['manBlocking'])}; el resto, otros esquemas. EPA del modelo de NGS.",
    f"Fuente: NFL Next Gen Trends · NFL {SEASON} · hasta la semana {SEM}")
guardar(fig, salida(f"ngs_bloqueo_{SEASON}.png", SEASON))
