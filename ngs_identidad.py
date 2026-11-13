"""
ngs_identidad.py
La identidad de los 32 segun NFL Next Gen Stats: los 3 rasgos de ataque y
los 3 de defensa que NGS marca como identidad de cada equipo (`identity`),
con su puesto en la liga. Mosaico por divisiones.

Titular automatico: el equipo con la identidad mas extrema (menor suma de
puestos en sus 6 rasgos). Datos de datos_web/ngs_trends_equipos_*
(`python copiar_ngs_trends.py`). 13-nov-2026.
"""

import glob
import os

import pandas as pd
from matplotlib.patches import FancyBboxPatch

from pbp_loader import salida, season_cli
from estilo import *

SEASON = season_cli() or 2026
RAIZ = os.path.dirname(os.path.abspath(__file__))
ES = {"11Personnel": "11 personnel", "12Personnel": "12 personnel", "13Personnel": "13 personnel",
      "21Personnel": "21 personnel", "motion": "motion", "earlyDownPlayAction": "play action",
      "calledPassRate": "pase pedido", "tempo": "tempo", "quickPassing": "quick pass",
      "inBreakingTargets": "rutas dentro", "outBreakingTargets": "rutas fuera",
      "downfieldPassing": "pase 10+ yd", "manBlocking": "bloqueo man",
      "gapBlocking": "bloqueo gap", "insideZoneBlocking": "inside zone",
      "outsideZoneBlocking": "outside zone", "base": "base", "nickel": "nickel", "dime": "dime",
      "manCoverage": "man", "zoneCoverage": "zona", "pressCoverage": "press",
      "singleHigh": "single-high", "splitSafety": "split-safety", "cover0": "Cover 0",
      "blitzRate": "blitz", "pressureRateBlitz": "presión+blitz",
      "pressureRateNoBlitz": "presión sin bl."}
DIVS = [("AFC Este", ["BUF", "MIA", "NE", "NYJ"]), ("AFC Norte", ["BAL", "CIN", "CLE", "PIT"]),
        ("AFC Sur", ["HOU", "IND", "JAX", "TEN"]), ("AFC Oeste", ["DEN", "KC", "LV", "LAC"]),
        ("NFC Este", ["DAL", "NYG", "PHI", "WAS"]), ("NFC Norte", ["CHI", "DET", "GB", "MIN"]),
        ("NFC Sur", ["ATL", "CAR", "NO", "TB"]), ("NFC Oeste", ["ARI", "LA", "SF", "SEA"])]


def ultimo(patron):
    fs = sorted(glob.glob(os.path.join(RAIZ, "datos_web", patron)))
    if not fs:
        raise SystemExit(f"Falta datos_web/{patron}: lanza `python copiar_ngs_trends.py`")
    return fs[-1]


e = pd.read_csv(ultimo(f"ngs_trends_equipos_{SEASON}_w*.csv"))
SEM = int(e["semana"].mode()[0])
i = e[e["identidad"]].copy()
falta = set(i["metrica"]) - set(ES)
if falta:
    raise SystemExit(f"Métricas sin traducir: {falta}")
i["nom"] = i["metrica"].map(ES)
i = i.sort_values(["equipo", "lado", "puesto"])
extremo = i.groupby("equipo")["puesto"].sum().sort_values()
print(extremo.head(8).to_string())
for t in extremo.index[:3]:
    print(t, i[i["equipo"] == t][["lado", "nom", "uso", "puesto"]].to_string(index=False))

t0 = extremo.index[0]
r0 = i[i["equipo"] == t0].sort_values("puesto")
primeros = r0[r0["puesto"] == 1]["nom"].tolist()
rasgos = primeros if len(primeros) >= 2 else r0["nom"].head(3).tolist()
y_lista = ", ".join(rasgos[:-1]) + " y " + rasgos[-1]
cap = lambda s: s[0].upper() + s[1:]
tit = (f"{cap(nombre_equipo(t0))}, la identidad más marcada: "
       + (f"1º de la liga en {y_lista}" if len(primeros) >= 2 else f"top 3 en {y_lista}"))

fig = lienzo()
antetitulo(fig, f"La identidad de los 32 · NFL {SEASON}")
y = titular(fig, tit, y=0.93, size=19)
y = contexto(fig, [("Los rasgos que Next Gen Stats marca como identidad", "fuerte"),
                   ("de cada equipo: ataque a la izquierda (blanco), defensa a la derecha (gris),"
                    " con su puesto en la liga.", None)], y=y - 0.01)

top, bot = y - 0.025, 0.085
alto = (top - bot) / len(DIVS)
ancho = (R - L + 0.04) / 4
x0 = L - 0.02
for d, (div, eqs) in enumerate(DIVS):
    ytop = top - d * alto
    fig.text(x0 + 0.004, ytop - 0.002, div.upper(), family=TXT, size=5.6, color=GRIS2,
             va="top", weight="bold")
    for j, t in enumerate(eqs):
        cx, cy = x0 + j * ancho, ytop - 0.016
        h = alto - 0.02
        fig.add_artist(FancyBboxPatch((cx + 0.003, cy - h), ancho - 0.006, h,
                                      boxstyle="round,pad=0,rounding_size=0.006",
                                      transform=fig.transFigure, fc=CARD, ec="none"))
        poner_logo(fig, t, cx + 0.02, cy - 0.013, zoom=0.0105)
        fig.text(cx + 0.038, cy - 0.013, t, family=DISP, size=8.5, color=FG, va="center",
                 weight="bold")
        rs = i[i["equipo"] == t]
        mitad = (ancho - 0.006) / 2
        for k, (lado, c) in enumerate((("ataque", FG), ("defensa", GRIS1))):
            xa = cx + 0.010 + k * mitad
            yl = cy - 0.031
            for _, r in rs[rs["lado"] == lado].iterrows():
                top3 = r["puesto"] <= 3
                fig.text(xa, yl, r["nom"], family=TXT, size=5.0, color=c, va="center")
                fig.text(xa + mitad - 0.012, yl, f"{int(r['puesto'])}º", family=TXT, size=5.0,
                         color=RYG[2] if top3 else c, va="center", ha="right",
                         weight="bold" if top3 else "normal")
                yl -= (h - 0.03) / 3

pie(fig, "En verde, top 3 de la liga. Uso de cada rasgo frente a la media; coberturas y"
         " blitz, sobre jugadas de pase.",
    f"Fuente: NFL Next Gen Trends · NFL {SEASON} · hasta la semana {SEM}")
guardar(fig, salida(f"ngs_identidad_{SEASON}.png", SEASON))
