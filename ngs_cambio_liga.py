"""
ngs_cambio_liga.py
Como ha cambiado la NFL desde 2018: las 8 tendencias de Next Gen Stats que
mas se han movido (en puntos de %), temporada a temporada, en pequenos
graficos. El titular nombra la que mas ha cambiado desde 2018 y la que mas
ha subido este ano en proporcion.

Datos de datos_web/ngs_trends_liga_* y _liga_resumen_* (`python
copiar_ngs_trends.py`). La temporada en curso va hasta la ultima semana
copiada. Para contexto (podcasts, offseason). 13-nov-2026.
"""

import glob
import os

import pandas as pd

from pbp_loader import salida, season_cli
from estilo import *

SEASON = season_cli() or 2026
RAIZ = os.path.dirname(os.path.abspath(__file__))
N = 8
ES = {"11Personnel": "11 personnel", "12Personnel": "12 personnel", "13Personnel": "13 personnel",
      "21Personnel": "21 personnel", "motion": "Motion", "earlyDownPlayAction": "Play action",
      "calledPassRate": "Pase pedido", "tempo": "Tempo", "quickPassing": "Quick passing",
      "inBreakingTargets": "Rutas hacia dentro", "outBreakingTargets": "Rutas hacia fuera",
      "downfieldPassing": "Pases de 10+ yardas", "manBlocking": "Bloqueo man",
      "gapBlocking": "Bloqueo gap", "insideZoneBlocking": "Inside zone",
      "outsideZoneBlocking": "Outside zone", "base": "Defensa base", "nickel": "Nickel",
      "dime": "Dime", "manCoverage": "Cobertura man", "zoneCoverage": "Cobertura en zona",
      "pressCoverage": "Press", "singleHigh": "Single-high", "splitSafety": "Split-safety",
      "cover0": "Cover 0", "blitzRate": "Blitz", "pressureRateBlitz": "Presión con blitz",
      "pressureRateNoBlitz": "Presión sin blitz"}


def ultimo(patron):
    fs = sorted(glob.glob(os.path.join(RAIZ, "datos_web", patron)))
    if not fs:
        raise SystemExit(f"Falta datos_web/{patron}: lanza `python copiar_ngs_trends.py`")
    return fs[-1]


ruta = ultimo(f"ngs_trends_liga_{SEASON}_w*.csv")
SEM = int(ruta.rsplit("_w", 1)[1][:2])
l = pd.read_csv(ruta).pivot(index="metrica", columns="temporada", values="uso_liga") * 100
a0, a1 = l.columns.min(), l.columns.max()
cambio = (l[a1] - l[a0]).sort_values(key=abs, ascending=False)
yoy = (l[a1] / l[a1 - 1] - 1).sort_values(ascending=False)
print(pd.DataFrame({"desde_" + str(a0): cambio.round(1), "yoy_rel": yoy.round(2)}).to_string())
sel = cambio.index[:N]

m1, m2 = cambio.index[0], yoy.index[0]
baja = lambda t: t[0].lower() + t[1:]
fp = lambda x: f"{x:.0f}%" if x >= 10 else f"{x:.1f}%".replace(".", ",")
tit = (f"{ES[m1]}: del {fp(l.loc[m1, a0])} al {fp(l.loc[m1, a1])} desde {a0}. Y este año, el"
       f" {baja(ES[m2])}, del {fp(l.loc[m2, a1 - 1])} al {fp(l.loc[m2, a1])}")

fig = lienzo()
antetitulo(fig, f"Cómo ha cambiado la NFL · {a0}-{a1}")
y = titular(fig, tit, y=0.93, size=19)
y = contexto(fig, [(f"Las {N} tendencias que más se han movido desde {a0}", "fuerte"),
                   (f"(Next Gen Stats, media de la liga, % de las jugadas en que se puede dar)."
                    f" {a1}, hasta la semana {SEM}.", None)], y=y - 0.01)

top, bot = y - 0.03, 0.10
cols, filas = 2, N // 2
w = (R - L) / cols
h = (top - bot) / filas
tarjeta(fig, bot - 0.01, top + 0.01)
for k, met in enumerate(sel):
    c, f = k % cols, k // cols
    x0, y0 = L + c * w, top - (f + 1) * h
    ax = fig.add_axes([x0 + 0.03, y0 + 0.03, w - 0.07, h - 0.075])
    estilo_ejes(ax, x=True, y=False)
    s = l.loc[met]
    sube = s.iloc[-1] > s.iloc[0]
    col = ACCENT if sube else GRIS1
    ax.plot(s.index, s.values, color=col, lw=1.8)
    ax.fill_between(s.index, s.values, s.min() - (s.max() - s.min()) * 0.15, color=col, alpha=0.12)
    ax.scatter([s.index[0], s.index[-1]], [s.iloc[0], s.iloc[-1]], color=col, s=10, zorder=3)
    ax.set_ylim(s.min() - (s.max() - s.min()) * 0.15, s.max() + (s.max() - s.min()) * 0.35)
    ax.set_xticks([a0, a1])
    ax.set_xticklabels([str(a0), str(a1)])
    ax.text(s.index[0], s.iloc[0], f"{s.iloc[0]:.0f}%  ", ha="right", va="center", size=6.5,
            color=GRIS1)
    ax.text(s.index[-1], s.iloc[-1], f"  {s.iloc[-1]:.0f}%", ha="left", va="center", size=7,
            color=FG, weight="bold")
    ax.set_xlim(a0 - 0.9, a1 + 1.1)
    fig.text(x0 + 0.03, y0 + h - 0.022, ES[met], family=TXT, size=8, color=FG, weight="bold")
    fig.text(x0 + w - 0.04, y0 + h - 0.022, f"{num(cambio[met], 1)} pts", family=TXT, size=7,
             color=col, ha="right", weight="bold")

pie(fig, "Personal y motion, sobre todas las jugadas; coberturas, sobre pases; bloqueos,"
         " sobre carreras.",
    f"Fuente: NFL Next Gen Trends · {a0}-{a1} (semana {SEM})")
guardar(fig, salida(f"ngs_cambio_liga_{SEASON}.png", SEASON))
