"""PNG "como defiende Seattle" del duelo del sabado 10-oct-2026 (SF @ SEA): el
tuit 2 del hilo. Misma plantilla que lab/sf_sea_png.py.

Fuente: tablas de Sharp Football Analysis copiadas a datos_web/ por
copiar_sharp_defensa.py (actualizadas el 06-oct-2026, semanas 1-4). El
% de blitz cuadra con nuestro FTN (21% en los dos).
- Medio abierto = dos safeties profundos tras el snap (cover 2, 4 y 6). Sharp
  NO separa cover 2 de cover 4: por eso el PNG dice "dos safeties profundos" y
  nunca "cover 2".
- Sub package = 5 o mas defensive backs en el campo.
- Son rasgos de identidad, no de calidad: la casilla del puesto va en el color
  del equipo, sin verde ni rojo. Puesto 1 = el que mas lo hace.
En empates el puesto es el mejor de los empatados (rank "min").

El titular es la frase que abre el tuit 2 del hilo
(salidas/2026/w05/textos/borradores_sabado.md)."""
import os
import sys

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from pbp_loader import salida   # noqa: E402
from estilo import *            # noqa: E402,F401,F403

HASTA = 4
EQ = "SEA"


def web(nombre):
    return pd.read_csv(os.path.join(RAIZ, "datos_web", f"sharp_{nombre}_2026_w{HASTA:02d}.csv")
                       ).set_index("team")


cob, ten, dl = web("coberturas"), web("tendencias_defensa"), web("linea_defensiva")
FILAS = [  # (rotulo, aclaracion, tabla, columna)
    ("Cobertura en zona", "al hombre, solo el {man}%", cob, "zona_pct"),
    ("Dos safeties profundos", "medio del campo abierto tras el snap", cob, "medio_abierto_pct"),
    ("Cinco o más defensive backs", "sub package", ten, "sub_package_pct"),
    ("Caja ligera", "seis o menos junto a la línea", ten, "caja_ligera_pct"),
    ("Blitz", "manda cinco o más", ten, "blitz_pct"),
    ("Presión sin blitzear", "% de dropbacks con presión mandando cuatro", dl, "presion_sin_blitz_pct"),
]
datos = []
for rot, acl, t, c in FILAS:
    r = int(t[c].rank(ascending=False, method="min")[EQ])
    datos.append(dict(rot=rot, acl=acl.format(man=num(cob.loc[EQ, "man_pct"], 1, signo=False)),
                      v=t.loc[EQ, c], rk=r, media=t[c].mean()))
    print(f"{rot}: {t.loc[EQ, c]} · {r}º de 32 · media {t[c].mean():.1f}")
print("man", cob.loc[EQ, "man_pct"], int(cob.man_pct.rank(ascending=False, method="min")[EQ]),
      "| actualizado", cob.actualizado.iloc[0])
zona = datos[0]

fig = lienzo()
antetitulo(fig, "Duelo · SF @ SEA · domingo")
y = titular(fig, "Seattle no te persigue: te espera en zona.", y=0.925) - 0.012
y = contexto(fig, [("Juega zona en el", None), (f"{zona['v']:.0f}%", "fuerte"),
                   ("de los pases, la que más de la NFL, y casi nunca blitzea.", None)], y=y)
y_top = y - 0.025
yc = y_top - 0.045
poner_logo(fig, EQ, L + 0.035, yc, zoom=0.022)
fig.text(L + 0.08, yc, EQ, family=TXT, weight="bold", size=10, color=color_equipo(EQ), va="center")
fig.text(L + 0.15, yc, "·  CÓMO DEFIENDE", family=TXT, weight="semibold", size=8.5, color=GRIS1,
         va="center")
yf = yc - 0.07
PASO = 0.072
for i, d in enumerate(datos):
    yy = yf - i * PASO
    if i:
        linea_h(fig, yy + PASO / 2 - 0.002)
    fig.text(L + 0.01, yy + 0.008, d["rot"], family=TXT, weight="semibold", size=9.5, color=FG,
             va="center")
    nota(fig, d["acl"], L + 0.01, yy - 0.014)
    fig.text(0.60, yy, f"{d['v']:.0f}%", family=DISP, weight=800, size=24, color=FG,
             va="center", ha="right")
    fig.text(0.635, yy + 0.012, ordinal(d["rk"]), family=TXT, weight="bold", size=8, color=FG,
             va="center")
    fig.text(R - 0.01, yy + 0.012, f"media NFL {d['media']:.0f}%", family=TXT, size=7.2,
             color=GRIS2, va="center", ha="right")
    pista32(fig, 0.635, yy - 0.014, R - 0.01 - 0.635, d["rk"], color=color_equipo(EQ))
tarjeta(fig, yf - (len(datos) - 1) * PASO - 0.05, y_top)
pie(fig, "1º = el que más lo hace · Dos safeties profundos = cover 2, 4 y 6",
    f"NFL 2026 · datos hasta sem. {HASTA} · Sharp Football Analysis")
guardar(fig, salida("duelo_sf_sea_como_defiende_2026.png", 2026, 5))
