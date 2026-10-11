"""EJEMPLO (11-oct-2026) de previa con tarjeta de cruce al estilo de SumerSports:
una situacion, ataque a la izquierda y defensa a la derecha, fila a fila, con
los logos en la cabecera en vez de fotos. CHI @ GB, 3er down: ataque de Green
Bay contra defensa de Chicago. No sustituye a Previas.py: es una prueba.

3er down = marcas de nflverse (third_down_converted / third_down_failed), igual
que Previas.py. Blitz = FTN n_blitzers > 0 sobre dropbacks. 3er down largo = 7
o mas yardas por ganar. Cobertura al hombre: Sharp, toda la temporada (no hay
cobertura por situacion). En empates el puesto es el mejor de los empatados."""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)
from pbp_loader import cargar_pbp, cargar_ftn, cargar_lesiones   # noqa: E402
from estilo import *            # noqa: E402,F401,F403

HASTA, AT, DF = 4, "GB", "CHI"
df, _ = cargar_pbp(2026)
df = df[df.week <= HASTA]
ftn, _ = cargar_ftn(2026)
t = df[(df.third_down_converted == 1) | (df.third_down_failed == 1)].merge(
    ftn[["nflverse_game_id", "nflverse_play_id", "n_blitzers"]],
    left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="left")
t["drop"] = t.qb_dropback == 1
t["largo"] = t.ydstogo >= 7
t["blitz"] = t.n_blitzers > 0


def lado(col):
    g, d = t.groupby(col), t[t["drop"]].groupby(col)
    lg = t[t.largo].groupby(col)
    return pd.DataFrame({"n": g.size(), "hechos": g.third_down_converted.sum(),
                         "conv": g.third_down_converted.mean(), "pase": g["drop"].mean(),
                         "ytg": g.ydstogo.mean(), "n_drop": d.size(), "blitz": d.blitz.mean(),
                         "sacks": d.sack.sum(), "largo_n": lg.size(),
                         "largo_hechos": lg.third_down_converted.sum(),
                         "largo_conv": lg.third_down_converted.mean()})


def rk(s, eq, asc=False):
    return int(s.rank(ascending=asc, method="min")[eq])


A, D = lado("posteam"), lado("defteam")
a, d = A.loc[AT], D.loc[DF]
tg = t[(t.posteam == AT) & (t.pass_attempt == 1) & (t.sack == 0)]
top = tg.receiver_player_name.value_counts()
cob = pd.read_csv(os.path.join("datos_web", f"sharp_coberturas_2026_w{HASTA:02d}.csv")).set_index("team")
R_ = dict(a_conv=rk(A.conv, AT), d_conv=rk(D.conv, DF, True), a_pase=rk(A.pase, AT),
          a_ytg=rk(A.ytg, AT), a_sacks=rk(A.sacks, AT, True), d_sacks=rk(D.sacks, DF),
          d_blitz=rk(D.blitz, DF), d_largo=rk(D.largo_conv, DF, True), man=rk(cob.man_pct, DF))
print(a.to_dict(), "\n", d.to_dict(), "\n", R_, "\n", top.head(4).to_dict(), len(tg),
      "| man", cob.loc[DF, "man_pct"], "media", round(cob.man_pct.mean(), 1),
      "| blitz media", round(D.blitz.mean(), 3))
try:
    les, _ = cargar_lesiones(2026)
    x = les[(les.team == AT) & (les.week == les.week.max())]
    print("lesiones", AT, "sem.", les.week.max(), x[x.position.isin(["WR", "TE"])][
        ["full_name", "report_status", "practice_status"]].to_dict("records"))
except Exception as e:                                             # noqa: BLE001
    print("lesiones no disponibles:", e)

COL = colores_pareja(AT, DF)
XM, XI, XD = 0.5, L, R                     # eje central y bordes de las dos columnas
fig = lienzo()

# Cabecera: cada mitad teñida con el color de su equipo y su logo en grande
for eq, x0 in ((AT, 0.0), (DF, 0.5)):
    ax = fig.add_axes([x0, 0.80, 0.5, 0.20], zorder=-2)
    g = np.zeros((60, 2, 4))
    g[..., :3] = to_rgb(COL[eq])
    g[..., 3] = np.linspace(0.50, 0.0, 60)[:, None]
    ax.imshow(g, aspect="auto", extent=(0, 1, 0, 1))
    ax.axis("off")
    poner_logo(fig, eq, x0 + 0.25, 0.895, zoom=0.105)
fig.text(XM, 0.895, "vs", family=TXT, weight="bold", size=9, color=GRIS1, ha="center", va="center")

antetitulo(fig, "Previa · Semana 5 · CHI @ GB · 3er down", y=0.785)
y = titular(fig, "Green Bay no convierte un 3er down\ny Chicago no los concede.", y=0.762, size=21)

y_top = y - 0.022
yh = y_top - 0.03
for eq, rol, x, ha in ((AT, "ATAQUE", XI, "left"), (DF, "DEFENSA", XD, "right")):
    fig.text(x, yh, rol, family=TXT, weight="bold", size=7.5, color=COL[eq], va="center", ha=ha)
    fig.text(x, yh - 0.026, nombre_equipo(eq), family=DISP, weight="bold", size=14, color=FG,
             va="center", ha=ha)
yn = yh - 0.085
numero(fig, pct(a.conv), XI, yn, size=40, color=color_rango(R_["a_conv"]))
numero(fig, pct(d.conv), XD, yn, size=40, color=color_rango(R_["d_conv"]), ha="right")
fig.text(XM, yn, "vs", family=TXT, weight="bold", size=8, color=GRIS2, ha="center", va="center")
nota(fig, f"3er downs convertidos · {a.hechos:.0f} de {a.n:.0f} · {R_['a_conv']}º", XI, yn - 0.043)
nota(fig, f"3er downs concedidos · {d.hechos:.0f} de {d.n:.0f} · {R_['d_conv']}ª", XD, yn - 0.043,
     ha="right")

ALTO_C = 0.098
y0 = yn - 0.068
ANCHO_B = 0.20


def barra(x, yy, frac, color, ha="left"):
    xx = x if ha == "left" else x - ANCHO_B
    fig.add_artist(Rectangle((xx, yy - 0.004), ANCHO_B, 0.008, transform=fig.transFigure,
                             color=GRID, lw=0))
    fig.add_artist(Rectangle((xx, yy - 0.004), ANCHO_B * frac, 0.008, transform=fig.transFigure,
                             color=color, lw=0))


def celda(i, x, ha, rot, valor, acl, bar=None, color=FG):
    yy = y0 - i * ALTO_C
    fig.text(x, yy - 0.022, rot.upper(), family=TXT, weight="semibold", size=7, color=GRIS1,
             va="center", ha=ha)
    fig.text(x, yy - 0.052, valor, family=DISP, weight="bold", size=17, color=color, va="center",
             ha=ha)
    nota(fig, acl, x, yy - 0.08, ha=ha)
    if bar is not None:
        sep = 0.12 if ha == "left" else -0.12
        barra(x + sep, yy - 0.052, bar[0], bar[1], ha)


def juicio(r):
    """Color solo si el puesto es extremo; si no, FG."""
    c = color_rango(r)
    return FG if c == GRIS1 else c


filas = [
    (("Pasa", pct(a.pase), f"el que más pasa en 3er down · {R_['a_pase']}º", (a.pase, COL[AT]), FG),
     ("Blitz", pct(d.blitz), f"{R_['d_blitz']}ª · media NFL {pct(D.blitz.mean())}",
      (d.blitz, COL[DF]), FG)),
    (("Yardas por ganar", num(a.ytg, 1, signo=False),
      f"de media · {R_['a_ytg']}ª distancia más larga", None, FG),
     ("3er down largo (7+)", f"{d.largo_hechos:.0f} de {d.largo_n:.0f}",
      "convertidos · la que menos concede", None, juicio(R_["d_largo"]))),
    (("Sacks recibidos", f"{a.sacks:.0f}", f"en {a.n_drop:.0f} dropbacks · {R_['a_sacks']}º que menos",
      None, juicio(R_["a_sacks"])),
     ("Sacks", f"{d.sacks:.0f}", f"en {d.n_drop:.0f} dropbacks · {R_['d_sacks']}ª que más", None,
      juicio(R_["d_sacks"]))),
    (("A quién va el pase", f"{top.index[0]} {top.iloc[0]} · {top.index[1]} {top.iloc[1]}",
      f"targets, de {len(tg)} pases", None, FG),
     ("Cobertura al hombre", f"{cob.loc[DF, 'man_pct']:.0f}%".replace(".", ","),
      f"{R_['man']}ª que más · toda la temporada", None, FG)),
]
for i, (izq, der) in enumerate(filas):
    linea_h(fig, y0 - i * ALTO_C)
    celda(i, XI, "left", *izq)
    celda(i, XD, "right", *der)
y_fin = y0 - len(filas) * ALTO_C
linea_v(fig, XM, y_fin + 0.012, y0 - 0.012)
tarjeta(fig, y_fin - 0.005, y_top)
pie(fig, "Color solo en los 5 mejores y los 5 peores de 32 · Blitz: FTN · Cobertura: Sharp",
    f"NFL 2026 · datos hasta sem. {HASTA}")
guardar(fig, os.path.join("lab", "previa_cruce_GB_CHI_ejemplo.png"))
