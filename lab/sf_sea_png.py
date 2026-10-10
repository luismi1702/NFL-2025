"""PNG del duelo del sabado 10-oct-2026: el pase corto de San Francisco contra
la defensa de Seattle, que cierra el pase largo y obliga a lanzar corto.
Estilo docs/estilo-visual.md, misma plantilla que lab/den_sf_png.py.

Definiciones: pase = intento con air_yards (sin sacks ni spikes); corto = menos
de 10 yardas por el aire, largo = 10 o mas; exito = EPA > 0. Congelado en las
semanas 1-4: el TNF de la semana 5 mueve la media de la liga y descuadraria el
texto.
OJO con el 28º de Seattle en EPA permitido en pase corto (el dato de
SumerSports, que nos sale igual): en yardas por pase es la 7ª que menos concede
y en exito la 14ª. El EPA lo suben 4 TD, varios 4º down convertidos y ninguna
intercepcion. Por eso el PNG pinta el matiz y el texto no habla de "grieta".

Los titulares son las frases que abren los tuits 2/3 y 3/3 del hilo
(salidas/2026/w05/textos/borradores_sabado.md)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pbp_loader import cargar_pbp, salida   # noqa: E402
from estilo import *                         # noqa: E402,F401,F403

HASTA = 4
CORTO = 10


def rk(s, t, asc=False):
    s = s.sort_values(ascending=asc)
    return list(s.index).index(t) + 1


df, _ = cargar_pbp(2026)
df = df[df.week <= HASTA]
d = df[(df.qb_dropback == 1) & (df.qb_spike != 1) & df.epa.notna() & df.posteam.notna()]
p = d[(d.pass_attempt == 1) & (d.sack != 1) & d.air_yards.notna()]
c, lg = p[p.air_yards < CORTO], p[p.air_yards >= CORTO]

# San Francisco en pase corto
e, s, y = (c.groupby("posteam").epa.mean(), c.groupby("posteam").success.mean(),
           c.groupby("posteam").yards_gained.mean())
sf = dict(n=int((c.posteam == "SF").sum()), epa=e["SF"], rk_epa=rk(e, "SF"),
          exito=s["SF"], rk_exito=rk(s, "SF"), yds=y["SF"], rk_yds=rk(y, "SF"),
          liga_epa=c.epa.mean(), liga_exito=c.success.mean(), liga_yds=c.yards_gained.mean())
sk = d.groupby("posteam").sack.sum()
sacks_sf, db_sf, sin_sacks = int(sk["SF"]), int((d.posteam == "SF").sum()), int((sk == 0).sum())

# Seattle contra el pase largo y el corto
sea = {}
for k, x in (("largo", lg), ("corto", c)):
    e = x.groupby("defteam").epa.mean()
    z = x[x.defteam == "SEA"]
    sea[k] = dict(n=len(z), epa=e["SEA"], rk=rk(e, "SEA", asc=True), liga=x.epa.mean(),
                  comp=int(z.complete_pass.sum()), td=int(z.touchdown.sum()),
                  ints=int(z.interception.sum()), yds=z.yards_gained.mean())
yc = c.groupby("defteam").yards_gained.mean()
sea["corto"]["rk_yds"] = rk(yc, "SEA", asc=True)
cuota = p.assign(c=p.air_yards < CORTO).groupby("defteam").c.mean()
cuota_sea = dict(pct=cuota["SEA"], rk=rk(cuota, "SEA"))
print("SF corto", sf)
print("SF sacks", sacks_sf, "en", db_sf, "| equipos sin sacks:", sin_sacks)
print("SEA", sea)
print("SEA cuota de pases cortos recibidos", cuota_sea)

SELLO = f"NFL 2026 · datos hasta sem. {HASTA} · nflverse PBP"
PIE = "Pase corto = menos de 10 yardas por el aire · largo = 10 o más"


def bloque(fig, x, y, etiq, valor, unidad, rk_, nota_txt):
    etiqueta(fig, etiq, x + 0.01, y)
    t = numero(fig, valor, x + 0.005, y - 0.065, size=44)
    if unidad:
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        fig.text(bb.x1 / fig.bbox.width + 0.004, y - 0.08, unidad, family=DISP,
                 weight="semibold", size=19, color=FG, va="center")
    fig.text(x + 0.01, y - 0.13, ordinal(rk_), family=TXT, weight="bold", size=9,
             color=color_rango(rk_), va="center")
    nota(fig, nota_txt, x + 0.01, y - 0.158)
    pista32(fig, x + 0.01, y - 0.195, 0.36, rk_)


def cabecera(fig, team, rotulo, y_top, dx):
    yc = y_top - 0.045
    poner_logo(fig, team, L + 0.035, yc, zoom=0.022)
    fig.text(L + 0.08, yc, team, family=TXT, weight="bold", size=10, color=color_equipo(team),
             va="center")
    fig.text(L + dx, yc, rotulo, family=TXT, weight="semibold", size=8.5, color=GRIS1,
             va="center")
    return yc - 0.065


# ── PNG 1: el pase corto de San Francisco
fig = lienzo()
antetitulo(fig, "Duelo · SF @ SEA · domingo")
y = titular(fig, "Nadie saca más del pase corto que San Francisco.", y=0.925) - 0.012
y = contexto(fig, [("Gana", None), (num(sf["yds"], 1, signo=False), "fuerte"),
                   ("yardas por cada pase corto, con la media de la NFL en", None),
                   (num(sf["liga_yds"], 1, signo=False) + ".", "fuerte")], y=y)
y_top = y - 0.025
yb = cabecera(fig, "SF", "·  PASE CORTO", y_top, 0.14)
bloque(fig, L, yb, "EPA por pase corto", num(sf["epa"]), "", sf["rk_epa"],
       f"{sf['n']} pases · media NFL {num(sf['liga_epa'])}")
bloque(fig, 0.53, yb, "Pases cortos con éxito", f"{sf['exito'] * 100:.0f}", "%", sf["rk_exito"],
       f"{sf['n']} pases · media NFL {sf['liga_exito'] * 100:.0f}%")
linea_v(fig, 0.5, yb - 0.21, yb + 0.015)
ym = yb - 0.26
linea_h(fig, ym + 0.025)
etiqueta(fig, "Y además", L + 0.01, ym)
unico = "el único equipo de la NFL sin ninguno." if sin_sacks == 1 else "sin ninguno todavía."
texto_rico(fig, L + 0.01, ym - 0.022, [(f"{sacks_sf} sacks", FG, "bold"),
                                        (f"en {db_sf} dropbacks: {unico}", GRIS1, "normal")],
           size=10, ancho=R - L - 0.03)
tarjeta(fig, ym - 0.085, y_top)
pie(fig, PIE + " · Éxito = EPA > 0", SELLO)
guardar(fig, salida("duelo_sf_sea_ataque_2026.png", 2026, 5))

# ── PNG 2: la defensa de Seattle
fig = lienzo()
antetitulo(fig, "Duelo · SF @ SEA · domingo")
y = titular(fig, "Seattle cierra el pase largo y obliga a lanzar corto.", y=0.925) - 0.012
y = contexto(fig, [("El", None), (f"{cuota_sea['pct'] * 100:.0f}%", "fuerte"),
                   (f"de los pases que le lanzan son cortos: {cuota_sea['rk']}º de la NFL.", None)],
             y=y)
y_top = y - 0.025
yb = cabecera(fig, "SEA", "·  DEFENSA CONTRA EL PASE", y_top, 0.15)
a, b = sea["largo"], sea["corto"]
bloque(fig, L, yb, "EPA permitido · pase largo", num(a["epa"]), "", a["rk"],
       f"{a['comp']} completos y {a['ints']} INT en {a['n']} · media NFL {num(a['liga'])}")
bloque(fig, 0.53, yb, "EPA permitido · pase corto", num(b["epa"]), "", b["rk"],
       f"{b['n']} pases · media NFL {num(b['liga'])}")
linea_v(fig, 0.5, yb - 0.21, yb + 0.015)
ym = yb - 0.26
linea_h(fig, ym + 0.025)
etiqueta(fig, "El matiz", L + 0.01, ym)
texto_rico(fig, L + 0.01, ym - 0.022,
           [(f"{num(b['yds'], 1, signo=False)} yardas por pase corto,", FG, "bold"),
            (f"{b['rk_yds']}ª que menos concede. Lo que le cuesta: {b['td']} TD y "
             f"{b['ints']} intercepciones.", GRIS1, "normal")],
           size=10, ancho=R - L - 0.03)
tarjeta(fig, ym - 0.085, y_top)
pie(fig, PIE + " · 1º = el que menos concede", SELLO)
guardar(fig, salida("duelo_sf_sea_defensa_2026.png", 2026, 5))
