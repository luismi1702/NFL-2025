"""PNG 3 y 4 del duelo del sabado 10-oct-2026 (SF @ SEA): la carrera segun la
caja y el 21 personnel de San Francisco. Misma plantilla que lab/sf_sea_png.py;
los numeros de exploracion salen de lab/sf_sea_carrera.py.

Definiciones:
- Carrera = carrera disenada (play_type == "run", sin scrambles ni rodillas).
- Caja = n_defense_box de FTN (16 partidos de 16 en las semanas 1-4). "7" es
  exactamente 7; Seattle solo ha puesto 8 o mas en 2 carreras de 96.
- Personal: fichero jugada a jugada de Sharp (semanas 1-4, sin rodillas ni
  spikes). 21 = 2 RB y 1 TE. Exito = EPA > 0.
- Congelado en las semanas 1-4, como los dos primeros PNG.
OJO: Seattle con caja ligera concede 4,4 yardas (14ª) en 38 carreras, pero 10
de ellas (a 7,3 yardas) son de los Chargers en la semana 4; sin ese partido son
28 carreras a 3,3. Por eso el PNG 3 se apoya en San Francisco (54 carreras con
caja ligera, a mas de 4,4 yardas en los cuatro partidos) y de Seattle solo
pinta el dato con siete en la caja. Contra el 21, 33 jugadas, 22 tambien de
los Chargers: el PNG 4 lo dice.

Los titulares son las frases que abren los tuits 4 y 5 del hilo
(salidas/2026/w05/textos/borradores_sabado.md)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pbp_loader import cargar_pbp, cargar_ftn, cargar_calendario, salida   # noqa: E402
import copiar_personal_web as cpw                                         # noqa: E402
from estilo import *                                                      # noqa: E402,F401,F403

HASTA = 4


def rk(s, t, asc=False):
    s = s.dropna().sort_values(ascending=asc)
    return list(s.index).index(t) + 1


# ── Carrera y caja (PBP + FTN)
df, _ = cargar_pbp(2026)
df = df[df.week <= HASTA]
ftn, _ = cargar_ftn(2026)
df = df.merge(ftn[["nflverse_game_id", "nflverse_play_id", "n_defense_box"]],
              left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"],
              how="left")
car = df[(df.play_type == "run") & (df.qb_scramble != 1) & (df.qb_kneel != 1) & df.epa.notna()
         & df.n_defense_box.notna()].copy()
siete, ligera = car[car.n_defense_box == 7], car[car.n_defense_box <= 6]
sea = {}
for k, x in (("siete", siete), ("ligera", ligera)):
    y, e = x.groupby("defteam").yards_gained.mean(), x.groupby("defteam").success.mean()
    z = x[x.defteam == "SEA"]
    sea[k] = dict(n=len(z), yds=y["SEA"], rk_yds=rk(y, "SEA", asc=True), exito=e["SEA"],
                  rk_exito=rk(e, "SEA", asc=True), liga=x.yards_gained.mean())
cuota = car.assign(o=car.n_defense_box <= 6).groupby("posteam").o.mean()
y, e = ligera.groupby("posteam").yards_gained.mean(), ligera.groupby("posteam").success.mean()
sf = dict(n=int((ligera.posteam == "SF").sum()), cuota=cuota["SF"], rk_cuota=rk(cuota, "SF"),
          yds=y["SF"], rk_yds=rk(y, "SF"), exito=e["SF"], rk_exito=rk(e, "SF"),
          yds7=siete[siete.posteam == "SF"].yards_gained.mean(),
          rk_yds7=rk(siete.groupby("posteam").yards_gained.mean(), "SF"),
          exito7=siete[siete.posteam == "SF"].success.mean(),
          n7=int((siete.posteam == "SF").sum()))
sea_car = car[car.defteam == "SEA"]
sea_cuota7 = (sea_car.n_defense_box == 7).mean()
sin_lac = ligera[(ligera.defteam == "SEA") & (ligera.posteam != "LAC")]
print("SEA % de carreras con 7 en la caja:", round(sea_cuota7, 3), "| caja ligera sin LAC:",
      len(sin_lac), round(sin_lac.yards_gained.mean(), 2), round(sin_lac.success.mean(), 2))
print("SF por partido, caja ligera:", ligera[ligera.posteam == "SF"].groupby("week").yards_gained
      .agg(["size", "mean"]).round(2).to_dict("index"))
ocho = int(((car.defteam == "SEA") & (car.n_defense_box >= 8)).sum())
glob = car.groupby("defteam").agg(y=("yards_gained", "mean"), e=("success", "mean"))
print("SEA global", len(car[car.defteam == "SEA"]), glob.loc["SEA"].round(3).to_dict(),
      "puesto yds", rk(glob.y, "SEA", asc=True), "exito", rk(glob.e, "SEA", asc=True))
print("SEA caja", sea, "| carreras con 8+:", ocho)
print("SF caja ligera", sf)
lac = ligera[(ligera.defteam == "SEA") & (ligera.posteam == "LAC")]
print("SEA caja ligera ante LAC:", len(lac), round(lac.yards_gained.mean(), 2))

# ── 21 personnel (Sharp)
pl, generado = cpw.sharp_jugadas(2026)
raw = __import__("json").loads(cpw.bajar(cpw.SHARP.format(season=2026)))
inv = {ch: i for i, ch in enumerate(raw["alphabet"])}
pl["pase"] = [inv[ch] == 1 for ch in raw["cols"]["b"]]     # b == 1: pase (cuadra con el PBP)
pl = pl[pl.week <= HASTA]
cal, _ = cargar_calendario(2026)
cal = cal[cal.game_type == "REG"]
riv = {}
for _, g in cal.iterrows():
    riv[(int(g.week), g.home_team)] = g.away_team
    riv[(int(g.week), g.away_team)] = g.home_team
pl["rival"] = [riv.get((w, t)) for w, t in zip(pl.week, pl.team)]
uso = (pl.grupo == 21).groupby(pl.team).mean()
p21 = pl[pl.grupo == 21]
a, d = p21[p21.team == "SF"], p21[p21.rival == "SEA"]
per = dict(uso=uso["SF"], rk_uso=rk(uso, "SF"), uso_liga=(pl.grupo == 21).mean(), n=len(a),
           segundo=uso.sort_values(ascending=False).index[1],
           uso_segundo=uso.sort_values(ascending=False).iloc[1],
           car_n=int((~a.pase).sum()), car_yds=a[~a.pase].yds.mean(),
           car_exito=a[~a.pase].exito.mean(), car_liga=p21[~p21.pase].yds.mean(),
           pase_n=int(a.pase.sum()), pase_exito=a[a.pase].exito.mean(),
           pase_yds=a[a.pase].yds.mean(),
           sea_n=len(d), sea_lac=int((d.team == "LAC").sum()), sea_car_n=int((~d.pase).sum()),
           sea_car_yds=d[~d.pase].yds.mean(), sea_car_exito=d[~d.pase].exito.mean(),
           sea_pase_n=int(d.pase.sum()), sea_pase_exito=d[d.pase].exito.mean())
print("21 personnel", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in per.items()},
      "| Sharp generado", generado)

ANTE = "Duelo · SF @ SEA · domingo"


def bloque(fig, x, y, etiq, valor, unidad, rk_txt, color, nota_txt, rk_=None):
    etiqueta(fig, etiq, x + 0.01, y)
    t = numero(fig, valor, x + 0.005, y - 0.065, size=44)
    if unidad:
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        fig.text(bb.x1 / fig.bbox.width + 0.004, y - 0.08, unidad, family=DISP,
                 weight="semibold", size=19, color=FG, va="center")
    fig.text(x + 0.01, y - 0.13, rk_txt, family=TXT, weight="bold", size=9, color=color,
             va="center")
    nota(fig, nota_txt, x + 0.01, y - 0.158)
    if rk_:
        pista32(fig, x + 0.01, y - 0.195, 0.36, rk_)


def cabecera(fig, team, rotulo, y_top, dx):
    yc = y_top - 0.045
    poner_logo(fig, team, L + 0.035, yc, zoom=0.022)
    fig.text(L + 0.08, yc, team, family=TXT, weight="bold", size=10, color=color_equipo(team),
             va="center")
    fig.text(L + dx, yc, rotulo, family=TXT, weight="semibold", size=8.5, color=GRIS1,
             va="center")
    return yc - 0.065


def y1(v):
    return num(v, 1, signo=False)


# ── PNG 3: la carrera segun la caja (el dato solido es el de San Francisco)
s7 = sea["siete"]
fig = lienzo()
antetitulo(fig, ANTE)
y = titular(fig, "San Francisco corre según la caja que le pongan.", y=0.925) - 0.012
y = contexto(fig, [("Contra seis o menos gana", None), (y1(sf["yds"]), "fuerte"),
                   ("yardas por carrera. Contra siete,", None), (y1(sf["yds7"]) + ".", "fuerte")],
             y=y)
y_top = y - 0.025
yb = cabecera(fig, "SF", "·  CARRERA SEGÚN LA CAJA", y_top, 0.14)
bloque(fig, L, yb, "Yardas por carrera · 6 o menos", y1(sf["yds"]), "", ordinal(sf["rk_yds"]),
       color_rango(sf["rk_yds"]),
       f"{sf['n']} carreras · {sf['exito'] * 100:.0f}% con éxito · media NFL "
       f"{y1(sea['ligera']['liga'])}", sf["rk_yds"])
bloque(fig, 0.53, yb, "Yardas por carrera · 7 en la caja", y1(sf["yds7"]), "",
       ordinal(sf["rk_yds7"]), color_rango(sf["rk_yds7"]),
       f"{sf['n7']} carreras · {sf['exito7'] * 100:.0f}% con éxito · media NFL {y1(s7['liga'])}",
       sf["rk_yds7"])
linea_v(fig, 0.5, yb - 0.21, yb + 0.015)
ym = yb - 0.26
linea_h(fig, ym + 0.025)
etiqueta(fig, "Enfrente", L + 0.01, ym)
texto_rico(fig, L + 0.01, ym - 0.022,
           [(f"Seattle con siete en la caja: {y1(s7['yds'])} yardas por carrera,", FG, "bold"),
            (f"{s7['rk_yds']}ª de la NFL. Y pone siete en el {sea_cuota7 * 100:.0f}% de las "
             f"carreras que recibe.", GRIS1, "normal")],
           size=10, ancho=R - L - 0.03)
tarjeta(fig, ym - 0.085, y_top)
pie(fig, "Carreras diseñadas, sin scrambles · Caja = defensores junto a la línea al snap (FTN)",
    f"NFL 2026 · datos hasta sem. {HASTA} · nflverse PBP + FTN")
guardar(fig, salida("duelo_sf_sea_carrera_2026.png", 2026, 5))

# ── PNG 4: el 21 personnel
fig = lienzo()
antetitulo(fig, ANTE)
y = titular(fig, "Nadie juega con dos RB tanto como San Francisco.", y=0.925) - 0.012
y = contexto(fig, [("El", None), (f"{per['uso'] * 100:.0f}%", "fuerte"),
                   ("de sus jugadas salen con 2 RB y 1 TE. La media de la NFL es el", None),
                   (f"{per['uso_liga'] * 100:.0f}%.", "fuerte")], y=y)
y_top = y - 0.025
yb = cabecera(fig, "SF", "·  21 PERSONNEL (2 RB, 1 TE)", y_top, 0.14)
bloque(fig, L, yb, "Jugadas en 21 personnel", f"{per['uso'] * 100:.0f}", "%",
       ordinal(per["rk_uso"]), color_rango(per["rk_uso"]),
       f"{per['n']} jugadas · el 2º, {per['segundo']}, {per['uso_segundo'] * 100:.0f}%",
       per["rk_uso"])
bloque(fig, 0.53, yb, "Yardas por carrera desde 21", y1(per["car_yds"]), "",
       f"{per['car_exito'] * 100:.0f}% con éxito", FG,
       f"{per['car_n']} carreras · media NFL desde 21: {y1(per['car_liga'])}")
linea_v(fig, 0.5, yb - 0.21, yb + 0.015)
ym = yb - 0.26
linea_h(fig, ym + 0.025)
etiqueta(fig, "Enfrente", L + 0.01, ym)
texto_rico(fig, L + 0.01, ym - 0.022,
           [(f"Seattle contra el 21: {y1(per['sea_car_yds'])} yardas por carrera", FG, "bold"),
            (f"y un {per['sea_car_exito'] * 100:.0f}% de éxito en {per['sea_car_n']} carreras. "
             f"Solo ha visto {per['sea_n']} jugadas así, {per['sea_lac']} ante los Chargers.",
             GRIS1, "normal")],
           size=10, ancho=R - L - 0.03)
tarjeta(fig, ym - 0.085, y_top)
pie(fig, "21 personnel = 2 running backs (RB o FB) y 1 tight end · Éxito = EPA > 0",
    f"NFL 2026 · datos hasta sem. {HASTA} · Sharp Football Analysis")
guardar(fig, salida("duelo_sf_sea_personal_2026.png", 2026, 5))
