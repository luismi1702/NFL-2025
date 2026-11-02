"""
premios_temporada.py
Premios de la temporada en curso (pensado para la mitad, nov-2026): 5 PNGs
con el top 5 de cada premio, un titular automatico y el ganador destacado.

  MVP              QB con 5+ partidos Y 200+ jugadas, por EPA/jugada + CPOE/100
                   (la convencion EPA+CPOE de la auditoria de jul-2026)
  JUGADOR OFENSIVO cualquiera MENOS los QB: EPA total de recepcion (cada
                   target, credito completo) + carrera
  DEFENSA          EPA que quitan sus jugadas decisivas: sacks (medio sack a
                   medias), intercepciones y fumbles forzados que acaban en
                   perdida (a medias si lo recupera otro defensor). Mismo
                   reparto que MVPsSemana/MVPsSeason (14-oct-2026); cada jugada,
                   una vez por jugador
  ROOKIE ATAQUE    como JUGADOR OFENSIVO, solo rookies; un QB rookie entra
                   con el EPA total de sus jugadas
  ROOKIE DEFENSA   como DEFENSA, solo rookies

Decisiones de Luis (02-nov-2026): el MVP es para un QB y el Jugador Ofensivo
para alguien que no lo sea; minimo de 5 partidos y 200 jugadas para el MVP
(con "o" se colaba Lamar Jackson: 5 partidos pero 118 jugadas). Los que serian
1os por jugada sin llegar al minimo salen en una nota aparte.

Va por GSIS id, no por nombre (MVPsSeason cruza por "F.Apellido").
Uso: python premios_temporada.py [--season 2026] [--week 8]
"""

import sys
import numpy as np
import pandas as pd
from pbp_loader import cargar_pbp, cargar_rosters, salida, season_cli, week_cli, sello
from estilo import *   # noqa: F401,F403  (estilo comun, docs/estilo-visual.md)

sys.stdout.reconfigure(encoding="utf-8")

SEASON = season_cli()
MIN_PARTIDOS, MIN_JUGADAS = 5, 200
TOP = 5


# ── DATOS ─────────────────────────────────────────────────────────────────────
pbp, SEASON = cargar_pbp(SEASON)
SEMANA = week_cli()
if SEMANA:
    pbp = pbp[pbp["week"] <= SEMANA]
for c in ["epa", "cpoe", "yards_gained", "sack", "fumble_lost", "qb_dropback",
          "complete_pass", "pass_touchdown", "rush_touchdown", "interception"]:
    pbp[c] = pd.to_numeric(pbp[c], errors="coerce")
p = pbp[pbp["epa"].notna()].copy()
print(f"PBP {SEASON}: {len(p):,} jugadas con EPA, semanas {int(p.week.min())}-{int(p.week.max())}")

ros, _ = cargar_rosters(SEASON)
ros = ros.sort_values("week").drop_duplicates("gsis_id", keep="last").set_index("gsis_id")
POS = ros["position"]
# Lo que se ve en el PNG: CB y S separados (regla de CLAUDE.md), nunca "DB"
_DC = ros["depth_chart_position"].replace({"FS": "S", "SS": "S"})
POS_VIS = ros["position"].where(ros["position"] != "DB", _DC)
NOMBRE = ros["full_name"]
ROOKIES = set(ros.index[pd.to_numeric(ros["rookie_year"], errors="coerce") == SEASON])
print(f"Rookies {SEASON} en rosters: {len(ROOKIES)}")


def equipo_mas_comun(df, col_pid, col_eq):
    return df.groupby(col_pid)[col_eq].agg(lambda s: s.mode().iat[0])


# MVP: dropbacks del QB (pase, sack, scramble) + sus carreras diseñadas
db = p[p["qb_dropback"] == 1].copy()
db["pid"] = db["passer_player_id"].fillna(db["rusher_player_id"])
corre = p[(p["play_type"] == "run") & (p["qb_dropback"] != 1)].copy()
corre["pid"] = corre["rusher_player_id"]
corre = corre[corre["pid"].map(POS) == "QB"]
qbj = pd.concat([db[["pid", "epa", "cpoe", "posteam", "game_id"]],
                 corre[["pid", "epa", "posteam", "game_id"]]])
qbj = qbj[qbj["pid"].map(POS) == "QB"]
qb = qbj.groupby("pid").agg(jugadas=("epa", "size"), epa_tot=("epa", "sum"),
                            epa_j=("epa", "mean"), cpoe=("cpoe", "mean"),
                            partidos=("game_id", "nunique"))
qb["eq"] = equipo_mas_comun(qbj, "pid", "posteam")
qb["valor"] = qb["epa_j"] + qb["cpoe"] / 100
partidos_eq = pd.concat([p.drop_duplicates("game_id")["home_team"],
                         p.drop_duplicates("game_id")["away_team"]]).value_counts()
qb["partidos_eq"] = qb["eq"].map(partidos_eq)
mvp_ok = qb[(qb["partidos"] >= MIN_PARTIDOS) & (qb["jugadas"] >= MIN_JUGADAS)]
mvp = mvp_ok.sort_values("valor", ascending=False)
fuera = qb[(qb["jugadas"] >= 100) & ~qb.index.isin(mvp_ok.index)
           & (qb["valor"] > mvp["valor"].iloc[0])].sort_values("valor", ascending=False)

# Ataque sin QB: recepcion (cada target) + carrera, credito completo
rec = p[(p["play_type"] == "pass") & p["receiver_player_id"].notna()].copy()
rec["pid"] = rec["receiver_player_id"]
rec["yds"] = np.where(rec["complete_pass"] == 1, rec["yards_gained"], 0)
rec["td"] = rec["pass_touchdown"].fillna(0)
rec["rec"] = rec["complete_pass"].fillna(0)
car = p[(p["play_type"] == "run") & p["rusher_player_id"].notna()].copy()
car["pid"] = car["rusher_player_id"]
car["yds"] = car["yards_gained"]
car["td"] = car["rush_touchdown"].fillna(0)
car["rec"] = 0
rec["tipo"], car["tipo"] = "rec", "car"
toq = pd.concat([rec, car])[["pid", "tipo", "epa", "yds", "td", "rec", "posteam", "game_id"]]
of = toq.groupby("pid").agg(epa_tot=("epa", "sum"), toques=("epa", "size"), td=("td", "sum"),
                           partidos=("game_id", "nunique"))
for t in ("rec", "car"):
    s = toq[toq["tipo"] == t].groupby("pid")
    of[f"epa_{t}"] = s["epa"].sum()
    of[f"yds_{t}"] = s["yds"].sum()
    of[f"n_{t}"] = s["epa"].size()
of["recepciones"] = toq[toq["tipo"] == "rec"].groupby("pid")["rec"].sum()
of = of.fillna(0)
of["eq"] = equipo_mas_comun(toq, "pid", "posteam")
of["pos"] = of.index.map(POS)
of["valor"] = of["epa_tot"]
opoy = of[of["pos"] != "QB"].sort_values("valor", ascending=False)

# Rookie ataque: los no QB como arriba; un QB rookie, con su EPA total
rq = qb[qb.index.isin(ROOKIES)][["epa_tot", "eq", "jugadas", "partidos", "epa_j", "cpoe"]].copy()
rq["valor"], rq["pos"] = rq["epa_tot"], "QB"
roy_of = pd.concat([of[of.index.isin(ROOKIES) & (of["pos"] != "QB")], rq])
roy_of = roy_of.sort_values("valor", ascending=False)

# Defensa: jugadas decisivas (mismo reparto que MVPsSemana/MVPsSeason)
por_jugada = {}


def dar(idx, pid, frac):
    if pd.isna(pid):
        return
    j = por_jugada.setdefault(idx, {})
    j[pid] = max(j.get(pid, 0.0), frac)


for idx, r in p[p["interception_player_id"].notna()].iterrows():
    dar(idx, r["interception_player_id"], 1.0)
sk = p[p["sack"] == 1]
for idx, r in sk[sk["sack_player_id"].notna()].iterrows():
    dar(idx, r["sack_player_id"], 1.0)
for idx, r in sk[sk["half_sack_1_player_id"].notna()].iterrows():
    dar(idx, r["half_sack_1_player_id"], 0.5)
    dar(idx, r["half_sack_2_player_id"], 0.5)
ff = p[(p["fumble_lost"] == 1) & p["forced_fumble_player_1_player_id"].notna()]
for idx, r in ff.iterrows():
    rec_id = r["fumble_recovery_1_player_id"]
    if (pd.notna(rec_id) and r["fumble_recovery_1_team"] == r["defteam"]
            and rec_id != r["forced_fumble_player_1_player_id"]):
        dar(idx, r["forced_fumble_player_1_player_id"], 0.5)
        dar(idx, rec_id, 0.5)
    else:
        dar(idx, r["forced_fumble_player_1_player_id"], 1.0)

filas = [(pid, idx, frac) for idx, j in por_jugada.items() for pid, frac in j.items()]
dj = pd.DataFrame(filas, columns=["pid", "idx", "frac"])
dj["epa"] = -p.loc[dj["idx"], "epa"].values * dj["frac"]
dj["eq"] = p.loc[dj["idx"], "defteam"].values
de = dj.groupby("pid").agg(valor=("epa", "sum"), jugadas=("epa", "size"))
de["eq"] = equipo_mas_comun(dj, "pid", "eq")
# Recuento para el lector (lo oficial: medio sack = 0,5)
de["sacks"] = (sk["sack_player_id"].value_counts()
               .add(sk["half_sack_1_player_id"].value_counts() * 0.5, fill_value=0)
               .add(sk["half_sack_2_player_id"].value_counts() * 0.5, fill_value=0))
de["ints"] = p["interception_player_id"].value_counts()
de["ff"] = ff["forced_fumble_player_1_player_id"].value_counts()
de = de.fillna(0)
de["pos"] = de.index.map(POS_VIS)
dpoy = de.sort_values("valor", ascending=False)
roy_de = de[de.index.isin(ROOKIES)].sort_values("valor", ascending=False)

for tabla in (mvp, fuera, opoy, dpoy, roy_of, roy_de):
    tabla["nombre"] = tabla.index.map(NOMBRE)


def mostrar(titulo, t, cols):
    print(f"\n== {titulo}")
    print(t.head(8)[["nombre", "eq"] + cols].round(3).to_string())


mostrar("MVP (QB)", mvp, ["partidos", "jugadas", "epa_j", "cpoe", "valor", "epa_tot"])
if len(fuera):
    mostrar("Fuera del MVP por minimo", fuera, ["partidos", "partidos_eq", "jugadas", "valor"])
mostrar("JUGADOR OFENSIVO (no QB)", opoy, ["pos", "partidos", "recepciones", "yds_rec", "yds_car",
                                           "td", "epa_rec", "epa_car", "valor"])
mostrar("DEFENSA", dpoy, ["pos", "sacks", "ints", "ff", "valor"])
mostrar("ROOKIE ATAQUE", roy_of, ["pos", "valor"])
mostrar("ROOKIE DEFENSA", roy_de, ["pos", "sacks", "ints", "ff", "valor"])


# ── DIBUJO ────────────────────────────────────────────────────────────────────
def apellido(nombre):
    partes = str(nombre).split()
    sufijos = {"Jr.", "Jr", "Sr.", "II", "III", "IV", "V"}
    while len(partes) > 1 and partes[-1] in sufijos:
        partes = partes[:-1]
    return " ".join(partes[1:]) if len(partes) > 1 else partes[0]


def etiqueta_equipo(team):
    """Ciudad; los que la comparten, con el nombre entero (NEW YORK JETS), no
    "los Jets" de nombre_equipo, que es para frases."""
    if team in ("LA", "LAC", "NYG", "NYJ"):
        try:
            return equipos().loc[team, "team_name"]
        except Exception:
            return team
    return nombre_equipo(team)


def miles(v):
    return f"{int(round(v)):,}".replace(",", ".")


def cuenta(v):
    """8,5 sacks / 2 sacks"""
    return (f"{v:.1f}".replace(".", ",") if v % 1 else f"{int(v)}")


def linea_ofensiva(r):
    trozos = []
    if r.get("n_rec", 0):
        trozos.append(f"{int(r['recepciones'])} rec. · {miles(r['yds_rec'])} yd")
    if r.get("n_car", 0) >= 10:
        trozos.append(f"{int(r['n_car'])} carreras · {miles(r['yds_car'])} yd")
    if r.get("td", 0):
        trozos.append(f"{int(r['td'])} TD")
    return " · ".join(trozos)


def linea_defensiva(r):
    t = []
    if r["sacks"]:
        t.append(f"{cuenta(r['sacks'])} sack{'s' if r['sacks'] != 1 else ''}")
    if r["ints"]:
        t.append(f"{int(r['ints'])} INT")
    if r["ff"]:
        t.append(f"{int(r['ff'])} fumble{'s' if r['ff'] != 1 else ''} forzado{'s' if r['ff'] != 1 else ''}")
    return " · ".join(t)


def linea_qb(r):
    return (f"{num(r['epa_j'])} EPA/jugada · CPOE {num(r['cpoe'], 1)} · "
            f"{int(r['partidos'])} partidos")


def linea_rookie(r):
    if r["pos"] == "QB":
        return f"QB · {int(r['jugadas'])} jugadas · {num(r['epa_j'])} EPA/jugada"
    return linea_ofensiva(r)


def ficha(clave, antet, titular_txt, contexto_trozos, t, fmt_valor, etiqueta_valor,
          linea, definicion, notas=(), empate=False):
    """empate=True: el 2º se pinta tambien como 1º (empate tecnico, decision
    de Luis del 02-nov-2026: en el tuit se elige a uno y se explica por que)."""
    top = t.head(TOP)
    fig = lienzo()
    antetitulo(fig, antet)
    y = titular(fig, titular_txt, y=0.925)
    y = contexto(fig, contexto_trozos, y=y - 0.018)

    # Ganador
    g = top.iloc[0]
    y1 = y - 0.012
    y0 = y1 - 0.205
    tarjeta(fig, y0, y1)
    col = color_equipo(g["eq"])
    poner_logo(fig, g["eq"], L + 0.085, (y0 + y1) / 2, zoom=0.05)
    fig.text(L + 0.19, y1 - 0.045, "1º", family=DISP, weight=800, size=13, color=col, va="center")
    fig.text(L + 0.235, y1 - 0.045, str(g["nombre"]).upper(), family=DISP, weight="bold",
             size=21, color=FG, va="center")
    etiqueta(fig, f"{g.get('pos', 'QB') if isinstance(g.get('pos'), str) else 'QB'} · "
             f"{etiqueta_equipo(g['eq'])}", L + 0.19, y1 - 0.083)
    numero(fig, fmt_valor(g["valor"]), L + 0.19, y0 + 0.068, size=36)
    etiqueta(fig, etiqueta_valor, L + 0.19, y0 + 0.025)
    fig.text(R, y0 + 0.025, linea(g), family=TXT, weight="semibold", size=8.5, color=FG,
             va="center", ha="right")

    # 2º-5º
    vmax = top["valor"].max()
    filas_y = y0 - 0.055
    paso = 0.088
    for i, (_, r) in enumerate(top.iloc[1:].iterrows(), start=2):
        yy = filas_y - (i - 2) * paso
        emp = empate and i == 2
        fig.text(L, yy, "1º" if emp else f"{i}º", family=DISP, weight=800, size=13,
                 color=color_equipo(r["eq"]) if emp else GRIS1, va="center")
        poner_logo(fig, r["eq"], L + 0.085, yy, zoom=0.022)
        fig.text(L + 0.14, yy + 0.012, str(r["nombre"]), family=TXT, weight="bold", size=10,
                 color=FG, va="center")
        fig.text(L + 0.14, yy - 0.016, linea(r), family=TXT, size=7.3, color=GRIS1, va="center")
        fig.text(R, yy + 0.012, fmt_valor(r["valor"]), family=DISP, weight="bold", size=14,
                 color=FG, va="center", ha="right")
        # barra proporcional al ganador
        w = 0.20 * max(r["valor"], 0) / vmax if vmax > 0 else 0
        fig.add_artist(Rectangle((R - 0.20, yy - 0.020), 0.20, 0.006, transform=fig.transFigure,
                                 color=GRID, lw=0))
        fig.add_artist(Rectangle((R - w, yy - 0.020), w, 0.006, transform=fig.transFigure,
                                 color=color_equipo(r["eq"]), lw=0))
        if i < len(top):
            linea_h(fig, yy - 0.044)

    yn = filas_y - (len(top) - 2) * paso - 0.068
    for n in notas:
        yn = texto_rico(fig, L, yn, [(n, GRIS1, "normal")], size=7.5, ancho=R - L) - 0.004

    pie(fig, definicion, sello(SEASON))
    return guardar(fig, salida(f"premios_{clave}_{SEASON}.png", SEASON, SEMANA))


ANTE = "PREMIOS DE MITAD DE TEMPORADA"
fe = lambda v: num(v, 1)            # EPA total: +60,3
fq = lambda v: num(v, 3)            # EPA+CPOE: +0,327


def ventaja(t):
    return t["valor"].iloc[0] - t["valor"].iloc[1]


# MVP
a, b, c = mvp.iloc[0], mvp.iloc[1], mvp.iloc[2]
hueco = a["valor"] - c["valor"]
if hueco < 0.01:
    tit = (f"{apellido(a['nombre'])}, por la mínima: {apellido(b['nombre'])} y "
           f"{apellido(c['nombre'])}, a un suspiro")
else:
    tit = f"{apellido(a['nombre'])} es el QB más eficiente de la temporada"
notas_mvp = [f"{r['nombre']} ({r['eq']}) sería 1º con {fq(r['valor'])}, pero suma "
             f"{int(r['jugadas'])} jugadas en {int(r['partidos'])} de "
             f"{int(r['partidos_eq'])} partidos."
             for _, r in fuera.iterrows()]
ficha("mvp", f"{ANTE} · MVP", tit,
      [("QB con", None), (f"{MIN_PARTIDOS}+ partidos", "fuerte"), ("y", None),
       (f"{MIN_JUGADAS}+ jugadas,", "fuerte"),
       ("ordenados por lo que suma cada jugada y la precisión de sus pases.", None)],
      mvp, fq, "EPA por jugada + CPOE", linea_qb,
      "EPA+CPOE = EPA por jugada (pases, sacks y carreras del QB) + CPOE/100",
      notas_mvp)

# Jugador ofensivo
a, b = opoy.iloc[0], opoy.iloc[1]
ficha("jugador_ofensivo", f"{ANTE} · JUGADOR OFENSIVO",
      f"Nadie fuera del QB suma más que {apellido(a['nombre'])}: {fe(ventaja(opoy))} de EPA "
      f"sobre el segundo",
      [("Sin QB. EPA total de cada balón que le buscan y cada carrera:", None),
       (fe(a["valor"]), "fuerte"), ("el de " + apellido(a["nombre"]) + ".", None)],
      opoy, fe, "EPA total (recepción + carrera)", linea_ofensiva,
      "EPA en recepción = todos los pases que le lanzan (completos o no)")

# Defensa
a, b = dpoy.iloc[0], dpoy.iloc[1]
EMPATE_EPA = 1.0                     # menos de 1 punto de EPA = empate tecnico
emp_de = ventaja(dpoy) < EMPATE_EPA
tit_de = (f"{apellido(a['nombre'])} y {apellido(b['nombre'])}, en empate técnico: les separa "
          f"{num(ventaja(dpoy), 1, signo=False)} de EPA" if emp_de else
          f"{apellido(a['nombre'])} es el defensor que más puntos ha quitado con sus jugadas decisivas")
ficha("defensa", f"{ANTE} · JUGADOR DEFENSIVO", tit_de,
      [("EPA que restan sus", None), ("sacks,", "fuerte"),
       ("intercepciones", "fuerte"), ("y", None), ("fumbles forzados", "fuerte"),
       ("que acaban en pérdida.", None)],
      dpoy, fe, "EPA quitado", linea_defensiva,
      "Medio sack, mitad del EPA. Fumble recuperado por otro defensor, a medias",
      empate=emp_de)

# Rookies
a = roy_of.iloc[0]
ficha("rookie_ataque", f"{ANTE} · ROOKIE OFENSIVO",
      f"{apellido(a['nombre'])} es el rookie que más suma en ataque",
      [("EPA total de recepción y carrera; un QB rookie, con todas sus jugadas.", None)],
      roy_of, fe, "EPA total", linea_rookie,
      "EPA en recepción = todos los pases que le lanzan (completos o no)")

a = roy_de.iloc[0]
ficha("rookie_defensa", f"{ANTE} · ROOKIE DEFENSIVO",
      f"{apellido(a['nombre'])} es el rookie que más puntos ha quitado en defensa",
      [("EPA que restan sus", None), ("sacks,", "fuerte"),
       ("intercepciones", "fuerte"), ("y", None), ("fumbles forzados", "fuerte"),
       ("que acaban en pérdida.", None)],
      roy_de, fe, "EPA quitado", linea_defensiva,
      "Medio sack, mitad del EPA. Fumble recuperado por otro defensor, a medias")
