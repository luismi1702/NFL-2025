"""
cruces.py — los cruces ataque contra defensa más fuertes de cada partido (03-oct-2026)

Sustituto en temporada de matchup_intel.py (que necesita personal y coberturas
de pbp_participation, que no llegan hasta febrero). Solo usa datos GRATIS en
temporada: PBP y charting de FTN (blitzers, hombres en la caja, play action,
motion, pantallas).

Para cada partido de la jornada y cada lado del balón, busca:
  - VENTAJA: el ataque está en el top 10 de una situación y la defensa rival
    entre los 10 peores defendiéndola (o al revés: ventaja de la defensa)
  - CHOQUE: los dos en el top 8 (fuerza contra fuerza)
y solo cuenta la situación si se da de verdad en el partido: contra el blitz
importa si la defensa blitzea; con play action, si el ataque lo usa.

Métrica: % de jugadas con éxito (EPA > 0), que aguanta las muestras pequeñas
mejor que el EPA (en el duelo DEN @ SF, un lateral de 80 yardas inflaba el
EPA de SF contra el blitz). El EPA va al lado, de referencia.

Salidas:
  textos/cruces_semana_N.txt  — todos los partidos, los 3 cruces más fuertes
                                 de cada uno (herramienta interna y ganchos
                                 para el redactor del domingo)
  --partido VIS LOC            — además, el PNG del mejor cruce de ese partido
                                 (o el que se pida con --faceta), estilo duelo

    python cruces.py --week 5
    python cruces.py --week 5 --partido DEN SF
    python cruces.py --week 5 --partido DEN SF --faceta blitz --ataca SF
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from pbp_loader import (cargar_pbp, cargar_ftn, cargar_calendario, salida, season_cli,
                        week_cli, proxima_semana, DatosNoDisponibles)
from estilo import *   # noqa: F401,F403  (estilo comun, docs/estilo-visual.md)

sys.stdout.reconfigure(encoding="utf-8")

MIN_N = 12            # jugadas mínimas de cada lado para dar un puesto
VENTAJA = 10          # top 10 contra los 10 peores
CHOQUE = 8            # top 8 contra top 8
USO_MIN = 0.85        # el uso de la situación tiene que llegar al 85% de la media de la liga

# clave: (etiqueta, base, filtro, quién decide cuánto se da, frase del ataque, frase de la defensa)
FACETAS = {
    "blitz": ("Pase contra el blitz", "db", "bl", "def", "contra el blitz", "cuando blitzea"),
    "sinbl": ("Pase sin blitz", "db", "~bl", "def", "en el pase sin blitz", "en el pase sin blitzear"),
    "pa":    ("Pase con play action", "db", "pa", "atk", "con play action", "contra el play action"),
    "scr":   ("Pantallas", "db", "scr", "atk", "en pantallas", "contra las pantallas"),
    "carg":  ("Carrera contra caja cargada (8+)", "dis", "box_carg", "def",
              "corriendo contra caja cargada", "contra la carrera con caja cargada"),
    "lig":   ("Carrera contra caja ligera (6 o menos)", "dis", "box_lig", "def",
              "corriendo contra caja ligera", "contra la carrera con caja ligera"),
    "mot":   ("Jugadas con motion", "all", "mot", "atk", "con motion", "contra el motion"),
    "d3":    ("3er down", "all", "d3", None, "en 3er down", "en 3er down"),
    "rz":    ("Red zone", "all", "rz", None, "en la red zone", "en su red zone"),
}


# Cómo se cuenta el uso de cada situación ({e} = equipo que la decide)
USO_TXT = {
    "blitz": "{e} blitzea en el {p} de los pases", "sinbl": "{e} no blitzea en el {p} de los pases",
    "carg": "{e} pone 8 o más en la caja en el {p} de las carreras",
    "lig": "{e} pone 6 o menos en la caja en el {p} de las carreras",
    "pa": "{e} usa play action en el {p} de sus pases", "scr": "{e} lanza pantallas en el {p} de sus pases",
    "mot": "{e} usa motion en el {p} de sus jugadas",
}


def cargar(season, hasta):
    df, season = cargar_pbp(season)
    ftn, _ = cargar_ftn(season)
    df = df[df.week <= hasta]
    pr = df[df.play_type.isin(["pass", "run"]) & df.epa.notna() & df.posteam.notna()].copy()
    pr = pr[(pr.qb_kneel != 1) & (pr.qb_spike != 1)]
    pr = pr.merge(ftn[["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_defense_box",
                       "is_play_action", "is_motion", "is_screen_pass"]],
                  left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"],
                  how="left")
    pr["db"] = pr.qb_dropback == 1
    pr["dis"] = (pr.play_type == "run") & (pr.qb_scramble != 1)
    pr["all"] = True
    pr["bl"] = pd.to_numeric(pr.n_blitzers, errors="coerce") >= 1
    box = pd.to_numeric(pr.n_defense_box, errors="coerce")
    pr["box_carg"] = box >= 8
    pr["box_lig"] = box <= 6
    pr["pa"] = pr.is_play_action == True          # noqa: E712
    pr["mot"] = pr.is_motion == True              # noqa: E712
    pr["scr"] = pr.is_screen_pass == True         # noqa: E712
    pr["rz"] = pr.yardline_100 <= 20
    pr["d3"] = pr.down == 3
    return pr, season


def tablas(pr):
    """{faceta: dict(of, df, uso_of, uso_df)} con éxito, EPA, n y puesto."""
    out = {}
    for k, (_, base, filtro, _, _, _) in FACETAS.items():
        b = pr[pr[base]]
        neg = filtro.startswith("~")
        m = ~b[filtro[1:]] if neg else b[filtro]
        s = b[m]
        res = {}
        for lado, col, asc in (("of", "posteam", False), ("df", "defteam", True)):
            g = s.groupby(col).agg(exito=("success", "mean"), epa=("epa", "mean"), n=("epa", "size"))
            ok = g[g.n >= MIN_N]
            g["rk"] = ok.exito.rank(ascending=asc, method="min")
            res[lado] = g
        res["uso_of"] = m.groupby(b.posteam).mean()
        res["uso_df"] = m.groupby(b.defteam).mean()
        res["liga"] = s.success.mean()
        out[k] = res
    return out


def cruces_de(T, atk, dfn):
    """Cruces de un lado del balón, ordenados por fuerza."""
    lista = []
    for k, (lab, _, _, decide, corto, corto_d) in FACETAS.items():
        t = T[k]
        if atk not in t["of"].index or dfn not in t["df"].index:
            continue
        o, d = t["of"].loc[atk], t["df"].loc[dfn]
        if pd.isna(o.rk) or pd.isna(d.rk):
            continue
        o_rk, d_rk = int(o.rk), int(d.rk)
        tipo = None
        if o_rk <= VENTAJA and d_rk > 32 - VENTAJA:
            tipo = "ventaja ataque"
        elif o_rk > 32 - VENTAJA and d_rk <= VENTAJA:
            tipo = "ventaja defensa"
        elif o_rk <= CHOQUE and d_rk <= CHOQUE:
            tipo = "choque"
        if not tipo:
            continue
        # Relevancia: la situación tiene que darse en este partido
        if decide == "def":
            uso, uso_media, quien = t["uso_df"].get(dfn, np.nan), t["uso_df"].mean(), dfn
        elif decide == "atk":
            uso, uso_media, quien = t["uso_of"].get(atk, np.nan), t["uso_of"].mean(), atk
        else:
            uso, uso_media, quien = np.nan, np.nan, None
        if decide and (pd.isna(uso) or uso < USO_MIN * uso_media):
            continue
        fuerza = (abs(o_rk - d_rk) if tipo != "choque" else 33 - max(o_rk, d_rk)) \
            + (10 * (uso / uso_media - 1) if decide else 0)
        lista.append(dict(faceta=k, lab=lab, corto=corto, corto_d=corto_d, tipo=tipo, atk=atk, dfn=dfn,
                          o=o, d=d, o_rk=o_rk, d_rk=d_rk, uso=uso, uso_media=uso_media,
                          quien=quien, liga=t["liga"], fuerza=fuerza))
    return sorted(lista, key=lambda c: -c["fuerza"])


def frase(c):
    """Titular del cruce: solo puestos, que son hechos."""
    A, B = nombre_equipo(c["atk"]), nombre_equipo(c["dfn"])
    o, d = c["o_rk"], c["d_rk"]
    if c["tipo"] == "ventaja ataque":
        s = f"El ataque de {A} es {o}º {c['corto']}; la defensa de {B}, {d}º."
    elif c["tipo"] == "ventaja defensa":
        s = f"La defensa de {B} es {d}º {c['corto_d']}; el ataque de {A}, {o}º."
    else:   # choque: "Fuerza contra fuerza" va en el antetitulo
        s = f"{A} {v(A, 'es', 'son')} {o}º {c['corto']}; {B}, {d}º {c['corto_d']}."
    return s[0].upper() + s[1:]


def linea_txt(c):
    uso = ""
    if c["quien"]:
        uso = " · " + USO_TXT[c["faceta"]].format(e=c["quien"], p=pct(c["uso"]))             + f" (media {pct(c['uso_media'])})"
    return (f"  [{c['tipo'].upper()}] {c['lab']}: {c['atk']} {pct(c['o'].exito)} de éxito "
            f"({c['o_rk']}º, {int(c['o'].n)} jugadas, EPA {num(c['o'].epa)}) contra "
            f"{c['dfn']} {pct(c['d'].exito)} permitido ({c['d_rk']}º, {int(c['d'].n)} jugadas, "
            f"EPA {num(c['d'].epa)}){uso}\n      Titular: {frase(c)}")


def dibujar(c, season, week, hasta, vis, loc):
    """Un PNG por cruce: ataque a la izquierda, defensa a la derecha."""
    cols = colores_pareja(c["atk"], c["dfn"])
    fig = lienzo()
    antetitulo(fig, f"Duelo · {vis} @ {loc} · semana {week}"
               + (" · fuerza contra fuerza" if c["tipo"] == "choque" else ""))
    y = titular(fig, frase(c), y=0.925) - 0.012
    liga = c["liga"]
    y = contexto(fig, [(c["lab"] + ":", None), ("la media de la NFL acierta en el", None),
                       (pct(liga) + ".", "fuerte")], y=y)
    y_top = y - 0.025
    tarjeta_y0 = None

    yb = y_top - 0.06
    for x, lado, eq, rol, rk, v, etq in (
            (L, "o", c["atk"], "ATAQUE", c["o_rk"], c["o"], "Jugadas con éxito"),
            (0.53, "d", c["dfn"], "DEFENSA", c["d_rk"], c["d"], "Éxito permitido")):
        poner_logo(fig, eq, x + 0.03, yb + 0.012, zoom=0.02)
        fig.text(x + 0.07, yb + 0.012, eq, family=TXT, weight="bold", size=9.5, color=cols[eq],
                 va="center")
        fig.text(x + 0.135, yb + 0.012, f"·  {rol}", family=TXT, weight="semibold", size=8.3,
                 color=GRIS1, va="center")
        etiqueta(fig, etq, x + 0.01, yb - 0.035)
        t = numero(fig, f"{v.exito * 100:.0f}", x + 0.005, yb - 0.1, size=44)
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        fig.text(bb.x1 / fig.bbox.width + 0.004, yb - 0.115, "%", family=DISP, weight="semibold",
                 size=19, color=FG, va="center")
        fig.text(x + 0.01, yb - 0.165, ordinal(rk), family=TXT, weight="bold", size=9,
                 color=color_rango(rk), va="center")
        nota(fig, f"{int(v.n)} jugadas · EPA {num(v.epa)}", x + 0.01, yb - 0.193)
        pista32(fig, x + 0.01, yb - 0.23, 0.36, rk)
    linea_v(fig, 0.5, yb - 0.245, yb + 0.03)

    ym = yb - 0.29
    linea_h(fig, ym + 0.025)
    if c["quien"]:
        etiqueta(fig, "¿Se va a dar?", L + 0.01, ym)
        antes, despues = USO_TXT[c["faceta"]].split("{p}")
        texto_rico(fig, L + 0.01, ym - 0.022, [
            (antes.format(e=c["quien"]).strip(), GRIS1, "normal"),
            (pct(c["uso"]), FG, "bold"),
            (despues.strip() + f" (media de la NFL, {pct(c['uso_media'])}).", GRIS1, "normal")],
            size=10, ancho=R - L - 0.03)
        tarjeta_y0 = ym - 0.075
    else:
        tarjeta_y0 = ym + 0.01
    tarjeta(fig, tarjeta_y0, y_top)
    pie(fig, "Éxito = EPA > 0 · Nº = puesto entre 32 (en defensa, el que menos concede) · charting de FTN",
        f"NFL {season} · datos hasta sem. {hasta} · nflverse PBP + FTN")
    nombre = f"duelo_{vis.lower()}_{loc.lower()}_{c['faceta']}_{c['atk'].lower()}_{season}.png"
    guardar(fig, salida(nombre, season, week))


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--partido", nargs=2, metavar=("VIS", "LOC"))
    ap.add_argument("--faceta", choices=list(FACETAS))
    ap.add_argument("--ataca")
    args, _ = ap.parse_known_args()
    season = season_cli()
    week = week_cli() or proxima_semana()
    hasta = week - 1
    try:
        pr, season = cargar(season, hasta)
    except DatosNoDisponibles as e:
        raise SystemExit(f"Sin charting de FTN todavía: {e}")
    fsem = pr.groupby("week").nflverse_game_id.nunique()
    print(f"Charting FTN por semana: {fsem.to_dict()}")
    T = tablas(pr)

    cal, _ = cargar_calendario(season)
    cal = cal[(cal.game_type == "REG") & (pd.to_numeric(cal.week, errors="coerce") == week)]
    lineas = [f"CRUCES DE LA SEMANA {week} — {season}",
              f"Datos de las semanas 1-{hasta} (PBP + charting de FTN: {fsem.to_dict()}).",
              "Métrica: % de jugadas con éxito (EPA > 0); EPA de referencia. VENTAJA = top "
              f"{VENTAJA} contra los {VENTAJA} peores; CHOQUE = los dos en el top {CHOQUE}. Solo "
              "situaciones que se dan en el partido (uso >= 85% de la media).",
              "Herramienta interna: para elegir el duelo del sábado y ganchos para el domingo. "
              "Cualquier número que se publique se verifica como siempre.", ""]
    elegido = None
    for _, g in cal.iterrows():
        vis, loc = g.away_team, g.home_team
        todos = cruces_de(T, loc, vis) + cruces_de(T, vis, loc)
        todos.sort(key=lambda c: -c["fuerza"])
        lineas.append(f"== {vis} @ {loc} ({g.get('weekday', '')} {g.get('gametime', '')})")
        if not todos:
            lineas.append("  (ningún cruce claro)")
        for c in todos[:3]:
            lineas.append(linea_txt(c))
        lineas.append("")
        if args.partido and {vis, loc} == set(args.partido):
            cand = [c for c in todos if (not args.faceta or c["faceta"] == args.faceta)
                    and (not args.ataca or c["atk"] == args.ataca)]
            elegido = (cand[0], vis, loc) if cand else ("nada", vis, loc)

    txt = salida(f"cruces_semana_{season}.txt", season, week)
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas))
    print(f"Guardado: {txt}")

    if args.partido:
        if elegido is None:
            raise SystemExit(f"{args.partido[0]} @ {args.partido[1]} no se juega en la semana {week}")
        if elegido[0] == "nada":
            raise SystemExit("Ese partido no tiene ningún cruce que cumpla los filtros")
        dibujar(elegido[0], season, week, hasta, elegido[1], elegido[2])


if __name__ == "__main__":
    main()
