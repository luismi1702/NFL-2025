"""
personal_jornada.py — ¿alguien destrozó a otro con un personal concreto esta
jornada? (05-oct-2026). Herramienta A DEMANDA, solo texto, sin PNG ni batch.

Fuente: el fichero jugada a jugada de Sharp (el de su página de personal
ofensivo), que trae la semana de cada jugada. SumerSports no sirve: su filtro
por semana es de pago. Sharp lo publica el martes, a hora variable (29-sep 16:32, 06-oct 06:19), así que la
jornada que acaba de jugarse no está hasta entonces.

Criterio, como DatoSemana.py: para cada ataque y personal de la jornada
(mínimo MIN_JUG jugadas), cuánto se aleja su % de ÉXITO de lo normal en un
partido con ese personal (z robusto sobre todos los partidos de la temporada),
encogido por el tamaño de muestra. Éxito y no EPA: con 15 jugadas, una sola
de 80 yardas hace que el EPA parezca una paliza (SF-ARI, semana 3: +0,64 de
EPA con un 38% de éxito). El EPA va al lado. Al lado, el rival: lo que concede
contra ese personal el RESTO de la temporada, para saber si fue mérito del
ataque o un agujero conocido de la defensa.

    python personal_jornada.py             # la última jornada que tenga Sharp
    python personal_jornada.py --week 3
"""
import argparse
import sys

import numpy as np
import pandas as pd

from pbp_loader import cargar_calendario, temporada_actual, season_cli
from copiar_personal_web import sharp_jugadas, GRUPOS

sys.stdout.reconfigure(encoding="utf-8")
MIN_JUG = 12          # jugadas mínimas de un personal en un partido
TOP = 6
QUIEN = {11: "11 (1 RB, 1 TE)", 12: "12 (1 RB, 2 TE)", 13: "13 (1 RB, 3 TE)",
         21: "21 (2 RB, 1 TE)", 22: "22 (2 RB, 2 TE)"}


def z_robusto(s):
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    return (s - med) / (mad if mad > 0 else s.std() or 1)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--week", type=int)
    args, _ = ap.parse_known_args()
    season = season_cli() or temporada_actual()
    pl, generado = sharp_jugadas(season)
    semanas = sorted(pl.week.unique())
    week = args.week or max(semanas)
    if week not in semanas:
        raise SystemExit(f"Sharp aún no tiene la semana {week} (tiene {semanas}, generado "
                         f"{generado}). La publica el martes (hora variable).")

    cal, _ = cargar_calendario(season)
    cal = cal[cal.game_type == "REG"]
    rival = {}
    for _, g in cal.iterrows():
        rival[(int(g.week), g.home_team)] = g.away_team
        rival[(int(g.week), g.away_team)] = g.home_team
    pl["rival"] = [rival.get((w, t)) for w, t in zip(pl.week, pl.team)]
    pl = pl[pl.grupo.isin([int(x) for x in GRUPOS])]

    # Cada partido de cada equipo con cada personal
    cel = (pl.groupby(["week", "team", "rival", "grupo"])
             .agg(n=("epa", "size"), epa=("epa", "mean"), exito=("exito", "mean"))
             .reset_index())
    cel = cel[cel.n >= MIN_JUG].copy()
    cel["z"] = cel.groupby("grupo").exito.transform(z_robusto)
    cel["z_aj"] = cel.z * np.sqrt(cel.n / (cel.n + 15))
    med = cel.groupby("grupo").exito.median()

    jor = cel[cel.week == week].copy()
    print(f"PERSONAL EN LA SEMANA {week} — {season} · Sharp, generado {generado}")
    print(f"Partidos con {MIN_JUG}+ jugadas de un personal: {len(jor)}. Comparado con "
          f"{len(cel)} partido-personal de las semanas {semanas[0]}-{semanas[-1]}.\n")

    def rival_resto(r, grupo):
        """Lo que concede el rival con ese personal en sus OTROS partidos."""
        otros = pl[(pl.rival == r) & (pl.grupo == grupo) & (pl.week != week)]
        if len(otros) < 20:
            return f"{r} no tiene 20 jugadas contra el {grupo} en otras semanas"
        por_def = (pl[(pl.week != week) & (pl.grupo == grupo)].groupby("rival").epa
                   .agg(["mean", "size"]))
        por_def = por_def[por_def["size"] >= 20]["mean"].sort_values()
        rk = int(por_def.index.get_loc(r)) + 1 if r in por_def.index else None
        return (f"{r} concede {otros.epa.mean():+.2f} con el {grupo} en el resto de la "
                f"temporada ({len(otros)} jugadas" + (f", {rk}º de {len(por_def)})" if rk else ")"))

    for titulo, asc in (("ATAQUES QUE DESTROZARON CON UN PERSONAL", False),
                        ("ATAQUES HUNDIDOS CON UN PERSONAL (= la defensa rival lo cerró)", True)):
        print(f"== {titulo}")
        for _, c in jor.sort_values("z_aj", ascending=asc).head(TOP).iterrows():
            print(f"  {c.team} contra {c.rival} con {QUIEN[c.grupo]}: {c.exito * 100:.0f}% de éxito "
                  f"(lo normal en un partido: {med[c.grupo] * 100:.0f}%), {c.epa:+.2f} EPA por jugada, "
                  f"{c.n} jugadas · z {c.z_aj:+.1f}")
            print(f"      {rival_resto(c.rival, c.grupo)}")
        print()
    print("Éxito = EPA > 0; EPA de nflverse calculado por Sharp (no el de SumerSports). z: "
          "distancia a lo normal en % de éxito, "
          "encogida por muestra; a partir de ±2 es raro de verdad. Verificar en la web de Sharp "
          "(filtro de semana) antes de publicar.")


if __name__ == "__main__":
    main()
