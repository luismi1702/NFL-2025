"""
cruce_externo.py — compara las cifras básicas que publicamos con una fuente
EXTERNA: el boxscore de ESPN (13-nov-2026).

Por qué: la red zone del resumen de partido y de las previas salió mal desde
agosto (contaba el extra point, que se patea desde la 15, como visita a la red
zone: SF 25/28 en vez de 20/23). El verificador comprobaba cada texto contra
nuestro propio script, así que siempre le cuadraba. Esto compara lo que PINTAN
los scripts con lo que dice otra fuente.

  Partido (resumen_partido.py, de su resumen_numeros_*.json): red zone
    (TD/visitas), 3er down, pérdidas, INT, sacks, completos/intentos — EXACTOS;
    yardas totales con margen (ESPN resta las de los sacks y cuenta
    arrodilladas).
  Temporada (Previas.py, de previas_numeros_*.txt): red zone de ataque y de
    defensa (TD/visitas) sumando el boxscore de ESPN de cada partido de
    temporada regular anterior a la jornada — EXACTO.

Salida: textos/cruce_externo_{season}_wNN.txt con una línea OK o ❌ por cifra.
Código de salida 1 si algo no cuadra (el verificador lo lee igual).
El boxscore de ESPN se guarda en pbp_cache/espn_box/ (un partido acabado no
cambia): una sola descarga por partido.

    python cruce_externo.py --season 2026 --week 10
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.request

import pandas as pd

from pbp_loader import cargar_calendario, salida

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(RAIZ, "pbp_cache", "espn_box")
URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={}"
# ESPN usa otras siglas
ESPN_A_NFLVERSE = {"WSH": "WAS", "LAR": "LA", "JAX": "JAX"}
MARGEN_YARDAS = 25


def calendario(season):
    c = cargar_calendario(season)
    return c[0] if isinstance(c, tuple) else c


def boxscore(espn_id):
    """{sigla nflverse: {stat: displayValue}} del partido, con caché."""
    os.makedirs(CACHE, exist_ok=True)
    ruta = os.path.join(CACHE, f"{espn_id}.json")
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    req = urllib.request.Request(URL.format(espn_id), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    estado = d.get("header", {}).get("competitions", [{}])[0].get("status", {}).get("type", {})
    out = {}
    for t in d.get("boxscore", {}).get("teams", []):
        sig = t["team"]["abbreviation"]
        out[ESPN_A_NFLVERSE.get(sig, sig)] = {s["name"]: s["displayValue"] for s in t["statistics"]}
    if estado.get("completed") and out:
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(out, f)
    time.sleep(0.3)
    return out


def par(v):
    """'7-14' o '23/33' -> (7, 14)."""
    a, b = re.split(r"[-/]", str(v))[:2]
    return int(a), int(b)


def cruce_partidos(season, week, lineas):
    """Lo que pinta resumen_partido.py frente al boxscore de ESPN."""
    cal = calendario(season).set_index("game_id")
    rutas = sorted(glob.glob(os.path.join(RAIZ, "salidas", str(season), f"w{week:02d}",
                                          "textos", "resumen_numeros_*.json")))
    if not rutas:
        lineas.append(f"(sin resumen_numeros_*.json en la semana {week}: relanza resumen_partido.py)")
        return 0
    malas = 0
    for ruta in rutas:
        with open(ruta, encoding="utf-8") as f:
            r = json.load(f)
        gid = r["game_id"]
        espn = cal.loc[gid, "espn"] if gid in cal.index else None
        if pd.isna(espn):
            lineas.append(f"❌ {gid}: sin id de ESPN en el calendario")
            malas += 1
            continue
        box = boxscore(int(espn))
        lineas.append(f"\n{gid} (ESPN {int(espn)}) — resumen_partido.py")
        for t, s in r["equipos"].items():
            e = box.get(t)
            if not e:
                lineas.append(f"  ❌ {t}: no está en el boxscore de ESPN")
                malas += 1
                continue
            sk, _ = par(e["sacksYardsLost"])
            comp, att = par(e["completionAttempts"])
            pruebas = [
                ("red zone (TD/visitas)", f"{s['rz_tds']}/{s['rz_trips']}", "{}/{}".format(*par(e["redZoneAttempts"]))),
                ("3er down", f"{s['tdc']}/{s['tdc'] + s['tdf']}", "{}/{}".format(*par(e["thirdDownEff"]))),
                ("pérdidas", s["turnovers"], int(e["turnovers"])),
                ("INT lanzadas", s["ints"], int(e["interceptions"])),
                ("sacks recibidos", s["sacks"], sk),
                ("completos/intentos", f"{s['comp']}/{s['att']}", f"{comp}/{att}"),
            ]
            for nombre, nos, ellos in pruebas:
                ok = str(nos) == str(ellos)
                malas += not ok
                lineas.append(f"  {'OK' if ok else '❌'} {t} {nombre}: nuestro {nos} · ESPN {ellos}")
            dy = s["total_yds"] - int(e["totalYards"])
            ok = abs(dy) <= MARGEN_YARDAS
            malas += not ok
            lineas.append(f"  {'OK' if ok else '❌'} {t} yardas totales: nuestro {s['total_yds']} · "
                          f"ESPN {e['totalYards']} (margen ±{MARGEN_YARDAS}: ESPN resta sacks y suma arrodilladas)")
    return malas


def cruce_temporada(season, week, lineas):
    """La red zone de temporada de previas_numeros_*.txt frente a la suma de
    los boxscores de ESPN de las semanas anteriores (temporada regular)."""
    rutas = glob.glob(os.path.join(RAIZ, "salidas", str(season), f"w{week:02d}",
                                   "textos", f"previas_numeros_{season}*.txt"))
    if not rutas:
        lineas.append(f"\n(sin previas_numeros_{season}_w{week:02d}.txt: nada de temporada que cruzar)")
        return 0
    with open(rutas[0], encoding="utf-8") as f:
        txt = f.read()
    nos = {}
    equipo = None
    for l in txt.splitlines():
        m = re.match(r"^\s{2}([A-Z]{2,3}) \(", l)
        if m:
            equipo = m.group(1)
        m = re.search(r"red zone ataque (\d+) TD en (\d+) visitas · red zone rival (\d+) TD en (\d+) visitas", l)
        if m and equipo:
            a, b, c, d = map(int, m.groups())
            nos[equipo] = ((a, b), (c, d))
    cal = calendario(season)
    cal = cal[(cal.game_type == "REG") & (cal.week < week) & cal.result.notna()]
    of, de = {}, {}
    for g in cal.itertuples():
        box = boxscore(int(g.espn))
        for t, rival in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
            if t not in box:
                continue
            td, vis = par(box[t]["redZoneAttempts"])
            of[t] = tuple(x + y for x, y in zip(of.get(t, (0, 0)), (td, vis)))
            de[rival] = tuple(x + y for x, y in zip(de.get(rival, (0, 0)), (td, vis)))
    lineas.append(f"\nTemporada hasta la sem. {week - 1} — Previas.py (previas_numeros) frente a ESPN")
    malas = 0
    for t in sorted(nos):
        (a, b), (c, d) = nos[t]
        for lado, n, e in (("ataque", (a, b), of.get(t)), ("defensa", (c, d), de.get(t))):
            ok = e is not None and tuple(n) == tuple(e)
            malas += not ok
            lineas.append(f"  {'OK' if ok else '❌'} {t} red zone {lado}: nuestro {n[0]}/{n[1]} · "
                          f"ESPN {e[0]}/{e[1]}" if e else f"  ❌ {t} red zone {lado}: sin ESPN")
    return malas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    a = ap.parse_args()
    lineas = [f"CRUCE EXTERNO — {a.season} sem. {a.week} — boxscore de ESPN "
              f"(site.api.espn.com), {time.strftime('%Y-%m-%d %H:%M')}",
              "Una cifra con ❌ NO se publica: o nuestro cálculo está mal o mide otra cosa."]
    malas = cruce_partidos(a.season, a.week, lineas) + cruce_temporada(a.season, a.week, lineas)
    lineas.insert(2, f"RESULTADO: {'TODO CUADRA' if not malas else f'{malas} cifras NO cuadran'}")
    ruta = salida(f"cruce_externo_{a.season}.txt", a.season, a.week)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
    print("\n".join(lineas[:3]))
    print(f"-> {ruta}")
    sys.exit(1 if malas else 0)


if __name__ == "__main__":
    main()
