"""
copiar_personal_web.py — copia a datos_web/ las tablas de personal ofensivo de
dos webs gratuitas (05-oct-2026). A demanda, no va en el batch: los scripts
de personal lo llaman solos (refrescar) antes de leer datos_web/.

  SumerSports  /teams/offensive/personnel-tendency   ataque: uso y EPA de cada
               /teams/defensive/personnel-tendency   personal; defensa: el EPA
                                                    que concede contra cada uno
               Se actualiza cada noche. Sin % de éxito.
  Sharp        /stats-nfl/nfl-offensive-personnel   ataque: EPA, yardas y % de
               éxito por personal (modo "Efficiency"). Se publica los martes.
               La página calcula la tabla en el navegador con un fichero de
               datos público; aquí se hace la misma cuenta que su JavaScript
               (EPA y éxito medios por equipo y agrupación, sin rodillas ni
               spikes, que ya vienen fuera del fichero).

Salidas (la semana va en el nombre, como el resto de datos_web/):
  datos_web/sumersports_personal_ataque_{season}_wNN.csv
  datos_web/sumersports_personal_defensa_{season}_wNN.csv
  datos_web/sharp_personal_ataque_{season}_wNN.csv

    python copiar_personal_web.py
"""
import html
import json
import os
import re
import sys
import urllib.request
from datetime import date

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(RAIZ, "datos_web")
UA = {"User-Agent": "Mozilla/5.0"}
SUMER = "https://sumersports.com/teams/{lado}/personnel-tendency/"
SHARP = "https://rmsummerlin.github.io/SFAStatsPages/data/personnel_grouping_{season}.json"
SHARP_PAG = "https://www.sharpfootballanalysis.com/stats-nfl/nfl-offensive-personnel/"
GRUPOS = [11, 12, 13, 21, 22]


def bajar(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read().decode("utf-8", errors="ignore")


# Siglas de nflverse (las del PBP y el calendario): los Rams son LA, no LAR
NFLVERSE = {"LAR": "LA"}


def siglas():
    eq = pd.read_parquet(os.path.join(RAIZ, "pbp_cache", "teams.parquet"))
    eq = eq[eq.team_abbr.isin(["LA"]) | ~eq.team_name.duplicated(keep=False)]
    return {n: NFLVERSE.get(t, t) for n, t in zip(eq.team_name, eq.team_abbr)}


def semana_de_jugadas(jugadas_web, lado):
    """Hasta qué semana llega la web: la del PBP cuyo total de jugadas por
    equipo se parece más al de la tabla (cuadra a ±2-3 jugadas)."""
    from pbp_loader import cargar_pbp
    df, _ = cargar_pbp(columns=["week", "posteam", "defteam", "play_type"], avisar=False)
    pr = df[df.play_type.isin(["pass", "run"])]
    col = "posteam" if lado == "offensive" else "defteam"
    mejor = None
    for w in sorted(pr.week.unique()):
        n = pr[pr.week <= w].groupby(col).size()
        err = (n.reindex(jugadas_web.index) - jugadas_web).abs().mean()
        if mejor is None or err < mejor[1]:
            mejor = (int(w), err)
    return mejor


def sumersports(season):
    nom = siglas()
    for lado, sufijo in (("offensive", "ataque"), ("defensive", "defensa")):
        # La web sirve a veces una version vieja de la tabla (05-oct-2026: la
        # misma mañana devolvia la de las 01:04 o la de las 01:55, con otro
        # EPA). Se pide varias veces y se queda la de "Last Updated" mas reciente
        def hora(html_):
            m = re.search(r"Last Updated\s*(?:<[^>]+>\s*)*(\d\d)-(\d\d)-(\d{4}) (\d\d):(\d\d) ([AP]M)",
                          html_)
            if not m:
                return ""
            mm, dd, yy, hh, mi, ap = m.groups()
            return f"{yy}{mm}{dd}{(int(hh) % 12) + (12 if ap == 'PM' else 0):02d}{mi}"
        t = max((bajar(SUMER.format(lado=lado)) for _ in range(6)), key=hora)
        txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", re.sub(
            r"<(script|style)[^>]*>.*?</>", "", t, flags=re.S))))
        act = re.search(r"Last Updated (\d\d-\d\d-\d{4} [\d:]+ [AP]M)", txt)
        # Celda a celda: una agrupacion sin pases o sin carreras deja vacio su
        # EPA por pase o por carrera
        filas = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", t, flags=re.S):
            c = [html.unescape(re.sub(r"<[^>]+>", "", x)).strip()
                 for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)]
            if len(c) >= 13 and c[1] == str(season):
                filas.append([re.sub(r"^\d+\.", "", c[0])] + c[2:13])
        cols = ["equipo", "personal", "jugadas_equipo", "jugadas", "uso_pct", "epa_total",
                "epa_rk", "epa_pase", "epa_carrera", "pase_pct", "uso_liga_pct", "epa_liga"]
        d = pd.DataFrame(filas, columns=cols)
        d.insert(0, "team", d.equipo.map(nom))
        for c in cols[2:]:
            d[c] = pd.to_numeric(d[c].str.rstrip("%").replace("", None), errors="coerce")
        d["epa_jugada"] = (d.epa_total / d.jugadas).round(3)
        if d.team.isna().any():
            raise SystemExit(f"Equipos sin sigla: {d[d.team.isna()].equipo.unique()}")
        tot = d.groupby("team").jugadas_equipo.first()
        w, err = semana_de_jugadas(tot, lado)
        d["fuente"] = SUMER.format(lado=lado)
        d["actualizado"] = act.group(1) if act else ""
        d["copiado"] = str(date.today())
        ruta = os.path.join(DIR, f"sumersports_personal_{sufijo}_{season}_w{w:02d}.csv")
        d.drop(columns="equipo").to_csv(ruta, index=False, encoding="utf-8")
        print(f"SumerSports {sufijo}: {len(d)} filas, actualizado {d.actualizado.iloc[0]}, "
              f"semana {w} (desvío medio {err:.1f} jugadas) -> {ruta}")


def sharp_jugadas(season):
    """Las jugadas del fichero de Sharp (team, week, grupo, epa, yds, exito) y
    la hora a la que se genero. Lo usa tambien personal_jornada.py."""
    d = json.loads(bajar(SHARP.format(season=season)))
    inv = {ch: i for i, ch in enumerate(d["alphabet"])}
    B = len(d["alphabet"])
    col = lambda s: [inv[ch] for ch in s]                          # noqa: E731
    def pair(hi, lo, off):
        return [h * B + l - off for h, l in zip(col(hi), col(lo))]
    c = d["cols"]
    pers = [d["pers"][i] for i in col(c["p"])]
    pl = pd.DataFrame({
        "team": [NFLVERSE.get(d["teams"][i], d["teams"][i]) for i in col(c["t"])],
        "week": col(c["w"]),
        "grupo": [(v // 100) * 10 + (v // 10) % 10 for v in pers],
        "epa": [x / d["epa_scale"] for x in pair(c["eh"], c["el"], d["epa_offset"])],
        "yds": pair(c["yh"], c["yl"], d["yds_offset"]),
        "exito": col(c["sc"]),
    })
    return pl, d["generated_utc"]


def sharp(season):
    pl, generado = sharp_jugadas(season)
    semanas = sorted(set(pl.week))
    g = pl.groupby(["team", "grupo"]).agg(jugadas=("epa", "size"), epa_jugada=("epa", "mean"),
                                          yardas_jugada=("yds", "mean"), exito=("exito", "mean"))
    g["uso_pct"] = g.jugadas / g.groupby(level=0).jugadas.transform("sum") * 100
    g = g.reset_index()
    g = g[g.grupo.isin(GRUPOS)].copy()
    g["exito_pct"] = (g.exito * 100).round(1)
    g["minimo_20"] = g.jugadas >= 20      # la web solo enseña la eficiencia con 20+
    g = g.drop(columns="exito").round({"epa_jugada": 3, "yardas_jugada": 2, "uso_pct": 1})
    g["fuente"] = SHARP_PAG
    g["actualizado"] = generado
    g["copiado"] = str(date.today())
    w = max(semanas)
    ruta = os.path.join(DIR, f"sharp_personal_ataque_{season}_w{w:02d}.csv")
    g.to_csv(ruta, index=False, encoding="utf-8")
    print(f"Sharp: {len(g)} filas, semanas {semanas}, generado {generado} -> {ruta}")


def refrescar(season):
    """Copia lo último antes de usarlo. Sin red (o si una web cambia), avisa y
    se sigue con la última copia que haya en datos_web/."""
    os.makedirs(DIR, exist_ok=True)
    for nombre, f in (("SumerSports", sumersports), ("Sharp", sharp)):
        try:
            f(season)
        except Exception as e:      # noqa: BLE001 — la copia vieja sirve igual
            print(f"  Aviso: no se pudo refrescar {nombre} ({type(e).__name__}: {e}); "
                  "se usa la última copia de datos_web/")


if __name__ == "__main__":
    from pbp_loader import temporada_actual
    refrescar(temporada_actual())
