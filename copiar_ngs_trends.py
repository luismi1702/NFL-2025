"""
copiar_ngs_trends.py — copia a datos_web/ TODO lo que NFL Next Gen Trends
(nfl.com/stats/ngs/trends, web gratis de Next Gen Stats lanzada el 12-nov-2026)
trae en abierto en su HTML. A demanda, como copiar_personal_web.py.

Lee la portada y las 32 páginas de equipo (sin navegador: los datos van en el
HTML) y deja, con NN = la última semana con datos:

  ngs_trends_equipos_{s}_wNN.csv     28 tendencias por equipo: jugadas, elegibles,
                                     % de uso, puesto, percentil, EPA, % de éxito,
                                     % de big plays, media de la liga y, en motion,
                                     play action, pase pedido y quick passing,
                                     lo mismo SIN la tendencia
  ngs_trends_partidos_{s}_wNN.csv    partido a partido de cada equipo: EPA total,
                                     de pase y de carrera (ataque y defensa),
                                     % de éxito y de big plays propios y del rival
  ngs_trends_temporada_{s}_wNN.csv   una fila por equipo: EPA/jugada y puestos,
                                     puntos, yardas, jugadas, playoffs (prob.,
                                     división, nº 1, Super Bowl, si gana / si
                                     pierde el próximo, victorias proyectadas),
                                     equipos parecidos en ataque y defensa
  ngs_trends_entrenadores_{s}.csv    HC, OC y DC de cada equipo y quién llama
                                     las jugadas en ataque y en defensa
  ngs_trends_liga_{s}_wNN.csv        media de la liga de cada tendencia,
                                     temporada a temporada desde 2018
  ngs_trends_insights_{s}.csv        los "insights" de NGS (datos de partido
                                     redactados por su equipo, con jugador,
                                     rival y semana): se ACUMULAN, la web solo
                                     enseña los últimos de cada equipo. Son
                                     IDEAS, como el boletín de SumerSports: su
                                     ángulo, nuestros números, nunca copiados

OJO:
- Su EPA es el modelo de NGS, NO el de nflverse (BUF w09: +0,107 frente a
  +0,140). No se mezcla con nuestro PBP en la misma frase ni gráfico.
- El denominador cambia según la métrica (`elegibles`): personal y motion,
  todas las jugadas; coberturas y blitz, jugadas de pase; bloqueos, carreras;
  play action, early downs; tempo, jugadas sin huddle posibles.
- `partidos` = los jugados por cada equipo (los byes cambian la muestra).
- Lo que la web carga por su API (temporadas anteriores por equipo, evolución
  de un equipo año a año, gráficos de dispersión) NO se copia: pide un token.
  Para tener serie propia, lanzar el copiador cada semana (cada wNN se guarda).

    python copiar_ngs_trends.py
"""
import json
import os
import re
import sys
import urllib.request
from datetime import datetime

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(RAIZ, "datos_web")
UA = {"User-Agent": "Mozilla/5.0"}
BASE = "https://www.nfl.com/stats/ngs/trends"
FUENTE = "NFL Next Gen Trends (nfl.com/stats/ngs/trends)"
NFLVERSE = {"LAR": "LA", "AZ": "ARI"}  # siglas del PBP


def sigla(a):
    return NFLVERSE.get(a, a)


def bajar(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read().decode("utf-8", errors="ignore")


def flight(html_):
    """Junta los trozos self.__next_f.push([1,"..."]) ya des-escapados."""
    trozos = re.findall(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)', html_)
    return "".join(json.loads(t) for t in trozos)


def consultas(texto):
    """{nombre de la consulta: data} de cada consulta deshidratada de la página."""
    dec, out = json.JSONDecoder(), {}
    for m in re.finditer(r'\{"dehydratedAt":', texto):
        try:
            o, _ = dec.raw_decode(texto, m.start())
        except json.JSONDecodeError:
            continue
        clave = o.get("queryKey") or [""]
        if str(clave[0]).startswith("useFetchNgs"):
            out.setdefault(clave[0], []).append(o.get("state", {}).get("data"))
    return out


def tendencias(obj):
    eq, out = sigla(obj["teamAbbr"]), []
    for lado, d in obj["tendencies"].items():
        ident = d.get("identity", [])
        for met, v in d.items():
            if met == "identity":
                continue
            sin = v.get("without", {})
            out.append({
                "equipo": eq, "lado": "ataque" if lado == "offense" else "defensa",
                "metrica": met, "identidad": met in ident,
                "jugadas": v.get("playCount"), "elegibles": v.get("eligiblePlayCount"),
                "uso": v.get("frequency"), "seg_entre_jugadas": v.get("avgSecondsBetweenPlays"),
                "puesto": v.get("rank"), "percentil": v.get("percentile"),
                "epa": v.get("avgOffenseEPA", v.get("avgDefenseEPA")),
                "exito": v.get("successRate", v.get("successRateAllowed")),
                "big_play": v.get("explosiveRate", v.get("explosiveRateAllowed")),
                "uso_liga": v.get("leagueAverageFrequency"),
                "sin_jugadas": sin.get("playCount"), "sin_epa": sin.get("avgOffenseEPA"),
                "sin_exito": sin.get("successRate"), "sin_big_play": sin.get("explosiveRate"),
                "partidos": obj.get("stats", {}).get("gamesPlayed"),
                "semana": obj["throughWeek"], "temporada": obj["season"],
            })
    return out


def temporada(obj):
    st, po = obj.get("stats", {}), dict(obj.get("playoffs", {}))
    gi = po.pop("gameInfo", {}) or {}
    fila = {"equipo": sigla(obj["teamAbbr"]), "semana": obj["throughWeek"],
            "temporada": obj["season"], **st, **po,
            "proximo_rival": sigla(gi.get("opponentTeamAbbr", "")),
            "proximo_fecha": gi.get("gameDate"), "proximo_campo": gi.get("gameLocation")}
    for lado, lista in (obj.get("similarTeamsCurrent") or {}).items():
        fila[f"parecidos_{'ataque' if lado == 'offense' else 'defensa'}"] = "; ".join(
            f"{sigla(x['teamAbbr'])} {x['similarityScore']}" for x in lista)
    return fila


def partidos(obj):
    out = []
    for w in obj.get("weeks", []):
        gi = w.get("gameInfo") or {}
        if gi.get("byeWeek") or gi.get("gameState") != "POST":
            continue
        out.append({"equipo": sigla(obj["teamAbbr"]), "temporada": obj["season"],
                    "semana": w["week"], "rival": sigla(gi.get("opponentTeamAbbr", "")),
                    "campo": gi.get("gameLocation"), "resultado": gi.get("result"),
                    "puntos": gi.get("teamTotalPoints"), "puntos_rival": gi.get("opponentTotalPoints"),
                    "fecha": gi.get("gameDate"), **(w.get("stats") or {})})
    return out


def entrenadores(datos):
    out = []
    for d in datos:
        for c in (d or {}).get("coaches", []):
            out.append({"equipo": sigla(c["teamAbbr"]), "nombre": c["fullName"], "cargo": c.get("role"),
                        "head_coach": c.get("headCoach"), "llama_ataque": c.get("offensePlayCaller"),
                        "llama_defensa": c.get("defensePlayCaller"),
                        "oc": c.get("offensiveCoordinator"), "dc": c.get("defensiveCoordinator"),
                        "desde": (c.get("publishDate") or "")[:10]})
    return out


def insights(datos):
    out = []
    for d in datos:
        for b in (d or {}).get("blurbs", []):
            out.append({"id": b["id"], "temporada": b.get("season"), "semana": b.get("week"),
                        "fecha": b.get("date"), "equipo": sigla(b.get("teamAbbr", "")),
                        "lado": b.get("teamType", "jugador" if b.get("playerName") else ""),
                        "rival": sigla(b.get("secondTeamAbbr", "") or ""),
                        "jugador": b.get("playerName", ""), "posicion": b.get("position", ""),
                        "gsis_id": b.get("gsisId", ""), "titulo": b.get("title", ""),
                        "detalle": b.get("subNote1", ""), "grafico": b.get("imageUrl", "")})
    return out


def guardar(df, nombre, que):
    df = df.copy()
    df["fuente"] = FUENTE
    df["copiado"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    ruta = os.path.join(DIR, nombre)
    df.to_csv(ruta, index=False, encoding="utf-8")
    print(f"  {que}: {len(df)} filas -> datos_web/{nombre}")
    return ruta


def main():
    os.makedirs(DIR, exist_ok=True)
    portada_html = bajar(BASE)
    portada = consultas(flight(portada_html))
    slugs = sorted(set(re.findall(r'href="/stats/ngs/trends/teams/([a-z0-9-]+)"', portada_html)))
    if len(slugs) != 32:
        print(f"Aviso: {len(slugs)} equipos en la portada (se esperaban 32)")

    filas_t, filas_s, filas_p, coach, ins = [], [], [], [], []
    semanas = set()
    for s in slugs:
        q = consultas(flight(bajar(f"{BASE}/teams/{s}")))
        res = [d for d in q.get("useFetchNgsTrendsTeamSummarySeason", []) if d and "tendencies" in d]
        if not res:
            raise ValueError(f"{s}: no aparece el resumen de temporada (¿ha cambiado la página?)")
        obj = res[0]
        semanas.add((obj["season"], obj["throughWeek"]))
        filas_t += tendencias(obj)
        filas_s.append(temporada(obj))
        for d in q.get("useFetchNgsTrendsTeamSummaryWeek", []):
            filas_p += partidos(d)
        coach += entrenadores(q.get("useFetchNgsTrendsCoaches", []))
        ins += insights(q.get("useFetchNgsTrendsInsights", []))
    ins += insights(portada.get("useFetchNgsTrendsLeagueInsights", []))

    season, w = max(semanas)
    if len(semanas) != 1:
        atras = sorted(f["equipo"] for f in filas_s if f["semana"] != w)
        print(f"Nota: datos hasta la semana {w}; {', '.join(atras)} se quedan en la anterior "
              "porque descansaron en la última jornada (los byes previos ya están dentro)")
    print(f"NGS Trends {season}, semana {w}:")
    guardar(pd.DataFrame(filas_t), f"ngs_trends_equipos_{season}_w{w:02d}.csv", "tendencias")
    guardar(pd.DataFrame(filas_p).sort_values(["semana", "equipo"]),
            f"ngs_trends_partidos_{season}_w{w:02d}.csv", "partidos")
    guardar(pd.DataFrame(filas_s), f"ngs_trends_temporada_{season}_w{w:02d}.csv", "temporada y playoffs")
    guardar(pd.DataFrame(coach), f"ngs_trends_entrenadores_{season}.csv", "entrenadores")

    liga = [d for d in portada.get("useFetchNgsTrendsLeagueAveragesHistory", []) if d]
    if liga:
        hist = pd.DataFrame(liga[0]["tendencies"]).rename(
            columns={"metric": "metrica", "season": "temporada", "frequency": "uso_liga"})
        guardar(hist.sort_values(["metrica", "temporada"]), f"ngs_trends_liga_{season}_w{w:02d}.csv",
                "medias de la liga desde 2018")

    # Insights: se acumulan (la web solo enseña los últimos)
    ruta_ins = os.path.join(DIR, f"ngs_trends_insights_{season}.csv")
    nuevos = pd.DataFrame(ins).drop_duplicates("id") if ins else pd.DataFrame(columns=["id"])
    if os.path.exists(ruta_ins):
        viejos = pd.read_csv(ruta_ins, dtype=str)
        n_antes = len(viejos)
        nuevos = pd.concat([viejos.drop(columns=["fuente", "copiado"], errors="ignore"),
                            nuevos.astype(str)]).drop_duplicates("id", keep="first")
        print(f"  insights nuevos: {len(nuevos) - n_antes}")
    guardar(nuevos.sort_values(["semana", "equipo"]), f"ngs_trends_insights_{season}.csv", "insights (acumulados)")


if __name__ == "__main__":
    main()
