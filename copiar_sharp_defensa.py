"""Copia a datos_web/ las tablas de Sharp que describen COMO defiende cada
equipo (10-oct-2026; nacio para el duelo SF @ SEA y se usa cada sabado). Son tablas HTML de temporada,
una fila por equipo; la semana no viene en la pagina, se pasa a mano.

  coberturas          % hombre / zona y medio del campo cerrado / abierto
  tendencias_defensa  % de blitz, caja ligera, caja pesada y sub package (5+ DB)
  cobertura_posicion  yardas por target concedidas a WR, TE, RB, exterior y slot
  linea_defensiva     % de presion (y sin blitz), yardas antes del contacto, stuffs
  linea_ofensiva      lo mismo, del lado del ataque
  tendencias_ataque   movimiento, play action, air yards, shotgun, no huddle

Medio abierto (MOFO) = dos safeties profundos tras el snap: cover 2, 4 y 6.
Medio cerrado (MOFC) = uno: cover 1 y 3. Sharp NO separa cover 2 de cover 4.

    python copiar_sharp_defensa.py 4          # 4 = semana hasta la que llegan
"""
import html
import os
import re
import sys
import urllib.request
from datetime import date

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
RAIZ = os.path.dirname(os.path.abspath(__file__))
import copiar_personal_web as cpw

BASE = "https://www.sharpfootballanalysis.com/stats-nfl/"
PAGINAS = {
    "coberturas": ("nfl-coverage-schemes", ["man_pct", "zona_pct", "medio_cerrado_pct",
                                            "medio_abierto_pct"]),
    "tendencias_defensa": ("nfl-defensive-tendencies", ["blitz_pct", "caja_ligera_pct",
                                                        "caja_pesada_pct", "sub_package_pct"]),
    "cobertura_posicion": ("nfl-coverage-stats-by-position", ["ypt_wr", "ypt_te", "ypt_rb",
                                                              "ypt_exterior", "ypt_slot"]),
    "linea_defensiva": ("nfl-defensive-line-stats", ["presion_pct", "presion_sin_blitz_pct",
                                                     "yardas_antes_contacto_rb", "stuff_pct"]),
    "linea_ofensiva": ("nfl-offensive-line-stats", None),
    "tendencias_ataque": ("nfl-offensive-tendencies-stats", ["movimiento_pct", "play_action_pct",
                                                             "air_yards_intento", "shotgun_pct",
                                                             "no_huddle_pct"]),
}


def limpia(x):
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", x))).strip()


def tabla(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=cpw.UA), timeout=60) as r:
        s = r.read().decode("utf-8", errors="ignore")
    mod = re.search(r'"dateModified":"([^"]+)"', s)
    for tb in re.findall(r"<table.*?</table>", s, flags=re.S):
        filas = [[limpia(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", f, flags=re.S)]
                 for f in re.findall(r"<tr.*?</tr>", tb, flags=re.S)]
        filas = [f for f in filas if f]
        if len(filas) >= 30:
            return filas[0], [f for f in filas[1:] if len(f) == len(filas[0])], mod.group(1)
    raise SystemExit(f"Sin tabla de equipos en {url}")


def main():
    semana = int(sys.argv[1])
    sig = cpw.siglas()
    for nombre, (slug, cols) in PAGINAS.items():
        url = BASE + slug + "/"
        cab, filas, mod = tabla(url)
        d = pd.DataFrame(filas, columns=["equipo"] + (cols or [limpia(c) for c in cab[1:]]))
        d.insert(0, "team", d.equipo.map(sig))
        if d.team.isna().any() or len(d) != 32:
            raise SystemExit(f"{nombre}: {len(d)} filas, sin sigla {d[d.team.isna()].equipo.tolist()}")
        for c in d.columns[2:]:
            d[c] = pd.to_numeric(d[c].str.replace("%", ""), errors="coerce")
        d["fuente"], d["actualizado"], d["copiado"] = url, mod, str(date.today())
        dest = os.path.join(RAIZ, "datos_web", f"sharp_{nombre}_2026_w{semana:02d}.csv")
        d.drop(columns="equipo").to_csv(dest, index=False)
        print(f"{nombre}: 32 equipos, {list(cab[1:])}, modificado {mod} -> {os.path.basename(dest)}")


if __name__ == "__main__":
    main()
