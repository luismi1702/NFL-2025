"""
ataque_por_personal_ofensivo.py
Los 32 ataques según el personal con el que juegan (11, 12, 13, 21, 22):
cuánto lo usan y qué sacan con él. Estilo docs/estilo-visual.md.

De dónde salen los datos (05-oct-2026):
  - Temporada en curso: tablas gratuitas copiadas a datos_web/ con
    `copiar_personal_web.py`. EPA por jugada de SumerSports (su propio modelo
    de EPA, que no se mezcla con el de nflverse) o, con --exito, % de jugadas
    con éxito de Sharp (va una semana por detrás).
  - Temporadas cerradas: participación de nflverse (llega en febrero).

    python ataque_por_personal_ofensivo.py
    python ataque_por_personal_ofensivo.py --exito
    python ataque_por_personal_ofensivo.py --season 2025
"""
import argparse
import re

import numpy as np
import pandas as pd

from pbp_loader import (cargar_pbp, cargar_participation, cargar_web, DatosNoDisponibles,
                        salida, season_cli, temporada_actual)
from estilo import *   # noqa: F401,F403  (estilo comun, docs/estilo-visual.md)

GRUPOS = ["11", "12", "13", "21", "22"]
QUIEN = {"11": "1 RB · 1 TE", "12": "1 RB · 2 TE", "13": "1 RB · 3 TE",
         "21": "2 RB · 1 TE", "22": "2 RB · 2 TE"}
FRASE = {"11": "con tres receptores (11 personal)", "12": "con dos TE (12 personal)",
         "13": "con tres TE (13 personal)", "21": "con dos RB (21 personal)",
         "22": "con dos RB y dos TE (22 personal)"}
MIN_JUG = 20          # jugadas mínimas por celda (el mismo corte que Sharp)
COLOR_MIN = 25        # columnas con menos equipos válidos, sin color de juicio
TIT_MIN = 15          # equipos válidos mínimos para que una columna dé titular
TIT_JUG = 40          # jugadas mínimas de la celda del titular


def parse_personnel(s):
    """'1 RB, 1 TE, 3 WR' -> '11'. El FB cuenta como RB."""
    if pd.isna(s):
        return None
    rb = sum(int(m.group(1)) for pos in ("RB", "FB")
             for m in [re.search(r"(\d+)\s*" + pos + r"(?:[,\s]|$)", str(s), re.I)] if m)
    te = re.search(r"(\d+)\s*TE", str(s), re.I)
    return f"{rb}{te.group(1)}" if rb > 0 and te else None


def desde_participacion(season):
    """Tabla larga team, grupo, jugadas, uso_pct, valor con el PBP de nflverse."""
    pbp, season = cargar_pbp(season)
    part, _ = cargar_participation(season)
    part = part[["nflverse_game_id", "play_id", "offense_personnel"]].rename(
        columns={"nflverse_game_id": "game_id"})
    p = pbp.merge(part, on=["game_id", "play_id"])
    p = p[p.play_type.isin(["pass", "run"]) & p.epa.notna() & p.posteam.notna()
          & (p.qb_kneel != 1) & (p.qb_spike != 1)]
    p["grupo"] = p.offense_personnel.apply(parse_personnel)
    p = p[p.grupo.notna()]
    g = p.groupby(["posteam", "grupo"]).agg(jugadas=("epa", "size"), epa=("epa", "mean"),
                                           exito=("success", "mean")).reset_index()
    g["uso_pct"] = g.jugadas / g.groupby("posteam").jugadas.transform("sum") * 100
    g["exito"] *= 100
    return g.rename(columns={"posteam": "team"}), season, None


def desde_web(season, exito):
    from copiar_personal_web import refrescar
    refrescar(season)       # siempre lo último: estos scripts se lanzan a demanda
    if exito:
        d, sem = cargar_web("sharp_personal_ataque", season)
        d = d.rename(columns={"exito_pct": "exito"})
        d["grupo"] = d.grupo.astype(str)
        fuente = "Sharp Football Analysis (personal ofensivo)"
    else:
        d, sem = cargar_web("sumersports_personal_ataque", season)
        d = d.rename(columns={"personal": "grupo", "epa_jugada": "epa"})
        fuente = "SumerSports (Personnel Tendency, EPA propio)"
    return d, season, (sem, fuente, d.actualizado.iloc[0])


def titular_txt(tab, n, rk, metrica):
    """La celda (no 11) que más se separa de la mediana de su columna, solo
    entre las de TIT_JUG+ jugadas (con 23 jugadas el valor bailaba de un dia
    para otro). El puesto se cuenta sobre TODAS las celdas que enseña la
    tabla, para que el titular nunca contradiga al PNG."""
    mejor = None
    for gr in GRUPOS[1:]:
        col = tab[gr].dropna()
        if len(col) < TIT_MIN:
            continue
        for tm in col.index:
            if n.loc[tm, gr] < TIT_JUG:
                continue
            dif = (col[tm] - col.median()) / (100 if metrica == "exito" else 1)
            if mejor is None or dif > mejor[0]:
                mejor = (dif, gr, tm)
    if not mejor:
        return None
    _, gr, tm = mejor
    T, val, r = nombre_equipo(tm), tab.loc[tm, gr], int(rk[gr][tm])
    dato = (f"{pct(val / 100)} de jugadas con éxito" if metrica == "exito"
            else f"{num(val)} de EPA por jugada")
    jug = int(n.loc[tm, gr])
    if r == 1:
        s = f"Ningún ataque juega mejor {FRASE[gr]} que {T}: {dato} en {jug} jugadas."
    else:
        s = (f"{T} {v(T, 'saca', 'sacan')} {dato} {FRASE[gr]}, "
             f"{r}º de {int(tab[gr].notna().sum())}, en {jug} jugadas.")
    return s[0].upper() + s[1:]


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--exito", action="store_true", help="%% de éxito (Sharp) en vez de EPA")
    args, _ = ap.parse_known_args()
    metrica = "exito" if args.exito else "epa"
    season = season_cli() or temporada_actual()

    try:
        d, season, web = desde_participacion(season)
    except DatosNoDisponibles:
        try:
            d, season, web = desde_web(season, args.exito)
        except DatosNoDisponibles as e:
            raise SystemExit(f"\n  Sin datos de personal: {e}\n  Lanza antes "
                             "`python copiar_personal_web.py`.\n")

    d = d[d.grupo.isin(GRUPOS)]
    tab = d.pivot(index="team", columns="grupo", values=metrica).reindex(columns=GRUPOS)
    n = d.pivot(index="team", columns="grupo", values="jugadas").reindex(columns=GRUPOS)
    tab = tab.where(n >= MIN_JUG)
    for gr in GRUPOS:
        tab[f"uso_{gr}"] = d[d.grupo == gr].set_index("team").uso_pct
    rk = {gr: tab[gr].rank(ascending=False, method="min") for gr in GRUPOS}
    tab = tab.sort_values("11", ascending=False)

    fig = lienzo()
    antetitulo(fig, "Ataques · personal ofensivo")
    tit = titular_txt(tab, n, rk, metrica)
    y_top = (titular(fig, tit, y=0.925, size=19) - 0.025) if tit else 0.90
    x0, ancho_c = L + 0.17, (R - L - 0.17) / len(GRUPOS)
    yh = y_top - 0.03
    for j, gr in enumerate(GRUPOS):
        xc = x0 + (j + 0.5) * ancho_c
        fig.text(xc, yh + 0.008, gr, family=DISP, weight=700, size=12, color=FG, ha="center",
                 va="center")
        fig.text(xc, yh - 0.016, QUIEN[gr], family=TXT, size=6, color=GRIS1, ha="center",
                 va="center")
    linea_h(fig, yh - 0.03)
    alto = (yh - 0.045 - 0.12) / len(tab)
    for i, tm in enumerate(tab.index):
        y = yh - 0.045 - (i + 0.5) * alto
        poner_logo(fig, tm, L + 0.025, y, zoom=0.0095)
        fig.text(L + 0.05, y, tm, family=TXT, weight="bold", size=7.2, color=FG, va="center")
        for j, gr in enumerate(GRUPOS):
            xc = x0 + (j + 0.5) * ancho_c
            v, uso = tab.loc[tm, gr], tab.loc[tm, f"uso_{gr}"]
            if pd.isna(v):
                fig.text(xc, y, "–", family=TXT, size=7, color=GRIS2, ha="center", va="center")
                continue
            r = int(rk[gr][tm])
            txt = pct(v / 100) if metrica == "exito" else num(v)
            n_col = int(tab[gr].notna().sum())
            fig.text(xc + 0.012, y, txt, family=TXT, weight="bold", size=7.2,
                     color=color_rango(r, n_col) if n_col >= COLOR_MIN else FG,
                     ha="right", va="center")
            fig.text(xc + 0.017, y, f"{uso:.0f}%", family=TXT, size=5.8, color=GRIS2,
                     ha="left", va="center")
    y_fin = yh - 0.045 - len(tab) * alto
    tarjeta(fig, y_fin - 0.01, y_top)
    que = "% de éxito (EPA > 0)" if metrica == "exito" else "EPA por jugada"
    if web:
        sem, fuente, act = web
        sello_txt = f"NFL {season} · hasta sem. {sem} · {fuente}"
    else:
        sello_txt = f"NFL {season} · nflverse PBP + participación"
    pie(fig, f"{que} · gris: % de uso · color: 5 mejores y 5 peores ({COLOR_MIN}+ equipos) · "
             f"–: menos de {MIN_JUG} jugadas", sello_txt)
    nombre = f"ataque_por_personal_ofensivo{'_exito' if args.exito else ''}_{season}.png"
    ruta = salida(nombre, season)
    guardar(fig, ruta)
    if tit:
        print("Titular:", tit)
    print("Guardado:", ruta)


if __name__ == "__main__":
    main()
