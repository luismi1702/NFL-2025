"""
defensa_vs_personal_ofensivo.py
Las 32 defensas según el personal ofensivo del rival (11, 12, 13, 21, 22):
cuánto se lo encuentran y qué EPA conceden contra él. Estilo
docs/estilo-visual.md.

De dónde salen los datos (05-oct-2026):
  - Temporada en curso: SumerSports, Personnel Tendency de defensa, copiado a
    datos_web/ con `copiar_personal_web.py`. Su EPA es de su propio modelo:
    no se mezcla con el de nflverse. OJO: su "EPA Rank" de defensa va al revés
    (1 = la que más concede), así que aquí el puesto se calcula de nuevo.
  - Temporadas cerradas: participación de nflverse (llega en febrero).

    python defensa_vs_personal_ofensivo.py
    python defensa_vs_personal_ofensivo.py --season 2025
"""
import re

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
MIN_JUG = 20          # jugadas mínimas por celda
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
    pbp, season = cargar_pbp(season)
    part, _ = cargar_participation(season)
    part = part[["nflverse_game_id", "play_id", "offense_personnel"]].rename(
        columns={"nflverse_game_id": "game_id"})
    p = pbp.merge(part, on=["game_id", "play_id"])
    p = p[p.play_type.isin(["pass", "run"]) & p.epa.notna() & p.defteam.notna()
          & (p.qb_kneel != 1) & (p.qb_spike != 1)]
    p["grupo"] = p.offense_personnel.apply(parse_personnel)
    p = p[p.grupo.notna()]
    g = p.groupby(["defteam", "grupo"]).agg(jugadas=("epa", "size"),
                                           epa=("epa", "mean")).reset_index()
    g["uso_pct"] = g.jugadas / g.groupby("defteam").jugadas.transform("sum") * 100
    return g.rename(columns={"defteam": "team"}), season, None


def desde_web(season):
    d, sem = cargar_web("sumersports_personal_defensa", season)
    d = d.rename(columns={"personal": "grupo", "epa_jugada": "epa"})
    return d, season, (sem, "SumerSports (Personnel Tendency, EPA propio)")


def titular_txt(tab, n, rk):
    """Como en el de ataque: la celda (no 11) de TIT_JUG+ jugadas que más
    mejora la mediana de su columna; el puesto, sobre todas las de la tabla."""
    mejor = None
    for gr in GRUPOS[1:]:
        col = tab[gr].dropna()
        if len(col) < TIT_MIN:
            continue
        for tm in col.index:
            if n.loc[tm, gr] < TIT_JUG:
                continue
            dif = col.median() - col[tm]
            if mejor is None or dif > mejor[0]:
                mejor = (dif, gr, tm)
    if not mejor:
        return None
    _, gr, tm = mejor
    val, r, jug = tab.loc[tm, gr], int(rk[gr][tm]), int(n.loc[tm, gr])
    if r == 1:
        s = (f"Ninguna defensa frena mejor a los ataques {FRASE[gr]} que la de "
             f"{nombre_equipo(tm)}: los deja en {num(val)} de EPA por jugada en {jug} jugadas.")
    else:
        s = (f"La defensa de {nombre_equipo(tm)} deja a los ataques {FRASE[gr]} en {num(val)} "
             f"de EPA por jugada, {r}ª de {int(tab[gr].notna().sum())}, en {jug} jugadas.")
    return s[0].upper() + s[1:]


def main():
    season = season_cli() or temporada_actual()
    try:
        d, season, web = desde_participacion(season)
    except DatosNoDisponibles:
        try:
            d, season, web = desde_web(season)
        except DatosNoDisponibles as e:
            raise SystemExit(f"\n  Sin datos de personal: {e}\n  Lanza antes "
                             "`python copiar_personal_web.py`.\n")

    d = d[d.grupo.isin(GRUPOS)]
    tab = d.pivot(index="team", columns="grupo", values="epa").reindex(columns=GRUPOS)
    n = d.pivot(index="team", columns="grupo", values="jugadas").reindex(columns=GRUPOS)
    tab = tab.where(n >= MIN_JUG)
    for gr in GRUPOS:
        tab[f"uso_{gr}"] = d[d.grupo == gr].set_index("team").uso_pct
    rk = {gr: tab[gr].rank(ascending=True, method="min") for gr in GRUPOS}   # 1 = concede menos
    tab = tab.sort_values("11", ascending=True)

    fig = lienzo()
    antetitulo(fig, "Defensas · contra cada personal ofensivo")
    tit = titular_txt(tab, n, rk)
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
            n_col = int(tab[gr].notna().sum())
            fig.text(xc + 0.012, y, num(v), family=TXT, weight="bold", size=7.2,
                     color=color_rango(int(rk[gr][tm]), n_col) if n_col >= COLOR_MIN else FG,
                     ha="right", va="center")
            fig.text(xc + 0.017, y, f"{uso:.0f}%", family=TXT, size=5.8, color=GRIS2,
                     ha="left", va="center")
    y_fin = yh - 0.045 - len(tab) * alto
    tarjeta(fig, y_fin - 0.01, y_top)
    if web:
        sem, fuente = web
        sello_txt = f"NFL {season} · hasta sem. {sem} · {fuente}"
    else:
        sello_txt = f"NFL {season} · nflverse PBP + participación"
    pie(fig, "EPA concedido por jugada (menos = mejor) · gris: % de jugadas · color: 5 mejores y "
             f"5 peores ({COLOR_MIN}+ equipos) · –: menos de {MIN_JUG} jugadas", sello_txt)
    ruta = salida(f"defensa_vs_personal_ofensivo_{season}.png", season)
    guardar(fig, ruta)
    if tit:
        print("Titular:", tit)
    print("Guardado:", ruta)


if __name__ == "__main__":
    main()
