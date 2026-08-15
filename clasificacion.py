"""
clasificacion.py
Cuadro de playoffs: los 7 clasificados de cada conferencia mas los que estan
en la pelea, con los desempates reales de la NFL aplicados.

Uso:
  python clasificacion.py                    # temporada y semana autodetectadas
  python clasificacion.py --season 2025 --week 12
  python clasificacion.py --divisiones       # vista por divisiones en vez de sembrado

Desempates implementados, en el orden oficial:
  Division  -> enfrentamiento directo, record divisional, partidos comunes,
               record de conferencia, fuerza de las victorias, diferencial
  Comodin   -> (si son de la misma division, primero se resuelve la division)
               enfrentamiento directo, record de conferencia, partidos comunes,
               fuerza de las victorias, diferencial

No implementa los criterios finales que casi nunca se alcanzan (fuerza del
calendario, ranking combinado de puntos, sorteo). Cuando dos equipos llegan
empatados hasta ahi, el orden lo decide el diferencial de puntos y se avisa.
"""
import os
import sys
from itertools import combinations

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from pbp_loader import (cargar_calendario, cargar_equipos, salida, sello,
                        season_cli, week_cli, ultima_semana)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON    = season_cli()
LOGOS_DIR = "logos"

BG     = "#0f1115"
CARD   = "#151924"
FG     = "#EDEDED"
GRID   = "#2a2f3a"
ACCENT = "#2d6cdf"
VERDE  = "#06d6a0"
AMBAR  = "#ffd166"
ROJO   = "#d84a4a"
DPI    = 170

N_PLAYOFF = 7      # 4 campeones de division + 3 comodines
N_PELEA   = 3      # cuantos equipos fuera del corte se muestran


# ── LOGOS ──────────────────────────────────────────────────────────────────────
def load_logo(team, zoom=0.055):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        # Recorta margenes transparentes: algunos archivos traen mucho aire
        # (NYJ: tinta 3768x1186 en lienzo 4096x4096) y sin recorte salen enanos
        if img.ndim == 3 and img.shape[2] == 4:
            ys, xs = np.where(img[:, :, 3] > 0.02)
            if len(ys):
                img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = img.shape[:2]
        # Normaliza por el area de tinta real; los wordmarks apaisados
        # pueden ensancharse hasta 1.8x para compensar su poca altura
        z = zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * z > 900.0 * zoom:
            z = 900.0 * zoom / w
        return OffsetImage(img, zoom=z, resample=True)
    except Exception:
        return None


# ── DATOS ──────────────────────────────────────────────────────────────────────
def partidos_por_equipo(sched):
    """Una fila por equipo y partido jugado: rival, puntos a favor/en contra."""
    filas = []
    for lado, prop, rival in (("home", "home", "away"), ("away", "away", "home")):
        sub = sched.rename(columns={
            f"{prop}_team": "equipo", f"{prop}_score": "pf",
            f"{rival}_team": "rival",  f"{rival}_score": "pc",
        })[["week", "equipo", "rival", "pf", "pc", "div_game"]]
        filas.append(sub)
    df = pd.concat(filas, ignore_index=True)
    df["pf"] = pd.to_numeric(df["pf"], errors="coerce")
    df["pc"] = pd.to_numeric(df["pc"], errors="coerce")
    df = df.dropna(subset=["pf", "pc"])
    df["gana"]  = (df["pf"] > df["pc"]).astype(int)
    df["pierde"] = (df["pf"] < df["pc"]).astype(int)
    df["empata"] = (df["pf"] == df["pc"]).astype(int)
    return df


def pct(g, p, e):
    """Porcentaje de victorias de la NFL: el empate vale medio partido."""
    total = g + p + e
    return (g + 0.5 * e) / total if total else 0.0


def tabla(juegos, equipos):
    """Registro global, divisional y de conferencia por equipo."""
    meta = equipos.set_index("team_abbr")
    j = juegos.copy()
    j["conf_rival"] = j["rival"].map(meta["team_conf"])
    j["conf"]       = j["equipo"].map(meta["team_conf"])
    j["mismo_conf"] = j["conf"] == j["conf_rival"]

    filas = {}
    for eq, g in j.groupby("equipo"):
        div_g = g[g["div_game"] == 1]
        cnf_g = g[g["mismo_conf"]]
        filas[eq] = {
            "G": int(g["gana"].sum()), "P": int(g["pierde"].sum()),
            "E": int(g["empata"].sum()),
            "PF": int(g["pf"].sum()),  "PC": int(g["pc"].sum()),
            "pct":     pct(g["gana"].sum(), g["pierde"].sum(), g["empata"].sum()),
            "pct_div": pct(div_g["gana"].sum(), div_g["pierde"].sum(), div_g["empata"].sum()),
            "pct_cnf": pct(cnf_g["gana"].sum(), cnf_g["pierde"].sum(), cnf_g["empata"].sum()),
            "div": meta.loc[eq, "team_division"],
            "conf": meta.loc[eq, "team_conf"],
        }
    t = pd.DataFrame(filas).T
    t["dif"] = t["PF"] - t["PC"]
    return t


# ── DESEMPATES ─────────────────────────────────────────────────────────────────
def h2h(juegos, a, b):
    """Balance de a contra b. >0 si a manda, <0 si manda b, 0 si no aplica."""
    d = juegos[(juegos["equipo"] == a) & (juegos["rival"] == b)]
    if d.empty:
        return 0.0
    return float(d["gana"].sum() - d["pierde"].sum())


def h2h_grupo(juegos, grupo, eq):
    """Porcentaje de un equipo contra el resto del grupo empatado."""
    d = juegos[(juegos["equipo"] == eq) & (juegos["rival"].isin(set(grupo) - {eq}))]
    if d.empty:
        return None
    return pct(d["gana"].sum(), d["pierde"].sum(), d["empata"].sum())


def partidos_comunes(juegos, grupo, eq, minimo=4):
    """Registro contra los rivales a los que se ha enfrentado TODO el grupo."""
    rivales = [set(juegos[juegos["equipo"] == e]["rival"]) - set(grupo) for e in grupo]
    comunes = set.intersection(*rivales) if rivales else set()
    d = juegos[(juegos["equipo"] == eq) & (juegos["rival"].isin(comunes))]
    if len(d) < minimo:
        return None
    return pct(d["gana"].sum(), d["pierde"].sum(), d["empata"].sum())


def fuerza_victorias(juegos, t, eq):
    """Media del % de victorias de los rivales a los que se ha ganado."""
    ganados = juegos[(juegos["equipo"] == eq) & (juegos["gana"] == 1)]["rival"]
    vals = [t.loc[r, "pct"] for r in ganados if r in t.index]
    return float(np.mean(vals)) if vals else 0.0


def ordenar(grupo, juegos, t, ambito):
    """Ordena un grupo de equipos empatados. ambito: 'div' o 'conf'.

    Devuelve (lista ordenada, motivo del desempate). La NFL resuelve los
    empates de tres o mas por eliminacion: se aplica el criterio, se queda el
    mejor, y los demas vuelven a empezar el proceso desde el principio.
    """
    if len(grupo) == 1:
        return list(grupo), ""

    if ambito == "div":
        criterios = [
            ("enfrentamiento directo", lambda e: h2h_grupo(juegos, grupo, e)),
            ("récord divisional",      lambda e: t.loc[e, "pct_div"]),
            ("partidos comunes",       lambda e: partidos_comunes(juegos, grupo, e)),
            ("récord de conferencia",  lambda e: t.loc[e, "pct_cnf"]),
            ("fuerza de victorias",    lambda e: fuerza_victorias(juegos, t, e)),
        ]
    else:
        criterios = [
            ("enfrentamiento directo", lambda e: h2h_grupo(juegos, grupo, e)),
            ("récord de conferencia",  lambda e: t.loc[e, "pct_cnf"]),
            ("partidos comunes",       lambda e: partidos_comunes(juegos, grupo, e)),
            ("fuerza de victorias",    lambda e: fuerza_victorias(juegos, t, e)),
        ]

    for nombre, f in criterios:
        vals = {e: f(e) for e in grupo}
        if any(v is None for v in vals.values()):
            continue                      # criterio no aplicable a todo el grupo
        mejor = max(vals.values())
        lideres = [e for e in grupo if vals[e] == mejor]
        if len(lideres) < len(grupo):     # el criterio separa
            resto, _ = ordenar([e for e in grupo if e not in lideres], juegos, t, ambito)
            cabeza, _ = ordenar(lideres, juegos, t, ambito) if len(lideres) > 1 else (lideres, "")
            return list(cabeza) + resto, nombre

    # Nadie separa: cae al diferencial de puntos (y se avisa en el pie)
    return sorted(grupo, key=lambda e: -t.loc[e, "dif"]), "diferencial (sin resolver)"


def sembrado(t, juegos, conf):
    """Los 7 clasificados de una conferencia, en orden de siembra."""
    sub = t[t["conf"] == conf]
    avisos = []

    # 1. Campeon de cada division
    campeones = []
    for div, g in sub.groupby("div"):
        mejor = g["pct"].max()
        empatados = list(g[g["pct"] == mejor].index)
        orden, motivo = ordenar(empatados, juegos, t, "div")
        if len(empatados) > 1:
            avisos.append(f"{div}: {orden[0]} por {motivo}")
        campeones.append(orden[0])

    # 2. Siembra 1-4: campeones ordenados entre si (criterios de comodin)
    campeones = ordenar_por_pct(campeones, juegos, t, "conf", avisos)

    # 3. Comodines: el resto, 3 mejores
    resto = [e for e in sub.index if e not in campeones]
    resto = ordenar_por_pct(resto, juegos, t, "conf", avisos)

    return campeones + resto, avisos


def ordenar_por_pct(equipos, juegos, t, ambito, avisos):
    """Ordena por % de victorias resolviendo los empates PLAZA A PLAZA.

    Es importante hacerlo de una en una y no ordenar el bloque entero de golpe:
    cuando dos empatados son de la misma division, la NFL elimina al peor solo
    para esa plaza — vuelve a entrar en la puja por la siguiente.
    """
    restantes = list(equipos)
    orden = []
    while restantes:
        mejor = max(t.loc[e, "pct"] for e in restantes)
        bloque = [e for e in restantes if t.loc[e, "pct"] == mejor]

        if len(bloque) == 1:
            ganador = bloque[0]
        else:
            # Un representante por division: dentro de cada una se aplican
            # primero los desempates divisionales
            por_div = {}
            for e in bloque:
                por_div.setdefault(t.loc[e, "div"], []).append(e)
            representantes = []
            for div, miembros in por_div.items():
                o, motivo = ordenar(miembros, juegos, t, "div")
                representantes.append(o[0])
                if len(miembros) > 1:
                    avisos.append(f"{div}: {o[0]} sobre {', '.join(o[1:])} por {motivo}")
            if len(representantes) > 1:
                o, motivo = ordenar(representantes, juegos, t, ambito)
                avisos.append(f"{'/'.join(o)} por {motivo}")
            else:
                o = representantes
            ganador = o[0]

        orden.append(ganador)
        restantes.remove(ganador)
    return orden


# ── DIBUJO ─────────────────────────────────────────────────────────────────────
def fmt_record(r):
    return f"{int(r['G'])}-{int(r['P'])}" + (f"-{int(r['E'])}" if r["E"] else "")


def etiqueta(i, t, eq):
    if i < 4:
        return t.loc[eq, "div"].replace("AFC ", "").replace("NFC ", "")
    if i < N_PLAYOFF:
        return "Comodín"
    return ""


def panel(ax, conf, orden, t, sem):
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    n_filas = N_PLAYOFF + N_PELEA
    # Los que estan fuera del corte se bajan un poco: deja una banda limpia
    # bajo la linea de playoffs para su etiqueta, sin pisar filas
    ax.set_ylim(n_filas + 0.9, -1.5)
    ax.axis("off")

    ax.text(0.2, -1.0, conf, fontsize=20, fontweight="bold", color=FG, va="center")
    ax.text(9.8, -1.0, "REG  ·  DIF", fontsize=8.5, color="#767E90",
            va="center", ha="right")

    corte_g = t.loc[orden[N_PLAYOFF - 1], "G"] if len(orden) >= N_PLAYOFF else 0
    corte_p = t.loc[orden[N_PLAYOFF - 1], "P"] if len(orden) >= N_PLAYOFF else 0

    for i, eq in enumerate(orden[:n_filas]):
        r = t.loc[eq]
        dentro = i < N_PLAYOFF
        alpha  = 1.0 if dentro else 0.45
        y_fila = i if dentro else i + 0.5   # hueco bajo la linea de playoffs

        ax.add_patch(plt.Rectangle((0.15, y_fila - 0.38), 9.7, 0.76,
                                   facecolor=CARD if dentro else BG,
                                   edgecolor=GRID, linewidth=0.8, alpha=alpha,
                                   zorder=1))
        # Numero de siembra
        col_semilla = VERDE if i < 4 else (AMBAR if dentro else "#5A6172")
        ax.text(0.6, y_fila, str(i + 1), fontsize=13, fontweight="bold",
                color=col_semilla, va="center", ha="center", alpha=alpha, zorder=3)

        im = load_logo(eq, zoom=0.05)
        if im is not None:
            im.image.set_alpha(alpha)
            ax.add_artist(AnnotationBbox(im, (1.5, y_fila), frameon=False, zorder=3))
        else:
            ax.text(1.5, y_fila, eq, fontsize=11, color=FG, va="center", ha="center")

        ax.text(2.35, y_fila, eq, fontsize=12, fontweight="bold", color=FG,
                va="center", alpha=alpha, zorder=3)
        ax.text(3.5, y_fila, etiqueta(i, t, eq), fontsize=9, color="#8A93A6",
                va="center", alpha=alpha, zorder=3)

        ax.text(7.9, y_fila, fmt_record(r), fontsize=12, color=FG, va="center",
                ha="right", alpha=alpha, zorder=3, fontfamily="monospace")
        dif = int(r["dif"])
        ax.text(9.7, y_fila, f"{dif:+d}", fontsize=11, va="center", ha="right",
                color=VERDE if dif > 0 else (ROJO if dif < 0 else "#8A93A6"),
                alpha=alpha, zorder=3, fontfamily="monospace")

        # A cuantos partidos del corte
        if not dentro:
            atras = ((corte_g - r["G"]) + (r["P"] - corte_p)) / 2
            ax.text(6.1, y_fila, f"a {atras:g}", fontsize=9, color="#767E90",
                    va="center", ha="right", alpha=0.8, zorder=3)

    # Linea de playoffs. La etiqueta va a la izquierda y BAJO la linea: a la
    # derecha pisaba el diferencial del septimo clasificado.
    y = N_PLAYOFF - 0.5
    ax.plot([0.15, 9.85], [y, y], color=ACCENT, linewidth=1.6, zorder=4)
    ax.text(0.25, y + 0.24, "línea de playoffs", fontsize=8.5, color=ACCENT,
            ha="left", va="center", style="italic", zorder=5)


def main():
    equipos = cargar_equipos()
    sched, season = cargar_calendario(SEASON)
    sched = sched[sched["game_type"] == "REG"].copy()

    # La semana la manda --week; si no, hasta donde ha llegado la liga; si eso
    # tampoco se puede verificar, la ultima con resultado en el calendario.
    jugadas = pd.to_numeric(
        sched.dropna(subset=["home_score"])["week"], errors="coerce").dropna()
    if jugadas.empty:
        raise SystemExit(
            f"\n  Todavia no se ha jugado ningun partido de {season}.\n"
            f"  La clasificacion necesita al menos una jornada.\n")

    semana = week_cli() or ultima_semana(season) or int(jugadas.max())
    sched = sched[pd.to_numeric(sched["week"], errors="coerce") <= semana]

    juegos = partidos_por_equipo(sched)
    if juegos.empty:
        raise SystemExit(
            f"\n  No hay partidos jugados en {season} hasta la semana {semana}.\n")

    t = tabla(juegos, equipos)
    print(f"NFL {season} — clasificacion hasta la semana {semana} "
          f"({len(juegos)//2} partidos)")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.6), dpi=DPI)
    fig.patch.set_facecolor(BG)

    todos_avisos = []
    for ax, conf in zip(axes, ["AFC", "NFC"]):
        orden, avisos = sembrado(t, juegos, conf)
        panel(ax, conf, orden, t, semana)
        todos_avisos += avisos
        print(f"\n{conf}:")
        for i, eq in enumerate(orden[:N_PLAYOFF]):
            print(f"  {i+1}. {eq:4} {fmt_record(t.loc[eq]):8} {etiqueta(i, t, eq)}")

    fig.text(0.5, 0.975, f"Cuadro de playoffs — NFL {season}, semana {semana}",
             ha="center", va="top", fontsize=20, fontweight="bold", color=FG)
    fig.text(0.5, 0.935,
             "Siembra 1-4: campeones de división  ·  5-7: comodines  ·  "
             "desempates oficiales aplicados",
             ha="center", va="top", fontsize=10.5, color="#8A93A6", style="italic")

    if todos_avisos:
        print("\nDesempates aplicados:")
        for a in todos_avisos:
            print(f"  - {a}")
        sin_resolver = [a for a in todos_avisos if "sin resolver" in a]
        if sin_resolver:
            fig.text(0.5, 0.045,
                     "Algún empate llegó al diferencial de puntos: la NFL usaría "
                     "criterios posteriores no implementados",
                     ha="center", fontsize=8.5, color=AMBAR, style="italic")

    fig.text(0.012, 0.015, f"Fuente: nflverse-data · schedules  ·  {sello(season)}",
             ha="left", va="bottom", fontsize=7.5, color="#555555", style="italic")
    fig.text(0.988, 0.015, "@CuartayDato", ha="right", va="bottom",
             color="#888888", fontsize=9, alpha=0.85, style="italic")

    plt.subplots_adjust(top=0.90, bottom=0.07, left=0.02, right=0.98, wspace=0.10)
    out = salida(f"clasificacion_{season}.png", season, semana)
    plt.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()
