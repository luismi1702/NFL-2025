# waddle_reparto.py — que cambio en el ataque de Denver entre la semana 1 y la
# 2 cuando empezaron a buscar a Jaylen Waddle. Saca DOS PNG:
#   waddle_reparto_*.png  — la causa: el reparto de objetivos
#   waddle_impacto_*.png  — el efecto: que le paso al juego de pase
#
# El angulo sale de la previa de El Nickel del 17-sep-2026, que avisaba de que
# Denver no podia repetir lo de Waddle con una recepcion y dos yardas. Los
# numeros son nuestros, recalculados con el PBP.
#
# La cuota se calcula como en destacados.py: objetivos del jugador entre los
# objetivos del equipo CON receptor identificado (no entre pases intentados),
# para que el numero del post y el del PNG sean el mismo.
#
#   python lab/waddle_reparto.py [--season 2026] [--week 2]

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pbp_loader import cargar_pbp, salida, season_cli, week_cli, sello

BG, CARD, FG, GRID, ACCENT = "#0f1115", "#151924", "#EDEDED", "#2a2f3a", "#2d6cdf"
RYG = ["#d84a4a", "#ffd166", "#06d6a0"]
EQUIPO, FIGURA = "DEN", "J.Waddle"
APODO = "Jaylen Waddle"


def reparto(df, week):
    """[(receptor, cuota %, recepciones, yardas), ...] ordenado por cuota."""
    pas = df[(df["week"] == week) & (df["posteam"] == EQUIPO)
             & (df["pass_attempt"] == 1) & df["receiver_player_name"].notna()]
    total = len(pas)
    filas = []
    for nombre, g in pas.groupby("receiver_player_name"):
        cogidas = g[g["complete_pass"] == 1]
        filas.append((nombre, len(g) / total * 100, len(g), len(cogidas),
                      int(cogidas["yards_gained"].sum())))
    return sorted(filas, key=lambda f: f[1]), total


def panel(ax, datos, titulo, subtitulo, tope):
    nombres = [d[0] for d in datos]
    cuotas = [d[1] for d in datos]
    colores = [RYG[2] if n == FIGURA else "#3a4457" for n in nombres]
    y = np.arange(len(nombres))
    ax.barh(y, cuotas, color=colores, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(nombres, fontsize=11)
    # set_yticklabels no admite lista de colores: hay que iterar
    for etiqueta, nombre in zip(ax.get_yticklabels(), nombres):
        etiqueta.set_color(FG if nombre == FIGURA else "#9aa3b2")
        if nombre == FIGURA:
            etiqueta.set_fontweight("bold")

    for yi, (nombre, cuota, obj, rec, yds) in zip(y, datos):
        txt = f"{cuota:.1f}%  ·  {rec}/{obj}  ·  {yds} yd"
        ax.text(cuota + 1.2, yi, txt, va="center", fontsize=10.5,
                color=FG if nombre == FIGURA else "#9aa3b2",
                fontweight="bold" if nombre == FIGURA else "normal")

    ax.set_facecolor(CARD)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("bottom", "left"):
        ax.spines[lado].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)
    # MISMO tope en los dos paneles: si no, las barras no se pueden
    # comparar de un lado al otro, que es justo lo que cuenta el grafico
    ax.set_xlim(0, tope)
    ax.tick_params(colors="#9aa3b2", labelsize=10)
    ax.set_xlabel("cuota de objetivos (%)", color="#9aa3b2", fontsize=10.5)
    ax.set_title(titulo, color=FG, fontsize=15, fontweight="bold", pad=34)
    ax.text(0.5, 1.01, subtitulo, transform=ax.transAxes, ha="center",
            va="bottom", color="#9aa3b2", fontsize=11)


def impacto(df, week):
    """Los numeros del juego de pase de esa jornada, con y sin Waddle."""
    d = df[(df["week"] == week) & (df["posteam"] == EQUIPO)]
    db = d[(d["pass_attempt"] == 1) | (d["sack"] == 1)]
    for c in ("qb_kneel", "qb_spike"):
        if c in db.columns:
            db = db[db[c] != 1]
    wad = db[db["receiver_player_name"] == FIGURA]
    expl = db[(db["complete_pass"] == 1) & (db["yards_gained"] >= 20)]
    return {
        "epa_db": db["epa"].mean(),
        "adot": wad["air_yards"].mean() if len(wad) else 0.0,
        "epa_wad": wad["epa"].sum(),
        "expl": len(expl),
        "fd_wad": int(wad["first_down"].fillna(0).sum()),
        "fd_tot": int(db["first_down"].fillna(0).sum()),
    }


def mini(ax, titulo, v1, v2, fmt, mejor_alto=True):
    """Dos barras, semana 1 y semana 2, con su propia escala."""
    colores = ["#3a4457", RYG[2] if (v2 > v1) == mejor_alto else RYG[0]]
    ax.bar([0, 1], [v1, v2], color=colores, width=0.62)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["sem. 1", "sem. 2"], fontsize=11)
    ax.set_title(titulo, color=FG, fontsize=12.5, fontweight="bold", pad=12)
    lim = max(abs(v1), abs(v2)) * 1.45 or 1
    ax.set_ylim(min(0, min(v1, v2) * 1.45), lim)
    for x, v in ((0, v1), (1, v2)):
        ax.text(x, v + (lim * 0.05 if v >= 0 else -lim * 0.05), fmt.format(v),
                ha="center", va="bottom" if v >= 0 else "top",
                color=FG, fontsize=13, fontweight="bold")
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_facecolor(CARD)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_yticks([])          # el valor ya va escrito sobre cada barra
    ax.tick_params(colors="#9aa3b2", labelsize=10, left=False, labelleft=False)


def figura_impacto(df, SEASON, WEEK):
    a, b = impacto(df, WEEK - 1), impacto(df, WEEK)
    fig, axes = plt.subplots(1, 4, figsize=(16, 6.4), facecolor=BG)
    fig.subplots_adjust(top=0.88, bottom=0.16, left=0.05, right=0.97, wspace=0.28)
    apellido = APODO.split()[1]
    mini(axes[0], "EPA por intento de pase\n(todo el equipo)",
         a["epa_db"], b["epa_db"], "{:+.2f}")
    mini(axes[1], f"Profundidad media del envío\na {apellido} (yardas)",
         a["adot"], b["adot"], "{:.1f}")
    mini(axes[2], f"EPA generado en los\nenvíos a {apellido}",
         a["epa_wad"], b["epa_wad"], "{:+.1f}")
    mini(axes[3], "Pases explosivos del equipo\n(20 yardas o más)",
         a["expl"], b["expl"], "{:.0f}")

    # Sin titulo ni subtitulo: el texto del post ya los cuenta (21-sep-2026)
    fig.text(0.5, 0.055, f"En la semana 2, {b['fd_wad']} de los {b['fd_tot']} primeros downs "
             f"de pase de Denver salieron de un envío a {APODO.split()[1]}",
             ha="center", color=FG, fontsize=13)
    fig.text(0.05, 0.02, sello(SEASON), ha="left", color="#888888", fontsize=9, alpha=0.8)
    fig.text(0.97, 0.02, "@CuartayDato", ha="right", color="#888888", fontsize=9,
             alpha=0.8, fontstyle="italic")
    out = salida(f"waddle_impacto_{SEASON}.png", SEASON, WEEK)
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Guardado: {out}")
    for etiqueta, d in (("semana 1", a), ("semana 2", b)):
        print(f"  {etiqueta}: EPA/dropback {d['epa_db']:+.3f} · ADOT a Waddle {d['adot']:.1f} "
              f"· EPA de Waddle {d['epa_wad']:+.1f} · explosivas {d['expl']}")


def main():
    SEASON = season_cli(2026)
    WEEK = week_cli(2)
    df, _ = cargar_pbp(SEASON, solo_reg=True, avisar=False)

    d1, tot1 = reparto(df, WEEK - 1)
    d2, tot2 = reparto(df, WEEK)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.8), facecolor=BG)
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.97, wspace=0.42)
    tope = max(c for datos in (d1, d2) for _, c, *_ in datos) + 17
    panel(axes[0], d1, f"Semana {WEEK - 1}",
          f"{tot1} objetivos · derrota 31-10 en Kansas City", tope)
    panel(axes[1], d2, f"Semana {WEEK}",
          f"{tot2} objetivos · victoria 20-13 ante Jacksonville", tope)

    # Sin titulo ni subtitulo: el texto del post ya los cuenta (21-sep-2026)

    axes[1].text(0.99, 0.01, "@CuartayDato", transform=axes[1].transAxes,
                 ha="right", va="bottom", color="#888888", fontsize=9,
                 alpha=0.8, fontstyle="italic")
    axes[0].text(0.0, -0.11, sello(SEASON), transform=axes[0].transAxes,
                 ha="left", va="top", color="#888888", fontsize=9, alpha=0.8)

    out = salida(f"waddle_reparto_{SEASON}.png", SEASON, WEEK)
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Guardado: {out}")
    figura_impacto(df, SEASON, WEEK)
    for etiqueta, datos in ((f"semana {WEEK-1}", d1), (f"semana {WEEK}", d2)):
        w = [d for d in datos if d[0] == FIGURA]
        if w:
            n, cuota, obj, rec, yds = w[0]
            print(f"  {etiqueta}: {cuota:.1f}% de cuota · {rec}/{obj} · {yds} yd")


if __name__ == "__main__":
    main()
