# johnson_peor_partido.py — los 70 partidos de Ben Johnson llamando jugadas,
# ordenados en el tiempo, con el 31,9% del domingo en Chicago marcado abajo.
#
# Continuacion del hilo Johnson vs Flores del sabado 19-sep: aquella pieza
# anticipaba el duelo y esta cuenta como acabo. La idea sale del boletin Week 2
# Review de SumerSports (21-sep-2026), pero la cifra NO: esta recalculada con
# nuestro PBP, como manda docs/decisiones.md (14-sep-2026).
#
#   python lab/johnson_peor_partido.py [--season 2026] [--week 2]

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pbp_loader import cargar_pbp, salida, season_cli, week_cli, sello

BG, CARD, FG, GRID, ACCENT = "#0f1115", "#151924", "#EDEDED", "#2a2f3a", "#2d6cdf"
RYG = ["#d84a4a", "#ffd166", "#06d6a0"]
LOGOS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logos")

# Ben Johnson canta las jugadas en Detroit desde 2022 y en Chicago desde 2025
# (verificado en web el 21-sep-2026: sigue siendo el quien las canta como HC)
ERAS = [(2022, "DET"), (2023, "DET"), (2024, "DET"), (2025, "CHI"), (2026, "CHI")]
MIN_JUGADAS = 30


def load_logo(team, zoom=0.030):
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
        z = zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * z > 900.0 * zoom:
            z = 900.0 * zoom / w
        return OffsetImage(img, zoom=z, resample=True)
    except Exception:
        return None


def tasa_exito(season, equipo):
    """Un punto por partido: tasa de exito ofensiva, como en destacados.py."""
    df, _ = cargar_pbp(season, solo_reg=True, avisar=False)
    o = df[(df["posteam"] == equipo) & df["play_type"].isin(["pass", "run"])
           & df["epa"].notna()]
    for c in ("qb_kneel", "qb_spike"):
        if c in o.columns:
            o = o[o[c] != 1]
    g = (o.groupby(["week", "game_id"])["success"]
           .agg(["mean", "size"]).reset_index())
    g = g[g["size"] >= MIN_JUGADAS]
    g["sr"] = g["mean"] * 100
    g["season"], g["equipo"] = season, equipo
    return g.sort_values("week")


def main():
    SEASON = season_cli(2026)
    WEEK = week_cli(2)

    t = pd.concat([tasa_exito(s, e) for s, e in ERAS], ignore_index=True)
    t = t.sort_values(["season", "week"]).reset_index(drop=True)
    t["n"] = np.arange(1, len(t) + 1)

    peor = t.loc[t["sr"].idxmin()]
    media = t["sr"].mean()
    corte = t[t["equipo"] == "CHI"]["n"].min()          # salto Detroit -> Chicago

    fig, ax = plt.subplots(figsize=(13.5, 7.6), facecolor=BG)
    fig.subplots_adjust(top=0.90, left=0.07, right=0.93, bottom=0.12)
    ax.set_facecolor(CARD)
    # Margen abajo para que el punto marcado no caiga sobre la marca de agua
    ax.set_ylim(t["sr"].min() - 8, t["sr"].max() + 4)
    for lado in ("top", "right", "bottom", "left"):
        ax.spines[lado].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)

    # Frontera de eras y media de su carrera
    ax.axvline(corte - 0.5, color=GRID, linewidth=1.4, linestyle="--")
    ax.axhline(media, color=FG, linewidth=1.0, alpha=0.35)
    ax.text(len(t) + 0.4, media, f" media {media:.1f}%", color=FG, alpha=0.55,
            fontsize=10, va="center")

    es_peor = t["n"] == peor["n"]
    ax.scatter(t.loc[~es_peor, "n"], t.loc[~es_peor, "sr"], s=58,
               c=ACCENT, alpha=0.75, edgecolors="none", zorder=3)
    ax.scatter(t.loc[es_peor, "n"], t.loc[es_peor, "sr"], s=190,
               c=RYG[0], edgecolors="white", linewidths=1.4, zorder=5)

    # El punto del domingo, contado. Sin flecha (21-sep-2026): el texto va
    # pegado al punto, a su izquierda y a su misma altura, en la banda vacia
    # de abajo, que es la unica zona del grafico sin puntos
    ax.text(peor["n"] - 1.6, peor["sr"],
            f"MIN 9-3 CHI · semana {WEEK}\n"
            f"{peor['sr']:.1f}% de jugadas exitosas: el peor de sus {len(t)} partidos",
            color=FG, fontsize=12.5, fontweight="bold", linespacing=1.6,
            va="center", ha="right")

    # Logos dentro del grafico, en la banda alta que tampoco tiene puntos
    alto = ax.get_ylim()[1]
    for team, x in ((ERAS[0][1], corte / 2), (ERAS[-1][1], (corte + len(t)) / 2)):
        logo = load_logo(team, zoom=0.052)
        if logo:
            ax.add_artist(AnnotationBbox(logo, (x, alto - 1.2), frameon=False,
                                         box_alignment=(0.5, 1.0)))

    ax.set_xlim(0, len(t) + 2)
    ax.set_xlabel("partidos llamando jugadas, en orden (Detroit 2022-24 · Chicago 2025-26)",
                  color="#9aa3b2", fontsize=11, labelpad=10)
    ax.set_ylabel("jugadas exitosas (%)", color="#9aa3b2", fontsize=11)
    ax.tick_params(colors="#9aa3b2", labelsize=10)

    # Sin titulo grande (21-sep-2026): el texto del post ya lo cuenta
    ax.set_title("Tasa de éxito ofensiva, partido a partido, desde que dirige un ataque en la NFL",
                 color="#9aa3b2", fontsize=12.5, pad=16)

    ax.text(0.01, 0.01, sello(SEASON), transform=ax.transAxes, ha="left",
            va="bottom", color="#888888", fontsize=9, alpha=0.8)
    ax.text(0.99, 0.01, "@CuartayDato", transform=ax.transAxes, ha="right",
            va="bottom", color="#888888", fontsize=9, alpha=0.8, fontstyle="italic")

    out = salida(f"johnson_peor_partido_{SEASON}.png", SEASON, WEEK)
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"Guardado: {out}")
    print(f"  partidos: {len(t)} · media {media:.1f}% · peor {peor['sr']:.1f}% "
          f"({peor['season']} sem. {int(peor['week'])})")
    print(t.nsmallest(5, "sr")[["season", "equipo", "week", "sr"]]
           .to_string(index=False, float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()
