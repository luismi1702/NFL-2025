"""
qb_agresividad.py
Quien arriesga: tiempo hasta lanzar (X) frente a agresividad (Y, % de pases
a ventana cerrada, con un defensor a menos de una yarda) de Next Gen Stats
(nflverse `ngs_passing`, fila de temporada `week == 0`).

El equipo de cada QB sale de nuestro PBP (con el que mas ha lanzado), no de
NGS: la fila de temporada trae el equipo ACTUAL (en nov-2026, Cousins MIN y
Brissett BAL, cuando sus jugadas son de LV y ARI). 13-nov-2026.
"""

import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.ticker import MultipleLocator

from pbp_loader import cargar_pbp, cargar_ngs, salida, season_cli
from estilo import *

SEASON = season_cli()
MIN_INTENTOS = 140


def separar(xs, ys, x_tol, y_tol):
    orden = sorted(range(len(xs)), key=lambda i: -ys[i])
    out = list(ys)
    for p, i in enumerate(orden):
        for j in orden[:p]:
            if abs(xs[i] - xs[j]) < x_tol and abs(out[i] - out[j]) < y_tol:
                out[i] = min(out[i], out[j] - y_tol)
    return out


df, SEASON = cargar_pbp(SEASON)
ngs = cargar_ngs("passing", SEASON)
ngs = ngs[0] if isinstance(ngs, tuple) else ngs
ngs = ngs[ngs["season_type"].fillna("REG") == "REG"]
SEM = int(ngs["week"].max())
g = ngs[ngs["week"] == 0].copy()
g = g[g["attempts"] >= MIN_INTENTOS]
d = df[(pd.to_numeric(df["pass"], errors="coerce") == 1) & df["passer_id"].notna()]
equipo = d.groupby("passer_id")["posteam"].agg(lambda s: s.value_counts().index[0])
g["team"] = g["player_gsis_id"].map(equipo).fillna(g["team_abbr"].replace({"LAR": "LA"}))
g["ap"] = g["player_last_name"]
w = g["attempts"]
lg_tt = (g["avg_time_to_throw"] * w).sum() / w.sum()
lg_ag = (g["aggressiveness"] * w).sum() / w.sum()
print(g.sort_values("aggressiveness", ascending=False)[
    ["player_display_name", "team", "attempts", "aggressiveness", "avg_time_to_throw",
     "avg_intended_air_yards"]].round(2).to_string(index=False))
print(f"media (ponderada, estos QBs): {lg_tt:.2f} s · {lg_ag:.1f}% · NGS hasta sem. {SEM}")

ag = g.sort_values("aggressiveness", ascending=False).iloc[0]
pr = g.sort_values("aggressiveness").iloc[0]
tit = (f"{ag['ap']} se la juega en el {ag['aggressiveness']:.0f}% de sus pases."
       f" {pr['ap']}, en el {pr['aggressiveness']:.0f}%: el que menos arriesga")

fig = lienzo()
antetitulo(fig, f"Quarterbacks · quién arriesga · NFL {SEASON}")
y = titular(fig, tit, y=0.93, size=20)
contexto(fig, [("Agresividad", "fuerte"),
               ("(% de pases a ventana cerrada: el defensor a menos de una yarda) frente al",
                None), ("tiempo hasta lanzar", "fuerte"),
               (f". Next Gen Stats; mín. {MIN_INTENTOS} intentos.", None)],
         y=y - 0.012)

tarjeta(fig, 0.085, 0.765)
ax = fig.add_axes([L + 0.045, 0.125, R - L - 0.05, 0.62])
estilo_ejes(ax, x=True, y=True)
xs, ys = g["avg_time_to_throw"].values, g["aggressiveness"].values
xp, yp = (xs.max() - xs.min()) * 0.08, (ys.max() - ys.min()) * 0.10
x0, x1, y0, y1 = xs.min() - xp, xs.max() + xp, ys.min() - yp * 1.4, ys.max() + yp
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
ax.set_axisbelow(True)
ax.axvline(lg_tt, color=GRIS2, lw=0.9, ls="--", zorder=1)
ax.axhline(lg_ag, color=GRIS2, lw=0.9, ls="--", zorder=1)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_major_formatter(lambda v, _: num(v, 1, signo=False) + " s")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.set_xlabel("Tiempo hasta lanzar", color=GRIS1, size=7.5, labelpad=4)
ax.set_ylabel("Agresividad (% a ventana cerrada)", color=GRIS1, size=7.5, labelpad=2)
q = dict(size=7, color=GRIS2, style="italic")
ax.text(x1 - (x1 - x0) * 0.02, y1 - (y1 - y0) * 0.02, "aguantan y se la juegan",
        ha="right", va="top", **q)
ax.text(x0 + (x1 - x0) * 0.02, y0 + (y1 - y0) * 0.02, "rápido y a lo seguro",
        ha="left", va="bottom", **q)

for _, r in g.iterrows():
    ax.scatter(r["avg_time_to_throw"], r["aggressiveness"], s=150, color=FG, alpha=0.10, lw=0,
               zorder=3)
    im = logo(r["team"], zoom=0.0115)
    if im is not None:
        ax.add_artist(AnnotationBbox(im, (r["avg_time_to_throw"], r["aggressiveness"]),
                                     frameon=False, zorder=4))
off = (y1 - y0) * 0.026
lys = separar(list(xs), [v - off for v in ys], (x1 - x0) * 0.07, (y1 - y0) * 0.025)
for (_, r), ly in zip(g.iterrows(), lys):
    dest = r.name in (ag.name, pr.name)
    ax.text(r["avg_time_to_throw"], ly, r["ap"], ha="center", va="top",
            size=7 if dest else 6.3, color=FG if dest else GRIS1,
            weight="bold" if dest else "normal", zorder=5,
            path_effects=[pe.Stroke(linewidth=1.6, foreground=CARD), pe.Normal()])

pie(fig, f"Media de estos QBs: {num(lg_tt, 2, False)} s y {lg_ag:.0f}% de agresividad."
         f" Fuente: NFL Next Gen Stats (vía nflverse).",
    f"NFL {SEASON} · semanas 1-{SEM}")
guardar(fig, salida(f"qb_agresividad_{SEASON}.png", SEASON))
