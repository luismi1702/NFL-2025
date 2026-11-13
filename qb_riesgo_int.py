"""
qb_riesgo_int.py
Suerte y riesgo de cada QB: pases que merecian intercepcion (FTN
`is_interception_worthy`) frente a intercepciones reales, por cada 100
intentos. Por debajo de la diagonal, le han perdonado; por encima, le han
castigado de mas.

Intentos = `pass_attempt == 1` sin sacks, atribuidos a `passer_id`. Solo
semanas con el FTN charteado entero (partidos con `n_pass_rushers > 0`).
Equipo = con el que mas ha lanzado (los traspasados de noviembre-2026,
Cousins y Brissett, salen con LV y ARI). 13-nov-2026.
"""

import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.ticker import MultipleLocator

from pbp_loader import cargar_pbp, cargar_ftn, salida, season_cli, sello
from estilo import *

SEASON = season_cli()
MIN_INTENTOS = 130


def apellido(n):
    return n.split(".", 1)[1] if isinstance(n, str) and "." in n[:3] else n


def separar(xs, ys, x_tol, y_tol):
    orden = sorted(range(len(xs)), key=lambda i: -ys[i])
    out = list(ys)
    for p, i in enumerate(orden):
        for j in orden[:p]:
            if abs(xs[i] - xs[j]) < x_tol and abs(out[i] - out[j]) < y_tol:
                out[i] = min(out[i], out[j] - y_tol)
    return out


df, SEASON = cargar_pbp(SEASON)
ftn = cargar_ftn(SEASON)
ftn = ftn[0] if isinstance(ftn, tuple) else ftn
pbp_p = df.groupby("week")["game_id"].nunique()
ftn_p = ftn[ftn["n_pass_rushers"] > 0].groupby("week")["nflverse_game_id"].nunique()
SEM = 0
while SEM + 1 in pbp_p.index and ftn_p.get(SEM + 1, 0) >= pbp_p[SEM + 1]:
    SEM += 1

f = ftn[["nflverse_game_id", "nflverse_play_id", "is_interception_worthy"]].rename(
    columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
f["play_id"] = pd.to_numeric(f["play_id"], errors="coerce")
df = df[df["week"] <= SEM].copy()
df["play_id"] = pd.to_numeric(df["play_id"], errors="coerce")
a = df[(df["pass_attempt"] == 1) & (df["sack"] != 1) & df["passer_id"].notna()].merge(
    f, on=["game_id", "play_id"], how="left")
a["iw"] = a["is_interception_worthy"].astype(float).fillna(0)
a["int"] = pd.to_numeric(a["interception"], errors="coerce").fillna(0)
a["perdon"] = ((a["iw"] == 1) & (a["int"] == 0)).astype(int)

g = a.groupby("passer_id").agg(nombre=("passer", lambda s: s.value_counts().index[0]),
                               team=("posteam", lambda s: s.value_counts().index[0]),
                               n=("iw", "size"), iw=("iw", "sum"), ints=("int", "sum"), perdon=("perdon", "sum"))
g = g[g["n"] >= MIN_INTENTOS].copy()
g["iw100"], g["int100"] = 100 * g["iw"] / g["n"], 100 * g["ints"] / g["n"]
g["ap"] = g["nombre"].map(apellido)
lg_iw, lg_int = 100 * a["iw"].sum() / len(a), 100 * a["int"].sum() / len(a)
print(g.sort_values("iw100", ascending=False).round(2).to_string())
print(f"liga: {lg_iw:.2f} IW y {lg_int:.2f} INT por 100 intentos; sem 1-{SEM}")

riesgo = g.sort_values("iw100", ascending=False).iloc[0]
perd = g.sort_values(["perdon", "iw"], ascending=False).iloc[0]
tit = (f"{riesgo['ap']} es el que más balones regala. A {perd['ap']} le han perdonado"
       f" {int(perd['perdon'])} intercepciones")

fig = lienzo()
antetitulo(fig, f"Quarterbacks · suerte y riesgo · NFL {SEASON}")
y = titular(fig, tit, y=0.93, size=20)
contexto(fig, [("Pases que merecían intercepción", "fuerte"), ("(FTN) frente a", None),
               ("intercepciones reales", "fuerte"),
               (f", por cada 100 intentos. Bajo la línea, menos castigo de lo normal.", None)],
         y=y - 0.012)

tarjeta(fig, 0.085, 0.765)
ax = fig.add_axes([L + 0.045, 0.125, R - L - 0.05, 0.62])
estilo_ejes(ax, x=True, y=True)
tope = max(g["iw100"].max(), g["int100"].max()) * 1.08
ax.set_xlim(-0.15, tope)
ax.set_ylim(-0.25, tope)
ax.plot([0, tope], [0, tope * lg_int / lg_iw], color=GRIS2, lw=0.9, ls="--", zorder=1)
ax.text(tope * 0.97, tope * lg_int / lg_iw * 0.97 + tope * 0.02, "lo normal en la liga", ha="right", size=6.5, color=GRIS2)
ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
ax.set_axisbelow(True)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_major_formatter(lambda v, _: num(v, 0, signo=False))
ax.yaxis.set_major_formatter(lambda v, _: num(v, 0, signo=False))
ax.set_xlabel("Pases de intercepción por 100 intentos", color=GRIS1, size=7.5, labelpad=4)
ax.set_ylabel("Intercepciones por 100 intentos", color=GRIS1, size=7.5, labelpad=2)
q = dict(size=7, color=GRIS2, style="italic")
ax.text(tope * 0.97, tope * 0.06, "le han perdonado más", ha="right", **q)
ax.text(tope * 0.03, tope * 0.94, "le han castigado de más", ha="left", va="top", **q)

for _, r in g.iterrows():
    ax.scatter(r["iw100"], r["int100"], s=150, color=FG, alpha=0.10, lw=0, zorder=3)
    im = logo(r["team"], zoom=0.0115)
    if im is not None:
        ax.add_artist(AnnotationBbox(im, (r["iw100"], r["int100"]), frameon=False, zorder=4))
off = tope * 0.03
lys = separar(list(g["iw100"]), [v - off for v in g["int100"]], tope * 0.07, tope * 0.026)
for (_, r), ly in zip(g.iterrows(), lys):
    dest = r.name in (riesgo.name, perd.name)
    ax.text(r["iw100"], ly, r["ap"], ha="center", va="top", size=7 if dest else 6.3,
            color=FG if dest else GRIS1, weight="bold" if dest else "normal", zorder=5,
            path_effects=[pe.Stroke(linewidth=1.6, foreground=CARD), pe.Normal()])

pie(fig, f"Liga: {num(lg_iw, 1, False)} pases de intercepción y {num(lg_int, 1, False)}"
         f" intercepciones por 100 intentos. Mín. {MIN_INTENTOS} intentos. Fuente: FTN y nflverse.",
    f"NFL {SEASON} · semanas 1-{SEM}")
guardar(fig, salida(f"qb_riesgo_int_{SEASON}.png", SEASON))
