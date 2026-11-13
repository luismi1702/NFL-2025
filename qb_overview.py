"""
qb_overview.py
Todos los QBs de la liga: EPA por jugada de pase (X) frente a CPOE (Y),
con el logo de su equipo en cada punto y las medias de liga como cruz.
Titular automatico: quien genera mas y quien es el mas preciso.

Jugada de pase = `pass == 1` de nflverse (dropback: incluye sacks y
scrambles), atribuida a `passer_id`. Minimo MIN_JUGADAS.
Migrado al estilo de estilo.py el 13-nov-2026 (antes: 13x9 sin titular y
color por "EPA bajo presion" con qb_hit+sack como proxy: se quita, en
temporada no hay presion real por QB en nflverse).
"""

import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.offsetbox import AnnotationBbox

from pbp_loader import cargar_pbp, salida, season_cli, sello
from estilo import *

SEASON = season_cli()
MIN_JUGADAS = 150


def apellido(nombre):
    """'B.Purdy' -> 'Purdy'; 'A.St. Brown' -> 'St. Brown'."""
    if not isinstance(nombre, str):
        return ""
    return nombre.split(".", 1)[1] if "." in nombre[:3] else nombre


def separar(xs, ys, x_tol, y_tol):
    """Baja las etiquetas que caerian casi encima de otra (el punto no se mueve)."""
    orden = sorted(range(len(xs)), key=lambda i: -ys[i])
    out = list(ys)
    for p, i in enumerate(orden):
        for j in orden[:p]:
            if abs(xs[i] - xs[j]) < x_tol and abs(out[i] - out[j]) < y_tol:
                out[i] = min(out[i], out[j] - y_tol)
    return out


# ── Datos
df, SEASON = cargar_pbp(SEASON)
for c in ["epa", "cpoe", "pass"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
d = df[(df["pass"] == 1) & df["epa"].notna() & df["passer_id"].notna()]

g = (d.groupby("passer_id")
     .agg(nombre=("passer", lambda s: s.value_counts().index[0]),
          team=("posteam", lambda s: s.value_counts().index[0]),
          n=("epa", "size"), epa=("epa", "mean"), cpoe=("cpoe", "mean"))
     .reset_index())
g = g[g["n"] >= MIN_JUGADAS].copy()
g["ap"] = g["nombre"].map(apellido)
lg_epa, lg_cpoe = d["epa"].mean(), d["cpoe"].mean()

top_e = g.sort_values("epa", ascending=False).iloc[0]
top_c = g.sort_values("cpoe", ascending=False).iloc[0]
print(g.sort_values("epa", ascending=False)[["nombre", "team", "n", "epa", "cpoe"]]
      .round(3).to_string(index=False))
print(f"liga: EPA {lg_epa:+.3f} · CPOE {lg_cpoe:+.2f} · {len(g)} QBs")

if top_e["passer_id"] == top_c["passer_id"]:
    tit = f"{top_e['ap']}, el mejor en las dos cosas: el que más genera y el más preciso"
else:
    tit = f"{top_e['ap']} es el QB que más genera. {top_c['ap']}, el más preciso"

# ── Figura
fig = lienzo()
antetitulo(fig, f"Quarterbacks · NFL {SEASON}")
y = titular(fig, tit, y=0.93, size=21)
contexto(fig, [("EPA por jugada de pase", "fuerte"),
               ("(con sacks y scrambles) frente a", None),
               ("CPOE", "fuerte"),
               (f"(acierto sobre lo esperado). Mín. {MIN_JUGADAS} jugadas;"
                f" líneas: media de la liga.", None)],
         y=y - 0.012)

tarjeta(fig, 0.085, 0.765)
ax = fig.add_axes([L + 0.045, 0.125, R - L - 0.05, 0.62])
estilo_ejes(ax, x=True, y=True)

xs, ys = g["epa"].values, g["cpoe"].values
xp, yp = (xs.max() - xs.min()) * 0.08, (ys.max() - ys.min()) * 0.10
x0, x1, y0, y1 = xs.min() - xp, xs.max() + xp, ys.min() - yp * 1.4, ys.max() + yp
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
ax.set_axisbelow(True)
ax.axvline(lg_epa, color=GRIS2, lw=0.9, ls="--", zorder=1)
ax.axhline(lg_cpoe, color=GRIS2, lw=0.9, ls="--", zorder=1)
from matplotlib.ticker import MultipleLocator
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_major_formatter(lambda v, _: num(v, 1))
ax.yaxis.set_major_formatter(lambda v, _: num(v, 0) + "%")
ax.set_xlabel("EPA por jugada de pase", color=GRIS1, size=7.5, labelpad=4)
ax.set_ylabel("CPOE", color=GRIS1, size=7.5, labelpad=2)

q = dict(size=7, color=GRIS2, style="italic", alpha=0.9)
ax.text(x1 - (x1 - x0) * 0.02, y1 - (y1 - y0) * 0.02, "generan y aciertan", ha="right",
        va="top", **q)
ax.text(x0 + (x1 - x0) * 0.02, y0 + (y1 - y0) * 0.02, "ni una cosa ni otra", ha="left",
        va="bottom", **q)

for _, r in g.iterrows():
    ax.scatter(r["epa"], r["cpoe"], s=150, color=FG, alpha=0.10, lw=0, zorder=3)
    im = logo(r["team"], zoom=0.0115)
    if im is not None:
        ax.add_artist(AnnotationBbox(im, (r["epa"], r["cpoe"]), frameon=False, zorder=4))

off = (y1 - y0) * 0.026
lys = separar(list(xs), [v - off for v in ys], (x1 - x0) * 0.07, (y1 - y0) * 0.025)
for (_, r), ly in zip(g.iterrows(), lys):
    destacado = r["passer_id"] in (top_e["passer_id"], top_c["passer_id"])
    ax.text(r["epa"], ly, r["ap"], ha="center", va="top", size=6.3 if not destacado else 7,
            color=FG if destacado else GRIS1, weight="bold" if destacado else "normal",
            zorder=5, path_effects=[pe.Stroke(linewidth=1.6, foreground=CARD), pe.Normal()])

pie(fig, f"Liga: {num(lg_epa, 3)} EPA por jugada de pase y {num(lg_cpoe, 1)}% de CPOE."
         f" Fuente: nflverse.", sello(SEASON))
guardar(fig, salida(f"qb_overview_{SEASON}.png", SEASON))
