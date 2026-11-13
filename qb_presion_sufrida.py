"""
qb_presion_sufrida.py
¿Culpa suya o de la linea? % de dropbacks con presion de cada QB (PFR:
hurries + golpes + sacks, `times_pressured` semanal) frente a su EPA por
jugada de pase (nflverse, `pass == 1`). Arriba a la derecha, los que rinden
aunque les lleguen.

Union PFR-nflverse por partido + equipo + apellido. El denominador de la
presion es el de PFR (times_pressured / times_pressured_pct); en partidos
sin presion, nuestros dropbacks. Se corta en la ultima semana que trae PFR,
y el EPA, en la misma. 13-nov-2026. Sustituye en temporada a qb_presion.py,
que necesita la participacion de febrero.
"""

import re
import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.ticker import MultipleLocator

from pbp_loader import cargar_pbp, cargar_pfr, salida, season_cli
from estilo import *

SEASON = season_cli()
MIN_JUGADAS = 150


def apellido(n):
    return n.split(".", 1)[1] if isinstance(n, str) and "." in n[:3] else n


def clave(s):
    s = re.sub(r"\b(Jr|Sr|II|III|IV)\.?$", "", str(s).strip()).strip()
    return re.sub(r"[^a-z]", "", s.split()[-1].lower() if " " in s else s.lower())


def separar(xs, ys, x_tol, y_tol):
    orden = sorted(range(len(xs)), key=lambda i: -ys[i])
    out = list(ys)
    for p, i in enumerate(orden):
        for j in orden[:p]:
            if abs(xs[i] - xs[j]) < x_tol and abs(out[i] - out[j]) < y_tol:
                out[i] = min(out[i], out[j] - y_tol)
    return out


df, SEASON = cargar_pbp(SEASON)
pfr = cargar_pfr("pass", SEASON, semanal=True)
pfr = pfr[0] if isinstance(pfr, tuple) else pfr
pfr = pfr[pfr["game_type"].fillna("REG") == "REG"].copy()
SEM = int(pfr["week"].max())
falta = set(df[df["week"] <= SEM]["game_id"]) - set(pfr["game_id"])
if falta:
    raise SystemExit(f"PFR no trae {len(falta)} partidos hasta la sem. {SEM}: {sorted(falta)[:5]}")

df = df[df["week"] <= SEM].copy()
for c in ["epa", "pass"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
d = df[(df["pass"] == 1) & df["epa"].notna() & df["passer_id"].notna()]
pg = (d.groupby(["game_id", "posteam", "passer_id"])
      .agg(nombre=("passer", "first"), db=("epa", "size")).reset_index())
pg["k"] = pg["nombre"].map(lambda n: clave(apellido(n)))
pfr["k"] = pfr["pfr_player_name"].map(clave)
pfr["tp"] = pd.to_numeric(pfr["times_pressured"], errors="coerce").fillna(0)
pct = pd.to_numeric(pfr["times_pressured_pct"], errors="coerce")
pct = pct.where(pct <= 1, pct / 100)
pfr["db_pfr"] = (pfr["tp"] / pct).where(pct > 0)
m = pg.merge(pfr[["game_id", "team", "k", "tp", "db_pfr"]],
             left_on=["game_id", "posteam", "k"], right_on=["game_id", "team", "k"], how="left")
sin = m[m["tp"].isna() & (m["db"] >= 10)]
if len(sin):
    print("Sin cruce con PFR (>=10 dropbacks):", sin[["game_id", "posteam", "nombre", "db"]]
          .to_string(index=False))
m["den"] = m["db_pfr"].fillna(m["db"])

g = (d.groupby("passer_id")
     .agg(nombre=("passer", lambda s: s.value_counts().index[0]),
          team=("posteam", lambda s: s.value_counts().index[0]),
          n=("epa", "size"), epa=("epa", "mean")))
p = m.dropna(subset=["tp"]).groupby("passer_id").agg(tp=("tp", "sum"), den=("den", "sum"))
g = g.join(p)
g = g[g["n"] >= MIN_JUGADAS].copy()
g["pres"] = 100 * g["tp"] / g["den"]
g["ap"] = g["nombre"].map(apellido)
lg_epa = d["epa"].mean()
lg_pres = 100 * m["tp"].sum() / m.dropna(subset=["tp"])["den"].sum()
print(g.sort_values("pres", ascending=False).round(3).to_string())
print(f"liga: presion {lg_pres:.1f}% · EPA {lg_epa:+.3f} · sem 1-{SEM}")

mas = g.sort_values("pres", ascending=False).iloc[0]
aguanta = g.sort_values("pres").iloc[0]          # el mas protegido
tit = (f"A {mas['ap']} le llega la presión en el {mas['pres']:.0f}% de sus dropbacks."
       f" A {aguanta['ap']}, solo en el {aguanta['pres']:.0f}%")

fig = lienzo()
antetitulo(fig, f"Quarterbacks · ¿culpa suya o de la línea? · NFL {SEASON}")
y = titular(fig, tit, y=0.93, size=20)
contexto(fig, [("% de dropbacks con presión", "fuerte"),
               ("(PFR: hurries, golpes y sacks) frente a", None),
               ("EPA por jugada de pase", "fuerte"),
               (f". Mín. {MIN_JUGADAS} jugadas; líneas: media de la liga.", None)],
         y=y - 0.012)

tarjeta(fig, 0.085, 0.765)
ax = fig.add_axes([L + 0.045, 0.125, R - L - 0.05, 0.62])
estilo_ejes(ax, x=True, y=True)
xs, ys = g["pres"].values, g["epa"].values
xp, yp = (xs.max() - xs.min()) * 0.08, (ys.max() - ys.min()) * 0.10
x0, x1, y0, y1 = xs.min() - xp, xs.max() + xp, ys.min() - yp * 1.4, ys.max() + yp
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
ax.set_axisbelow(True)
ax.axvline(lg_pres, color=GRIS2, lw=0.9, ls="--", zorder=1)
ax.axhline(lg_epa, color=GRIS2, lw=0.9, ls="--", zorder=1)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.yaxis.set_major_formatter(lambda v, _: num(v, 1))
ax.set_xlabel("% de dropbacks con presión", color=GRIS1, size=7.5, labelpad=4)
ax.set_ylabel("EPA por jugada de pase", color=GRIS1, size=7.5, labelpad=2)
q = dict(size=7, color=GRIS2, style="italic")
ax.text(x1 - (x1 - x0) * 0.02, y1 - (y1 - y0) * 0.02, "rinden aunque les lleguen",
        ha="right", va="top", **q)
ax.text(x0 + (x1 - x0) * 0.02, y0 + (y1 - y0) * 0.02, "tiempo de sobra y no rinden",
        ha="left", va="bottom", **q)

for _, r in g.iterrows():
    ax.scatter(r["pres"], r["epa"], s=150, color=FG, alpha=0.10, lw=0, zorder=3)
    im = logo(r["team"], zoom=0.0115)
    if im is not None:
        ax.add_artist(AnnotationBbox(im, (r["pres"], r["epa"]), frameon=False, zorder=4))
off = (y1 - y0) * 0.026
lys = separar(list(xs), [v - off for v in ys], (x1 - x0) * 0.07, (y1 - y0) * 0.025)
for (_, r), ly in zip(g.iterrows(), lys):
    dest = r.name in (mas.name, aguanta.name)
    ax.text(r["pres"], ly, r["ap"], ha="center", va="top", size=7 if dest else 6.3,
            color=FG if dest else GRIS1, weight="bold" if dest else "normal", zorder=5,
            path_effects=[pe.Stroke(linewidth=1.6, foreground=CARD), pe.Normal()])

pie(fig, f"Liga: presión en el {lg_pres:.0f}% de los dropbacks y {num(lg_epa, 3)} EPA por"
         f" jugada de pase. Fuente: Pro Football Reference y nflverse.",
    f"NFL {SEASON} · semanas 1-{SEM}")
guardar(fig, salida(f"qb_presion_sufrida_{SEASON}.png", SEASON))
