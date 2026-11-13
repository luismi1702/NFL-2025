"""
qb_situaciones.py
Los QBs cuando aprieta: EPA por jugada de pase de cada QB en global, en
3er down, en red zone y contra el blitz, en una tabla con su puesto.
Mezcla de QBsTotalEPA.py (red zone y 3er down) con el blitz de FTN.

Jugada de pase = `pass == 1` (dropback: sacks y scrambles incluidos),
atribuida a `passer_id`. Red zone = jugada desde la 19 hacia dentro
(`yardline_100 < 20`, la misma frontera que cruza con ESPN). Blitz =
`n_blitzers > 0` de FTN. Solo semanas con FTN charteado entero (se cuentan
los partidos con `n_pass_rushers > 0`, ver CLAUDE.md), para que las cuatro
columnas salgan del mismo periodo.
Una celda con menos de MIN_CELDA jugadas sale en gris y no entra en el
ranking de esa columna. 13-nov-2026.
"""

import pandas as pd
from matplotlib.patches import FancyBboxPatch

from pbp_loader import cargar_pbp, cargar_ftn, salida, season_cli, sello
from estilo import *

SEASON = season_cli()
MIN_JUGADAS = 150
MIN_CELDA = 25
COLS = [("all", "Global"), ("3d", "3er down"), ("rz", "Red zone"), ("bl", "Contra blitz")]

# ── Datos
df, SEASON = cargar_pbp(SEASON)
ftn = cargar_ftn(SEASON)
ftn = ftn[0] if isinstance(ftn, tuple) else ftn
pbp_p = df.groupby("week")["game_id"].nunique()
ftn_p = ftn[ftn["n_pass_rushers"] > 0].groupby("week")["nflverse_game_id"].nunique()
completas = [w for w in pbp_p.index if ftn_p.get(w, 0) >= pbp_p[w]]
SEM = max(w for w in completas if all(x in completas for x in range(1, w + 1)))
if SEM < pbp_p.index.max():
    print(f"FTN completo hasta la sem. {SEM}: se corta ahí (PBP llega a la {pbp_p.index.max()})")

f = ftn[["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_pass_rushers"]].rename(
    columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
f["play_id"] = pd.to_numeric(f["play_id"], errors="coerce")
df = df[df["week"] <= SEM].copy()
df["play_id"] = pd.to_numeric(df["play_id"], errors="coerce")
for c in ["epa", "pass", "down", "yardline_100"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
d = df[(df["pass"] == 1) & df["epa"].notna() & df["passer_id"].notna()].merge(
    f, on=["game_id", "play_id"], how="left")
print(f"Pases sin FTN: {d['n_blitzers'].isna().mean():.1%}")

mask = {"all": pd.Series(True, index=d.index), "3d": d["down"] == 3,
        "rz": d["yardline_100"] < 20, "bl": d["n_blitzers"] > 0}
g = d.groupby("passer_id").agg(nombre=("passer", lambda s: s.value_counts().index[0]),
                               team=("posteam", lambda s: s.value_counts().index[0]),
                               n=("epa", "size"))
liga = {}
for k, m in mask.items():
    x = d[m].groupby("passer_id")["epa"].agg(["mean", "size"])
    g[k], g["n_" + k] = x["mean"], x["size"]
    liga[k] = d[m]["epa"].mean()
g = g[g["n"] >= MIN_JUGADAS].copy()
for k, _ in COLS:
    valido = g["n_" + k].fillna(0) >= MIN_CELDA
    g["rk_" + k] = g.loc[valido, k].rank(ascending=False, method="min")
    g["N_" + k] = int(valido.sum())
g = g.sort_values("all", ascending=False)
g["ap"] = g["nombre"].map(lambda s: s.split(".", 1)[1] if "." in s[:3] else s)
print(g.drop(columns=["nombre"]).round(3).to_string())
print("liga:", {k: round(v, 3) for k, v in liga.items()})

# ── Titular automatico: el mejor de cada situacion y quien es top 10 en todas
mejor = {k: g.sort_values("rk_" + k).iloc[0]["ap"] for k, _ in COLS[1:]}
todo = g[(g[["rk_" + k for k, _ in COLS]] <= 10).all(axis=1)]["ap"].tolist()
partes = []
if todo:
    quien = " y ".join(todo) if len(todo) <= 2 else ", ".join(todo[:-1]) + " y " + todo[-1]
    partes.append(f"{quien}, top 10 en todo.")
partes.append(f"{mejor['3d']} manda en 3er down y {mejor['bl']}, contra el blitz")
tit = " ".join(partes)

# ── Figura
fig = lienzo()
antetitulo(fig, f"Quarterbacks cuando aprieta · NFL {SEASON}")
y = titular(fig, tit, y=0.93, size=19)
y = contexto(fig, [("EPA por jugada de pase", "fuerte"),
                   (f"(sacks y scrambles incluidos) y su puesto. Mín. {MIN_JUGADAS} jugadas;"
                    f" en gris, las celdas de menos de {MIN_CELDA}.", None)],
             y=y - 0.01)

top = y - 0.035
X_LOGO, X_NOM = L + 0.015, L + 0.045
XC = [0.47, 0.60, 0.73, 0.86]
WC = 0.115
fila = (top - 0.105) / len(g)
tarjeta(fig, 0.085, top + 0.03)
for (k, nom), xc in zip(COLS, XC):
    etiqueta(fig, nom, xc, top + 0.012, ha="center")

for i, (_, r) in enumerate(g.iterrows()):
    yy = top - (i + 0.5) * fila
    poner_logo(fig, r["team"], X_LOGO, yy, zoom=0.0105)
    fig.text(X_NOM, yy, r["ap"], family=TXT, size=7.8, color=FG, va="center")
    for (k, _), xc in zip(COLS, XC):
        v, rk, n = r[k], r["rk_" + k], r["N_" + k]
        if pd.isna(rk):
            col, txt, tc = CARD, (num(v, 2) if pd.notna(v) else "–"), GRIS2
        else:
            col = color_rango(int(rk), n)
            col = col if col != GRIS1 else NEUTRO
            txt = f"{num(v, 2)}  ·  {int(rk)}º"
            tc = BG if col in RYG else FG
        fig.add_artist(FancyBboxPatch((xc - WC / 2, yy - fila * 0.40), WC, fila * 0.80,
                                      boxstyle="round,pad=0,rounding_size=0.004",
                                      transform=fig.transFigure, fc=col, ec="none"))
        fig.text(xc, yy, txt, family=TXT, size=6.2, color=tc, va="center", ha="center",
                 weight="semibold" if col in RYG else "normal")

pie(fig, f"Liga: global {num(liga['all'])}, 3er down {num(liga['3d'])}, red zone "
         f"{num(liga['rz'])}, contra blitz {num(liga['bl'])}. Red zone: desde la 19."
         f" Blitz: FTN. Fuente: nflverse.",
    sello(SEASON).split("·")[0].strip() + f" · semanas 1-{SEM}")
guardar(fig, salida(f"qb_situaciones_{SEASON}.png", SEASON))
