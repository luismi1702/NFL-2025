"""
destacados.py
Rastrea una jornada y ORDENA lo que merece post, con sus numeros.

Nace en sep-2026: durante toda la semana 1 los angulos de los posts salieron de
mirar los datos a mano. Esto automatiza esa busqueda para que el redactor del
lunes tenga de donde tirar sin inventar nada.

Cuatro familias, decididas por Luis:
  1. EXTREMOS       — el mejor y el peor de la jornada en cada faceta
  2. CONTRADICCIONES— gano casi todo y perdio, o gano jugando mal
  3. IDENTIDAD      — quien cambio su forma de jugar respecto al ano pasado
  4. JUGADORES      — cuotas y EPA fuera de lo normal

Salida por consola (el batch la vuelca a destacados.txt). NO genera PNG.

Uso:
  python destacados.py [--season N] [--week N]
"""

import numpy as np
import pandas as pd

from pbp_loader import (cargar_pbp, season_cli, week_cli, sello, ultima_semana,
                        orden_partido)

SEASON = season_cli()
MIN_JUGADAS = 20        # por equipo y partido, para no destacar ruido
TOP = 5


# ── METRICAS (mismas definiciones que ficha_tactica.py) ────────────────────────
def _m(s):
    return s.mean() if len(s) else np.nan


def metricas(o):
    o = o[o["play_type"].isin(["pass", "run"]) & o["epa"].notna()]
    for c in ("qb_kneel", "qb_spike"):
        if c in o.columns:
            o = o[o[c] != 1]
    if len(o) < MIN_JUGADAS:
        return None
    es_db = (o["pass_attempt"].fillna(0) > 0) | (o["sack"].fillna(0) > 0) | \
            (o["qb_scramble"].fillna(0) == 1)
    db = o[es_db]
    ru = o[(o["rush_attempt"] == 1) & (o["qb_scramble"] != 1)]
    pas = o[o["pass_attempt"] == 1]
    rz = o[o["yardline_100"] <= 20]
    expl = (pas["yards_gained"].fillna(0) >= 20).sum() + \
           (ru["yards_gained"].fillna(0) >= 10).sum()
    return {
        "epa": _m(o["epa"]), "sr": _m(o["success"]) * 100,
        "exp": expl / len(o) * 100,
        "epa_db": _m(db["epa"]), "sr_db": _m(db["success"]) * 100,
        "sack": db["sack"].fillna(0).sum() / len(db) * 100 if len(db) else np.nan,
        "epa_ru": _m(ru["epa"]), "sr_ru": _m(ru["success"]) * 100,
        "epa_early": _m(o[o["down"] <= 2]["epa"]),
        "sr_late": _m(o[o["down"] >= 3]["success"]) * 100,
        "epa_rz": _m(rz["epa"]),
        "uc": (o["shotgun"] == 0).mean() * 100 if "shotgun" in o.columns else np.nan,
        "n": len(o),
    }


CALIDAD = [("epa", True), ("sr", True), ("exp", True), ("epa_db", True),
           ("sr_db", True), ("sack", False), ("epa_ru", True), ("sr_ru", True),
           ("epa_early", True), ("sr_late", True), ("epa_rz", True)]

ETIQUETA = {"epa": "EPA/jugada", "sr": "jugadas exitosas", "exp": "explosivas",
            "epa_db": "EPA/dropback", "sr_db": "dropbacks exitosos",
            "sack": "sacks encajados", "epa_ru": "EPA/carrera",
            "sr_ru": "carreras exitosas", "epa_early": "EPA en 1er y 2o down",
            "sr_late": "exito en 3er y 4o down", "epa_rz": "EPA en Red Zone"}

ES_PCT = {"sr", "exp", "sr_db", "sack", "sr_ru", "sr_late", "uc"}


def fmt(clave, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.0f}%" if clave in ES_PCT else f"{v:+.2f}"


# ── DATOS ──────────────────────────────────────────────────────────────────────
df, SEASON = cargar_pbp(SEASON, solo_reg=False, avisar=False)
week = week_cli() or ultima_semana(SEASON) or int(df["week"].max())
w = df[(df["week"] == week) & (df["season_type"] == "REG")]
if w.empty:
    raise SystemExit(f"Sin datos de la semana {week} de {SEASON}.")

prev, _ = cargar_pbp(SEASON - 1, avisar=False)

partidos = {}
for gid, g in w.groupby("game_id"):
    away, home = g["away_team"].iloc[0], g["home_team"].iloc[0]
    pts = {home: float(g["total_home_score"].max()),
           away: float(g["total_away_score"].max())}
    m = {t: metricas(g[g["posteam"] == t]) for t in (away, home)}
    if any(v is None for v in m.values()):
        continue
    idx = orden_partido(SEASON, week, away, home)
    partidos[gid] = dict(away=away, home=home, pts=pts, met=m,
                         idx=idx[0] if idx else 99, jugadas=g)

print()
print("=" * 74)
print(f"  DESTACADOS DE LA JORNADA — semana {week} de la NFL {SEASON}")
print(f"  {sello(SEASON)}  ·  {len(partidos)} partidos con datos")
print("=" * 74)
print("  Cada numero de aqui sale del play-by-play de nflverse y es publicable.")

# ── 1. EXTREMOS ────────────────────────────────────────────────────────────────
print("\n" + "-" * 74)
print("  1. EXTREMOS DE LA JORNADA")
print("-" * 74)
filas = []
for gid, p in partidos.items():
    for t in (p["away"], p["home"]):
        rival = p["home"] if t == p["away"] else p["away"]
        fila = dict(equipo=t, rival=rival, gid=gid)
        fila.update(p["met"][t])
        filas.append(fila)
tabla = pd.DataFrame(filas)

for clave, mas_alto_mejor in CALIDAD:
    s = tabla.dropna(subset=[clave])
    if s.empty:
        continue
    mejor = s.loc[s[clave].idxmax() if mas_alto_mejor else s[clave].idxmin()]
    peor = s.loc[s[clave].idxmin() if mas_alto_mejor else s[clave].idxmax()]
    print(f"  {ETIQUETA[clave]:24} mejor {mejor.equipo:3} {fmt(clave, mejor[clave]):>7} "
          f"(vs {mejor.rival})   |   peor {peor.equipo:3} {fmt(clave, peor[clave]):>7} "
          f"(vs {peor.rival})")

# ── 2. CONTRADICCIONES ─────────────────────────────────────────────────────────
print("\n" + "-" * 74)
print("  2. CONTRADICCIONES (lo que el marcador esconde)")
print("-" * 74)
hubo = False
for gid, p in sorted(partidos.items(), key=lambda kv: kv[1]["idx"]):
    a, h = p["away"], p["home"]
    gana_a = sum(1 for c, hb in CALIDAD
                 if not np.isnan(p["met"][a][c]) and not np.isnan(p["met"][h][c])
                 and ((p["met"][a][c] > p["met"][h][c]) == hb))
    total = sum(1 for c, _ in CALIDAD
                if not np.isnan(p["met"][a][c]) and not np.isnan(p["met"][h][c]))
    ganador = a if p["pts"][a] > p["pts"][h] else h
    perdedor = h if ganador == a else a
    facetas_perdedor = gana_a if perdedor == a else total - gana_a
    if facetas_perdedor >= total - 3:
        hubo = True
        print(f"  {perdedor} gano {facetas_perdedor} de {total} facetas y PERDIO "
              f"{p['pts'][perdedor]:.0f}-{p['pts'][ganador]:.0f} ante {ganador}")
    if p["met"][ganador]["epa"] < 0:
        hubo = True
        print(f"  {ganador} gano con el ataque en {fmt('epa', p['met'][ganador]['epa'])} "
              f"EPA/jugada ({p['pts'][ganador]:.0f}-{p['pts'][perdedor]:.0f})")

    # Lo que costaron las perdidas: EPA con y sin ellas
    for t in (a, h):
        o = p["jugadas"]
        o = o[(o["posteam"] == t) & o["play_type"].isin(["pass", "run"]) & o["epa"].notna()]
        to = o[(o["interception"] == 1) | (o["fumble_lost"] == 1)]
        if len(to) >= 2 and len(o) - len(to) >= 20:
            resto = o.drop(to.index)
            if resto["epa"].mean() > 0.10 and to["epa"].sum() < -8:
                hubo = True
                print(f"  {t} jugo a {resto['epa'].mean():+.2f} EPA en {len(resto)} de sus "
                      f"{len(o)} jugadas; las {len(to)} perdidas costaron {to['epa'].sum():+.1f}")
if not hubo:
    print("  Sin contradicciones claras esta jornada.")

# ── 3. IDENTIDAD ───────────────────────────────────────────────────────────────
print("\n" + "-" * 74)
print(f"  3. CAMBIOS DE IDENTIDAD (respecto a su {SEASON - 1})")
print("-" * 74)


def uc_pct(d, week_=None):
    o = d[d["play_type"].isin(["pass", "run"]) & d["epa"].notna() & d["shotgun"].notna()]
    for c in ("qb_kneel", "qb_spike"):
        if c in o.columns:
            o = o[o[c] != 1]
    if week_ is not None:
        o = o[o["week"] == week_]
    tot = o.groupby("posteam").size()
    uc = o[o["shotgun"] == 0].groupby("posteam").size().reindex(tot.index).fillna(0)
    return uc / tot * 100


act, base = uc_pct(w), uc_pct(prev)
cam = (act - base.reindex(act.index)).dropna().sort_values(ascending=False)
print("  Bajo centro, mayores subidas:")
for eq, v in cam.head(TOP).items():
    print(f"    {eq:3} {act[eq]:5.1f}%  (era {base[eq]:5.1f}%)  {v:+5.1f} pp")
print("  Bajo centro, mayores caidas:")
for eq, v in cam.tail(3).items():
    print(f"    {eq:3} {act[eq]:5.1f}%  (era {base[eq]:5.1f}%)  {v:+5.1f} pp")

# Pase profundo: desviacion respecto a su temporada anterior
def deep(d, week_=None):
    o = d[(d["play_type"] == "pass") & d["epa"].notna() & (d["air_yards"] >= 15)]
    if week_ is not None:
        o = o[o["week"] == week_]
    g = o.groupby("posteam")["epa"].agg(["mean", "size"])
    return g[g["size"] >= 4]["mean"], g["size"]


d_act, n_act = deep(w)
d_base, _ = deep(prev)
dif = (d_act - d_base.reindex(d_act.index)).dropna().sort_values(ascending=False)
print("  Pase profundo (15+ yardas de aire), mayor salto y mayor desplome:")
for eq in list(dif.index[:2]) + list(dif.index[-2:]):
    print(f"    {eq:3} {d_act[eq]:+.2f} EPA en {int(n_act[eq])} intentos "
          f"(su {SEASON-1}: {d_base[eq]:+.2f})  {dif[eq]:+.2f}")

# ── 4. JUGADORES ───────────────────────────────────────────────────────────────
print("\n" + "-" * 74)
print("  4. JUGADORES DESATADOS")
print("-" * 74)
pas = w[(w["pass_attempt"] == 1) & w["epa"].notna()]
con = pas[pas["receiver_player_name"].notna()]
obj_eq = con.groupby("posteam").size()
rec = con.groupby(["posteam", "receiver_player_name"]).agg(
    obj=("pass_attempt", "size"), rec=("complete_pass", "sum"),
    yds=("receiving_yards", "sum"), epa=("epa", "sum"),
    adot=("air_yards", "mean")).reset_index()
rec["cuota"] = rec.apply(lambda r: r["obj"] / obj_eq[r["posteam"]] * 100, axis=1)

print("  Mas EPA generado como receptor:")
for _, r in rec.sort_values("epa", ascending=False).head(TOP).iterrows():
    print(f"    {r.posteam:3} {r.receiver_player_name:18} {r.epa:+6.1f} EPA · "
          f"{int(r.rec)}/{int(r.obj)} · {int(r.yds):3d} yd · cuota {r.cuota:.1f}%")
print("  Mayor cuota de objetivos (minimo 6):")
for _, r in rec[rec.obj >= 6].sort_values("cuota", ascending=False).head(TOP).iterrows():
    print(f"    {r.posteam:3} {r.receiver_player_name:18} {r.cuota:.1f}% de los objetivos · "
          f"{int(r.obj)} en total · ADOT {r.adot:.1f} · {r.epa:+.1f} EPA")

corr = w[(w["rush_attempt"] == 1) & w["epa"].notna() &
         w["rusher_player_name"].notna()]
qbs = set(w["passer_player_name"].dropna())
corr = corr[~corr["rusher_player_name"].isin(qbs)]
g = corr.groupby(["posteam", "rusher_player_name"]).agg(
    acar=("rush_attempt", "size"), yds=("rushing_yards", "sum"),
    td=("rush_touchdown", "sum"), epa=("epa", "sum")).reset_index()
print("  Corredores, mejor y peor dia por EPA (minimo 8 acarreos):")
g = g[g.acar >= 8].sort_values("epa", ascending=False)
for _, r in pd.concat([g.head(3), g.tail(2)]).iterrows():
    print(f"    {r.posteam:3} {r.rusher_player_name:18} {r.epa:+6.1f} EPA · "
          f"{int(r.acar)} acarreos · {int(r.yds):3d} yd · {int(r.td)} TD")

print("\n" + "=" * 74)
print("  Recuerda: un dato de una sola jornada no es una tendencia.")
print("=" * 74 + "\n")
