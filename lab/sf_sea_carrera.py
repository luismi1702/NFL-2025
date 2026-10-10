"""Numeros del duelo SF @ SEA (sabado 10-oct-2026), segunda vuelta: la carrera
de San Francisco contra la defensa de carrera de Seattle, y el cruce por
personal ofensivo. Solo texto: imprime lo que luego va al hilo.

Definiciones:
- Carrera = carrera disenada: play_type == "run", sin scrambles ni rodillas.
- Exito = EPA > 0 · explosiva = 10+ yardas · frenada = 0 yardas o menos.
- Congelado en las semanas 1-4 (ni SF ni SEA han jugado la 5; el TNF moveria
  la media y los puestos de la liga).
- Caja (n_defense_box), movimiento y play action: charting de FTN, 16 partidos
  de 16 en cada una de las semanas 1-4.
- Personal: fichero jugada a jugada de Sharp (semanas 1-4, sin rodillas ni
  spikes; EPA de nflverse redondeado a decimas) y tablas de SumerSports.

    python lab/sf_sea_carrera.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pbp_loader import cargar_pbp, cargar_ftn, cargar_calendario, cargar_web   # noqa: E402
import copiar_personal_web as cpw                                             # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
HASTA = 4
A, D = "SF", "SEA"


def rk(s, t, asc=False):
    s = s.dropna().sort_values(ascending=asc)
    return list(s.index).index(t) + 1 if t in s.index else None


def tabla(x, por):
    """n, EPA, exito, yardas, explosivas y frenadas por grupo."""
    return x.groupby(por).agg(n=("epa", "size"), epa=("epa", "mean"), exito=("success", "mean"),
                              yds=("yards_gained", "mean"), expl=("expl", "mean"),
                              fren=("fren", "mean"))


def linea(nombre, x, lado, equipo, mejor_alto, minimo=0):
    """Una fila con el valor del equipo y su puesto en cada metrica."""
    t = tabla(x, lado)
    t = t[t.n >= minimo]
    if equipo not in t.index:
        return f"  {nombre:<28} sin muestra"
    r = t.loc[equipo]
    # mejor_alto: ataque (mas es mejor). En defensa, 1º = el que menos concede.
    p = {c: rk(t[c], equipo, asc=not mejor_alto) for c in ("epa", "exito", "yds", "expl")}
    p["fren"] = rk(t.fren, equipo, asc=mejor_alto)
    return (f"  {nombre:<28} n={int(r.n):>3} | EPA {r.epa:+.3f} ({p['epa']}º) | exito {r.exito * 100:4.1f}% "
            f"({p['exito']}º) | yds {r.yds:4.2f} ({p['yds']}º) | 10+ {r.expl * 100:4.1f}% ({p['expl']}º) | "
            f"<=0 {r.fren * 100:4.1f}% ({p['fren']}º) | de {len(t)}")


df, _ = cargar_pbp(2026)
df = df[df.week <= HASTA]
ftn, _ = cargar_ftn(2026)
ftn = ftn[ftn.week <= HASTA]
print("FTN partidos por semana:", ftn.groupby("week").nflverse_game_id.nunique().to_dict())
df = df.merge(ftn[["nflverse_game_id", "nflverse_play_id", "qb_location", "n_offense_backfield",
                   "n_defense_box", "is_motion", "is_play_action", "is_rpo", "is_qb_sneak"]],
              left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"],
              how="left")

jug = df[df.play_type.isin(["pass", "run"]) & df.epa.notna() & df.posteam.notna()
         & (df.qb_kneel != 1) & (df.qb_spike != 1)].copy()
jug["expl"] = (jug.yards_gained >= 10).astype(float)
jug["fren"] = (jug.yards_gained <= 0).astype(float)
car = jug[(jug.play_type == "run") & (jug.qb_scramble != 1)].copy()
print(f"Carreras disenadas semanas 1-{HASTA}: {len(car)} | con caja FTN: {car.n_defense_box.notna().sum()}")

# ───────────────────────── 1. Carrera: global
print("\n== 1. CARRERA GLOBAL (liga: EPA %+.3f, exito %.1f%%, yds %.2f, 10+ %.1f%%, <=0 %.1f%%)" % (
    car.epa.mean(), car.success.mean() * 100, car.yards_gained.mean(), car.expl.mean() * 100,
    car.fren.mean() * 100))
print(linea(f"{A} ataque", car, "posteam", A, True))
print(linea(f"{D} defensa", car, "defteam", D, False))
tot = jug.groupby("posteam").size()
print(f"  {A} % de carrera sobre sus jugadas: {(car.posteam == A).sum() / tot[A] * 100:.1f}% "
      f"({rk(car.groupby('posteam').size() / tot, A)}º)")
neu = jug[(jug.down.isin([1, 2])) & (jug.wp.between(0.2, 0.8)) & (jug.half_seconds_remaining > 120)]
nr = neu.assign(c=(neu.play_type == "run") & (neu.qb_scramble != 1))
print(f"  {A} % de carrera en 1º-2º down neutro: {nr[nr.posteam == A].c.mean() * 100:.1f}% "
      f"({rk(nr.groupby('posteam').c.mean(), A)}º) | rivales de {D}: "
      f"{nr[nr.defteam == D].c.mean() * 100:.1f}% ({rk(nr.groupby('defteam').c.mean(), D)}º que mas)")

# ───────────────────────── 2. Por partido
print("\n== 2. PARTIDO A PARTIDO")
for eq, lado, otro in ((A, "posteam", "defteam"), (D, "defteam", "posteam")):
    x = car[car[lado] == eq]
    g = x.groupby(["week", otro]).agg(n=("epa", "size"), yds=("yards_gained", "sum"),
                                      ypc=("yards_gained", "mean"), epa=("epa", "mean"),
                                      exito=("success", "mean"), expl=("expl", "sum"),
                                      td=("touchdown", "sum"))
    print(f"  {eq} ({'ataque' if lado == 'posteam' else 'defensa'})")
    print(g.round(3).to_string())

# ───────────────────────── 3. Corredores
print(f"\n== 3. CORREDORES DE {A} y los que ha sufrido {D}")
x = car[car.posteam == A]
print(tabla(x, "rusher_player_name").sort_values("n", ascending=False).round(3).to_string())
x = car[car.defteam == D]
print(tabla(x, ["posteam", "rusher_player_name"]).sort_values("n", ascending=False).head(10)
      .round(3).to_string())

# ───────────────────────── 4. Formacion
print("\n== 4. BAJO CENTRO / SHOTGUN (PBP)")
car["form"] = car.shotgun.map({1: "shotgun", 0: "bajo centro"})
for f in ("bajo centro", "shotgun"):
    x = car[car.form == f]
    print(linea(f"{A} ataque · {f}", x, "posteam", A, True, 15))
    print(linea(f"{D} defensa · {f}", x, "defteam", D, False, 15))
uc = car.assign(u=car.shotgun == 0)
print(f"  % de carreras bajo centro: {A} {uc[uc.posteam == A].u.mean() * 100:.0f}% "
      f"({rk(uc.groupby('posteam').u.mean(), A)}º) | recibidas por {D} "
      f"{uc[uc.defteam == D].u.mean() * 100:.0f}% | liga {uc.u.mean() * 100:.0f}%")

# ───────────────────────── 5. Direccion y hueco
print("\n== 5. DIRECCION (run_location) Y HUECO (run_gap)")
car["hueco"] = car.run_location.fillna("?") + " " + car.run_gap.fillna("")
car.loc[car.run_location == "middle", "hueco"] = "middle"
car["fuera"] = car.run_gap.isin(["end"]).map({True: "exterior (end)", False: "interior"})
for col in ("run_location", "fuera", "hueco"):
    print(f"  -- {col}")
    for v in sorted(car[col].dropna().unique()):
        x = car[car[col] == v]
        if (x.posteam == A).sum() < 8:
            continue
        print(linea(f"{A} · {v}", x, "posteam", A, True, 8))
        print(linea(f"{D} def · {v}", x, "defteam", D, False, 8))

# ───────────────────────── 6. Down
print("\n== 6. POR DOWN")
car["sit"] = "2º"
car.loc[car.down == 1, "sit"] = "1º"
car.loc[car.down >= 3, "sit"] = "3º-4º"
for v in ("1º", "2º", "3º-4º"):
    x = car[car.sit == v]
    print(linea(f"{A} · {v} down", x, "posteam", A, True, 8))
    print(linea(f"{D} def · {v} down", x, "defteam", D, False, 8))
x = car[(car.down >= 3) & (car.ydstogo <= 2)]
for eq, lado in ((A, "posteam"), (D, "defteam")):
    z = x[x[lado] == eq]
    print(f"  {eq} 3º-4º y 1-2 por carrera: {int(z.first_down.sum())} de {len(z)} convertidos")

# ───────────────────────── 7. Caja (FTN)
print("\n== 7. CAJA (FTN n_defense_box)")
cj = car[car.n_defense_box.notna()].copy()
cj["caja"] = pd.cut(cj.n_defense_box, [0, 6, 7, 20], labels=["6 o menos", "7", "8 o mas"])
print(f"  Caja media: contra {A} {cj[cj.posteam == A].n_defense_box.mean():.2f} "
      f"({rk(cj.groupby('posteam').n_defense_box.mean(), A)}º mas cargada) | la que pone {D} "
      f"{cj[cj.defteam == D].n_defense_box.mean():.2f} "
      f"({rk(cj.groupby('defteam').n_defense_box.mean(), D)}º mas cargada) | liga "
      f"{cj.n_defense_box.mean():.2f}")
c8 = cj.assign(o=cj.n_defense_box >= 8)
print(f"  % de carreras con 8+ en la caja: contra {A} {c8[c8.posteam == A].o.mean() * 100:.0f}% "
      f"({rk(c8.groupby('posteam').o.mean(), A)}º) | {D} {c8[c8.defteam == D].o.mean() * 100:.0f}% "
      f"({rk(c8.groupby('defteam').o.mean(), D)}º) | liga {c8.o.mean() * 100:.0f}%")
c6 = cj.assign(o=cj.n_defense_box <= 6)
print(f"  % de carreras con 6 o menos: contra {A} {c6[c6.posteam == A].o.mean() * 100:.0f}% "
      f"({rk(c6.groupby('posteam').o.mean(), A)}º) | {D} {c6[c6.defteam == D].o.mean() * 100:.0f}% "
      f"({rk(c6.groupby('defteam').o.mean(), D)}º) | liga {c6.o.mean() * 100:.0f}%")
for v in ("6 o menos", "7", "8 o mas"):
    x = cj[cj.caja == v]
    print(linea(f"{A} · caja {v}", x, "posteam", A, True, 10))
    print(linea(f"{D} def · caja {v}", x, "defteam", D, False, 10))
# Caja en todas las jugadas (tambien pase): cuanto carga Seattle
tj = jug[jug.n_defense_box.notna()]
print(f"  Caja media en TODAS las jugadas: {D} {tj[tj.defteam == D].n_defense_box.mean():.2f} "
      f"({rk(tj.groupby('defteam').n_defense_box.mean(), D)}º mas cargada) | contra {A} "
      f"{tj[tj.posteam == A].n_defense_box.mean():.2f} "
      f"({rk(tj.groupby('posteam').n_defense_box.mean(), A)}º) | liga {tj.n_defense_box.mean():.2f}")

# ───────────────────────── 8. Movimiento y backfield (FTN)
print("\n== 8. MOVIMIENTO Y BACKFIELD EN CARRERA (FTN)")
mv = cj.assign(m=cj.is_motion.astype(float))
print(f"  % de carreras con movimiento: {A} {mv[mv.posteam == A].m.mean() * 100:.0f}% "
      f"({rk(mv.groupby('posteam').m.mean(), A)}º) | liga {mv.m.mean() * 100:.0f}%")
for v, n in ((1, "con movimiento"), (0, "sin movimiento")):
    x = mv[mv.m == v]
    print(linea(f"{A} · {n}", x, "posteam", A, True, 10))
    print(linea(f"{D} def · {n}", x, "defteam", D, False, 10))
for v in sorted(cj.n_offense_backfield.dropna().unique()):
    x = cj[cj.n_offense_backfield == v]
    if (x.posteam == A).sum() >= 10:
        print(linea(f"{A} · {int(v)} en backfield", x, "posteam", A, True, 10))
        print(linea(f"{D} def · {int(v)} en backfield", x, "defteam", D, False, 10))

# ───────────────────────── 9. Play action (FTN)
print("\n== 9. PLAY ACTION (FTN) — dropbacks")
db = jug[(jug.qb_dropback == 1) & jug.is_play_action.notna()].copy()
db["pa"] = db.is_play_action.astype(float)
print(f"  % de dropbacks con play action: {A} {db[db.posteam == A].pa.mean() * 100:.0f}% "
      f"({rk(db.groupby('posteam').pa.mean(), A)}º) | contra {D} "
      f"{db[db.defteam == D].pa.mean() * 100:.0f}% ({rk(db.groupby('defteam').pa.mean(), D)}º) | "
      f"liga {db.pa.mean() * 100:.0f}%")
for v, n in ((1, "con play action"), (0, "sin play action")):
    x = db[db.pa == v]
    print(linea(f"{A} · {n}", x, "posteam", A, True, 10))
    print(linea(f"{D} def · {n}", x, "defteam", D, False, 10))

# ───────────────────────── 10. Personal (Sharp jugada a jugada)
print("\n== 10. PERSONAL — Sharp jugada a jugada")
raw = json.loads(cpw.bajar(cpw.SHARP.format(season=2026)))
inv = {ch: i for i, ch in enumerate(raw["alphabet"])}
pl, generado = cpw.sharp_jugadas(2026)
for k in ("q", "d", "n", "z", "b"):
    pl[k] = [inv[ch] for ch in raw["cols"][k]]
print(f"  generado {generado} · semanas {sorted(pl.week.unique())} · {len(pl)} jugadas")
print("  valores de b:", pl.b.value_counts().to_dict(), "| d:", pl.d.value_counts().to_dict(),
      "| z:", pl.z.value_counts().to_dict())
cal, _ = cargar_calendario(2026)
cal = cal[cal.game_type == "REG"]
riv = {}
for _, g in cal.iterrows():
    riv[(int(g.week), g.home_team)] = g.away_team
    riv[(int(g.week), g.away_team)] = g.home_team
pl["rival"] = [riv.get((w, t)) for w, t in zip(pl.week, pl.team)]
pl = pl[pl.week <= HASTA]
# b: comprobar contra el PBP si separa pase de carrera
pbp_pase = jug.assign(p=(jug.play_type == "pass") | (jug.qb_scramble == 1)).groupby("posteam").p.mean()
for v in sorted(pl.b.unique()):
    s = pl.assign(o=pl.b == v).groupby("team").o.mean()
    print(f"  b=={v}: {A} {s[A] * 100:.1f}% · {D} {s[D] * 100:.1f}% | % de pase en el PBP: "
          f"{A} {pbp_pase[A] * 100:.1f}% · {D} {pbp_pase[D] * 100:.1f}% | corr con PBP "
          f"{s.corr(pbp_pase):+.2f}")

# b == 1 es pase (dropbacks, con scrambles): cuadra al 100% con el PBP
pl["pase"] = pl.b == 1
pl["tipo"] = pl.pase.map({True: "pase", False: "carrera"})
G = [11, 12, 13, 21, 22]
MIN = 15


def fila_personal(x, lado, eq, mejor_alto, minimo=MIN):
    t = x.groupby(lado).agg(n=("epa", "size"), epa=("epa", "mean"), exito=("exito", "mean"),
                            yds=("yds", "mean"))
    if eq not in t.index:
        return "sin jugadas"
    r = t.loc[eq]
    t = t[t.n >= minimo]
    if eq not in t.index:
        return f"n={int(r.n):>3} | EPA {r.epa:+.2f} | exito {r.exito * 100:4.1f}% | yds {r.yds:4.2f} (muestra corta)"
    return (f"n={int(r.n):>3} | EPA {r.epa:+.2f} ({rk(t.epa, eq, asc=not mejor_alto)}º) | exito "
            f"{r.exito * 100:4.1f}% ({rk(t.exito, eq, asc=not mejor_alto)}º) | yds {r.yds:4.2f} "
            f"({rk(t.yds, eq, asc=not mejor_alto)}º) | de {len(t)}")


def cruce(at, de):
    print(f"\n  ---- ATAQUE {at} contra DEFENSA {de}")
    na, nd = (pl.team == at).sum(), (pl.rival == de).sum()
    uso_a = pl.groupby("team").grupo.value_counts(normalize=True).unstack(fill_value=0)
    uso_d = pl.groupby("rival").grupo.value_counts(normalize=True).unstack(fill_value=0)
    for g in G:
        x = pl[pl.grupo == g]
        xa, xd = x[x.team == at], x[x.rival == de]
        print(f"  [{g}] liga: {len(x) / len(pl) * 100:.0f}% de uso, EPA {x.epa.mean():+.2f}, exito "
              f"{x.exito.mean() * 100:.0f}%, pase {x.pase.mean() * 100:.0f}% | carrera: EPA "
              f"{x[~x.pase].epa.mean():+.2f}, exito {x[~x.pase].exito.mean() * 100:.0f}%, "
              f"{x[~x.pase].yds.mean():.1f} yds | pase: EPA {x[x.pase].epa.mean():+.2f}, exito "
              f"{x[x.pase].exito.mean() * 100:.0f}%")
        print(f"     {at} ataque : uso {len(xa) / na * 100:4.1f}% ({rk(uso_a[g], at)}º), pase "
              f"{xa.pase.mean() * 100 if len(xa) else 0:.0f}% | {fila_personal(x, 'team', at, True)}")
        for tp in ("carrera", "pase"):
            print(f"        {tp:<8}: {fila_personal(x[x.tipo == tp], 'team', at, True, 10)}")
        print(f"     {de} defensa: recibe {len(xd) / nd * 100:4.1f}% ({rk(uso_d[g], de)}º), pase "
              f"{xd.pase.mean() * 100 if len(xd) else 0:.0f}% | {fila_personal(x, 'rival', de, False)}")
        for tp in ("carrera", "pase"):
            print(f"        {tp:<8}: {fila_personal(x[x.tipo == tp], 'rival', de, False, 10)}")
        if len(xd):
            pp = xd.groupby(["week", "team"]).agg(n=("epa", "size"), epa=("epa", "mean"),
                                                  exito=("exito", "mean"), yds=("yds", "mean"))
            print("        por partido: " + " · ".join(
                f"sem {w} {t}: {int(r.n)} jug, {r.exito * 100:.0f}% exito, {r.yds:.1f} yds"
                for (w, t), r in pp.iterrows()))


cruce(A, D)
cruce(D, A)

# Agrupado: un TE o menos de 3 WR ("pesado" = 12, 13, 21, 22) frente al 11
pl["pesado"] = pl.grupo.isin([12, 13, 21, 22])
pl["dos_rb"] = pl.grupo.isin([21, 22])
print("\n  ---- AGRUPADO")
for nombre, m in (("11", pl.grupo == 11), ("pesado (12,13,21,22)", pl.pesado), ("dos RB (21,22)", pl.dos_rb)):
    x = pl[m]
    print(f"  [{nombre}] liga EPA {x.epa.mean():+.2f}, exito {x.exito.mean() * 100:.0f}%")
    for tp, y in (("todo", x), ("carrera", x[~x.pase]), ("pase", x[x.pase])):
        print(f"     {tp:<8} {A} at: {fila_personal(y, 'team', A, True)}")
        print(f"     {tp:<8} {D} def: {fila_personal(y, 'rival', D, False)}")
u = pl.groupby("team").pesado.mean()
print(f"  % de jugadas en personal pesado: {A} {u[A] * 100:.0f}% ({rk(u, A)}º) | liga {pl.pesado.mean() * 100:.0f}%")
u = pl.groupby("team").dos_rb.mean()
print(f"  % de jugadas con dos RB (21+22): {A} {u[A] * 100:.0f}% ({rk(u, A)}º) | {D} {u[D] * 100:.0f}% ({rk(u, D)}º) | liga {pl.dos_rb.mean() * 100:.0f}%")
u = pl.groupby("rival").dos_rb.mean()
print(f"  % de jugadas con dos RB recibidas: {D} {u[D] * 100:.0f}% ({rk(u, D)}º) | {A} {u[A] * 100:.0f}%")

# SumerSports (EPA propio, hasta la semana 5: el TNF solo toca a TB y DAL)
print("\n== 11. PERSONAL — SumerSports (contraste)")
for lado, eq in (("ataque", A), ("defensa", D), ("ataque", D), ("defensa", A)):
    w = cargar_web(f"sumersports_personal_{lado}", 2026)
    w = w[0] if isinstance(w, tuple) else w
    x = w[(w.team == eq) & w.personal.astype(str).isin([str(g) for g in G])]
    print(f"  {eq} {lado} · actualizado {x.actualizado.iloc[0]}")
    print(x[["personal", "jugadas", "uso_pct", "uso_liga_pct", "epa_jugada", "epa_rk", "epa_pase",
             "epa_carrera", "pase_pct", "epa_liga"]].sort_values("jugadas", ascending=False).to_string(index=False))
