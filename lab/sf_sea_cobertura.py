"""Duelo SF @ SEA (10-oct-2026), tercera vuelta: COMO defiende Seattle y como
ataca San Francisco justo eso. Solo texto.

Como defiende Seattle: tablas de Sharp copiadas a datos_web/ por
copiar_sharp_defensa.py (coberturas, sub package, caja, blitz, presion,
yardas por target segun la posicion del receptor). Semanas 1-4.
Como ataca San Francisco eso: PBP + FTN, semanas 1-4. No hay cobertura jugada
a jugada hasta febrero, asi que el cruce se hace con lo que SI se ve en cada
jugada: si hay blitz (FTN), a quien va el pase (posicion del receptor, de los
rosters), a que zona (pass_location y air_yards) y las yardas tras la
recepcion.

    python lab/sf_sea_cobertura.py
"""
import glob
import os
import sys

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from pbp_loader import cargar_pbp, cargar_ftn, cargar_rosters   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
HASTA = 4
A, D = "SF", "SEA"


def rk(s, t, asc=False):
    s = s.dropna().sort_values(ascending=asc)
    return list(s.index).index(t) + 1 if t in s.index else None


# ───────────────────────── 1. Perfil de Sharp
print("== 1. COMO DEFIENDE SEATTLE (Sharp, puesto 1 = el valor mas alto)")
for f in sorted(glob.glob(os.path.join(RAIZ, "datos_web", f"sharp_*_2026_w{HASTA:02d}.csv"))):
    nombre = os.path.basename(f)[6:-13]
    if nombre == "personal_ataque":
        continue
    d = pd.read_csv(f).set_index("team")
    cols = [c for c in d.columns if c not in ("fuente", "actualizado", "copiado")]
    eq = [A] if nombre in ("linea_ofensiva", "tendencias_ataque") else [D]
    for e in eq:
        print(f"  {nombre:<20} {e}: " + " · ".join(
            f"{c} {d.loc[e, c]} ({rk(d[c], e)}º; media {d[c].mean():.1f})" for c in cols))
    print(f"  {'':<20} actualizado {d.actualizado.iloc[0]}")

# ───────────────────────── PBP + FTN
df, _ = cargar_pbp(2026)
df = df[df.week <= HASTA]
ftn, _ = cargar_ftn(2026)
df = df.merge(ftn[["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_pass_rushers",
                   "is_screen_pass", "is_play_action", "is_qb_out_of_pocket"]],
              left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"],
              how="left")
db = df[(df.qb_dropback == 1) & (df.qb_spike != 1) & df.epa.notna() & df.posteam.notna()].copy()
db["expl"] = (db.yards_gained >= 15).astype(float)
sg = df[df.play_type.isin(["pass", "run"]) & (df.qb_kneel != 1) & (df.qb_spike != 1)]
print(f"\n  (contraste con Sharp) shotgun de {A} en el PBP: "
      f"{sg[sg.posteam == A].shotgun.mean() * 100:.1f}%")


def fila(nombre, x, lado, eq, mejor_alto, minimo=10):
    t = x.groupby(lado).agg(n=("epa", "size"), epa=("epa", "mean"), exito=("success", "mean"),
                            yds=("yards_gained", "mean"))
    if eq not in t.index or t.loc[eq, "n"] < minimo:
        return f"  {nombre:<34} muestra corta ({int(t.loc[eq, 'n']) if eq in t.index else 0})"
    t = t[t.n >= minimo]
    r = t.loc[eq]
    return (f"  {nombre:<34} n={int(r.n):>3} | EPA {r.epa:+.2f} ({rk(t.epa, eq, asc=not mejor_alto)}º) | "
            f"exito {r.exito * 100:4.1f}% ({rk(t.exito, eq, asc=not mejor_alto)}º) | yds {r.yds:4.2f} "
            f"({rk(t.yds, eq, asc=not mejor_alto)}º) | de {len(t)}")


# ───────────────────────── 2. Blitz y numero de rushers (FTN)
print("\n== 2. BLITZ (FTN) — dropbacks")
b = db[db.n_blitzers.notna()].assign(bl=lambda x: x.n_blitzers > 0)
s = b.groupby("defteam").bl.mean()
print(f"  {D} blitzea en el {s[D] * 100:.0f}% de los dropbacks ({rk(s, D)}º; liga {b.bl.mean() * 100:.0f}%)"
      f" | a {A} le blitzean el {b[b.posteam == A].bl.mean() * 100:.0f}% "
      f"({rk(b.groupby('posteam').bl.mean(), A)}º)")
for v, n in ((False, "sin blitz"), (True, "con blitz")):
    x = b[b.bl == v]
    print(fila(f"{A} ataque · {n}", x, "posteam", A, True))
    print(fila(f"{D} defensa · {n}", x, "defteam", D, False))
r4 = db[db.n_pass_rushers.notna()].assign(c=lambda x: x.n_pass_rushers <= 4)
s = r4.groupby("defteam").c.mean()
print(f"  {D} manda 4 o menos en el {s[D] * 100:.0f}% ({rk(s, D)}º; liga {r4.c.mean() * 100:.0f}%)")
x = r4[r4.c]
print(fila(f"{A} ataque · contra 4 o menos", x, "posteam", A, True))
print(fila(f"{D} defensa · con 4 o menos", x, "defteam", D, False))
sk = x.groupby("defteam").sack.mean()
print(f"  {D} % de sack mandando 4 o menos: {sk[D] * 100:.1f}% ({rk(sk, D)}º) | {A} sufre "
      f"{x[x.posteam == A].sack.mean() * 100:.1f}%")

# ───────────────────────── 3. A quien va el pase
print("\n== 3. POSICION DEL RECEPTOR (targets; rosters)")
ro = cargar_rosters(2026)
ro = ro[0] if isinstance(ro, tuple) else ro
idc = "gsis_id" if "gsis_id" in ro.columns else "player_id"
pos = ro.dropna(subset=[idc]).drop_duplicates(idc).set_index(idc).position
tg = db[(db.pass_attempt == 1) & (db.sack != 1) & db.receiver_player_id.notna()].copy()
tg["pos"] = tg.receiver_player_id.map(pos).replace({"FB": "RB"})
print("  sin posicion:", tg.pos.isna().sum(), "de", len(tg), "|", tg.pos.value_counts().to_dict())
tg = tg[tg.pos.isin(["WR", "TE", "RB"])]
cu_a = tg.groupby("posteam").pos.value_counts(normalize=True).unstack(fill_value=0)
cu_d = tg.groupby("defteam").pos.value_counts(normalize=True).unstack(fill_value=0)
for p in ("RB", "TE", "WR"):
    x = tg[tg.pos == p]
    print(f"  [{p}] {A} le lanza el {cu_a.loc[A, p] * 100:.0f}% de sus pases ({rk(cu_a[p], A)}º) | a "
          f"{D} le lanzan ahi el {cu_d.loc[D, p] * 100:.0f}% ({rk(cu_d[p], D)}º) | liga "
          f"{(tg.pos == p).mean() * 100:.0f}%")
    print(fila(f"{A} ataque · pases a {p}", x, "posteam", A, True))
    print(fila(f"{D} defensa · pases a {p}", x, "defteam", D, False))
x = tg[tg.posteam == A]
print(x.groupby(["receiver_player_name", "pos"]).agg(
    n=("epa", "size"), yds=("yards_gained", "sum"), ypt=("yards_gained", "mean"),
    epa=("epa", "sum"), exito=("success", "mean"), ay=("air_yards", "mean"),
    yac=("yards_after_catch", "sum")).sort_values("n", ascending=False).head(9).round(2).to_string())

# ───────────────────────── 4. Zona del campo
print("\n== 4. ZONA DEL PASE (pass_location x profundidad; corto = menos de 10 por el aire)")
z = db[(db.pass_attempt == 1) & (db.sack != 1) & db.air_yards.notna() & db.pass_location.notna()].copy()
z["prof"] = (z.air_yards >= 10).map({True: "largo", False: "corto"})
z["lado"] = z.pass_location.map({"left": "exterior", "right": "exterior", "middle": "centro"})
z["zona"] = z.prof + " " + z.lado
cu_a = z.groupby("posteam").zona.value_counts(normalize=True).unstack(fill_value=0)
cu_d = z.groupby("defteam").zona.value_counts(normalize=True).unstack(fill_value=0)
for v in ("corto centro", "corto exterior", "largo centro", "largo exterior"):
    x = z[z.zona == v]
    print(f"  [{v}] {A} {cu_a.loc[A, v] * 100:.0f}% de sus pases ({rk(cu_a[v], A)}º) | contra {D} "
          f"{cu_d.loc[D, v] * 100:.0f}% ({rk(cu_d[v], D)}º) | liga {(z.zona == v).mean() * 100:.0f}%")
    print(fila(f"{A} ataque · {v}", x, "posteam", A, True, 8))
    print(fila(f"{D} defensa · {v}", x, "defteam", D, False, 8))
    zz = x[x.defteam == D]
    print(f"      {D}: {int(zz.complete_pass.sum())} de {len(zz)} completos, "
          f"{int(zz.interception.sum())} INT, {int(zz.touchdown.sum())} TD")

# ───────────────────────── 5. Yardas tras la recepcion
print("\n== 5. YARDAS TRAS LA RECEPCION (pases completos)")
c = db[db.complete_pass == 1]
ya, yd = c.groupby("posteam").yards_after_catch.mean(), c.groupby("defteam").yards_after_catch.mean()
print(f"  {A} gana {ya[A]:.2f} tras la recepcion ({rk(ya, A)}º) | {D} concede {yd[D]:.2f} "
      f"({rk(yd, D, asc=True)}ª que menos) | liga {c.yards_after_catch.mean():.2f}")
cc = c[c.air_yards < 10]
ya, yd = cc.groupby("posteam").yards_after_catch.mean(), cc.groupby("defteam").yards_after_catch.mean()
print(f"  solo en pase corto: {A} {ya[A]:.2f} ({rk(ya, A)}º) | {D} concede {yd[D]:.2f} "
      f"({rk(yd, D, asc=True)}ª que menos) | liga {cc.yards_after_catch.mean():.2f}")
cuota = c.assign(q=c.yards_after_catch / c.yards_gained.where(c.yards_gained > 0))
ty = c.groupby("posteam").agg(yac=("yards_after_catch", "sum"), y=("yards_gained", "sum"))
ty["q"] = ty.yac / ty.y
print(f"  % de las yardas de pase de {A} que llegan tras la recepcion: {ty.loc[A, 'q'] * 100:.0f}% "
      f"({rk(ty.q, A)}º)")

# ───────────────────────── 6. Los rivales de SF, segun como defienden
print(f"\n== 6. {A} PARTIDO A PARTIDO, con el perfil de cobertura del rival (Sharp)")
cob = pd.read_csv(os.path.join(RAIZ, "datos_web", f"sharp_coberturas_2026_w{HASTA:02d}.csv")).set_index("team")
x = db[db.posteam == A]
g = x.groupby(["week", "defteam"]).agg(n=("epa", "size"), epa=("epa", "mean"),
                                       exito=("success", "mean"), yds=("yards_gained", "mean"),
                                       ay=("air_yards", "mean")).reset_index()
g["zona_rival"] = g.defteam.map(cob.zona_pct)
g["abierto_rival"] = g.defteam.map(cob.medio_abierto_pct)
print(g.round(2).to_string(index=False))
print(f"  {D}: zona {cob.loc[D, 'zona_pct']}, medio abierto {cob.loc[D, 'medio_abierto_pct']}")
x = db[db.defteam == D]
g = x.groupby(["week", "posteam"]).agg(n=("epa", "size"), epa=("epa", "mean"),
                                       exito=("success", "mean"), yds=("yards_gained", "mean"),
                                       ay=("air_yards", "mean"), sacks=("sack", "sum"),
                                       ints=("interception", "sum")).reset_index()
print(f"  {D} defensa, dropbacks por partido")
print(g.round(2).to_string(index=False))
