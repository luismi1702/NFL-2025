# radiografia_equipo.py
# Radiografía de un equipo en cuatro PNG (14-oct-2026, hilo de Jacksonville):
#   1_facetas     su puesto en cada faceta (ataque, defensa, global)
#   2_defensa     cómo defiende: coberturas, paquetes, caja y blitz (Sharp + FTN)
#   3_ataque      cómo ataca: personal y formaciones (SumerSports)
#   4_reparto     a quién lanza su QB (targets y EPA por target)
#
#   python radiografia_equipo.py --team JAX
#
# Personal, formaciones y coberturas no están en nflverse en temporada: salen
# de datos_web/ (copiar_personal_web.py y las tablas de Sharp/SumerSports
# copiadas a mano con su fuente y hora). El post sale de los mismos CSV.

import argparse
import glob
import os

import numpy as np
import pandas as pd

from estilo import *
from pbp_loader import cargar_pbp, cargar_ftn, cargar_calendario, salida, sello, DatosNoDisponibles

ap = argparse.ArgumentParser()
ap.add_argument("--team", default="JAX")
ap.add_argument("--season", type=int, default=None)
args = ap.parse_args()
T = args.team

df, SEASON = cargar_pbp(args.season)
for c in ("epa", "week", "yards_gained", "air_yards", "down", "yardline_100", "qtr"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
SEM = int(df.week.max())
NOMBRE = nombre_equipo(T)
pl = df[df.play_type.isin(["pass", "run"]) & df.epa.notna()].copy()
pas = pl[pl.play_type == "pass"]
run = pl[(pl.play_type == "run") & (pl.qb_scramble != 1)]


def puesto(serie, menor_mejor=False):
    r = serie.rank(ascending=menor_mejor, method="min")
    return serie[T], int(r[T]), len(serie)


def ultimo(patron):
    fs = sorted(glob.glob(os.path.join("datos_web", patron)))
    return pd.read_csv(fs[-1]) if fs else None


def cab(fig, sub):
    cabecera(fig, T, (NOMBRE.upper(), sub))
    antetitulo(fig, f"Radiografía · semanas 1-{SEM}", y=Y_BAJO_CABECERA)


# ── Datos comunes
gp = pd.concat([df[["posteam", "game_id"]].rename(columns={"posteam": "t"}),
                df[["defteam", "game_id"]].rename(columns={"defteam": "t"})]).dropna() \
    .groupby("t").game_id.nunique()
cal, _ = cargar_calendario(SEASON)
cal = cal[(cal.season == SEASON) & (cal.game_type == "REG")].dropna(subset=["home_score"])
dif = {}
for _, r in cal.iterrows():
    dif[r.home_team] = dif.get(r.home_team, 0) + r.home_score - r.away_score
    dif[r.away_team] = dif.get(r.away_team, 0) + r.away_score - r.home_score
dif = pd.Series(dif)
w = int(((cal.home_team == T) & (cal.home_score > cal.away_score)).sum()
        + ((cal.away_team == T) & (cal.away_score > cal.home_score)).sum())
l_ = int(((cal.home_team == T) & (cal.home_score < cal.away_score)).sum()
         + ((cal.away_team == T) & (cal.away_score < cal.home_score)).sum())
RECORD = f"{w}-{l_}"


# Pérdida = INT + fumble perdido que suelta el propio ataque (un muff del
# rival recuperado por el ataque no lo es: Seattle, 13-oct-2026)
_perd = ((df.interception == 1) | ((df.fumble_lost == 1) & (df.fumbled_1_team == df.posteam))).astype(int)
per = _perd.groupby(df.posteam).sum()
forz = _perd.groupby(df.defteam).sum()
drv = df[df.drive.notna() & df.defteam.notna()].groupby(["defteam", "game_id", "drive"]).agg(
    rz=("yardline_100", lambda s: (s <= 20).any()), td=("touchdown", lambda s: (s == 1).any()))
rz_def = drv[drv.rz].groupby(level=0).td.mean()
st = df[df.play_type.isin(["kickoff", "punt", "field_goal", "extra_point"]) & df.epa.notna()]
st_net = (st.groupby("posteam").epa.sum().sub(st.groupby("defteam").epa.sum(), fill_value=0)) / gp
pen = df[(df.penalty == 1) & df.penalty_team.notna()].groupby("penalty_team").size() / gp

# ═══ 1. FACETAS ═══════════════════════════════════════════════════════════════
filas = [
    ("GLOBAL", None),
    ("Diferencial de puntos", puesto(dif), lambda v: f"{v:+.0f}"),
    ("Equipos especiales (EPA/partido)", puesto(st_net), num),
    ("Penalties por partido", puesto(pen, True), lambda v: num(v, 1, False)),
    ("ATAQUE", None),
    ("EPA por jugada", puesto(pl.groupby("posteam").epa.mean()), num),
    ("EPA por pase", puesto(pas.groupby("posteam").epa.mean()), num),
    ("EPA por carrera", puesto(run.groupby("posteam").epa.mean()), num),
    ("Pérdidas de balón", puesto(per, True), lambda v: f"{v:.0f}"),
    ("DEFENSA", None),
    ("EPA por jugada permitido", puesto(pl.groupby("defteam").epa.mean(), True), num),
    ("EPA por carrera permitido", puesto(run.groupby("defteam").epa.mean(), True), num),
    ("EPA por pase permitido", puesto(pas.groupby("defteam").epa.mean(), True), num),
    ("TD permitidos en red zone", puesto(rz_def, True), lambda v: pct(v)),
    ("Turnovers forzados", puesto(forz), lambda v: f"{v:.0f}"),
]
datos = [f for f in filas if f[1] is not None]
top10 = sum(1 for f in datos if f[1][1] <= 10)
fuera = [f[0].split(" (")[0].lower() for f in datos if f[1][1] > 10]

fig = lienzo()
cab(fig, f"{RECORD} · CÓMO ES")
fuera_rk = [f"{f[0].split(' (')[0].lower()} ({f[1][1]}º)" for f in datos if f[1][1] > 10]
tit = (f"Top 10 en {top10} de las {len(datos)} facetas. "
       + (f"Fuera: {' y '.join(fuera_rk)}." if fuera_rk else "En todas."))
y = titular(fig, tit, y=0.82, size=18.5) - 0.025
y0 = y
for f in filas:
    if f[1] is None:
        etiqueta(fig, f[0], L + 0.01, y - 0.008)
        linea_h(fig, y - 0.022)
        y -= 0.040
        continue
    lab, (v, rk, n), fmt = f
    fig.text(L + 0.01, y, lab, family=TXT, size=8.2, color=FG, va="center")
    fig.text(0.62, y, fmt(v), family=DISP, weight="bold", size=12, color=FG, va="center",
             ha="right")
    pista32(fig, 0.655, y - 0.0045, 0.20, rk, n)
    fig.text(R - 0.005, y, f"{rk}º", family=TXT, weight="bold", size=8.5, color=color_rango(rk, n),
             va="center", ha="right")
    y -= 0.038
tarjeta(fig, y + 0.01, y0 + 0.02)
pie(fig, "EPA: puntos esperados añadidos. Pérdidas: INT + fumbles perdidos. "
         "Red zone: % de series en la 20 que acaban en TD.", sello(SEASON))
guardar(fig, salida(f"radiografia_{T}_1_facetas_{SEASON}.png", SEASON))

# ═══ 2. DEFENSA ═══════════════════════════════════════════════════════════════
cob = ultimo("sharp_coberturas_*.csv")
ten = ultimo("sharp_tendencias_defensa_*.csv")
sem_sharp = None
if cob is not None:
    sem_sharp = cob.attrs.get("w")
try:
    ftn, _ = cargar_ftn(SEASON)
    con = ftn[ftn.n_pass_rushers > 0].groupby("week").nflverse_game_id.nunique()
    semanas_ftn = [int(s) for s, n in con.items() if n >= 13]
    ftn = ftn[ftn.week.isin(semanas_ftn)]
except DatosNoDisponibles:
    ftn, semanas_ftn = None, []

v_run, rk_run, n_run = puesto(run.groupby("defteam").epa.mean(), True)
fig = lienzo()
cab(fig, "CÓMO DEFIENDE")
caja = ten.set_index("team") if ten is not None else None
rk_caja = int(caja.caja_ligera_pct.rank(ascending=False, method="min")[T]) if caja is not None else None
tit = (f"La {'mejor' if rk_run == 1 else f'{rk_run}ª'} defensa contra la carrera"
       + (f" juega con la caja ligera: {caja.loc[T, 'caja_ligera_pct']:.0f}% de las jugadas"
          f" ({rk_caja}º)." if caja is not None and rk_caja <= 10 else "."))
y = titular(fig, tit, y=0.82, size=18.5) - 0.03
y_card = y
etiqueta(fig, "EPA por carrera permitido", L + 0.01, y - 0.005)
numero(fig, num(v_run), L + 0.01, y - 0.055, size=40, color=color_rango(rk_run))
fig.text(L + 0.30, y - 0.055, f"{rk_run}º de {n_run}", family=DISP, weight="bold", size=16,
         color=FG, va="center")
y -= 0.11
linea_h(fig, y + 0.01)
rows = []
if cob is not None:
    c = cob.set_index("team")
    rows += [("Zona", c.zona_pct), ("Individual (man)", c.man_pct),
             ("Dos safeties profundos", c.medio_abierto_pct)]
if caja is not None:
    rows += [("Paquete nickel o dime", caja.sub_package_pct), ("Caja ligera", caja.caja_ligera_pct),
             ("Blitz", caja.blitz_pct)]
for lab, s in rows:
    rk = int(s.rank(ascending=False, method="min")[T])
    fig.text(L + 0.01, y - 0.012, lab, family=TXT, size=8.2, color=FG, va="center")
    fig.text(0.62, y - 0.012, f"{s[T]:.0f}%", family=DISP, weight="bold", size=12, color=FG,
             va="center", ha="right")
    pista32(fig, 0.655, y - 0.0165, 0.20, rk, len(s), color=ACCENT)
    fig.text(R - 0.005, y - 0.012, f"{rk}º", family=TXT, weight="bold", size=8.5, color=GRIS1,
             va="center", ha="right")
    y -= 0.038
nota(fig, "Puesto en uso: 1º = el que más lo hace (no es mejor ni peor)", L + 0.01, y - 0.004)
y -= 0.035
if ftn is not None and len(semanas_ftn):
    m = pas.merge(ftn, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"])
    m = m[m.n_pass_rushers > 0]
    con_b = m[m.n_blitzers > 0].groupby("defteam").epa.mean()
    sin_b = m[m.n_blitzers == 0].groupby("defteam").epa.mean()
    linea_h(fig, y + 0.012)
    etiqueta(fig, "EPA por pase permitido", L + 0.01, y - 0.008)
    y -= 0.04
    for lab, s in (("Con blitz", con_b), ("Sin blitz", sin_b)):
        v, rk, n = s[T], int(s.rank(method="min")[T]), len(s)
        fig.text(L + 0.01, y, lab, family=TXT, size=8.2, color=FG, va="center")
        x0, ancho = 0.42, 0.30
        lim = 0.6
        xc = x0 + ancho / 2
        fig.add_artist(Rectangle((xc, y - 0.008), (max(-lim, min(lim, v)) / lim) * ancho / 2, 0.016,
                                 transform=fig.transFigure, color=color_rango(rk, n), lw=0))
        linea_v(fig, xc, y - 0.014, y + 0.014)
        fig.text(0.80, y, num(v), family=DISP, weight="bold", size=12, color=FG, va="center",
                 ha="right")
        fig.text(R - 0.005, y, f"{rk}º", family=TXT, weight="bold", size=8.5,
                 color=color_rango(rk, n), va="center", ha="right")
        y -= 0.038
tarjeta(fig, y + 0.005, y_card + 0.02)
fs = [f for f in ("sharp_coberturas", "sharp_tendencias_defensa") if ultimo(f + "_*.csv") is not None]
w_sharp = sorted(glob.glob("datos_web/sharp_coberturas_*.csv"))[-1][-6:-4] if fs else "?"
pie(fig, f"Coberturas, paquetes, caja y blitz: Sharp Football Analysis (hasta la sem. {int(w_sharp[1:])}). "
         f"Blitz con/sin: FTN, sem. {semanas_ftn[0]}-{semanas_ftn[-1]}.", sello(SEASON))
guardar(fig, salida(f"radiografia_{T}_2_defensa_{SEASON}.png", SEASON))

# ═══ 3. ATAQUE ════════════════════════════════════════════════════════════════
per_o = ultimo("sumersports_personal_ataque_*.csv")
form = ultimo(f"sumersports_formaciones_{T}_*.csv")
fig = lienzo()
cab(fig, "CÓMO ATACA")
fo = form[form.lado == "ataque"].set_index("formacion") if form is not None else None
# Solo formaciones con muestra (30+ jugadas): la 2x1 de 11 jugadas no cuenta
mejor = fo[fo.jugadas >= 30].epa_pase.idxmax() if fo is not None else None
peor = fo.loc[fo.uso_pct.idxmax()] if fo is not None else None
if fo is not None:
    tit = (f"Pasa mejor con tres receptores en un lado ({mejor.lower()}: {num(fo.loc[mejor, 'epa_pase'])}"
           f" por pase) que en 2x2 ({num(fo.loc['2X2', 'epa_pase'])}), su formación más usada.")
    if mejor not in ("1X3", "3X1"):
        tit = f"Su formación más usada es la {fo.uso_pct.idxmax()}."
else:
    tit = "Cómo ataca"
y = titular(fig, tit, y=0.82, size=17) - 0.03
y_card = y


def barras(y, titulo, filas, lim=0.5):
    etiqueta(fig, titulo, L + 0.01, y)
    nota(fig, "uso", 0.40, y, ha="right")
    nota(fig, "EPA", 0.60, y)
    linea_h(fig, y - 0.016)
    y -= 0.045
    for lab, uso, liga, v, rk, n in filas:
        fig.text(L + 0.01, y, lab, family=TXT, size=8.2, color=FG, va="center")
        fig.text(0.40, y, f"{uso:.0f}%", family=DISP, weight="bold", size=12, color=FG,
                 va="center", ha="right")
        if liga is not None:
            fig.text(0.41, y, f"liga {liga:.0f}%", family=TXT, size=6.5, color=GRIS2, va="center")
        xc, ancho = 0.67, 0.22
        fig.add_artist(Rectangle((xc, y - 0.008), (max(-lim, min(lim, v)) / lim) * ancho / 2, 0.016,
                                 transform=fig.transFigure,
                                 color=color_rango(rk, n) if rk else NEUTRO, lw=0))
        linea_v(fig, xc, y - 0.014, y + 0.014)
        fig.text(R - 0.005, y, num(v), family=DISP, weight="bold", size=11, color=FG,
                 va="center", ha="right")
        y -= 0.040
    return y - 0.02


if per_o is not None:
    filas_p = []
    for p, lab in (("11", "11 · 1 RB, 1 TE, 3 WR"), ("12", "12 · 1 RB, 2 TE, 2 WR"),
                   ("13", "13 · 1 RB, 3 TE, 1 WR")):
        x = per_o[per_o.personal == p].set_index("team")
        if T in x.index:
            rk = int(x.epa_jugada.rank(ascending=False, method="min")[T])
            filas_p.append((lab, x.loc[T, "uso_pct"], x.loc[T, "uso_liga_pct"], x.loc[T, "epa_jugada"],
                            rk, len(x)))
    y = barras(y, "Personal · EPA por jugada", filas_p, lim=0.3)
if fo is not None:
    filas_f = [(f"{f.lower()}", fo.loc[f, "uso_pct"], fo.loc[f, "uso_liga_pct"], fo.loc[f, "epa_pase"],
                None, None) for f in ("2X2", "1X3", "3X1") if f in fo.index]
    y = barras(y, "Formación · EPA por pase", filas_f, lim=0.6)
    nota(fig, "2x2: dos receptores a cada lado · 1x3 y 3x1: tres en un mismo lado (trips)",
         L + 0.01, y + 0.012)
    y -= 0.02
tarjeta(fig, y, y_card + 0.02)
pie(fig, "SumerSports, sem. 1-5 (13-oct). Verde/rojo: top/bottom 5 en EPA con ese personal.",
    sello(SEASON))
guardar(fig, salida(f"radiografia_{T}_3_ataque_{SEASON}.png", SEASON))

# ═══ 4. REPARTO ═══════════════════════════════════════════════════════════════
from pbp_loader import cargar_rosters
try:
    ros, _ = cargar_rosters(SEASON)
    NOMBRES = ros.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id").full_name
except DatosNoDisponibles:
    NOMBRES = pd.Series(dtype=str)
tg = pas[(pas.posteam == T) & pas.receiver_player_id.notna()].copy()
tg["nombre"] = tg.receiver_player_id.map(NOMBRES).fillna(tg.receiver_player_name)
qb_id = pas[pas.posteam == T].passer_player_id.value_counts().index[0]
qb = NOMBRES.get(qb_id, qb_id)
r = tg.groupby("nombre").agg(n=("epa", "size"), epa=("epa", "mean"),
                                          yds=("yards_gained", "sum")).sort_values("n", ascending=False)
r["cuota"] = r.n / len(tg)
r = r.head(6)
lider = r.index[0]
fig = lienzo()
cab(fig, "A QUIÉN LANZA")
tit = (f"{lider}, el primer objetivo de {qb}: {pct(r.loc[lider, 'cuota'])} de los targets.")
y = titular(fig, tit, y=0.82, size=18.5) - 0.03
y_card = y
etiqueta(fig, "Cuota de targets", L + 0.01, y)
etiqueta(fig, "EPA por target", R - 0.005, y, ha="right")
linea_h(fig, y - 0.016)
y -= 0.05
mx = r.cuota.max()
for nm, f in r.iterrows():
    fig.text(L + 0.01, y + 0.012, nm, family=TXT, weight="bold", size=8.5, color=FG, va="center")
    fig.text(L + 0.01, y - 0.012, f"{f.n:.0f} targets · {f.yds:.0f} yardas", family=TXT, size=6.8,
             color=GRIS2, va="center")
    x0 = 0.36
    fig.add_artist(Rectangle((x0, y - 0.009), 0.36 * f.cuota / mx, 0.018, transform=fig.transFigure,
                             color=color_equipo(T) if nm == lider else NEUTRO, lw=0))
    fig.text(x0 + 0.36 * f.cuota / mx + 0.01, y, pct(f.cuota), family=DISP, weight="bold", size=11,
             color=FG, va="center")
    fig.text(R - 0.005, y, num(f.epa), family=DISP, weight="bold", size=12, color=FG, va="center",
             ha="right")
    y -= 0.062
tarjeta(fig, y + 0.02, y_card + 0.02)
pie(fig, "Targets: pases con receptor designado (PBP de nflverse).", sello(SEASON))
guardar(fig, salida(f"radiografia_{T}_4_reparto_{SEASON}.png", SEASON))
