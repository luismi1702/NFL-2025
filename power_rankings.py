"""
power_rankings.py
Power Rankings semanales composites con logos.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbp_loader import cargar_pbp, salida, season_cli, week_cli, sello
from estilo import *   # noqa: F401,F403  (estilo comun, docs/estilo-visual.md)
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON = season_cli()   # None = auto-detectar última temporada
BG     = "#0f1115"
FG     = "#EDEDED"
GRID   = "#2a2f3a"
DPI    = 170
LOGOS_DIR    = "logos"
RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

# ── HELPERS ────────────────────────────────────────────────────────────────────
def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pick_col(df, *cands):
    for c in cands:
        if c and c in df.columns:
            return c
    return None


def load_logo(team, base_zoom=0.055):
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
        # Normaliza por el area de tinta real; los wordmarks apaisados
        # pueden ensancharse hasta 1.8x para compensar su poca altura
        zoom = base_zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * zoom > 900.0 * (base_zoom):
            zoom = 900.0 * (base_zoom) / w
        return OffsetImage(img, zoom=zoom, resample=True)
    except Exception:
        return None


def safe_norm(series):
    """Normalize a series 0-1; if all equal return 0.5."""
    mn = series.min()
    mx = series.max()
    rng = mx - mn
    if rng == 0:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / rng

# ── INPUT ──────────────────────────────────────────────────────────────────────
week = week_cli() or int(input("Semana (numero): ").strip())

# ── DATA ───────────────────────────────────────────────────────────────────────
# solo_reg=False: el filtro "hasta semana N" es del usuario
df, SEASON = cargar_pbp(SEASON, solo_reg=False)
print(f"PBP {SEASON}: {len(df):,} jugadas")
to_num(df, ["epa", "week"])

# Pesos (21-oct-2026, Luis). Antes: 35/35/10 y 20% de tendencia SOLO ofensiva,
# que hundia a SF (6-0, 1º en ataque) al 9º por haber empezado altisimo.
# Ahora la tendencia mira ataque y defensa y es solo un desempate (5%).
PESOS = {"off": 0.425, "def": 0.425, "st": 0.10, "trend": 0.05}


def calcular(df, week):
    """Ranking hasta la semana `week`, ordenado de mejor a peor."""
    d = df[df["week"].notna() & (df["week"] <= week) & df["epa"].notna()]
    jug = d[d["play_type"].isin(["pass", "run"]) &
            d["posteam"].notna() & d["defteam"].notna()]

    # A/B: EPA por jugada en ataque y en defensa (defensa: menos = mejor)
    off_epa = jug.groupby("posteam")["epa"].mean()
    def_epa = jug.groupby("defteam")["epa"].mean()

    # C: tendencia = ultimas 3 semanas frente a las anteriores, ataque y defensa
    ult3 = [w for w in (week - 2, week - 1, week) if w >= 1]
    rec, ant = jug[jug["week"].isin(ult3)], jug[jug["week"] < min(ult3)]
    if len(ant):
        t_off = (rec.groupby("posteam")["epa"].mean()
                 .sub(ant.groupby("posteam")["epa"].mean()))
        t_def = (ant.groupby("defteam")["epa"].mean()
                 .sub(rec.groupby("defteam")["epa"].mean()))   # bajar EPA cedido = mejorar
        trending = t_off.add(t_def, fill_value=0.0)
    else:
        trending = pd.Series(0.0, index=off_epa.index)

    # D: equipos especiales, EPA neto por partido. El EPA va del lado de posteam
    # (en punts y FG, quien patea; en kickoffs, quien recibe): neto = como
    # posteam menos como defteam. Por partido, no por jugada: el numero de
    # jugadas de ST depende del marcador, no del acierto.
    st = d[d["play_type"].isin(["kickoff", "punt", "field_goal", "extra_point"]) &
           d["posteam"].notna() & d["defteam"].notna()]
    st_net = (st.groupby("posteam")["epa"].sum()
              .sub(st.groupby("defteam")["epa"].sum(), fill_value=0.0))
    partidos = pd.concat([jug[["posteam", "game_id"]].rename(columns={"posteam": "t"}),
                          jug[["defteam", "game_id"]].rename(columns={"defteam": "t"})]
                         ).groupby("t")["game_id"].nunique()

    s = pd.DataFrame({"off_epa": off_epa, "def_epa": def_epa,
                      "st_epa": st_net / partidos, "trending": trending})
    s = s.dropna(subset=["off_epa", "def_epa"]).fillna({"st_epa": 0.0, "trending": 0.0})
    s["composite"] = (PESOS["off"] * safe_norm(s["off_epa"]) +
                      PESOS["def"] * (1.0 - safe_norm(s["def_epa"])) +
                      PESOS["st"] * safe_norm(s["st_epa"]) +
                      PESOS["trend"] * safe_norm(s["trending"]))
    s = s.sort_values("composite", ascending=False)
    s.index.name = "team"
    return s.reset_index()


stats = calcular(df, week)
print(f"Equipos en rankings: {len(stats)}")

# Puesto de la semana anterior con la MISMA formula: el borrador cuenta las
# subidas y bajadas con esto, no con el PNG de la semana pasada.
prev = calcular(df, week - 1) if week > 1 else None
puesto_prev = ({t: i + 1 for i, t in enumerate(prev["team"])} if prev is not None else {})
rk_o = stats["off_epa"].rank(ascending=False, method="min").astype(int)
rk_d = stats["def_epa"].rank(ascending=True, method="min").astype(int)
rk_s = stats["st_epa"].rank(ascending=False, method="min").astype(int)
rk_t = stats["trending"].rank(ascending=False, method="min").astype(int)
print()
print(f"Pesos: ataque {PESOS['off']:.1%} | defensa {PESOS['def']:.1%} | "
      f"especiales {PESOS['st']:.0%} | tendencia (ataque+defensa) {PESOS['trend']:.0%}")
print(f"{'#':>2} {'eq':<4}{'nota':>6} {'sem.ant':>8} {'mov':>5}   "
      f"{'ataque':>11} {'defensa':>11} {'especiales':>13} {'tendencia':>13}")
for i, r in stats.iterrows():
    p = puesto_prev.get(r["team"])
    mov = "" if p is None else (f"+{p - i - 1}" if p > i + 1 else ("=" if p == i + 1 else f"{p - i - 1}"))
    print(f"{i + 1:>2} {r['team']:<4}{r['composite'] * 100:>6.1f} {p if p else '-':>8} {mov:>5}   "
          f"{r['off_epa']:+.3f} ({rk_o[i]:>2}) {r['def_epa']:+.3f} ({rk_d[i]:>2}) "
          f"{r['st_epa']:+7.2f} ({rk_s[i]:>2}) {r['trending']:+7.3f} ({rk_t[i]:>2})")
print()

# ── RECORD W-L (desde el cache de schedules; opcional) ────────────────────────
def records_hasta_semana(season, max_week):
    # Por cargar_calendario y no leyendo el parquet a pelo: el cache sin
    # refrescar dejo sin record a KC y DEN (el Monday Night) en la sem. 1 de 2026
    try:
        from pbp_loader import cargar_calendario
        sch, _ = cargar_calendario(season)
    except Exception:
        return {}
    for c in ("result", "week", "season"):
        sch[c] = pd.to_numeric(sch[c], errors="coerce")
    reg = sch[(sch["season"] == season) & (sch["game_type"] == "REG") &
              (sch["week"] <= max_week) & sch["result"].notna()]
    recs = {}
    for t in set(reg["home_team"]) | set(reg["away_team"]):
        h = reg[reg["home_team"] == t]
        a = reg[reg["away_team"] == t]
        w = int((h["result"] > 0).sum() + (a["result"] < 0).sum())
        l = int((h["result"] < 0).sum() + (a["result"] > 0).sum())
        e = int((h["result"] == 0).sum() + (a["result"] == 0).sum())
        recs[t] = f"{w}-{l}" + (f"-{e}" if e else "")
    return recs


records = records_hasta_semana(SEASON, week)

# ── PLOT (estilo docs/estilo-visual.md) ────────────────────────────────────────
def nombre_equipo(team):
    """Ciudad para el titular; los equipos que comparten ciudad, por su apodo."""
    try:
        eq = pd.read_parquet("pbp_cache/teams.parquet").set_index("team_abbr")
        if team in ("LA", "LAC", "NYG", "NYJ"):
            return "los " + eq.loc[team, "team_nick"]
        return eq.loc[team, "team_name"].replace(" " + eq.loc[team, "team_nick"], "")
    except Exception:
        return team


stats = stats.reset_index(drop=True)
n_teams = len(stats)
rk_of = stats["off_epa"].rank(ascending=False, method="min").astype(int)
rk_df = stats["def_epa"].rank(ascending=True, method="min").astype(int)

primero, ultimo = stats.loc[0, "team"], stats.loc[n_teams - 1, "team"]
n1, n32 = nombre_equipo(primero), nombre_equipo(ultimo)
tit = f"{n1} {v(n1, 'encabeza', 'encabezan')} el ranking; {n32} lo {v(n32, 'cierra', 'cierran')}."
tit = tit[0].upper() + tit[1:]

fig = lienzo()
antetitulo(fig, f"Power rankings · semana {week}")
y_top = titular(fig, tit, y=0.925) - 0.03

por_col = (n_teams + 1) // 2
alto_fila = (y_top - 0.15) / por_col
ancho_col = (R - L) / 2
for idx in range(n_teams):
    col, fila_i = divmod(idx, por_col)
    x0 = L + col * ancho_col + (0.012 if col else 0)
    yy = y_top - 0.035 - fila_i * alto_fila
    team = stats.loc[idx, "team"]
    rank = idx + 1
    comp = stats.loc[idx, "composite"]
    fig.text(x0 + 0.032, yy, f"{rank}", family=TXT, weight="bold", size=8,
             color=color_rango(rank, n_teams), ha="right", va="center")
    poner_logo(fig, team, x0 + 0.058, yy, zoom=0.0105)
    fig.text(x0 + 0.083, yy + 0.006, team, family=TXT, weight="bold", size=7.6, color=FG,
             va="center")
    rec = records.get(team, "")
    fig.text(x0 + 0.083, yy - 0.0085, rec, family=TXT, size=6.2, color=GRIS1, va="center")
    # barra de la puntuacion (0-100), gris: el juicio va en el numero del puesto
    xb, wb = x0 + 0.145, 0.15
    fig.add_artist(Rectangle((xb, yy - 0.004), wb, 0.008, transform=fig.transFigure,
                             color=GRID, lw=0))
    fig.add_artist(Rectangle((xb, yy - 0.004), wb * max(min(comp, 1), 0), 0.008,
                             transform=fig.transFigure, color=GRIS1 if 5 < rank <= n_teams - 5
                             else color_rango(rank, n_teams), lw=0))
    fig.text(xb + wb + 0.008, yy, f"{comp * 100:.0f}", family=TXT, weight="semibold",
             size=7.6, color=FG, va="center")
    fig.text(xb + wb + 0.04, yy, f"A {rk_of[idx]}º · D {rk_df[idx]}º", family=TXT, size=6.2,
             color=GRIS2, va="center")
y_fin = y_top - 0.035 - (por_col - 1) * alto_fila - alto_fila / 2 - 0.01
linea_v(fig, L + ancho_col + 0.002, y_fin + 0.015, y_top - 0.015)

tarjeta(fig, y_fin, y_top)
pct = lambda x: f"{x * 100:g}".replace(".", ",") + "%"
pie(fig, f"Puntuación 0-100 ({pct(PESOS['off'])} ataque, {pct(PESOS['def'])} defensa, "
         f"{pct(PESOS['st'])} especiales, {pct(PESOS['trend'])} tendencia) · "
         "A/D = puesto en EPA de ataque y de defensa",
    f"NFL {SEASON} · datos hasta sem. {week} · nflverse PBP")

outfile = salida(f"power_rankings_{SEASON}.png", SEASON, week)
guardar(fig, outfile)
