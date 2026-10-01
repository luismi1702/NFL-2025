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

# Filter: up to input_week, pass+run plays, posteam and epa present
mask = (
    df["week"].notna() &
    (df["week"] <= week) &
    df["play_type"].isin(["pass", "run"]) &
    df["posteam"].notna() &
    df["epa"].notna()
)
plays = df[mask].copy()
print(f"Jugadas filtradas: {len(plays):,}")

# ── METRIC A: OFF EPA (mean EPA/play as posteam) ───────────────────────────────
off_epa = plays.groupby("posteam")["epa"].mean().rename("off_epa")

# ── METRIC B: DEF EPA (mean EPA/play as defteam, lower = better) ──────────────
def_mask = (
    df["week"].notna() &
    (df["week"] <= week) &
    df["play_type"].isin(["pass", "run"]) &
    df["defteam"].notna() &
    df["epa"].notna()
)
def_plays = df[def_mask].copy()
def_epa = def_plays.groupby("defteam")["epa"].mean().rename("def_epa")

# ── METRIC C: TRENDING (last 3 weeks vs previous) ─────────────────────────────
last3_weeks = sorted([w for w in [week - 2, week - 1, week] if w >= 1])
min_last3   = min(last3_weeks) if last3_weeks else week

off_last3 = (
    plays[plays["week"].isin(last3_weeks)]
    .groupby("posteam")["epa"]
    .mean()
    .rename("off_last3")
)

prev_plays = plays[plays["week"] < min_last3]
if len(prev_plays) > 0:
    off_prev = prev_plays.groupby("posteam")["epa"].mean().rename("off_prev")
else:
    off_prev = pd.Series(dtype=float, name="off_prev")

# ── METRIC D: EQUIPOS ESPECIALES (EPA neto por partido) ──────────────────────
# Kickoff, punt, FG y PAT. El EPA va del lado de posteam (en punts y FG es quien
# patea; en kickoffs, quien recibe), asi que el neto = EPA como posteam menos
# EPA como defteam cubre patear y retornar. Por partido y no por jugada: el
# numero de jugadas de ST depende del marcador, no del acierto.
st_mask = (
    df["week"].notna() &
    (df["week"] <= week) &
    df["play_type"].isin(["kickoff", "punt", "field_goal", "extra_point"]) &
    df["posteam"].notna() & df["defteam"].notna() &
    df["epa"].notna()
)
st_plays = df[st_mask]
st_net = (st_plays.groupby("posteam")["epa"].sum()
          .sub(st_plays.groupby("defteam")["epa"].sum(), fill_value=0.0))
partidos = pd.concat([plays[["posteam", "game_id"]].rename(columns={"posteam": "t"}),
                      def_plays[["defteam", "game_id"]].rename(columns={"defteam": "t"})]
                     ).groupby("t")["game_id"].nunique()
st_epa = (st_net / partidos).rename("st_epa")

# ── COMBINE METRICS ────────────────────────────────────────────────────────────
stats = pd.DataFrame({"off_epa": off_epa, "def_epa": def_epa, "st_epa": st_epa})
stats["st_epa"] = stats["st_epa"].fillna(0.0)
stats = stats.join(off_last3, how="left").join(off_prev, how="left")

if len(off_prev) > 0:
    stats["trending"] = stats["off_last3"].fillna(stats["off_epa"]) - stats["off_prev"].fillna(stats["off_epa"])
else:
    stats["trending"] = 0.0

stats["trending"] = stats["trending"].fillna(0.0)
stats = stats.dropna(subset=["off_epa", "def_epa"]).copy()

# ── NORMALIZE ─────────────────────────────────────────────────────────────────
stats["norm_off"]   = safe_norm(stats["off_epa"])
stats["norm_def"]   = 1.0 - safe_norm(stats["def_epa"])   # inverted: lower def_epa = better
stats["norm_trend"] = safe_norm(stats["trending"])
stats["norm_st"]    = safe_norm(stats["st_epa"])

# ── COMPOSITE ─────────────────────────────────────────────────────────────────
stats["composite"] = (
    0.35 * stats["norm_off"] +
    0.35 * stats["norm_def"] +
    0.10 * stats["norm_st"] +
    0.20 * stats["norm_trend"]
)

stats = stats.sort_values("composite", ascending=False).reset_index()
stats = stats.rename(columns={"posteam": "team", "index": "team"})

# Handle index column name (it may be "posteam" or "index")
if "posteam" in stats.columns:
    stats = stats.rename(columns={"posteam": "team"})
elif "index" in stats.columns:
    stats = stats.rename(columns={"index": "team"})

# Ensure team column exists and is clean
team_col_candidates = [c for c in stats.columns if c not in
                       ["off_epa","def_epa","st_epa","off_last3","off_prev","trending",
                        "norm_off","norm_def","norm_st","norm_trend","composite"]]
if team_col_candidates:
    stats = stats.rename(columns={team_col_candidates[0]: "team"})

stats = stats.reset_index(drop=True)
print(f"Equipos en rankings: {len(stats)}")

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
tit = f"{nombre_equipo(primero)} encabeza el ranking; {nombre_equipo(ultimo)} lo cierra."
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
pie(fig, "Puntuación 0-100 (35% ataque, 35% defensa, 10% especiales, 20% tendencia) · "
         "A/D = puesto en EPA de ataque y de defensa",
    f"NFL {SEASON} · datos hasta sem. {week} · nflverse PBP")

outfile = salida(f"power_rankings_{SEASON}.png", SEASON, week)
guardar(fig, outfile)
