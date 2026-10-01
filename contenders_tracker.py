"""
contenders_tracker.py
Tracker semanal de contenders: que equipos cumplen las 12 metricas
acumuladas hasta la semana indicada, proratadas a ritmo de 17 partidos.

Autosuficiente: descarga schedules frescos en cada ejecucion y el PBP de la
temporada solo cuando hay jornadas nuevas (cache pbp_contenders_{season}.parquet).
No depende de game_logs_all, player_stats ni newmetrics/fullmetrics.

Uso:
  python contenders_tracker.py                  # auto-detecta temporada y ultima semana
  python contenders_tracker.py --week 10
  python contenders_tracker.py --season 2024 --week 14
"""
import os, sys, argparse, warnings
import socket
import numpy as np

socket.setdefaulttimeout(30)   # una descarga colgada no debe congelar el script
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

warnings.filterwarnings("ignore")

CACHE = "pbp_cache"
LOGOS = "logos"
BG    = "#0f1115"
CARD  = "#151924"
FG    = "#EDEDED"
GRID  = "#2a2f3a"
GOLD  = "#FFD700"
GREEN = "#06d6a0"
YELLOW= "#ffd166"
RED   = "#d84a4a"
DIM   = "#444c5e"

SCHEDULE_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
PBP_URL      = ("https://github.com/nflverse/nflverse-data/releases/download/"
                "pbp/play_by_play_{season}.csv.gz")
TEAM_MAP     = {"OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX"}

PBP_COLS = [
    "week", "season_type", "posteam", "defteam", "play_type",
    "epa", "yards_gained", "success",
    "complete_pass", "incomplete_pass", "interception", "fumble_lost",
    "third_down_converted", "third_down_failed", "sack",
    "fixed_drive", "drive_ended_with_score",
]

# ── UMBRALES (peor campeon 2015-2025, conteos = ritmo 17 partidos) ─────────────
# Recalibrados 2026-06 con las definiciones autosuficientes de este script
# (sacks/YPA oficiales desde PBP): cada umbral = valor del peor campeon con
# pequeño margen flotante. Peores campeones: KC19 (def_epa -0.0158),
# TB20 (pts cedidos pace 377.2), LA21 (3er down 0.4104), PHI24 (sacks 45),
# NE18 (yds/jugada 5.7809).
THRESHOLDS = {
    "wins_pace":        {"umbral": 11.00,  "lb": False, "label": "Victorias (ritmo)"},
    "ptdiff_pace":      {"umbral": 59.00,  "lb": False, "label": "Dif. puntos (ritmo)"},
    "def_epa":          {"umbral": -0.015, "lb": True,  "label": "EPA defensivo"},
    "pts_allowed_pace": {"umbral": 378.00, "lb": True,  "label": "Pts cedidos (ritmo)"},
    "def_third_conv":   {"umbral": 0.415,  "lb": True,  "label": "3er down cedido"},
    "pts_scored_pace":  {"umbral": 355.00, "lb": False, "label": "Pts anotados (ritmo)"},
    "ypa":              {"umbral": 6.91,   "lb": False, "label": "Yds/intento pase"},
    # drive_score_rate recalibrado 2026-07: definicion corregida (drives por
    # (week, fixed_drive)); peor campeon DEN15 (0.3204)
    "drive_score_rate": {"umbral": 0.3184, "lb": False, "label": "% drives con puntos"},
    "turnovers_def_pace":{"umbral": 17.00, "lb": False, "label": "Turnovers forzados (ritmo)"},
    "sacks_allowed_pace":{"umbral": 45.50, "lb": True,  "label": "Sacks permitidos (ritmo)"},
    "success_rate_def": {"umbral": 0.45,   "lb": True,  "label": "Success rate DEF"},
    "yards_play_def":   {"umbral": 5.85,   "lb": True,  "label": "Yds/jugada cedidas"},
}
TOTAL = len(THRESHOLDS)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_logo(team, zoom=0.04):
    path = os.path.join(LOGOS, f"{team}.png")
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
        z = zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * z > 900.0 * zoom:
            z = 900.0 * zoom / w
        return OffsetImage(img, zoom=z, resample=True)
    except Exception:
        return None


def place_logo(ax, team, x, y, zoom=0.04, zorder=5):
    im = load_logo(team, zoom)
    if im is None:
        ax.text(x, y, team, ha="center", va="center", color=FG, fontsize=8, zorder=zorder)
        return
    ab = AnnotationBbox(im, (x, y), frameon=False, zorder=zorder)
    ax.add_artist(ab)


def meets(val, umbral, lb):
    if pd.isna(val):
        return None
    return val <= umbral if lb else val >= umbral


# ── DATOS FRESCOS ─────────────────────────────────────────────────────────────
def load_schedules() -> pd.DataFrame:
    """Descarga schedules frescos y actualiza el cache compartido."""
    print("Descargando schedules...")
    sch = pd.read_csv(SCHEDULE_URL, low_memory=False)
    sch["home_team"] = sch["home_team"].replace(TEAM_MAP)
    sch["away_team"] = sch["away_team"].replace(TEAM_MAP)
    try:
        sch.to_parquet(os.path.join(CACHE, "schedules.parquet"), index=False)
    except Exception as e:
        print(f"  Aviso: no se pudo actualizar schedules.parquet ({e})")
    return sch


def load_pbp(season: int, last_played_week: int) -> pd.DataFrame:
    """PBP REG de la temporada. Re-descarga solo si el cache va por detras
    de la ultima semana jugada segun schedules."""
    cache = os.path.join(CACHE, f"pbp_contenders_{season}.parquet")
    if os.path.exists(cache):
        pbp = pd.read_parquet(cache)
        if pbp["week"].max() >= last_played_week:
            return pbp
        print(f"Cache PBP {season} llega a semana {int(pbp['week'].max())}, "
              f"jugada la {last_played_week} — actualizando...")
    print(f"Descargando PBP {season}...")
    raw = pd.read_csv(PBP_URL.format(season=season), low_memory=False,
                      compression="infer", usecols=lambda c: c in PBP_COLS)
    for c in PBP_COLS:
        if c not in ("season_type", "posteam", "defteam", "play_type") and c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    pbp = raw[raw["season_type"] == "REG"].copy()
    pbp["posteam"] = pbp["posteam"].replace(TEAM_MAP)
    pbp["defteam"] = pbp["defteam"].replace(TEAM_MAP)
    pbp.to_parquet(cache, index=False)
    return pbp


# ── CALCULO ───────────────────────────────────────────────────────────────────
def compute(sch: pd.DataFrame, season: int, max_week: int):
    reg = sch[(sch["season"] == season) & (sch["game_type"] == "REG") &
              (sch["week"] <= max_week) & sch["home_score"].notna()].copy()

    if reg.empty:
        print(f"ERROR: Sin datos para {season} semana {max_week}")
        sys.exit(1)

    # ── Wins, puntos, diferencial (de schedules)
    home = reg[["week","home_team","home_score","away_score"]].copy()
    home.columns = ["week","team","pf","pa"]
    away = reg[["week","away_team","away_score","home_score"]].copy()
    away.columns = ["week","team","pf","pa"]
    games = pd.concat([home, away])
    games["won"] = (games["pf"] > games["pa"]).astype(int)

    sc = games.groupby("team").agg(
        wins=("won","sum"),
        pts_scored=("pf","sum"),
        pts_allowed=("pa","sum"),
        gp=("won","count"),
    ).reset_index()
    sc["pt_diff"] = sc["pts_scored"] - sc["pts_allowed"]
    # Prorratear a 17 partidos
    sc["wins_pace"]         = sc["wins"]        / sc["gp"] * 17
    sc["pts_scored_pace"]   = sc["pts_scored"]  / sc["gp"] * 17
    sc["pts_allowed_pace"]  = sc["pts_allowed"] / sc["gp"] * 17
    sc["ptdiff_pace"]       = sc["pt_diff"]     / sc["gp"] * 17

    # ── Metricas PBP (acumuladas hasta max_week)
    pbp = load_pbp(season, int(reg["week"].max()))
    pbp = pbp[pbp["week"] <= max_week]
    pr  = pbp[pbp["play_type"].isin(["pass", "run"])]

    # EPA def, 3er down cedido, turnovers forzados — por semana y luego agregado,
    # misma definicion que los game logs de Manning_bot con la que se calibraron los umbrales
    weekly = []
    for (week, team), d in pr.groupby(["week", "defteam"]):
        weekly.append({
            "week": week, "team": team,
            "def_epa": d["epa"].mean(),
            "def_third_conv": d["third_down_converted"].sum() /
                              max(1, d["third_down_converted"].sum() + d["third_down_failed"].sum()),
            "turnovers_def": d["interception"].sum() + d["fumble_lost"].sum(),
        })
    wk = pd.DataFrame(weekly)
    gl_agg = wk.groupby("team").agg(
        def_epa        = ("def_epa",        "mean"),
        def_third_conv = ("def_third_conv", "mean"),
        turnovers_def  = ("turnovers_def",  "sum"),
        gp_gl          = ("week",           "count"),
    ).reset_index()
    gl_agg["turnovers_def_pace"] = gl_agg["turnovers_def"] / gl_agg["gp_gl"] * 17

    ts = sc.merge(gl_agg, on="team", how="left")

    # ── YPA y sacks permitidos (estilo oficial: intentos sin sacks)
    passes = pr[pr["play_type"] == "pass"]
    qbt = passes.groupby("posteam").agg(
        completions = ("complete_pass",   "sum"),
        incompletes = ("incomplete_pass", "sum"),
        ints        = ("interception",    "sum"),
        sacks_taken = ("sack",            "sum"),
        pass_yards  = ("yards_gained",    lambda s: s[passes.loc[s.index, "complete_pass"] == 1].sum()),
    ).reset_index().rename(columns={"posteam": "team"})
    qbt["pass_att"] = qbt["completions"] + qbt["incompletes"] + qbt["ints"]
    qbt["ypa"]      = qbt["pass_yards"] / qbt["pass_att"].clip(lower=1)
    qbt = qbt.merge(sc[["team", "gp"]], on="team", how="left")
    qbt["sacks_allowed_pace"] = qbt["sacks_taken"] / qbt["gp"].clip(lower=1) * 17
    ts = ts.merge(qbt[["team","ypa","sacks_allowed_pace"]], on="team", how="left")

    # ── drive_score_rate, success_rate_def, yards_play_def
    rows = []
    for team in pr["posteam"].dropna().unique():
        off = pbp[pbp["posteam"] == team]
        plays_def = pr[pr["defteam"] == team]
        # Un equipo juega max 1 partido/semana → (week, fixed_drive) identifica
        # cada drive sin game_id (fixed_drive se reinicia cada partido).
        # Umbral recalibrado con esta definicion (lab/patch_caches_drive_score.py).
        drives = off.groupby(["week", "fixed_drive"]).first()
        rows.append({
            "team": team,
            "drive_score_rate": drives["drive_ended_with_score"].fillna(0).mean(),
            "success_rate_def": plays_def["success"].fillna(0).mean() if len(plays_def) else np.nan,
            "yards_play_def":   plays_def["yards_gained"].sum() / max(1, plays_def["yards_gained"].notna().sum()),
        })
    ts = ts.merge(pd.DataFrame(rows), on="team", how="left")

    # ── Evaluar cada equipo contra los 12 umbrales
    results = []
    for _, row in ts.iterrows():
        cumple = {}
        for metric, cfg in THRESHOLDS.items():
            val = row.get(metric, np.nan)
            cumple[metric] = meets(val, cfg["umbral"], cfg["lb"])
        n_ok   = sum(1 for v in cumple.values() if v is not None and bool(v))
        n_miss = sum(1 for v in cumple.values() if v is not None and not bool(v))
        n_na   = sum(1 for v in cumple.values() if v is None)
        results.append({
            "team": row["team"],
            "wins": row.get("wins", 0),
            "gp":   row.get("gp", 0),
            "wins_pace":    row.get("wins_pace", np.nan),
            "n_ok":  n_ok,
            "n_miss":n_miss,
            "n_na":  n_na,
            **{f"ok_{m}": v for m, v in cumple.items()},
            **{f"val_{m}": row.get(m, np.nan) for m in THRESHOLDS},
        })

    df = pd.DataFrame(results)
    df["contender"] = df["n_miss"] == 0
    df = df.sort_values(["n_ok","wins_pace"], ascending=[False, False]).reset_index(drop=True)
    return df, reg["week"].nunique()


# ── VISUALIZACION (estilo docs/estilo-visual.md) ─────────────────────────────
ETIQ_CORTA = {
    "wins_pace": "Victorias", "ptdiff_pace": "Dif. puntos", "def_epa": "EPA def.",
    "pts_allowed_pace": "Pts cedidos", "def_third_conv": "3er down ced.",
    "pts_scored_pace": "Pts anotados", "ypa": "Yds/pase", "drive_score_rate": "Drives c/ pts",
    "turnovers_def_pace": "Pérdidas forz.", "sacks_allowed_pace": "Sacks recib.",
    "success_rate_def": "Éxito ced.", "yards_play_def": "Yds/jug. ced.",
}


def _nombre(team):
    try:
        eq = pd.read_parquet("pbp_cache/teams.parquet").set_index("team_abbr")
        if team in ("LA", "LAC", "NYG", "NYJ"):
            return "los " + eq.loc[team, "team_nick"]
        return eq.loc[team, "team_name"].replace(" " + eq.loc[team, "team_nick"], "")
    except Exception:
        return team


def render(df, season, week, n_weeks_played):
    from estilo import (lienzo, guardar, antetitulo, titular, tarjeta, etiqueta, pie, poner_logo,
                        color_equipo, L, R, TXT, FG, GRIS1, GRIS2, GRID, NEUTRO)
    from matplotlib.patches import Rectangle, Ellipse
    cont = df[df["contender"]]["team"].tolist()
    if not cont:
        mejor = df.iloc[0]
        nm = _nombre(mejor['team'])
        tit = (f"Ningún equipo cumple todavía las {TOTAL} marcas del peor campeón; "
               f"{nm} {'cumplen' if nm.startswith('los ') else 'cumple'} {int(mejor['n_ok'])}.")
    elif len(cont) == 1:
        nm = _nombre(cont[0])
        tit = f"Solo {nm} {'cumplen' if nm.startswith('los ') else 'cumple'} las {TOTAL} marcas del peor campeón."
    else:
        tit = f"{len(cont)} equipos cumplen las {TOTAL} marcas del peor campeón."
    tit = tit[0].upper() + tit[1:]

    fig = lienzo()
    antetitulo(fig, f"Fórmula del campeón · semana {week}")
    y_top = titular(fig, tit, y=0.925) - 0.03

    keys = list(THRESHOLDS.keys())
    n = len(df)
    X_COLS = L + 0.20
    col_w = (R - 0.01 - X_COLS) / len(keys)
    y_head = y_top - 0.05
    for j, k in enumerate(keys):
        fig.text(X_COLS + (j + 0.5) * col_w, y_head, ETIQ_CORTA.get(k, k), family=TXT,
                 size=5.6, color=GRIS1, rotation=50, ha="left", va="bottom",
                 rotation_mode="anchor")
    alto = (y_head - 0.16) / n
    y = y_head - 0.02
    for i, (_, row) in enumerate(df.iterrows()):
        yy = y - i * alto
        es = bool(row["contender"])
        if es:
            fig.add_artist(Rectangle((L, yy - alto / 2), R - L, alto, transform=fig.transFigure,
                                     color="#1e2433", lw=0, zorder=-0.5))
        poner_logo(fig, row["team"], L + 0.02, yy, zoom=0.0085)
        fig.text(L + 0.045, yy, row["team"], family=TXT, weight="bold" if es else "normal",
                 size=7, color=color_equipo(row["team"]) if es else FG, va="center")
        fig.text(L + 0.17, yy, f"{int(row['n_ok'])}/{TOTAL}", family=TXT, weight="bold",
                 size=7, color=FG if es else GRIS1, va="center", ha="right")
        for j, k in enumerate(keys):
            cx = X_COLS + (j + 0.5) * col_w
            v = row.get(f"ok_{k}", None)
            if v is None or pd.isna(v):
                fig.text(cx, yy, "–", family=TXT, size=6, color=GRIS2, ha="center", va="center")
            else:
                d = 0.0115                     # diametro en ancho de figura
                asp = fig.get_figwidth() / fig.get_figheight()
                fig.add_artist(Ellipse((cx, yy), d, d * asp, transform=fig.transFigure,
                                       fc=FG if bool(v) else "none",
                                       ec=FG if bool(v) else GRIS2, lw=0.9))
    tarjeta(fig, y - (n - 1) * alto - alto / 2 - 0.01, y_top)
    pie(fig, "Marca = el valor del peor campeón 2015-2025 · conteos a ritmo de 17 partidos · "
             "punto blanco = cumple",
        f"NFL {season} · datos hasta sem. {week} · nflverse")

    from pbp_loader import salida
    out = salida(f"contenders_{season}.png", season, week)
    guardar(fig, out)
    return out


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week",   type=int, default=None)
    args = parser.parse_args()

    sch = load_schedules()

    reg_all = sch[(sch["game_type"] == "REG") & sch["home_score"].notna()]
    season  = args.season or int(reg_all["season"].max())
    reg     = reg_all[reg_all["season"] == season]
    week    = args.week or int(reg["week"].max())

    print(f"Calculando contenders: temporada {season}, semana {week}")

    df, n_weeks = compute(sch, season, week)

    contenders = df[df["contender"]]["team"].tolist()
    print(f"\nContenders ({len(contenders)}): {', '.join(sorted(contenders)) or 'ninguno'}")
    print(f"\nResumen por equipo:")
    print(f"{'Equipo':>5}  {'OK':>4}  {'Fallan'}")
    print("-" * 50)
    for _, row in df.iterrows():
        fallan = [THRESHOLDS[m]["label"] for m in THRESHOLDS
                  if pd.notna(row.get(f"ok_{m}")) and not bool(row.get(f"ok_{m}"))]
        status = "CONTENDER" if row["contender"] else ", ".join(fallan[:3])
        print(f"{row['team']:>5}  {int(row['n_ok'])}/{TOTAL}  {status}")

    render(df, season, week, n_weeks)


if __name__ == "__main__":
    main()
