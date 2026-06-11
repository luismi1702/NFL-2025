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
import numpy as np
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
HARD_PENALTY = {"NYJ": 4.5}

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
# TB20 (pts cedidos pace 377.2), LA21 (3a bajada 0.4104), PHI24 (sacks 45),
# NE18 (yds/jugada 5.7809).
THRESHOLDS = {
    "wins_pace":        {"umbral": 11.00,  "lb": False, "label": "Victorias (ritmo)"},
    "ptdiff_pace":      {"umbral": 59.00,  "lb": False, "label": "Dif. puntos (ritmo)"},
    "def_epa":          {"umbral": -0.015, "lb": True,  "label": "EPA defensivo"},
    "pts_allowed_pace": {"umbral": 378.00, "lb": True,  "label": "Pts cedidos (ritmo)"},
    "def_third_conv":   {"umbral": 0.415,  "lb": True,  "label": "3a bajada cedida"},
    "pts_scored_pace":  {"umbral": 355.00, "lb": False, "label": "Pts anotados (ritmo)"},
    "ypa":              {"umbral": 6.91,   "lb": False, "label": "Yds/intento pase"},
    "drive_score_rate": {"umbral": 0.3461, "lb": False, "label": "% drives con TD/FG"},
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
        h, w = img.shape[:2]
        aspect = w / float(h) if h else 1.0
        div = HARD_PENALTY.get(team, np.clip(1.0 + 0.6 * max(0.0, aspect - 1.3), 1.0, 2.2))
        return OffsetImage(img, zoom=zoom / div, resample=True)
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

    # EPA def, 3a bajada cedida, turnovers forzados — por semana y luego agregado,
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
        # Nota: groupby solo por fixed_drive replica la definicion de los antiguos
        # newmetrics con la que se calibro el umbral 0.3461 — no cambiar sin recalibrar
        drives = off.groupby("fixed_drive").first()
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


# ── VISUALIZACION ─────────────────────────────────────────────────────────────
def render(df, season, week, n_weeks_played):
    contenders = df[df["contender"]]
    n_cont = len(contenders)

    metric_labels = [cfg["label"] for cfg in THRESHOLDS.values()]
    metric_keys   = list(THRESHOLDS.keys())
    n_metrics     = len(metric_keys)

    n_rows = len(df)
    fig_h  = max(12, n_rows * 0.38 + 3)
    fig    = plt.figure(figsize=(16, fig_h), facecolor=BG)
    gs     = gridspec.GridSpec(2, 1, figure=fig,
                               height_ratios=[1, n_rows],
                               hspace=0.02, top=0.97, bottom=0.02,
                               left=0.01, right=0.99)

    # ── Cabecera
    ax_h = fig.add_subplot(gs[0])
    ax_h.set_facecolor(BG)
    ax_h.axis("off")
    ax_h.text(0.5, 0.80, "Contenders Tracker",
              ha="center", color=GOLD, fontsize=22, fontweight="bold",
              transform=ax_h.transAxes)
    ax_h.text(0.5, 0.45, f"Temporada {season}  ·  Semana {week}  ·  {n_weeks_played} jornadas jugadas",
              ha="center", color=FG, fontsize=12, alpha=0.85,
              transform=ax_h.transAxes)
    ax_h.text(0.5, 0.10,
              f"{n_cont} equipo{'s' if n_cont != 1 else ''} cumple{'n' if n_cont != 1 else ''} "
              f"las {TOTAL} metricas simultaneamente",
              ha="center", color=GREEN if n_cont > 0 else RED, fontsize=11,
              fontweight="bold", transform=ax_h.transAxes)

    # ── Tabla
    ax = fig.add_subplot(gs[1])
    ax.set_facecolor(BG)
    ax.axis("off")

    # Proporciones: logo | equipo | ok/total | metrica1 ... metrica12
    LOGO_X  = 0.022
    TEAM_X  = 0.065
    SCORE_X = 0.115
    COL_START = 0.155
    COL_W     = (0.99 - COL_START) / n_metrics

    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows + 1)

    # Cabeceras metricas
    for j, lbl in enumerate(metric_labels):
        x = COL_START + j * COL_W + COL_W / 2
        short = lbl.replace(" (ritmo)", "").replace("Yds/", "Y/")
        ax.text(x, n_rows + 0.65, short, ha="center", va="center",
                color=GOLD, fontsize=6.2, fontweight="bold", rotation=30)

    ax.axhline(n_rows + 0.1, color=GRID, lw=0.8, xmin=0.01, xmax=0.99)

    prev_contender = None
    for i, (_, row) in enumerate(df.iterrows()):
        y    = n_rows - i - 0.5
        row_y = n_rows - i - 1
        is_cont = row["contender"]

        # Separador contenders / no contenders
        if bool(prev_contender) and not is_cont:
            ax.axhline(n_rows - i, color=GOLD, lw=1.5, xmin=0.01, xmax=0.99, alpha=0.7)
        prev_contender = is_cont

        # Fondo
        bg = "#1a2535" if is_cont else (CARD if i % 2 == 0 else "#1a2030")
        rect = plt.Rectangle((0.01, row_y), 0.98, 1, facecolor=bg, zorder=1)
        ax.add_patch(rect)

        # Logo
        place_logo(ax, row["team"], LOGO_X, y, zoom=0.030, zorder=3)

        # Equipo
        team_col = GREEN if is_cont else FG
        ax.text(TEAM_X, y, row["team"], ha="center", va="center",
                color=team_col, fontsize=9, fontweight="bold" if is_cont else "normal", zorder=3)

        # Score ok/total
        score_col = GREEN if is_cont else (YELLOW if row["n_miss"] <= 2 else RED)
        ax.text(SCORE_X, y, f"{int(row['n_ok'])}/{TOTAL}",
                ha="center", va="center", color=score_col, fontsize=9,
                fontweight="bold", zorder=3)

        # Celdas por metrica
        for j, metric in enumerate(metric_keys):
            x    = COL_START + j * COL_W + COL_W / 2
            val  = row.get(f"ok_{metric}", None)
            if pd.isna(val):           # pd.isna y bool() para tolerar numpy bools
                color, sym = DIM, "–"
                tcol = "#666"
            elif bool(val):
                color, sym = "#06d6a0", "✓"
                tcol = "#000"
            else:
                color, sym = "#d84a4a", "✗"
                tcol = FG

            cell_w = COL_W * 0.85
            cell = FancyBboxPatch((x - cell_w/2, y - 0.35), cell_w, 0.70,
                                  boxstyle="round,pad=0.01",
                                  facecolor=color, alpha=0.85, zorder=2)
            ax.add_patch(cell)
            ax.text(x, y, sym, ha="center", va="center",
                    color=tcol, fontsize=10, fontweight="bold", zorder=3)

    # Leyenda
    for lx, col, txt in [(0.01, GREEN, "CONTENDER"), (0.09, YELLOW, "1-2 metricas"),
                          (0.19, RED, "3+ metricas"), (0.29, DIM, "sin datos")]:
        chip = FancyBboxPatch((lx, -0.85), 0.07, 0.60,
                              boxstyle="round,pad=0.01", facecolor=col, alpha=0.85, zorder=2)
        ax.add_patch(chip)
        ax.text(lx + 0.035, -0.55, txt, ha="center", va="center",
                color="#000" if col in (GREEN, YELLOW) else FG, fontsize=7.5,
                fontweight="bold", zorder=3)

    fig.text(0.99, 0.005, "@CuartayDato", ha="right", va="bottom",
             fontsize=9, color="#888888", alpha=0.8, fontstyle="italic")
    fig.text(0.01, 0.005, "Fuente: nflverse  ·  Conteos prorrateados a ritmo 17 PJ",
             ha="left", va="bottom", fontsize=7, color="#555555", fontstyle="italic")

    out = f"contenders_s{season}_w{week}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Guardado: {out}")
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
