# Manning_bot.py  v6
# Predictor de resultados NFL con Machine Learning — datos nflverse
# Mejoras v5: SOS (Strength of Schedule via Elo opponent quality), MIN_GAMES=2
# Mejoras v6: ensemble clasificador + regresion de margen, y features de
# estabilidad (success rate, EPA neutral, EPA downs 1-2, equipos especiales).
# Verificado en lab/manning_exp_bateria.py: 69,0% walk-forward 2022-2025
# contra 65,5% de v5 — paridad con la linea de apuestas (68,0%).

import os
import socket
import warnings
import pandas as pd

socket.setdefaulttimeout(30)   # una descarga colgada no debe congelar el script
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
# La temporada NO se fija a mano: se deriva de schedules en tiempo de ejecucion
# (ver resolver_temporadas). Un predictor con el año escrito a fuego predice
# partidos ya jugados en cuanto pasa un verano.
PRIMERA_TEMPORADA = 2015
SEASONS_TRAIN  = []          # se rellenan en resolver_temporadas()
SEASON_PRED    = None
ROLLING_N      = 5
ROLLING_SHORT  = 2
MIN_GAMES      = 2
# Codigo de salida cuando falta muestra para predecir (no es un error): el
# modelo se entreno con partidos en que ambos equipos llevaban >= MIN_GAMES
# jugados, asi que la primera jornada con picks es la MIN_GAMES + 1.
# semana_auto.py lo registra como SIN MUESTRA, no como FALLO.
EXIT_SIN_MUESTRA = 3
CACHE_DIR      = Path("pbp_cache")
MODEL_FILE     = "manning_bot_model.pkl"

SCHEDULE_URL     = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
PBP_URL          = ("https://github.com/nflverse/nflverse-data/releases/download/"
                    "pbp/play_by_play_{season}.csv.gz")
PLAYER_STATS_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
                    "player_stats/player_stats.csv.gz")

NEEDED_PBP_COLS = [
    "season", "week", "game_id", "season_type",
    "posteam", "defteam", "play_type", "epa",
    "pass_attempt", "rush_attempt", "complete_pass",
    "interception", "fumble_lost",
    "third_down_converted", "third_down_failed",
    "sack", "cpoe", "yardline_100",
    # v6 — features de estabilidad
    "success", "wp", "down", "special_teams_play",
]

TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX"}

# Elo
ELO_K       = 20.0
ELO_HFA     = 65.0
ELO_REGRESS = 1 / 3
ELO_MEAN    = 1505.0

FEATURE_COLS = [
    # PBP — diferenciales EPA
    "d_off_epa", "d_def_epa", "d_pass_epa", "d_rush_epa",
    "d_def_pass_epa", "d_def_rush_epa",
    "d_third_conv", "d_def_third_conv",
    "d_turnover_diff", "d_sack_rate_off", "d_sack_rate_def",
    "d_rz_epa", "d_cpoe",           # NUEVO: CPOE differential
    # v6 — estabilidad: success rate, EPA neutral, downs tempranos, especiales
    "d_succ_off", "d_succ_def", "d_neutral_off_epa", "d_neutral_def_epa",
    "d_early_pass_off", "d_early_pass_def", "d_st_epa",
    # Schedule
    "d_win_pct", "d_pts_for", "d_pts_against", "d_pythag",
    # Forma reciente
    "d_recent_win", "d_recent_pts",
    # Contexto
    "rest_diff", "div_game",
    # Elo — solo prob (elo_diff era redundante con elo_win_prob)
    "elo_win_prob",
    # Weather
    "is_dome", "temp_adj", "wind_adj", "high_wind", "cold_game",
    # QB rolling
    "d_qb_epa", "d_qb_dakota",
    # Moneyline — solo prob (spread_line era redundante con home_impl_prob)
    "home_impl_prob", "total_line",
    # Strength of schedule: calidad de rivales enfrentados (Elo-based)
    "d_sos",
]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def load_pbp(season: int) -> pd.DataFrame:
    """PBP con las columnas que necesita el bot, cacheado en pbp_cache/.

    El cache se valida por esquema, no solo por existencia: en el disco hay
    ficheros pbp_{season}.parquet escritos por versiones antiguas con otro
    juego de columnas, y devolverlos tal cual reventaba build_game_logs con un
    KeyError. Si faltan columnas se vuelve a descargar.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"pbp_{season}.parquet"
    if cache.exists():
        import pyarrow.parquet as pq
        cols = set(pq.read_schema(cache).names)
        faltan = [c for c in NEEDED_PBP_COLS if c not in cols]
        if not faltan:
            return pd.read_parquet(cache)
        print(f"  Cache PBP {season} obsoleto (faltan {len(faltan)} columnas) — "
              f"se vuelve a descargar")
        cache.unlink()
    url = PBP_URL.format(season=season)
    print(f"  Descargando PBP {season}...")
    df = pd.read_csv(url, low_memory=False, compression="infer",
                     usecols=list(NEEDED_PBP_COLS))
    df.to_parquet(cache, index=False)
    return df


def load_schedules() -> pd.DataFrame:
    """Schedules siempre frescos (resultados y lineas nuevas cada semana);
    el cache solo es fallback sin internet."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / "schedules.parquet"
    try:
        print("Descargando schedules...")
        df = pd.read_csv(SCHEDULE_URL, low_memory=False)
        df.to_parquet(cache, index=False)
    except Exception as e:
        if not cache.exists():
            raise
        print(f"  Aviso: sin conexion ({e}) — usando schedules cacheados")
        df = pd.read_parquet(cache)
    df["home_team"] = df["home_team"].replace(TEAM_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_MAP)
    return df


def resolver_temporadas(schedules):
    """Decide que temporada se predice y con cuales se entrena, mirando el calendario.

    La temporada a predecir es la ultima que aparece en schedules: nflverse
    publica el calendario completo meses antes del arranque, asi que en agosto
    ya devuelve la que viene, que es justo la que interesa pronosticar.
    Se entrena con todas las anteriores que tengan resultados.
    """
    global SEASONS_TRAIN, SEASON_PRED
    SEASON_PRED = int(pd.to_numeric(schedules["season"], errors="coerce").max())

    jugadas = schedules[(schedules["game_type"] == "REG") &
                        schedules["home_score"].notna()]
    ult_completa = int(pd.to_numeric(jugadas["season"], errors="coerce").max())
    # Si la temporada a predecir aun no ha empezado, se entrena hasta la anterior
    tope = min(SEASON_PRED - 1, ult_completa)
    SEASONS_TRAIN = list(range(PRIMERA_TEMPORADA, tope + 1))

    print(f"Temporada a predecir: {SEASON_PRED}")
    print(f"Entrenamiento: {SEASONS_TRAIN[0]}-{SEASONS_TRAIN[-1]}")
    return SEASONS_TRAIN, SEASON_PRED


MODEL_META = "manning_bot_model.meta"   # ultima temporada usada al entrenar


def _guardar_meta():
    with open(MODEL_META, "w", encoding="utf-8") as f:
        f.write(str(SEASONS_TRAIN[-1]))


def _modelo_caducado():
    """Ultima temporada del modelo guardado si se ha quedado atras, si no None."""
    try:
        with open(MODEL_META, encoding="utf-8") as f:
            entrenado_hasta = int(f.read().strip())
    except (OSError, ValueError):
        return None                      # modelo antiguo sin meta: no molestar
    return entrenado_hasta if entrenado_hasta < SEASONS_TRAIN[-1] else None


def folds_walk_forward(seasons_train, n=4):
    """Ultimas n temporadas como validacion, entrenando siempre con las previas."""
    validaciones = seasons_train[-n:]
    return [(yr, list(range(PRIMERA_TEMPORADA, yr))) for yr in validaciones]


def coerce(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2A — PBP GAME LOGS
# ══════════════════════════════════════════════════════════════════════════════

def build_game_logs(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = coerce(pbp, ["epa", "pass_attempt", "rush_attempt", "complete_pass",
                        "interception", "fumble_lost", "third_down_converted",
                        "third_down_failed", "sack", "cpoe", "yardline_100",
                        "success", "wp", "down", "special_teams_play"])
    plays = pbp[pbp["season_type"] == "REG"].copy()
    plays["posteam"] = plays["posteam"].replace(TEAM_MAP)
    plays["defteam"]  = plays["defteam"].replace(TEAM_MAP)
    pr = plays[plays["play_type"].isin(["pass", "run"])].copy()

    rows = []
    for (season, week, team), grp in pr.groupby(["season", "week", "posteam"]):
        grp_pass = grp[grp["play_type"] == "pass"]
        grp_run  = grp[grp["play_type"] == "run"]
        grp_rz   = grp[grp["yardline_100"] <= 20]
        pass_n   = len(grp_pass)

        d      = pr[(pr["season"] == season) & (pr["week"] == week) & (pr["defteam"] == team)]
        d_pass = d[d["play_type"] == "pass"]
        d_run  = d[d["play_type"] == "run"]

        rows.append({
            "season": season, "week": week, "team": team,
            "off_epa":        grp["epa"].mean(),
            "pass_epa":       grp_pass["epa"].mean()  if pass_n > 0 else np.nan,
            "rush_epa":       grp_run["epa"].mean()   if len(grp_run) > 0 else np.nan,
            "rz_epa":         grp_rz["epa"].mean()    if len(grp_rz) > 0 else np.nan,
            "third_conv":     grp["third_down_converted"].sum() /
                              max(1, grp["third_down_converted"].sum() + grp["third_down_failed"].sum()),
            "turnovers_off":  grp["interception"].sum() + grp["fumble_lost"].sum(),
            "sack_rate_off":  grp_pass["sack"].sum() / max(1, pass_n),
            "cpoe":           grp_pass["cpoe"].mean() if pass_n > 0 else np.nan,
            "def_epa":        d["epa"].mean()          if len(d) > 0 else np.nan,
            "def_pass_epa":   d_pass["epa"].mean()     if len(d_pass) > 0 else np.nan,
            "def_rush_epa":   d_run["epa"].mean()      if len(d_run) > 0 else np.nan,
            "def_third_conv": d["third_down_converted"].sum() /
                              max(1, d["third_down_converted"].sum() + d["third_down_failed"].sum())
                              if len(d) > 0 else np.nan,
            "turnovers_def":  (d["interception"].sum() + d["fumble_lost"].sum()) if len(d) > 0 else 0,
            "sack_rate_def":  d_pass["sack"].sum() / max(1, len(d_pass)) if len(d_pass) > 0 else np.nan,
        })
    logs = pd.DataFrame(rows)

    # ── v6: metricas de estabilidad (vectorizadas) ────────────────────────
    # El EPA medio es ruidoso (turnovers, jugadas largas); estas versiones
    # predicen mejor el futuro: success rate, EPA en situacion neutral
    # (wp 5-95%, fuera el garbage time), EPA de pase en downs 1-2 y equipos
    # especiales (fase entera que v5 ignoraba).
    neutral = pr[(pr["wp"] >= 0.05) & (pr["wp"] <= 0.95)]
    early_p = pr[(pr["play_type"] == "pass") & (pr["down"].isin([1, 2]))]
    st      = plays[plays["special_teams_play"] == 1]

    def _m(df, key, col, name):
        return df.groupby(["season", "week", key])[col].mean().rename(name)

    off = pd.concat([_m(pr, "posteam", "success", "succ_off"),
                     _m(neutral, "posteam", "epa", "neutral_off_epa"),
                     _m(early_p, "posteam", "epa", "early_pass_off"),
                     _m(st, "posteam", "epa", "st_epa")], axis=1)
    deff = pd.concat([_m(pr, "defteam", "success", "succ_def"),
                      _m(neutral, "defteam", "epa", "neutral_def_epa"),
                      _m(early_p, "defteam", "epa", "early_pass_def")], axis=1)
    deff.index.names = off.index.names
    extras = (off.join(deff, how="outer").reset_index()
                 .rename(columns={"posteam": "team"}))
    return logs.merge(extras, on=["season", "week", "team"], how="left")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2B — SCHEDULE LOGS (win%, puntos)
# ══════════════════════════════════════════════════════════════════════════════

def build_schedule_logs(schedules: pd.DataFrame) -> pd.DataFrame:
    sched = schedules[schedules["game_type"] == "REG"].copy()
    sched = coerce(sched, ["home_score", "away_score", "result"])
    sched = sched.dropna(subset=["home_score", "away_score"])
    rows = []
    for _, g in sched.iterrows():
        rows.append({"season": g["season"], "week": g["week"], "team": g["home_team"],
                     "win": 1 if g["result"] > 0 else 0,
                     "pts_for": g["home_score"], "pts_against": g["away_score"]})
        rows.append({"season": g["season"], "week": g["week"], "team": g["away_team"],
                     "win": 1 if g["result"] < 0 else 0,
                     "pts_for": g["away_score"], "pts_against": g["home_score"]})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2C — ELO RATINGS (NUEVO)
# ══════════════════════════════════════════════════════════════════════════════

def compute_elo(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Elo estilo FiveThirtyEight desde todos los años disponibles en schedules.
    Usa datos desde 1999 (16 años de warmup antes de 2015).
    Guarda valores PRE-partido (sin leakage).
    """
    reg = schedules[schedules["game_type"] == "REG"].copy()
    reg = coerce(reg, ["result"])
    reg = reg.dropna(subset=["result"]).sort_values(["season", "week"])

    cache = CACHE_DIR / "elo.parquet"
    if cache.exists():
        cached = pd.read_parquet(cache)
        if len(cached) >= len(reg):     # sin partidos nuevos → cache valido
            return cached
        print(f"Elo cache con {len(cached):,} partidos, jugados {len(reg):,} — recalculando...")

    print("Calculando Elo ratings...")

    teams = sorted(set(reg["home_team"]) | set(reg["away_team"]))
    elo = {t: 1500.0 for t in teams}
    prev_season = None
    records = []

    for _, g in reg.iterrows():
        season, ht, at, result = g["season"], g["home_team"], g["away_team"], g["result"]

        # Regresión a la media al inicio de cada temporada
        if prev_season is not None and season != prev_season:
            for t in elo:
                elo[t] = elo[t] - ELO_REGRESS * (elo[t] - ELO_MEAN)
        prev_season = season

        # Probabilidad esperada PRE-partido
        exp_home = 1 / (1 + 10 ** (-(elo[ht] - elo[at] + ELO_HFA) / 400))

        records.append({
            "season": season, "week": g["week"],
            "home_team": ht, "away_team": at,
            "elo_diff":     elo[ht] - elo[at] + ELO_HFA,
            "elo_win_prob": exp_home,
        })

        # Actualizar Elo post-partido
        actual = 1.0 if result > 0 else 0.0 if result < 0 else 0.5
        delta  = ELO_K * (actual - exp_home)
        elo[ht] += delta
        elo[at] -= delta

    df = pd.DataFrame(records)
    df.to_parquet(cache, index=False)
    print(f"  Elo calculado: {len(df):,} partidos")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2D — STRENGTH OF SCHEDULE (SOS via Elo)
# ══════════════════════════════════════════════════════════════════════════════

def compute_sos(elo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada (team, season, week), calcula el promedio EWMA de la fuerza
    de los rivales enfrentados ANTES de ese partido.
    'Fuerza del rival' = elo_win_prob del rival en ese partido (desde su perspectiva).
    No hay leakage: se usa shift(1) y Elo pre-partido ya calculado.
    """
    # Perspectiva local: su rival es el visitante → fuerza rival = 1 - elo_win_prob
    rows_h = elo_df[["season", "week", "home_team", "elo_win_prob"]].copy()
    rows_h = rows_h.rename(columns={"home_team": "team"})
    rows_h["opp_strength"] = 1 - rows_h["elo_win_prob"]

    # Perspectiva visitante: su rival es el local → fuerza rival = elo_win_prob
    rows_a = elo_df[["season", "week", "away_team", "elo_win_prob"]].copy()
    rows_a = rows_a.rename(columns={"away_team": "team"})
    rows_a["opp_strength"] = rows_a["elo_win_prob"]

    opp = pd.concat(
        [rows_h[["season", "week", "team", "opp_strength"]],
         rows_a[["season", "week", "team", "opp_strength"]]],
        ignore_index=True
    ).sort_values(["season", "team", "week"])

    opp["r_sos"] = opp.groupby(["season", "team"])["opp_strength"].transform(
        lambda s: s.shift(1).ewm(span=ROLLING_N, min_periods=1).mean()
    )
    return opp[["season", "week", "team", "r_sos"]]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2E — QB ROLLING STATS
# ══════════════════════════════════════════════════════════════════════════════

def load_qb_rolling() -> pd.DataFrame:
    """
    Descarga player_stats.csv.gz y calcula rolling EWMA de passing_epa y dakota
    por QB (player_id), cruzando temporadas para reflejar historial real.
    """
    import time
    cache = CACHE_DIR / "qb_stats.parquet"
    if cache.exists():
        edad_dias = (time.time() - os.path.getmtime(cache)) / 86400
        if edad_dias < 3:
            return pd.read_parquet(cache)
        print(f"qb_stats con {edad_dias:.0f} dias — refrescando...")

    print("Descargando player_stats (QB rolling)...")
    needed = ["player_id", "player_name", "position", "recent_team",
              "season", "week", "season_type", "attempts",
              "passing_epa", "dakota"]
    try:
        df = pd.read_csv(PLAYER_STATS_URL, low_memory=False, compression="infer",
                         usecols=needed)
    except Exception as e:
        if cache.exists():
            print(f"  Aviso: no se pudo refrescar player_stats ({e}) — usando cache")
            return pd.read_parquet(cache)
        raise

    qbs = df[
        (df["position"] == "QB") &
        (df["season_type"] == "REG") &
        (df["season"] >= 2014)
    ].copy()
    qbs = coerce(qbs, ["attempts", "passing_epa", "dakota"])

    # Starter = QB con más intentos por (season, week, team)
    qbs = (qbs.sort_values("attempts", ascending=False)
              .drop_duplicates(subset=["season", "week", "recent_team"], keep="first")
              .sort_values(["player_id", "season", "week"])
              .reset_index(drop=True))

    # Rolling EWMA por player_id (sin reset de temporada → historial real)
    for col in ["passing_epa", "dakota"]:
        qbs[f"r_qb_{col}"] = qbs.groupby("player_id")[col].transform(
            lambda s: s.shift(1).ewm(span=ROLLING_N, min_periods=1).mean()
        )

    result = qbs[["player_id", "season", "week",
                  "r_qb_passing_epa", "r_qb_dakota"]].copy()
    result.to_parquet(cache, index=False)
    print(f"  QB stats cargadas: {len(result):,} QB-weeks")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — ROLLING FEATURES (EWMA)
# ══════════════════════════════════════════════════════════════════════════════

PBP_STAT_COLS = [
    "off_epa", "pass_epa", "rush_epa", "rz_epa", "third_conv",
    "turnovers_off", "sack_rate_off", "cpoe",
    "def_epa", "def_pass_epa", "def_rush_epa", "def_third_conv",
    "turnovers_def", "sack_rate_def",
    # v6 — estabilidad
    "succ_off", "succ_def", "neutral_off_epa", "neutral_def_epa",
    "early_pass_off", "early_pass_def", "st_epa",
]
SCHED_STAT_COLS = ["win", "pts_for", "pts_against"]


def _ewm_shift(series, span, min_p=1):
    return series.shift(1).ewm(span=span, min_periods=min_p).mean()


def compute_rolling(game_logs: pd.DataFrame) -> pd.DataFrame:
    gl = game_logs.sort_values(["season", "team", "week"]).copy()
    for col in PBP_STAT_COLS:
        if col not in gl.columns:
            continue
        gl[f"r_{col}"] = gl.groupby(["season", "team"])[col].transform(
            lambda s: _ewm_shift(s, span=ROLLING_N))
    gl["games_available"] = gl.groupby(["season", "team"])["week"].transform(
        lambda s: s.shift(1).rolling(ROLLING_N, min_periods=1).count())
    return gl


def add_season_carryover(rolling: pd.DataFrame, weight: float = 0.5) -> pd.DataFrame:
    """
    Rellena los NaN de primeras semanas de cada temporada con el último valor
    rolling de la temporada anterior (ponderado por weight).
    Resuelve el problema de semanas 1-3 sin datos PBP.
    """
    gl = rolling.sort_values(["team", "season", "week"]).copy()
    r_cols = [f"r_{c}" for c in PBP_STAT_COLS if f"r_{c}" in gl.columns]
    for col in r_cols:
        last_vals = (gl.groupby(["team", "season"])[col]
                       .last().rename("last_val").reset_index())
        last_vals["season"] = last_vals["season"] + 1
        gl = gl.merge(last_vals, on=["team", "season"], how="left")
        is_early = gl[col].isna() & gl["last_val"].notna()
        gl.loc[is_early, col] = gl.loc[is_early, "last_val"] * weight
        gl = gl.drop(columns=["last_val"])
    return gl


def compute_schedule_rolling(sched_logs: pd.DataFrame) -> pd.DataFrame:
    sl = sched_logs.sort_values(["season", "team", "week"]).copy()
    for col in SCHED_STAT_COLS:
        sl[f"sr_{col}"] = sl.groupby(["season", "team"])[col].transform(
            lambda s: _ewm_shift(s, span=ROLLING_N))
        sl[f"sf_{col}"] = sl.groupby(["season", "team"])[col].transform(
            lambda s: _ewm_shift(s, span=ROLLING_SHORT))
    sl["sr_pythag"] = (sl["sr_pts_for"] ** 2) / (
        sl["sr_pts_for"] ** 2 + sl["sr_pts_against"] ** 2 + 1e-6)
    return sl


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — Weather & Moneyline
# ══════════════════════════════════════════════════════════════════════════════

def _add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_dome"]   = df["roof"].isin(["dome", "closed"]).astype(int) \
                      if "roof" in df.columns else 0
    temp = pd.to_numeric(df.get("temp", np.nan), errors="coerce")
    wind = pd.to_numeric(df.get("wind", np.nan), errors="coerce")
    df["temp_adj"]  = temp.where(df["is_dome"] == 0, 72.0).fillna(59.0)
    df["wind_adj"]  = wind.where(df["is_dome"] == 0, 0.0).fillna(7.0)
    df["high_wind"] = (df["wind_adj"] >= 15).astype(int)
    df["cold_game"] = (df["temp_adj"] <= 32).astype(int)
    return df


def _add_moneyline_features(df: pd.DataFrame) -> pd.DataFrame:
    def ml_to_prob(ml):
        ml = pd.to_numeric(ml, errors="coerce")
        pos = ml > 0
        neg = ml < 0
        prob = pd.Series(np.where(pos, 100 / (ml + 100),
                         np.where(neg, (-ml) / (-ml + 100), np.nan)),
                         index=ml.index)
        return prob

    if "home_moneyline" in df.columns and "away_moneyline" in df.columns:
        h_raw = ml_to_prob(df["home_moneyline"])
        a_raw = ml_to_prob(df["away_moneyline"])
        total = h_raw + a_raw
        df["home_impl_prob"] = (h_raw / total).fillna(0.5)
    else:
        df["home_impl_prob"] = 0.5

    df["total_line"] = pd.to_numeric(
        df.get("total_line", 45.0), errors="coerce").fillna(45.0)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4 — FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def build_features(schedules, rolling, sched_rolling,
                   elo_df, qb_rolling, sos_df=None,
                   seasons=None, include_target=True):

    if seasons is not None:
        sched = schedules[schedules["season"].isin(seasons)].copy()
    else:
        sched = schedules.copy()

    sched = sched[sched["game_type"] == "REG"].copy()
    if include_target:
        sched = coerce(sched, ["home_score", "away_score", "result"])
        sched = sched.dropna(subset=["home_score", "away_score"])
        sched["home_win"] = (sched["result"] > 0).astype(int)
        sched = sched[sched["result"] != 0]

    # ── Join PBP rolling ───────────────────────────────────────────────────
    rh = rolling.rename(columns={f"r_{c}": f"h_{c}" for c in PBP_STAT_COLS}
                        | {"games_available": "h_games"})
    ra = rolling.rename(columns={f"r_{c}": f"a_{c}" for c in PBP_STAT_COLS}
                        | {"games_available": "a_games"})

    df = sched.merge(
        rh[["season","week","team"] + [f"h_{c}" for c in PBP_STAT_COLS] + ["h_games"]],
        left_on=["season","week","home_team"], right_on=["season","week","team"], how="left"
    ).drop(columns=["team"])
    df = df.merge(
        ra[["season","week","team"] + [f"a_{c}" for c in PBP_STAT_COLS] + ["a_games"]],
        left_on=["season","week","away_team"], right_on=["season","week","team"], how="left"
    ).drop(columns=["team"])

    # ── Join Schedule rolling ─────────────────────────────────────────────
    sr_cols = ([f"sr_{c}" for c in SCHED_STAT_COLS] +
               [f"sf_{c}" for c in SCHED_STAT_COLS] + ["sr_pythag"])
    srh = sched_rolling.rename(columns={c: f"h_{c}" for c in sr_cols})
    sra = sched_rolling.rename(columns={c: f"a_{c}" for c in sr_cols})

    df = df.merge(
        srh[["season","week","team"] + [f"h_{c}" for c in sr_cols]],
        left_on=["season","week","home_team"], right_on=["season","week","team"], how="left"
    ).drop(columns=["team"])
    df = df.merge(
        sra[["season","week","team"] + [f"a_{c}" for c in sr_cols]],
        left_on=["season","week","away_team"], right_on=["season","week","team"], how="left"
    ).drop(columns=["team"])

    # ── Join Elo ──────────────────────────────────────────────────────────
    df = df.merge(
        elo_df[["season","week","home_team","away_team","elo_diff","elo_win_prob"]],
        on=["season","week","home_team","away_team"], how="left"
    )
    df["elo_diff"]     = df["elo_diff"].fillna(ELO_HFA)
    df["elo_win_prob"] = df["elo_win_prob"].fillna(0.5)

    # ── Join QB rolling ───────────────────────────────────────────────────
    qb_cols = ["player_id", "season", "week", "r_qb_passing_epa", "r_qb_dakota"]
    if qb_rolling is not None and "home_qb_id" in df.columns:
        df = df.merge(
            qb_rolling[qb_cols].rename(columns={
                "player_id": "home_qb_id",
                "r_qb_passing_epa": "h_r_qb_epa",
                "r_qb_dakota":      "h_r_qb_dakota",
            }),
            on=["home_qb_id","season","week"], how="left"
        )
        df = df.merge(
            qb_rolling[qb_cols].rename(columns={
                "player_id": "away_qb_id",
                "r_qb_passing_epa": "a_r_qb_epa",
                "r_qb_dakota":      "a_r_qb_dakota",
            }),
            on=["away_qb_id","season","week"], how="left"
        )
    else:
        df["h_r_qb_epa"] = df["a_r_qb_epa"] = 0.0
        df["h_r_qb_dakota"] = df["a_r_qb_dakota"] = 0.0

    # ── Diferenciales PBP ─────────────────────────────────────────────────
    df["d_off_epa"]        = df["h_off_epa"]        - df["a_off_epa"]
    df["d_def_epa"]        = df["h_def_epa"]         - df["a_def_epa"]
    df["d_pass_epa"]       = df["h_pass_epa"]        - df["a_pass_epa"]
    df["d_rush_epa"]       = df["h_rush_epa"]        - df["a_rush_epa"]
    df["d_def_pass_epa"]   = df["h_def_pass_epa"]    - df["a_def_pass_epa"]
    df["d_def_rush_epa"]   = df["h_def_rush_epa"]    - df["a_def_rush_epa"]
    df["d_third_conv"]     = df["h_third_conv"]       - df["a_third_conv"]
    df["d_def_third_conv"] = df["h_def_third_conv"]   - df["a_def_third_conv"]
    df["d_turnover_diff"]  = ((df["h_turnovers_def"] - df["h_turnovers_off"]) -
                              (df["a_turnovers_def"] - df["a_turnovers_off"]))
    df["d_sack_rate_off"]  = df["h_sack_rate_off"]   - df["a_sack_rate_off"]
    df["d_sack_rate_def"]  = df["h_sack_rate_def"]   - df["a_sack_rate_def"]
    df["d_rz_epa"]         = df["h_rz_epa"]          - df["a_rz_epa"]
    df["d_cpoe"]           = df["h_cpoe"].fillna(0)  - df["a_cpoe"].fillna(0)
    for _c in ("succ_off", "succ_def", "neutral_off_epa", "neutral_def_epa",
               "early_pass_off", "early_pass_def", "st_epa"):
        df[f"d_{_c}"] = df[f"h_{_c}"].fillna(0) - df[f"a_{_c}"].fillna(0)

    # ── Diferenciales Schedule ────────────────────────────────────────────
    df["d_win_pct"]     = df["h_sr_win"]         - df["a_sr_win"]
    df["d_pts_for"]     = df["h_sr_pts_for"]     - df["a_sr_pts_for"]
    df["d_pts_against"] = df["h_sr_pts_against"] - df["a_sr_pts_against"]
    df["d_pythag"]      = df["h_sr_pythag"]      - df["a_sr_pythag"]
    df["d_recent_win"]  = df["h_sf_win"]         - df["a_sf_win"]
    df["d_recent_pts"]  = df["h_sf_pts_for"]     - df["a_sf_pts_for"]

    # ── QB diferenciales ─────────────────────────────────────────────────
    df["d_qb_epa"]    = df["h_r_qb_epa"].fillna(0)    - df["a_r_qb_epa"].fillna(0)
    df["d_qb_dakota"] = df["h_r_qb_dakota"].fillna(0) - df["a_r_qb_dakota"].fillna(0)

    # ── Strength of Schedule ──────────────────────────────────────────────
    if sos_df is not None:
        df = df.merge(
            sos_df.rename(columns={"r_sos": "h_sos"}),
            left_on=["season", "week", "home_team"], right_on=["season", "week", "team"],
            how="left"
        ).drop(columns=["team"])
        df = df.merge(
            sos_df.rename(columns={"r_sos": "a_sos"}),
            left_on=["season", "week", "away_team"], right_on=["season", "week", "team"],
            how="left"
        ).drop(columns=["team"])
        df["d_sos"] = df["h_sos"].fillna(0.5) - df["a_sos"].fillna(0.5)
    else:
        df["d_sos"] = 0.0

    # ── Contexto ─────────────────────────────────────────────────────────
    df["rest_diff"]   = df["home_rest"].fillna(7) - df["away_rest"].fillna(7)
    df["div_game"]    = pd.to_numeric(df.get("div_game",  0), errors="coerce").fillna(0)
    df["spread_line"] = pd.to_numeric(df["spread_line"],      errors="coerce").fillna(0)

    # ── Weather & Moneyline ───────────────────────────────────────────────
    df = _add_weather_features(df)
    df = _add_moneyline_features(df)

    df = df[(df["h_games"] >= MIN_GAMES) & (df["a_games"] >= MIN_GAMES)].copy()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FASE 5 — ENTRENAMIENTO (Ensemble)
# ══════════════════════════════════════════════════════════════════════════════

class ModeloManningV6:
    """Ensemble de dos enfoques promediados:
    - clasificador (Voting XGB+LR+RF, el v5 de siempre)
    - regresion del margen de puntos (XGB+Ridge), convertida a probabilidad
      con una normal cuyo sigma sale de los residuos de entrenamiento.
    El margen ensena mas que el gano/perdio (27-24 no es 38-10) y los dos
    enfoques se equivocan distinto: su promedio gana a cualquiera por separado
    (verificado walk-forward en lab/manning_exp_bateria.py)."""

    def fit(self, X, y, margen):
        self.clf = _clasificador()
        self.clf.fit(X, y)
        self.reg = xgb.XGBRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.03,
            subsample=0.80, colsample_bytree=0.70, min_child_weight=5,
            gamma=1.0, reg_alpha=0.1, reg_lambda=2.0,
            random_state=42, n_jobs=-1, verbosity=0)
        self.rid = Pipeline([("scaler", StandardScaler()),
                             ("ridge", Ridge(alpha=10.0))])
        self.reg.fit(X, margen)
        self.rid.fit(X, margen)
        pred_tr = (self.reg.predict(X) + self.rid.predict(X)) / 2
        self.sigma = float(np.std(margen - pred_tr, ddof=1))
        return self

    def predict_proba(self, X):
        from scipy.stats import norm
        p_clf = self.clf.predict_proba(X)[:, 1]
        p_mar = norm.cdf((self.reg.predict(X) + self.rid.predict(X)) / 2 / self.sigma)
        p = (p_clf + p_mar) / 2
        return np.column_stack([1 - p, p])


def _clasificador():
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.03,
        subsample=0.80, colsample_bytree=0.70,
        min_child_weight=5, gamma=1.0,
        reg_alpha=0.1, reg_lambda=2.0,
        eval_metric="logloss", random_state=42,
        n_jobs=-1, verbosity=0,
    )
    lr_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.3, max_iter=2000, random_state=42)),
    ])
    rf_clf = RandomForestClassifier(
        n_estimators=400, max_depth=4, min_samples_leaf=10,
        random_state=42, n_jobs=-1,
    )
    return VotingClassifier(
        estimators=[("xgb", xgb_clf), ("lr", lr_clf), ("rf", rf_clf)],
        voting="soft", weights=[3, 1, 2],
    )


def train_model(X: pd.DataFrame, y: pd.Series, margen: pd.Series):
    return ModeloManningV6().fit(X, y, margen)


def evaluate(model, X_val, y_val, label="Val"):
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(y_val, preds)
    ll    = log_loss(y_val, probs)
    bs    = brier_score_loss(y_val, probs)
    print(f"  [{label}]  Acc={acc:.3f}  LogLoss={ll:.4f}  Brier={bs:.4f}")
    return acc, ll, bs


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def bench_vs_mercado(X_all, y_all, seas_all, m_all, n=4):
    """Banco de pruebas: el bot contra la linea de apuestas, walk-forward.

    No es un feature ni un visual: es el diagnostico honesto del modelo. El
    mercado ya esta dentro del bot como `home_impl_prob` (la feature mas
    importante con diferencia), asi que la pregunta util no es "cuanto acierta"
    sino "cuando se separa del mercado, quien tiene razon". Se entrena siempre
    con las temporadas previas a la que se evalua, nunca con ella.
    """
    print()
    print("=" * 72)
    print("  BANCO DE PRUEBAS — MANNING BOT vs MERCADO (moneyline)")
    print("=" * 72)

    filas = []
    for val_yr, train_yrs in folds_walk_forward(SEASONS_TRAIN, n=n):
        mask_tr = seas_all.isin(train_yrs)
        mask_vl = seas_all == val_yr
        modelo  = train_model(X_all[mask_tr], y_all[mask_tr], m_all[mask_tr])

        p_bot = modelo.predict_proba(X_all[mask_vl])[:, 1]
        p_mkt = X_all.loc[mask_vl, "home_impl_prob"].to_numpy()
        y_val = y_all[mask_vl].to_numpy()

        # Sin moneyline el mercado queda en 0.5 exacto: no es una opinion, es
        # un hueco. Se excluye de la comparacion en vez de contarlo como fallo.
        con_linea = p_mkt != 0.5
        if not con_linea.any():
            print(f"  {val_yr}: sin moneyline en los datos, se salta")
            continue
        p_bot, p_mkt, y_val = p_bot[con_linea], p_mkt[con_linea], y_val[con_linea]

        pick_bot = (p_bot >= 0.5).astype(int)
        pick_mkt = (p_mkt >= 0.5).astype(int)
        discrepan = pick_bot != pick_mkt

        filas.append({
            "season": val_yr,
            "n": len(y_val),
            "acc_bot": (pick_bot == y_val).mean(),
            "acc_mkt": (pick_mkt == y_val).mean(),
            "brier_bot": brier_score_loss(y_val, p_bot),
            "brier_mkt": brier_score_loss(y_val, p_mkt),
            "n_disc": int(discrepan.sum()),
            "acc_bot_disc": (pick_bot[discrepan] == y_val[discrepan]).mean()
                            if discrepan.any() else np.nan,
        })

    if not filas:
        print("  Sin temporadas evaluables.")
        return None

    b = pd.DataFrame(filas)
    print()
    print(f"  {'AÑO':<6}{'N':>5}{'BOT':>8}{'MERCADO':>9}{'BRIER B':>9}{'BRIER M':>9}"
          f"{'DISCREP':>9}{'BOT ahi':>9}")
    print("  " + "-" * 62)
    for _, r in b.iterrows():
        ad = "  n/d" if pd.isna(r['acc_bot_disc']) else f"{r['acc_bot_disc']*100:5.1f}%"
        print(f"  {int(r['season']):<6}{int(r['n']):>5}{r['acc_bot']*100:>7.1f}%"
              f"{r['acc_mkt']*100:>8.1f}%{r['brier_bot']:>9.4f}{r['brier_mkt']:>9.4f}"
              f"{int(r['n_disc']):>9}{ad:>9}")

    tot_disc  = b["n_disc"].sum()
    tot_n     = b["n"].sum()
    # Media ponderada por partidos, no media de medias: las temporadas con mas
    # discrepancias pesan mas en el veredicto.
    acc_disc  = float((b["acc_bot_disc"].fillna(0) * b["n_disc"]).sum() / tot_disc) \
                if tot_disc else float("nan")
    print("  " + "-" * 62)
    print(f"  Global   {tot_n:>4}{b['acc_bot'].mean()*100:>7.1f}%"
          f"{b['acc_mkt'].mean()*100:>8.1f}%"
          f"{b['brier_bot'].mean():>9.4f}{b['brier_mkt'].mean():>9.4f}"
          f"{tot_disc:>9}{acc_disc*100:>8.1f}%")
    print()
    print(f"  El bot se separa del mercado en {tot_disc}/{tot_n} partidos "
          f"({tot_disc/tot_n*100:.1f}%).")
    if acc_disc == acc_disc:
        veredicto = ("el bot gana esos duelos" if acc_disc > 0.5 else
                     "el mercado gana esos duelos" if acc_disc < 0.5 else
                     "empate tecnico")
        print(f"  Ahi acierta el {acc_disc*100:.1f}% — {veredicto}.")
        print("  (Por debajo de 50% la discrepancia es ruido, no ventaja:")
        print("   seguir al mercado seria mejor politica en esos partidos.)")
    print()
    return b


def _args():
    """Flags para poder correr sin teclado (cron, reentrenamiento programado)."""
    import argparse
    ap = argparse.ArgumentParser(description="Manning Bot — predictor NFL")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--retrain", action="store_true",
                   help="reentrena sin preguntar y guarda el modelo")
    g.add_argument("--no-retrain", action="store_true",
                   help="usa el modelo guardado sin preguntar")
    ap.add_argument("--week", type=int, default=None,
                    help="semana a pronosticar (por defecto, la proxima)")
    ap.add_argument("--bench", action="store_true",
                    help="compara el bot contra la linea de apuestas y sale")
    return ap.parse_args()


if __name__ == "__main__":
    import pickle

    ARGS = _args()
    INTERACTIVO = not (ARGS.retrain or ARGS.no_retrain or ARGS.week is not None
                       or ARGS.bench)

    # 1. Schedules
    schedules = load_schedules()
    print(f"Schedules cargados: {len(schedules):,} partidos")
    resolver_temporadas(schedules)

    # 2. PBP game logs (con cache)
    all_logs_cache = CACHE_DIR / "game_logs_all.parquet"
    logs_ok = False
    if all_logs_cache.exists():
        import pyarrow.parquet as pq
        logs_ok = "st_epa" in pq.read_schema(all_logs_cache).names
        if not logs_ok:
            print("Cache de game logs sin las columnas v6 — se reconstruye...")
            all_logs_cache.unlink()
    if logs_ok:
        print("Cargando game logs desde cache...")
        game_logs = pd.read_parquet(all_logs_cache)
    else:
        print("Construyendo game logs (primera vez, ~5-10 min)...")
        # La temporada a predecir solo tiene PBP si ya ha empezado; antes del
        # kickoff su fichero no existe en nflverse (404) y no hay nada que leer
        _sp = schedules[(schedules["game_type"] == "REG") &
                        (schedules["season"] == SEASON_PRED)]
        pred_empezada = pd.to_numeric(_sp["home_score"], errors="coerce").notna().any()
        temporadas = SEASONS_TRAIN + ([SEASON_PRED] if pred_empezada else [])
        if not pred_empezada:
            print(f"  ({SEASON_PRED} aun sin partidos — se omite su PBP)")
        dfs = []
        for season in temporadas:
            pbp  = load_pbp(season)
            logs = build_game_logs(pbp)
            dfs.append(logs)
            print(f"  {season}: {len(logs)} team-weeks")
        game_logs = pd.concat(dfs, ignore_index=True)
        game_logs.to_parquet(all_logs_cache, index=False)

    # 2b. Refrescar la temporada en curso si el cache va por detras de schedules
    sched_pred = schedules[(schedules["game_type"] == "REG") &
                           (schedules["season"] == SEASON_PRED)].copy()
    sched_pred = coerce(sched_pred, ["home_score", "week"])
    jugadas = sched_pred.dropna(subset=["home_score"])["week"]
    last_played = int(jugadas.max()) if len(jugadas) else 0
    en_logs = game_logs.loc[game_logs["season"] == SEASON_PRED, "week"]
    last_logs = int(en_logs.max()) if len(en_logs) else 0
    if last_played > last_logs:
        print(f"Game logs {SEASON_PRED} llegan a semana {last_logs}, "
              f"jugada la {last_played} — actualizando PBP...")
        pbp_cache = CACHE_DIR / f"pbp_{SEASON_PRED}.parquet"
        if pbp_cache.exists():
            pbp_cache.unlink()
        logs_new  = build_game_logs(load_pbp(SEASON_PRED))
        game_logs = pd.concat(
            [game_logs[game_logs["season"] != SEASON_PRED], logs_new],
            ignore_index=True)
        game_logs.to_parquet(all_logs_cache, index=False)
        print(f"  {SEASON_PRED}: {len(logs_new)} team-weeks actualizados")

    # 3. Todas las features de soporte
    rolling       = add_season_carryover(compute_rolling(game_logs))
    sched_logs    = build_schedule_logs(schedules)
    sched_rolling = compute_schedule_rolling(sched_logs)
    elo_df        = compute_elo(schedules)
    sos_df        = compute_sos(elo_df)
    qb_rolling    = load_qb_rolling()

    # 4. Feature matrix
    print("\nConstruyendo feature matrix...")
    train_df = build_features(schedules, rolling, sched_rolling,
                              elo_df, qb_rolling, sos_df, seasons=SEASONS_TRAIN)
    print(f"  Partidos con datos completos: {len(train_df):,}")

    X_all     = train_df[FEATURE_COLS].fillna(0)
    y_all     = train_df["home_win"]
    m_all     = pd.to_numeric(train_df["result"], errors="coerce")
    seas_all  = train_df["season"]

    if ARGS.bench:
        bench_vs_mercado(X_all, y_all, seas_all, m_all)
        raise SystemExit(0)

    # 5. Entrenamiento
    if os.path.exists(MODEL_FILE):
        # Un modelo guardado el año pasado sigue cargando sin quejarse pero le
        # falta la ultima temporada entera: hay que avisar, no dejarlo pasar.
        caduco = _modelo_caducado()
        if caduco:
            print(f"\n  AVISO: '{MODEL_FILE}' se entreno hasta {caduco}, pero ya hay "
                  f"datos hasta {SEASONS_TRAIN[-1]}.")
            print( "  Conviene reentrenar antes de publicar pronosticos.")
        if ARGS.retrain or ARGS.no_retrain:
            retrain = ARGS.retrain
            print(f"  (--{'retrain' if retrain else 'no-retrain'})")
        else:
            sugerencia = "S/n" if caduco else "s/N"
            resp = input(f"\nYa existe '{MODEL_FILE}'. Reentrenar? ({sugerencia}): ").strip().lower()
            retrain = (resp == "s") or (caduco and resp == "")
    else:
        retrain = True

    if retrain:
        # Walk-forward CV para evaluación honesta
        print("\nWalk-forward cross-validation...")
        folds = folds_walk_forward(SEASONS_TRAIN)
        wf_accs = []
        for val_yr, train_yrs in folds:
            mask_tr = seas_all.isin(train_yrs)
            mask_vl = seas_all == val_yr
            m_tmp   = train_model(X_all[mask_tr], y_all[mask_tr], m_all[mask_tr])
            acc, _, _ = evaluate(m_tmp, X_all[mask_vl], y_all[mask_vl], label=str(val_yr))
            wf_accs.append(acc)
        print(f"  Walk-forward mean Acc: {np.mean(wf_accs):.3f}")

        # Modelo final: entrenado con TODAS las temporadas de entrenamiento
        print(f"\nEntrenando Manning Bot v6 final "
              f"(todos los datos {SEASONS_TRAIN[0]}-{SEASONS_TRAIN[-1]})...")
        model = train_model(X_all, y_all, m_all)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        _guardar_meta()
        print(f"  Modelo guardado: {MODEL_FILE} (hasta {SEASONS_TRAIN[-1]})")
    else:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        print(f"Modelo cargado desde {MODEL_FILE}")
        # Evaluar en las 3 ultimas temporadas de entrenamiento, para referencia
        ref_desde = SEASONS_TRAIN[-3]
        mask_val  = seas_all >= ref_desde
        evaluate(model, X_all[mask_val], y_all[mask_val],
                 label=f"{ref_desde}-{SEASONS_TRAIN[-1]} (ref)")

    # 6. Feature importance (XGBoost del ensemble)
    fi = pd.Series(model.clf.named_estimators_["xgb"].feature_importances_,
                   index=FEATURE_COLS)
    print("\nTop features (XGBoost):")
    for feat, imp in fi.sort_values(ascending=False).head(14).items():
        bar = "#" * int(imp * 200)
        print(f"  {feat:<22}  {imp:.4f}  {bar}")

    # 7. Prediccion de la temporada en curso
    print(f"\nConstruyendo features para {SEASON_PRED}...")
    pred_df = build_features(schedules, rolling, sched_rolling,
                             elo_df, qb_rolling, sos_df,
                             seasons=[SEASON_PRED], include_target=False)

    sched_reg = schedules[
        (schedules["season"] == SEASON_PRED) & (schedules["game_type"] == "REG")
    ].copy()
    sched_reg      = coerce(sched_reg, ["home_score"])
    completed_reg  = set(sched_reg.dropna(subset=["home_score"])["week"].astype(int).unique())
    last_reg       = max(completed_reg) if completed_reg else 0
    available      = sorted(pred_df["week"].astype(int).unique())

    # Por defecto se pronostica la PROXIMA jornada, no la ultima jugada: esto es
    # un predictor, no un backtest. Las jugadas siguen disponibles a mano para
    # revisar aciertos.
    pendientes = [w for w in available if w not in completed_reg]
    por_defecto = min(pendientes) if pendientes else last_reg

    print(f"Semanas disponibles: {available}")
    print(f"Ultima semana completada: {last_reg}"
          + (f"  |  proxima jornada: {por_defecto}" if pendientes else "  |  temporada cerrada"))

    if not available:
        if last_reg == 0:
            print(f"\n  SIN MUESTRA: la temporada {SEASON_PRED} no ha empezado.")
        else:
            print(f"\n  SIN MUESTRA: jugadas {last_reg} jornada(s), y el bot necesita que"
                  f" cada equipo lleve {MIN_GAMES} partidos.")
        print(f"  Primeros picks: semana {MIN_GAMES + 1}. Primer balance: el martes"
              f" siguiente a esa jornada.\n")
        raise SystemExit(EXIT_SIN_MUESTRA)

    rango = f"{min(available)}-{max(available)}"
    if ARGS.week is not None:
        semana = ARGS.week
        print(f"\nSemana pedida por --week: {semana}")
    elif INTERACTIVO:
        semana_str = input(f"\nQue semana? ({rango}, Enter = {por_defecto}): ").strip()
        semana     = int(semana_str) if semana_str.isdigit() else por_defecto
    else:
        semana = por_defecto
        print(f"\nSemana: {semana} (por defecto, sin preguntar)")

    games_semana = pred_df[pred_df["week"].astype(int) == semana].copy()
    if games_semana.empty:
        # Antes salia con exit 0 y el batch daba por bueno un TXT sin picks
        print(f"\n  SIN MUESTRA para la semana {semana}: el bot solo predice partidos"
              f" en que cada equipo lleva {MIN_GAMES} jugados (semanas {rango}).\n")
        raise SystemExit(EXIT_SIN_MUESTRA)
    else:
        X_pred = games_semana[FEATURE_COLS].fillna(0)
        probs  = model.predict_proba(X_pred)[:, 1]
        games_semana["home_win_prob"]    = probs
        games_semana["away_win_prob"]    = 1 - probs
        games_semana["predicted_winner"] = [
            r["home_team"] if p >= 0.5 else r["away_team"]
            for p, (_, r) in zip(probs, games_semana.iterrows())
        ]
        games_semana["confidence"] = games_semana[["home_win_prob","away_win_prob"]].max(axis=1)

        if semana in completed_reg and "result" in games_semana.columns:
            res = pd.to_numeric(games_semana["result"], errors="coerce")
            games_semana["actual_winner"] = games_semana["home_team"].where(
                res > 0, games_semana["away_team"].where(res < 0, "TIE"))
            games_semana["correct"] = (
                games_semana["predicted_winner"] == games_semana["actual_winner"])
        else:
            games_semana["actual_winner"] = "?"
            games_semana["correct"] = None

        print(f"\n{'='*75}")
        print(f"  MANNING BOT v6 -- NFL {SEASON_PRED}  |  Semana {semana}")
        print(f"{'='*75}")
        print(f"  {'PARTIDO':<28}  {'PRED':>5}  {'P.HOME':>7}  {'P.AWAY':>7}  {'REAL':>5}  OK")
        print(f"  {'-'*68}")

        aciertos = total = 0
        for _, r in games_semana.sort_values("confidence", ascending=False).iterrows():
            matchup = f"{r['away_team']} @ {r['home_team']}"
            ph  = f"{r['home_win_prob']*100:.1f}%"
            pa  = f"{r['away_win_prob']*100:.1f}%"
            ok  = ""
            if r["correct"] is True:
                ok = "OK"; aciertos += 1; total += 1
            elif r["correct"] is False:
                ok = "X"; total += 1
            print(f"  {matchup:<28}  {r['predicted_winner']:>5}  {ph:>7}  {pa:>7}  {r['actual_winner']:>5}  {ok}")

        if total > 0:
            print(f"\n  Aciertos semana {semana}: {aciertos}/{total} ({aciertos/total*100:.1f}%)")
        print()
