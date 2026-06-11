"""
pbp_loader.py
Carga compartida de datos nflverse con cache local en pbp_cache/.
Único módulo común del proyecto (excepción documentada en CLAUDE.md).

Uso típico en un script:
    from pbp_loader import cargar_pbp
    SEASON = None                      # None = auto-detectar última temporada
    df, SEASON = cargar_pbp(SEASON)    # REG-only, cacheado

Funciones:
    cargar_pbp(season, columns, solo_reg, refrescar)  -> (DataFrame, season)
    cargar_stats(season, refrescar)                   -> (DataFrame, season)  player stats REG
    cargar_participation(season, refrescar)           -> (DataFrame, season)  FTN (rutas, presión, coberturas)
    temporada_actual()                                -> int

Cache y frescura:
    - PBP:  pbp_cache/pbp_full_{season}.parquet — re-descarga solo si schedules
      muestra una jornada jugada posterior a la del cache.
    - Stats/participación: re-descarga si el cache tiene >3 días y la temporada
      es la actual.
    - Sin internet: usa siempre el cache disponible.
"""
import os, time
from datetime import date
from urllib.request import urlretrieve

import pandas as pd

CACHE = "pbp_cache"

PBP_URL   = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_reg_{season}.csv.gz"
PART_URL  = "https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{season}.parquet"
SCHED_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

_sched_info = None   # (temporada, última semana REG jugada) — 1 descarga por ejecución


def _info_schedules():
    """(temporada actual, última semana REG jugada) o None si no hay internet."""
    global _sched_info
    if _sched_info is None:
        try:
            sch = pd.read_csv(SCHED_URL, low_memory=False,
                              usecols=["season", "game_type", "week", "home_score"])
            reg = sch[(sch["game_type"] == "REG") & sch["home_score"].notna()]
            _sched_info = (int(reg["season"].max()),
                           int(reg[reg["season"] == reg["season"].max()]["week"].max()))
        except Exception:
            _sched_info = False
    return _sched_info or None


def temporada_actual() -> int:
    info = _info_schedules()
    if info:
        return info[0]
    hoy = date.today()   # fallback sin internet: sep-dic = año en curso
    return hoy.year if hoy.month >= 9 else hoy.year - 1


def _descargar(url: str, destino: str, etiqueta: str):
    print(f"Descargando {etiqueta} (solo cuando hay datos nuevos)...")
    tmp = destino + ".tmp"
    urlretrieve(url, tmp)
    os.replace(tmp, destino)


def cargar_pbp(season=None, columns=None, solo_reg=True, refrescar=False):
    """Play-by-play de nflverse. Devuelve (df, season).

    season   None = auto-detectar última temporada con partidos
    columns  lista de columnas (None = todas)
    solo_reg True = solo temporada regular (sin playoffs)
    """
    if season is None:
        season = temporada_actual()
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"pbp_full_{season}.parquet")

    necesita = refrescar or not os.path.exists(cache)
    if not necesita:
        info = _info_schedules()
        if info and info[0] == season:
            try:
                max_cache = pd.read_parquet(cache, columns=["week"])["week"].max()
                necesita = max_cache < info[1]
            except Exception:
                necesita = True
    if necesita:
        _descargar(PBP_URL.format(season=season), cache, f"PBP {season}")

    leer = None
    if columns is not None:
        leer = list(dict.fromkeys(list(columns) + ["season_type"]))
    df = pd.read_parquet(cache, columns=leer)
    if solo_reg:
        df = df[df["season_type"] == "REG"]
    if columns is not None:
        df = df[[c for c in columns]]
    return df.copy(), season


def _cargar_auxiliar(season, refrescar, nombre_cache, url, etiqueta, es_csv=False):
    if season is None:
        season = temporada_actual()
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"{nombre_cache}_{season}.parquet")

    necesita = refrescar or not os.path.exists(cache)
    if not necesita:
        info = _info_schedules()
        en_curso = info and info[0] == season
        viejo = (time.time() - os.path.getmtime(cache)) > 3 * 86400
        necesita = bool(en_curso and viejo)
    if necesita:
        try:
            if es_csv:
                df = pd.read_csv(url.format(season=season), low_memory=False, compression="infer")
                df.to_parquet(cache, index=False)
            else:
                _descargar(url.format(season=season), cache, f"{etiqueta} {season}")
        except Exception as e:
            if not os.path.exists(cache):
                raise
            print(f"  Aviso: no se pudo refrescar {etiqueta} ({e}) — usando cache")
    return pd.read_parquet(cache), season


def cargar_stats(season=None, refrescar=False):
    """Player stats de temporada regular (stats_player_reg). Devuelve (df, season)."""
    if season is None:
        season = temporada_actual()
    print_needed = not os.path.exists(os.path.join(CACHE, f"stats_player_reg_{season}.parquet"))
    if print_needed:
        print(f"Descargando player stats {season}...")
    return _cargar_auxiliar(season, refrescar, "stats_player_reg", STATS_URL,
                            "player stats", es_csv=True)


def cargar_participation(season=None, refrescar=False):
    """Datos FTN de participación (rutas, presión, coberturas, personal). Devuelve (df, season)."""
    return _cargar_auxiliar(season, refrescar, "pbp_part", PART_URL, "participación")
