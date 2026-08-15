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
    cargar_ftn(season, refrescar)                     -> (DataFrame, season)  FTN charting (play-action... 2022+)
    temporada_actual()                                -> int

Cache y frescura:
    - PBP:  pbp_cache/pbp_full_{season}.parquet — re-descarga solo si schedules
      muestra una jornada jugada posterior a la del cache.
    - Stats/participación: re-descarga si el cache tiene >3 días y la temporada
      es la actual.
    - Sin internet: usa siempre el cache disponible, AVISANDO por consola.
      Nunca se sirve un cache sin verificar en silencio (ver aviso_frescura()).

Sello de datos:
    sello(season) devuelve "NFL 2026 · datos hasta sem. 7" para estampar al pie
    de los PNG. Si la frescura no se pudo verificar, añade "(sin verificar)".
"""
import os, sys, time
import socket
from datetime import date
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError

import pandas as pd


class DatosNoDisponibles(Exception):
    """El dataset pedido aún no existe en nflverse (típico al arrancar la temporada)."""


def _excepthook(tipo, valor, tb):
    """Un dato que aún no existe no es un fallo del programa: no merece 15 líneas
    de traceback que hagan pensar que algo está roto. Como todos los scripts
    importan este módulo, con engancharlo aquí queda cubierto el proyecto entero."""
    if issubclass(tipo, DatosNoDisponibles):
        print(f"\n  Todavia no se puede generar este grafico.\n  {valor}\n",
              file=sys.stderr)
        return
    _excepthook_previo(tipo, valor, tb)


_excepthook_previo = sys.excepthook
sys.excepthook = _excepthook

# Una descarga colgada no debe congelar el script para siempre
# (urlretrieve y read_csv de URLs respetan el timeout global del socket)
socket.setdefaulttimeout(30)

CACHE = "pbp_cache"

PBP_URL   = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_reg_{season}.csv.gz"
PART_URL  = "https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{season}.parquet"
FTN_URL   = "https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.parquet"
SCHED_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

_sched_info = None   # (temporada, última semana REG jugada) — 1 descarga por ejecución
_aviso_dado = False  # el aviso de frescura se imprime una sola vez por ejecución


def _info_schedules():
    """(temporada actual, última semana REG jugada) o None si no se pudo consultar.

    Cuando devuelve None NADIE puede saber si el cache está al día, así que
    todos los caminos que dependen de esto deben avisar (ver aviso_frescura).
    """
    global _sched_info
    if _sched_info is None:
        try:
            sch = pd.read_csv(SCHED_URL, low_memory=False,
                              usecols=["season", "game_type", "week", "home_score"])
            reg = sch[(sch["game_type"] == "REG") & sch["home_score"].notna()]
            _sched_info = (int(reg["season"].max()),
                           int(reg[reg["season"] == reg["season"].max()]["week"].max()))
        except Exception as e:
            _sched_info = False
            _aviso_sin_verificar(e)
    return _sched_info or None


def _aviso_sin_verificar(e=None):
    """Grita cuando la frescura no se puede comprobar. Un cache viejo servido en
    silencio es el fallo más peligroso del proyecto: el PNG sale perfecto y los
    datos son de hace semanas."""
    global _aviso_dado
    if _aviso_dado:
        return
    _aviso_dado = True
    print("")
    print("  " + "!" * 62)
    print("  !!  AVISO: no se ha podido comprobar si hay jornada nueva.")
    if e is not None:
        print(f"  !!  Motivo: {type(e).__name__}: {str(e)[:70]}")
    print("  !!  Se usara el CACHE LOCAL, que puede tener semanas de retraso.")
    print("  !!  NO publiques nada de esta ejecucion sin verificar la fecha.")
    print("  " + "!" * 62)
    print("")


def frescura_verificada() -> bool:
    """True si se ha podido confirmar contra schedules cuál es la última jornada."""
    return _info_schedules() is not None


def ultima_semana(season=None):
    """Última semana REG jugada de `season`, o None si no se pudo determinar."""
    info = _info_schedules()
    if not info:
        return None
    if season is None or season == info[0]:
        return info[1]
    return 18 if season < info[0] else None


def sello(season=None, prefijo="NFL"):
    """Texto para el pie de los PNG: 'NFL 2026 · datos hasta sem. 7'.

    Deja constancia EN LA IMAGEN de hasta dónde llegan los datos, para que un
    cache viejo no pueda pasar por fresco una vez publicado el gráfico."""
    if season is None:
        season = temporada_actual()
    wk = ultima_semana(season)
    if wk is None:
        return f"{prefijo} {season} · frescura sin verificar"
    return f"{prefijo} {season} · datos hasta sem. {wk}"


def temporada_actual() -> int:
    info = _info_schedules()
    if info:
        return info[0]
    hoy = date.today()   # fallback sin internet: sep-dic = año en curso
    return hoy.year if hoy.month >= 9 else hoy.year - 1


# ──────────────────────────────────────────────────────────────────────────────
# CLI comun, salidas archivadas y guardia de muestra
# ──────────────────────────────────────────────────────────────────────────────
SALIDAS = "salidas"
_cli = None


def cli():
    """Lee --season / --week / --raiz de la linea de comandos (todos opcionales).

    Sin argumentos todo sigue funcionando igual que antes: temporada
    autodetectada y la semana se pide por input() donde haga falta.
    """
    global _cli
    if _cli is None:
        import argparse
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--season", type=int, default=None)
        p.add_argument("--week", type=int, default=None)
        p.add_argument("--raiz", action="store_true",
                       help="guardar el PNG en la raiz en vez de salidas/")
        _cli, _ = p.parse_known_args()
    return _cli


def season_cli(defecto=None):
    """Temporada pedida por --season, o la de siempre."""
    return cli().season if cli().season is not None else defecto


def week_cli(defecto=None):
    """Semana pedida por --week, o None."""
    return cli().week if cli().week is not None else defecto


def salida(nombre, season=None, week=None):
    """Ruta de salida con la semana estampada: salidas/2026/w07/proe_2026_w07.png

    La semana por defecto es hasta donde llegan los datos, asi que el archivo
    queda ordenado solo y el gráfico de la semana 4 no pisa al de la semana 3.
    Con --raiz se conserva el comportamiento antiguo (PNG en la raiz).
    """
    if season is None:
        season = temporada_actual()
    if week is None:
        week = ultima_semana(season)

    if week is None:                       # frescura sin verificar: sin sufijo
        base, sufijo = nombre, ""
    else:
        week = int(week)
        raiz, ext = os.path.splitext(nombre)
        base, sufijo = f"{raiz}_w{week:02d}{ext}", f"w{week:02d}"

    if cli().raiz:
        return base
    destino = os.path.join(SALIDAS, str(season), sufijo) if sufijo \
        else os.path.join(SALIDAS, str(season))
    os.makedirs(destino, exist_ok=True)
    return os.path.join(destino, base)


def aviso_muestra(df, minimo=150, columna="posteam"):
    """Avisa (y devuelve el texto) si aun no hay muestra para un grafico de temporada.

    En la jornada 1 la autodeteccion ya apunta a la temporada nueva, asi que sin
    esto un ranking de 32 equipos se dibuja alegremente a partir de dos.
    """
    try:
        por_equipo = df[df[columna].notna()].groupby(columna).size()
    except Exception:
        return None
    if por_equipo.empty:
        return None
    mediana = int(por_equipo.median())
    if mediana >= minimo:
        return None
    texto = (f"MUESTRA BAJA: {len(por_equipo)} equipos, mediana de "
             f"{mediana} jugadas (minimo recomendado {minimo})")
    print("")
    print("  " + "-" * 62)
    print(f"  !!  {texto}")
    print( "  !!  Es demasiado pronto para un grafico de temporada completa.")
    print( "  !!  Publicalo solo si sabes lo que estas haciendo.")
    print("  " + "-" * 62)
    print("")
    return texto


def _descargar(url: str, destino: str, etiqueta: str):
    print(f"Descargando {etiqueta} (solo cuando hay datos nuevos)...")
    tmp = destino + ".tmp"
    try:
        urlretrieve(url, tmp)
    except HTTPError as e:
        _limpiar(tmp)
        if e.code == 404:
            raise DatosNoDisponibles(
                f"{etiqueta} todavia no esta publicado en nflverse.\n"
                f"    Es normal al arrancar la temporada: el play-by-play sale horas\n"
                f"    despues del partido, pero el charting (FTN, participacion) lo\n"
                f"    apuntan a mano y tarda dias o semanas.\n"
                f"    URL consultada: {url}") from None
        raise
    except (URLError, OSError) as e:
        _limpiar(tmp)
        raise DatosNoDisponibles(
            f"No se pudo descargar {etiqueta} ({type(e).__name__}: {str(e)[:60]}).\n"
            f"    Revisa la conexion y vuelve a intentarlo.") from None
    os.replace(tmp, destino)


def _limpiar(tmp):
    """Un .tmp a medias de una descarga fallida no debe quedarse en pbp_cache/."""
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except OSError:
        pass


def cargar_pbp(season=None, columns=None, solo_reg=True, refrescar=False,
               avisar=True):
    """Play-by-play de nflverse. Devuelve (df, season).

    season   None = auto-detectar última temporada con partidos
    columns  lista de columnas (None = todas)
    solo_reg True = solo temporada regular (sin playoffs)
    avisar   True = grita si aun no hay muestra para un grafico de temporada
    """
    if season is None:
        season = temporada_actual()
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"pbp_full_{season}.parquet")

    necesita = refrescar or not os.path.exists(cache)
    if not necesita:
        info = _info_schedules()
        if info is None:
            _aviso_sin_verificar()          # cache servido a ciegas: hay que avisar
        elif info[0] == season:
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
    if avisar:
        aviso_muestra(df)
    return df.copy(), season


def _cargar_auxiliar(season, refrescar, nombre_cache, url, etiqueta, es_csv=False):
    if season is None:
        season = temporada_actual()
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"{nombre_cache}_{season}.parquet")

    necesita = refrescar or not os.path.exists(cache)
    if not necesita:
        info = _info_schedules()
        if info is None:
            _aviso_sin_verificar()          # cache servido a ciegas: hay que avisar
        else:
            viejo = (time.time() - os.path.getmtime(cache)) > 3 * 86400
            necesita = bool(info[0] == season and viejo)
    if necesita:
        try:
            if es_csv:
                try:
                    df = pd.read_csv(url.format(season=season), low_memory=False,
                                     compression="infer")
                except HTTPError as e:
                    if e.code == 404:
                        raise DatosNoDisponibles(
                            f"{etiqueta} {season} todavia no esta publicado en nflverse."
                        ) from None
                    raise
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


def cargar_ftn(season=None, refrescar=False):
    """Charting FTN (is_play_action, blitzers, motion... solo 2022+).
    Join con PBP: nflverse_game_id + nflverse_play_id. Devuelve (df, season)."""
    return _cargar_auxiliar(season, refrescar, "ftn_charting", FTN_URL, "FTN charting")
