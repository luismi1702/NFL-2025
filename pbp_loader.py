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
    - PBP:  pbp_cache/pbp_full_{season}.parquet — re-descarga si schedules
      muestra una jornada posterior a la del cache O MAS PARTIDOS jugados que
      los que hay dentro (el jueves y el domingo son la misma semana).
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

_REL = "https://github.com/nflverse/nflverse-data/releases/download/"

PBP_URL   = _REL + "pbp/play_by_play_{season}.parquet"
STATS_URL = _REL + "stats_player/stats_player_reg_{season}.csv.gz"
PART_URL  = _REL + "pbp_participation/pbp_participation_{season}.parquet"
FTN_URL   = _REL + "ftn_charting/ftn_charting_{season}.parquet"
SCHED_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

# Fuentes por temporada que SI se actualizan cada pocas horas en temporada
# (verificado en los cron de nflverse-pfr / nflverse-rosters, ago-2026)
PFR_SEASON_URL = _REL + "pfr_advstats/advstats_season_{tipo}.parquet"
PFR_WEEK_URL   = _REL + "pfr_advstats/advstats_week_{tipo}_{season}.parquet"
SNAPS_URL      = _REL + "snap_counts/snap_counts_{season}.parquet"
INJ_URL        = _REL + "injuries/injuries_{season}.parquet"
TEAM_URL       = _REL + "stats_team/stats_team_week_{season}.parquet"

# Fuentes con TODAS las temporadas en un solo fichero (sin {season})
NGS_URL       = _REL + "nextgen_stats/ngs_{tipo}.parquet"
QBR_URL       = _REL + "espn_data/qbr_{nivel}_level.parquet"
CONTRATOS_URL = _REL + "contracts/historical_contracts.parquet"
EQUIPOS_URL   = _REL + "teams/teams_colors_logos.parquet"
ROSTER_URL    = _REL + "rosters/roster_{season}.parquet"

_sched_info = None   # (temporada, última semana REG jugada, partidos REG jugados)
_aviso_dado = False  # el aviso de frescura se imprime una sola vez por ejecución


def _info_schedules():
    """(temporada, última semana REG jugada, partidos REG jugados) o None.

    El conteo de partidos existe porque la semana NO basta: el jueves y el
    domingo de una misma jornada son la misma semana, así que un cache bajado
    tras el partido inaugural parecía al día con 2 de 16 partidos dentro.

    Cuando devuelve None NADIE puede saber si el cache está al día, así que
    todos los caminos que dependen de esto deben avisar (ver aviso_frescura).
    """
    global _sched_info
    if _sched_info is None:
        try:
            sch = pd.read_csv(SCHED_URL, low_memory=False,
                              usecols=["season", "game_type", "week", "home_score"])
            reg = sch[(sch["game_type"] == "REG") & sch["home_score"].notna()]
            cur = reg[reg["season"] == reg["season"].max()]
            _sched_info = (int(reg["season"].max()),
                           int(cur["week"].max()),
                           int(len(cur)))
        except Exception as e:
            _sched_info = False
            _aviso_sin_verificar(e)
    return _sched_info or None


def partidos_jugados(season=None):
    """Partidos REG jugados de `season` según el calendario, o None sin verificar."""
    info = _info_schedules()
    if not info or (season is not None and season != info[0]):
        return None
    return info[2]


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
                # Semana Y partidos: dentro de una misma jornada la semana no
                # cambia, pero el numero de partidos publicados sí (jueves 1,
                # domingo 15). Comparar solo la semana servia un cache a medias.
                loc = pd.read_parquet(cache, columns=["week", "game_id", "season_type"])
                reg = loc[loc["season_type"] == "REG"]
                necesita = (reg["week"].max() < info[1] or
                            reg["game_id"].nunique() < info[2])
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
        elif info[0] == season:
            edad = time.time() - os.path.getmtime(cache)
            necesita = edad > 3 * 86400
            # Mismo punto ciego que el PBP: dentro de una misma jornada el cache
            # no "envejece" pero la fuente sí crece (stats_team tenia 2 partidos
            # en cache y 15 publicados). Si trae game_id y le faltan partidos,
            # se refresca. El minimo de 6 h evita re-descargar en cada llamada
            # una fuente que va retrasada de origen (PFR semanal, QBR).
            if not necesita and edad > 6 * 3600:
                try:
                    loc = pd.read_parquet(cache)
                    if "game_id" in loc.columns:
                        if "season_type" in loc.columns:
                            es_reg = (loc["season_type"].astype(str)
                                      .str.upper().str.startswith("REG"))
                            loc = loc[es_reg]
                        necesita = loc["game_id"].nunique() < info[2]
                except Exception:
                    necesita = True
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
    Join con PBP: nflverse_game_id + nflverse_play_id. Devuelve (df, season).

    OJO: no trae cobertura ni personal — eso solo vive en cargar_participation."""
    return _cargar_auxiliar(season, refrescar, "ftn_charting", FTN_URL, "FTN charting")


# ──────────────────────────────────────────────────────────────────────────────
# Fuentes que el proyecto no usaba (ago-2026). Todas verificadas: se actualizan
# cada 6 h (PFR, snaps) o a diario (NGS, lesiones) durante la temporada.
# ──────────────────────────────────────────────────────────────────────────────
_PFR_TIPOS = ("def", "pass", "rec", "rush")


def cargar_pfr(tipo="def", season=None, semanal=False, refrescar=False):
    """Stats avanzadas de Pro Football Reference. Devuelve (df, season).

    tipo     def  -> presiones, placajes fallados y COBERTURA por defensor
                     (tgt, cmp_percent, yds_tgt, rat, dadot, m_tkl_percent...)
             pass -> pocket_time, pressure_pct, bad_throw_pct, on_tgt_pct por QB
             rec  -> ybc/yac separados, adot, brk_tkl, drop_percent
             rush -> ybc_att (yardas antes del contacto), yac_att, brk_tkl
    semanal  True = fichero de la temporada por semanas; False = acumulado

    Es la fuente que cubre los items 3 y 6 de docs/pff-wishlist.md sin pagar.
    """
    if tipo not in _PFR_TIPOS:
        raise ValueError(f"tipo debe ser uno de {_PFR_TIPOS}")
    if semanal:
        return _cargar_auxiliar(season, refrescar, f"pfr_week_{tipo}",
                                PFR_WEEK_URL.replace("{tipo}", tipo),
                                f"PFR {tipo} semanal")
    # El acumulado trae todas las temporadas en un fichero
    df = _cargar_global(refrescar, f"pfr_season_{tipo}",
                        PFR_SEASON_URL.format(tipo=tipo), f"PFR {tipo}")
    if season is None:
        season = temporada_actual()
    return df[df["season"] == season].copy(), season


def cargar_ngs(tipo="passing", season=None, refrescar=False):
    """Next Gen Stats de la NFL. Devuelve (df, season).

    passing   -> avg_time_to_throw, aggressiveness, avg_air_yards_to_sticks
    receiving -> avg_separation, avg_cushion, avg_yac_above_expectation
    rushing   -> rush_yards_over_expected, avg_time_to_los, %8+ en la caja
    """
    if tipo not in ("passing", "receiving", "rushing"):
        raise ValueError("tipo debe ser passing, receiving o rushing")
    df = _cargar_global(refrescar, f"ngs_{tipo}", NGS_URL.format(tipo=tipo),
                        f"NGS {tipo}")
    if season is None:
        season = temporada_actual()
    return df[df["season"] == season].copy(), season


def cargar_lesiones(season=None, refrescar=False):
    """Parte de lesiones semanal (estado, practica, parte del cuerpo)."""
    return _cargar_auxiliar(season, refrescar, "injuries", INJ_URL, "lesiones")


def cargar_snaps(season=None, refrescar=False):
    """Snaps por jugador y semana (ofensa / defensa / equipos especiales)."""
    return _cargar_auxiliar(season, refrescar, "snap_counts", SNAPS_URL, "snap counts")


def cargar_stats_equipo(season=None, refrescar=False):
    """Stats de equipo por semana, ya agregadas por nflverse."""
    return _cargar_auxiliar(season, refrescar, "stats_team_week", TEAM_URL,
                            "stats de equipo")


def cargar_qbr(nivel="week", season=None, refrescar=False):
    """Total QBR de ESPN. nivel = 'week' o 'season'. Devuelve (df, season).

    Trae qbr_total, pts_added y el desglose pass/run/sack/penalty.
    Metrica que la audiencia reconoce y que el proyecto no usaba.
    """
    if nivel not in ("week", "season"):
        raise ValueError("nivel debe ser 'week' o 'season'")
    df = _cargar_global(refrescar, f"qbr_{nivel}", QBR_URL.format(nivel=nivel),
                        f"QBR {nivel}")
    if season is None:
        season = temporada_actual()
    return df[df["season"] == season].copy(), season


def cargar_rosters(season=None, refrescar=False):
    """Roster de la temporada. Trae `depth_chart_position`, que es la unica
    fuente fiable para separar edge de interior: nflverse llama LB a Parsons y
    PFR le llama DL, pero el depth chart dice OLB."""
    return _cargar_auxiliar(season, refrescar, "roster", ROSTER_URL, "roster")


def cargar_equipos(refrescar=False):
    """Metadatos de los 32 equipos: conferencia, division, colores y nombres.
    Evita hardcodear las divisiones en los scripts."""
    return _cargar_global(refrescar, "teams", EQUIPOS_URL, "equipos")


def cargar_calendario(season=None, refrescar=False):
    """Calendario completo (schedules) de una temporada: resultados, lineas,
    div_game, descanso, entrenadores. Incluye los partidos aun no jugados."""
    if season is None:
        season = temporada_actual()
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "schedules.parquet")
    # Los resultados cambian cada jornada: se refresca a diario, no cada 3 dias
    necesita = refrescar or not os.path.exists(cache)
    if not necesita:
        if _info_schedules() is None:
            _aviso_sin_verificar()
        else:
            necesita = (time.time() - os.path.getmtime(cache)) > 86400
    if necesita:
        try:
            pd.read_csv(SCHED_URL, low_memory=False).to_parquet(cache, index=False)
        except Exception as e:
            if not os.path.exists(cache):
                raise DatosNoDisponibles(f"No se pudo descargar el calendario ({e}).")
            print(f"  Aviso: no se pudo refrescar el calendario ({e}) — usando cache")
    df = pd.read_parquet(cache)
    return df[df["season"] == season].copy(), season


def orden_partido(season, week, *equipos):
    """(indice de kickoff, visitante, local) de un partido dentro de su jornada.

    El indice existe para que la carpeta de salidas se lea como se jugo la
    jornada: 01 es el partido inaugural del miercoles, 16 el Monday Night. Los
    equipos vuelven como visitante/local, asi que el nombre del PNG no depende
    del orden en que se tecleen las siglas.

    Devuelve None si el calendario no se puede consultar o el partido no
    aparece; quien llama se queda entonces con su nombre de siempre.
    """
    try:
        sch, _ = cargar_calendario(season)
        w = sch[(sch["game_type"] == "REG") & (sch["week"] == int(week))]
    except Exception:
        return None
    if w.empty:
        return None
    por = [c for c in ("gameday", "gametime") if c in w.columns]
    if por:
        w = w.sort_values(por, kind="stable")
    buscados = {str(e).upper() for e in equipos}
    for i, fila in enumerate(w.itertuples(index=False), start=1):
        if buscados <= {fila.away_team, fila.home_team}:
            return i, fila.away_team, fila.home_team
    return None


def cargar_contratos(refrescar=False):
    """Contratos historicos de OverTheCap: apy, apy_cap_pct, guaranteed, draft.
    Se enlaza con el resto por gsis_id. Un solo fichero, todas las temporadas."""
    return _cargar_global(refrescar, "contratos", CONTRATOS_URL, "contratos")


def _cargar_global(refrescar, nombre_cache, url, etiqueta):
    """Ficheros sin {season}: todas las temporadas en uno. Refresco por antiguedad."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"{nombre_cache}.parquet")
    necesita = refrescar or not os.path.exists(cache)
    if not necesita:
        if _info_schedules() is None:
            _aviso_sin_verificar()
        else:
            necesita = (time.time() - os.path.getmtime(cache)) > 3 * 86400
    if necesita:
        try:
            _descargar(url, cache, etiqueta)
        except Exception as e:
            if not os.path.exists(cache):
                raise
            print(f"  Aviso: no se pudo refrescar {etiqueta} ({e}) — usando cache")
    return pd.read_parquet(cache)
