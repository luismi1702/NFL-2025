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

import numpy as np
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

    OJO (sep-2026): el acumulado NO trae la temporada en curso — nflverse la
    añade al acabar (la de 2025 llego en feb-2026). Sin esto, 8 scripts recibian
    un DataFrame vacio y algunos dibujaban presiones a 0 como si fueran reales.
    Si el acumulado viene vacio para def/pass, se reconstruye sumando el
    semanal (solo REG, como el oficial) con las mismas columnas. Lo que el
    semanal no trae queda en NaN: pocket_time, on_tgt, RPO, play action y air
    yards del QB; age, gs, loaded y bats del defensor. `pos` sale del roster.
    El DataFrame lleva df.attrs["reconstruido_semanal"] = True.

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
    out = df[df["season"] == season].copy()
    if out.empty and tipo in ("def", "pass"):
        try:
            sem, _ = cargar_pfr(tipo, season, semanal=True, refrescar=refrescar)
        except DatosNoDisponibles:
            return out, season
        if not sem.empty:
            out = _pfr_acumulado_desde_semanal(tipo, sem, season)
            print(f"  PFR {tipo} {season}: acumulado reconstruido desde el semanal "
                  f"(semanas {int(sem['week'].min())}-{int(sem['week'].max())})")
    return out, season


# Semanal -> nombre de columna del acumulado (solo las que se pueden sumar)
_PFR_DEF_SUMAS = {
    "def_ints": "int", "def_targets": "tgt", "def_completions_allowed": "cmp",
    "def_yards_allowed": "yds", "def_receiving_td_allowed": "td",
    "def_air_yards_completed": "air", "def_yards_after_catch": "yac",
    "def_times_blitzed": "bltz", "def_times_hurried": "hrry",
    "def_times_hitqb": "qbkd", "def_sacks": "sk", "def_pressures": "prss",
    "def_tackles_combined": "comb", "def_missed_tackles": "m_tkl",
}
_PFR_PASS_SUMAS = {
    "passing_drops": "drops", "passing_bad_throws": "bad_throws",
    "times_blitzed": "times_blitzed", "times_hurried": "times_hurried",
    "times_hit": "times_hit", "times_pressured": "times_pressured",
}
# depth_chart_position del roster -> etiqueta de PFR que usan los scripts
_POS_PFR = {"FS": "S", "SS": "S", "S": "S", "NB": "CB", "CB": "CB",
            "NT": "DT", "DT": "DT", "DE": "DE", "OLB": "OLB",
            "ILB": "LB", "MLB": "LB", "LB": "LB"}


def _pfr_acumulado_desde_semanal(tipo, sem, season):
    """Replica el formato del acumulado de PFR: una fila por jugador y equipo,
    y si jugo en varios, otra fila total con tm/team = '2TM', '3TM'..."""
    sem = sem[sem["game_type"].astype(str).str.upper() == "REG"].copy()
    # Partidos cojos: PFR a veces publica solo uno de los dos equipos (en 2025
    # faltaron tres del Thanksgiving y KC salia con un 10% menos de presiones)
    cojos = sem.groupby("game_id")["team"].nunique()
    cojos = sorted(cojos[cojos < 2].index)
    if cojos:
        print(f"  !! PFR {tipo} semanal: {len(cojos)} partido(s) con un solo equipo "
              f"({', '.join(cojos[:4])}) — sus rivales salen con datos de menos")
    sumas = _PFR_DEF_SUMAS if tipo == "def" else _PFR_PASS_SUMAS
    for c in list(sumas) + (["def_adot"] if tipo == "def" else []):
        sem[c] = pd.to_numeric(sem[c], errors="coerce")
    if tipo == "def":
        sem["_air_tgt"] = sem["def_adot"] * sem["def_targets"]   # para el aDOT ponderado

    extra = ["_air_tgt"] if tipo == "def" else []
    agg = {c: "sum" for c in list(sumas) + extra}
    agg["game_id"] = "nunique"
    agg["pfr_player_name"] = "last"
    por_eq = sem.groupby(["pfr_player_id", "team"], as_index=False).agg(agg)
    total = sem.groupby("pfr_player_id", as_index=False).agg(
        {**agg, "team": "nunique"})
    total = total[total["team"] > 1].copy()
    total["team"] = total["team"].astype(int).astype(str) + "TM"
    t = pd.concat([total, por_eq], ignore_index=True)
    t = t.rename(columns={**sumas, "pfr_player_id": "pfr_id",
                          "pfr_player_name": "player", "game_id": "g"})
    t["season"] = season

    # `g` no se puede contar en el semanal: solo trae fila si el jugador hizo
    # algo medible, y en 2025 salian 7.791 partidos frente a 12.170. Se cuentan
    # los partidos con algun snap (defensa, ataque o especiales), como PFR.
    try:
        sn, _ = cargar_snaps(season)
        sn = sn[sn["game_type"].astype(str).str.upper() == "REG"]
        jugo = sn[sn[["offense_snaps", "defense_snaps", "st_snaps"]]
                  .apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) > 0]
        g_eq = jugo.groupby(["pfr_player_id", "team"])["game_id"].nunique()
        g_tot = jugo.groupby("pfr_player_id")["game_id"].nunique()
        eq_col = "team"
        es_total = t[eq_col].str.endswith("TM")
        g_snaps = pd.Series(
            [g_tot.get(pid) if tot else g_eq.get((pid, eq))
             for pid, eq, tot in zip(t["pfr_id"], t[eq_col], es_total)],
            index=t.index, dtype="float")
        t["g"] = g_snaps.fillna(t["g"])
    except Exception:
        pass

    if tipo == "pass":
        for c in ("pass_attempts", "drop_pct", "bad_throw_pct", "pocket_time",
                  "pressure_pct", "on_tgt_pct"):
            t[c] = np.nan
        t.attrs["reconstruido_semanal"] = True
        return t

    t = t.rename(columns={"team": "tm"})
    tgt = t["tgt"].where(t["tgt"] > 0)
    # Placajes: el semanal omite partidos enteros de los DL (Darius Robinson,
    # 5 filas de 17 partidos) y en 2025 sumaba un 15% menos. Se toman de las
    # stats de nflverse (solo + asistidos), que cuadran ±2 con PFR en el 92%.
    # En jugadores traspasados, las filas por equipo se quedan con el semanal.
    try:
        st, _ = cargar_stats(season)
        ros, _ = cargar_rosters(season)
        gsis_pfr = (ros.dropna(subset=["pfr_id", "gsis_id"])
                       .drop_duplicates("gsis_id").set_index("gsis_id")["pfr_id"])
        st = st.assign(pfr_id=st["player_id"].map(gsis_pfr),
                       comb=pd.to_numeric(st["def_tackles_solo"], errors="coerce").fillna(0)
                            + pd.to_numeric(st["def_tackle_assists"], errors="coerce").fillna(0))
        comb_st = st.dropna(subset=["pfr_id"]).groupby("pfr_id")["comb"].sum()
        traspasados = set(t.loc[t["tm"].str.endswith("TM"), "pfr_id"])
        usar = ~(t["pfr_id"].isin(traspasados) & ~t["tm"].str.endswith("TM"))
        t.loc[usar, "comb"] = t.loc[usar, "pfr_id"].map(comb_st).fillna(t.loc[usar, "comb"])
    except Exception:
        pass
    t["cmp_percent"] = (t["cmp"] / tgt).round(3)
    t["yds_cmp"] = (t["yds"] / t["cmp"].where(t["cmp"] > 0)).round(1)
    t["yds_tgt"] = (t["yds"] / tgt).round(1)
    t["dadot"] = (t.pop("_air_tgt") / tgt).round(1)
    tackles = (t["comb"] + t["m_tkl"]).where(lambda x: x > 0)
    t["m_tkl_percent"] = (t["m_tkl"] / tackles).round(3)
    # Passer rating permitido (formula NFL, cada termino acotado a [0, 2.375])
    a = ((t["cmp"] / tgt - 0.3) * 5).clip(0, 2.375)
    b = ((t["yds"] / tgt - 3) * 0.25).clip(0, 2.375)
    c = (t["td"] / tgt * 20).clip(0, 2.375)
    d = (2.375 - t["int"] / tgt * 25).clip(0, 2.375)
    t["rat"] = ((a + b + c + d) / 6 * 100).round(1)
    for col in ("age", "gs", "loaded", "bats"):
        t[col] = np.nan
    try:
        ros, _ = cargar_rosters(season)
        ros = ros.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id", keep="last")
        pos = ros.set_index("pfr_id")["depth_chart_position"]
        t["pos"] = t["pfr_id"].map(pos).map(lambda p: _POS_PFR.get(p, p))
    except Exception:
        t["pos"] = np.nan
    t.attrs["reconstruido_semanal"] = True
    return t


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
    # Se refresca cuando al cache le faltan partidos jugados, no por antiguedad:
    # con la regla de "una vez al dia", un cache bajado el lunes a mediodia
    # seguia valiendo el martes a las 8:00 sin el Monday Night, y el power
    # ranking de la semana 1 de 2026 salio con KC y DEN sin record.
    necesita = refrescar or not os.path.exists(cache)
    if not necesita:
        info = _info_schedules()
        if info is None:
            _aviso_sin_verificar()
        else:
            loc = pd.read_parquet(cache, columns=["season", "game_type", "home_score"])
            jugados = int(((loc["season"] == info[0]) & (loc["game_type"] == "REG")
                           & loc["home_score"].notna()).sum())
            necesita = (jugados < info[2]
                        or (time.time() - os.path.getmtime(cache)) > 86400)
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
