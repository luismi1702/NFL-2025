"""
estado_datos.py
Semaforo de las fuentes de nflverse: que hay publicado para una temporada,
hasta que semana llega y cuando se actualizo por ultima vez.

Nace de un susto de ago-2026: `pbp_participation` no tiene cron (es un rebuild
manual que en toda la temporada 2025 se ejecuto UNA vez, ya acabada), y de ahi
salen cobertura, personal y rutas de 13 scripts. Este script hace visible esa
clase de problema antes de que se convierta en un grafico publicado con datos
de hace tres meses.

Uso:
  python estado_datos.py              # temporada autodetectada
  python estado_datos.py --season 2026
"""
import os
import sys
import time
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

import pbp_loader as pl
from pbp_loader import DatosNoDisponibles, season_cli, temporada_actual

VERDE = "OK  "
AMBAR = "VIEJO"
ROJO  = "NO  "

ACUM = "acum"   # fuente de temporada acumulada: no tiene semanas, y es correcto


def _sem_desde_game_id(df):
    """La participacion no trae columna week; la semana va en el game_id
    (2025_07_SF_SEA). Es justo la fuente que mas importa vigilar."""
    g = df["nflverse_game_id"].astype(str).str.split("_", expand=True)
    return pd.to_numeric(g[1], errors="coerce")


# (etiqueta, funcion de carga, de donde sale la semana, critico_para)
FUENTES = [
    ("pbp",             lambda s: pl.cargar_pbp(s, solo_reg=False, avisar=False),
     "week",  "casi todo el catalogo"),
    ("stats_player",    lambda s: pl.cargar_stats(s),
     ACUM,    "comparadores y rankings"),
    ("ftn_charting",    lambda s: pl.cargar_ftn(s),
     "week",  "play_action, blitz"),
    ("pbp_participation", lambda s: pl.cargar_participation(s),
     _sem_desde_game_id, "cobertura, personal, rutas (13 scripts) — SIN CRON"),
    ("pfr_advstats def", lambda s: pl.cargar_pfr("def", s),
     ACUM,    "cobertura CB/S, presiones, placajes fallados"),
    ("pfr def semanal", lambda s: pl.cargar_pfr("def", s, semanal=True),
     "week",  "lo mismo, jornada a jornada"),
    ("pfr_advstats pass", lambda s: pl.cargar_pfr("pass", s),
     ACUM,    "pocket time, presion sufrida por QB"),
    ("nextgen_stats",   lambda s: pl.cargar_ngs("receiving", s),
     "week",  "separacion, YAC sobre esperado"),
    ("snap_counts",     lambda s: pl.cargar_snaps(s),
     "week",  "reparto de snaps"),
    ("injuries",        lambda s: pl.cargar_lesiones(s),
     "week",  "parte de lesiones"),
    ("qbr (ESPN)",      lambda s: pl.cargar_qbr("week", s),
     "game_week", "QBR semanal"),
    ("stats_team",      lambda s: pl.cargar_stats_equipo(s),
     "week",  "clasificacion y tendencias"),
]


def edad_cache(nombre):
    """Dias desde que se descargo el parquet, o None si no esta en cache."""
    for f in os.listdir(pl.CACHE) if os.path.isdir(pl.CACHE) else []:
        if f.startswith(nombre) and f.endswith(".parquet"):
            dias = (time.time() - os.path.getmtime(os.path.join(pl.CACHE, f))) / 86400
            return dias
    return None


def main():
    season = season_cli() or temporada_actual()
    sem_liga = pl.ultima_semana(season)

    print()
    print(f"  ESTADO DE LOS DATOS — temporada {season}")
    if sem_liga is None:
        print("  Frescura sin verificar (no se pudo consultar el calendario)")
    else:
        print(f"  La liga va por la semana {sem_liga}")
    print("  " + "-" * 76)
    print(f"  {'FUENTE':22} {'':5} {'SEMANAS':9} {'RETRASO':9} CRITICO PARA")
    print("  " + "-" * 76)

    problemas = []
    for etiqueta, carga, col_sem, critico in FUENTES:
        try:
            df, _ = carga(season)
        except DatosNoDisponibles:
            print(f"  {etiqueta:22} {ROJO:5} {'—':9} {'—':9} {critico}")
            problemas.append(f"{etiqueta}: aun no publicado para {season}")
            continue
        except Exception as e:
            print(f"  {etiqueta:22} {ROJO:5} {'error':9} {'—':9} {type(e).__name__}")
            problemas.append(f"{etiqueta}: {type(e).__name__}")
            continue

        if df is None or len(df) == 0:
            print(f"  {etiqueta:22} {ROJO:5} {'vacio':9} {'—':9} {critico}")
            problemas.append(f"{etiqueta}: publicado pero vacio")
            continue

        # Hasta que semana llegan los datos
        sem = None
        if col_sem is ACUM:
            print(f"  {etiqueta:22} {VERDE:5} {'acumulado':9} {f'{len(df):,} filas':9} {critico}")
            continue
        if callable(col_sem):
            s = col_sem(df).dropna()
            sem = int(s.max()) if len(s) else None
        elif col_sem and col_sem in df.columns:
            s = pd.to_numeric(df[col_sem], errors="coerce").dropna()
            sem = int(s.max()) if len(s) else None

        if sem is None:
            txt_sem, retraso, estado = "sin col.", "?", AMBAR
        elif sem_liga is None:
            txt_sem, retraso, estado = f"1-{sem}", "?", VERDE
        else:
            d = sem_liga - sem
            txt_sem = f"1-{sem}"
            retraso = "al dia" if d <= 0 else f"-{d} sem"
            estado = VERDE if d <= 0 else (AMBAR if d == 1 else ROJO)
            if d >= 2:
                problemas.append(
                    f"{etiqueta}: {d} semanas por detras de la liga ({critico})")

        print(f"  {etiqueta:22} {estado:5} {txt_sem:9} {retraso:9} {critico}")

    print("  " + "-" * 76)
    if problemas:
        print()
        print("  AVISOS:")
        for p in problemas:
            print(f"    - {p}")
        print()
        print("  Un grafico construido sobre una fuente atrasada saldra perfecto")
        print("  y sera falso. Comprueba el sello del pie antes de publicar.")
    else:
        print("  Todas las fuentes al dia.")
    print()


if __name__ == "__main__":
    main()
