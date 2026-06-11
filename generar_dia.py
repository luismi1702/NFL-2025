# generar_dia.py
# Genera los PNGs de draft del día (radar + grid + ranking spotlight) y los mueve a draft_calendar/
# Uso: python generar_dia.py              → usa la fecha de hoy
#      python generar_dia.py 2026-04-05   → fecha específica

import sys
import os
import shutil
from datetime import date

# Asegurar que el cwd es la carpeta del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

import draft_r1_radar      as radar_mod
import draft_grid          as grid_mod
import draft_ranking_equipos as ranking_mod

CAL_DIR = os.path.join(BASE_DIR, "draft_calendar")

CALENDARIO = {
    "2026-04-03": ("BAL", "CHI"),
    "2026-04-04": ("CIN", "DET"),
    "2026-04-05": ("CLE", "GB"),
    "2026-04-06": ("PIT", "MIN"),
    "2026-04-07": ("BUF", "DAL"),
    "2026-04-08": ("MIA", "NYG"),
    "2026-04-09": ("NE",  "PHI"),
    "2026-04-10": ("NYJ", "WAS"),
    "2026-04-11": ("HOU", "ATL"),
    "2026-04-12": ("IND", "CAR"),
    "2026-04-13": ("JAX", "NO"),
    "2026-04-14": ("TEN", "TB"),
    "2026-04-15": ("DEN", "ARI"),
    "2026-04-16": ("KC",  "LA"),
    "2026-04-17": ("LAC", "SEA"),
    "2026-04-18": ("LV",  "SF"),
}


def find_folder(fecha_str, teams):
    for entry in os.listdir(CAL_DIR):
        if entry.startswith(fecha_str):
            return os.path.join(CAL_DIR, entry)
    folder = os.path.join(CAL_DIR, f"{fecha_str}_{teams[0]}_{teams[1]}")
    os.makedirs(folder, exist_ok=True)
    return folder


def generar_equipo(team, out_dir, df_radar, df_grid, df_ranking):
    print(f"\n  [{team}]")

    radar_mod.plot_team(df_radar, team)
    src = os.path.join(BASE_DIR, "draft_radar_equipo.png")
    dst = os.path.join(out_dir, f"{team}_radar.png")
    shutil.move(src, dst)
    print(f"    OK {team}_radar.png")

    grid_mod.plot_equipo(df_grid, team)
    src = os.path.join(BASE_DIR, "draft_grid_equipo.png")
    dst = os.path.join(out_dir, f"{team}_grid.png")
    shutil.move(src, dst)
    print(f"    OK {team}_grid.png")

    ranking_mod.plot_ranking_spotlight(df_ranking, team)
    src = os.path.join(BASE_DIR, "draft_ranking_spotlight.png")
    dst = os.path.join(out_dir, f"{team}_ranking.png")
    shutil.move(src, dst)
    print(f"    OK {team}_ranking.png")


if __name__ == "__main__":
    fecha = sys.argv[1] if len(sys.argv) > 1 else str(date.today())

    if fecha not in CALENDARIO:
        print(f"Fecha '{fecha}' no está en el calendario.")
        print("Fechas disponibles:")
        for f in sorted(CALENDARIO):
            print(f"  {f}  →  {CALENDARIO[f][0]} + {CALENDARIO[f][1]}")
        sys.exit(1)

    teams   = CALENDARIO[fecha]
    out_dir = find_folder(fecha, teams)

    print(f"\nFecha  : {fecha}")
    print(f"Equipos: {teams[0]} + {teams[1]}")
    print(f"Carpeta: {out_dir}")

    print("\nCargando datos (radar)...")
    df_radar = radar_mod.load_data()

    print("Cargando datos (grid)...")
    df_grid = grid_mod.load_data()

    print("Cargando datos (ranking)...")
    df_ranking = ranking_mod.load_data()

    for team in teams:
        generar_equipo(team, out_dir, df_radar, df_grid, df_ranking)

    print(f"\nListo — {len(teams) * 3} PNGs guardados en:")
    print(f"  {out_dir}")
