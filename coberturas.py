# coberturas.py
# Coberturas defensivas por equipo — datos FTN via nflverse pbp_participation
# Cover 0, Cover 1, 2-Man, Cover 2, Cover 3, Cover 4, Cover 6, Cover 9, Combo

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pbp_loader import cargar_participation, DatosNoDisponibles, salida, season_cli

# === Config ===
LOGO_DIR  = "logos"
FIGSIZE   = (16, 18)
DPI       = 170
BG        = "#0f1115"
FG        = "#EDEDED"
SEASON    = season_cli()   # None = auto-detectar última temporada

# Orden visual y colores de coberturas
ORDER = ["COVER 0", "COVER 1", "2-MAN", "COVER 2", "COVER 3", "COVER 4", "COVER 6", "COVER 9", "COMBO"]
COLORS = {
    "COVER 0": "#e63946",
    "COVER 1": "#ff6b6b",
    "2-MAN":   "#f4a261",
    "COVER 2": "#e9c46a",
    "COVER 3": "#2a9d8f",
    "COVER 4": "#457b9d",
    "COVER 6": "#6a4c93",
    "COVER 9": "#9b5de5",
    "COMBO":   "#8ecae6",
}

# Mapa de valores nflverse -> etiqueta limpia
COV_MAP = {
    "COVER_0": "COVER 0",
    "COVER_1": "COVER 1",
    "COVER_2": "COVER 2",
    "COVER_3": "COVER 3",
    "COVER_4": "COVER 4",
    "COVER_6": "COVER 6",
    "COVER_9": "COVER 9",
    "2_MAN":   "2-MAN",
    "COMBO":   "COMBO",
}

# ─────────────────────────────────────────────
# 1. Carga y limpieza de datos
# ─────────────────────────────────────────────
def load_coverage(season):
    try:
        part, season = cargar_participation(season)
    except DatosNoDisponibles as e:
        raise SystemExit(
            "\n  No se puede generar este grafico todavia.\n"
            f"  {e}\n"
            "  Este visual necesita datos de participacion (coberturas charteadas),\n"
            "  que nflverse publica mas tarde que el play-by-play.\n")

    # Filtrar jugadas con cobertura registrada y valida
    part = part[
        part["defense_coverage_type"].notna() &
        (part["defense_coverage_type"] != "") &
        (part["defense_coverage_type"] != "BLOWN")
    ].copy()

    # Derivar equipo defensor desde el game_id ("2025_01_AWAY_HOME")
    split = part["nflverse_game_id"].str.split("_", expand=True)
    away_team = split[2]
    home_team = split[3]
    part["defteam"] = np.where(part["possession_team"] == away_team, home_team, away_team)
    part = part[part["defteam"].notna()]

    # Normalizar nombres de cobertura
    part["cobertura"] = part["defense_coverage_type"].map(COV_MAP)
    part = part[part["cobertura"].notna()]

    return part, season


def compute_pct(part: pd.DataFrame) -> pd.DataFrame:
    counts = part.groupby(["defteam", "cobertura"]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    pct = counts.div(totals, axis=0).mul(100).round(1)
    for cov in ORDER:
        if cov not in pct.columns:
            pct[cov] = 0.0
    return pct[ORDER]


# ─────────────────────────────────────────────
# 2. Logo helper
# ─────────────────────────────────────────────
# Normalizar todos los logos a ~60 px de pantalla independientemente de su tamaño real.
# Fórmula: zoom = TARGET_PX / max(h, w)
# Esto resuelve que NYJ tenga 4096x4096 px frente a los 500x500 del resto.
TARGET_PX = 30   # píxeles de pantalla deseados por logo

def add_logo(ax, team: str, y: float, x: float = -6.0):
    path = os.path.join(LOGO_DIR, f"{team}.png")
    if not os.path.exists(path):
        return
    img = plt.imread(path)
    h, w = img.shape[:2]
    zoom = TARGET_PX / max(h, w)
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom, resample=True),
                        (x, y), frameon=False, xycoords="data")
    ax.add_artist(ab)


# ─────────────────────────────────────────────
# 3. Grafico
# ─────────────────────────────────────────────
def plot_coberturas(pct_df: pd.DataFrame, season: int):
    # Ordenar por Cover 3 de mayor a menor (mas comun en la NFL)
    pct_df = pct_df.sort_values("COVER 3", ascending=True)
    teams = pct_df.index.tolist()
    n = len(teams)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bar_start =  0.0
    right_lim =  103.0
    bar_h     =  0.64

    ax.set_xlim(-8.5, right_lim)
    ax.set_ylim(-1, n)

    # Barras apiladas
    for i, team in enumerate(teams):
        x = bar_start
        for cov in ORDER:
            pct = float(pct_df.loc[team, cov])
            if pct <= 0:
                continue
            color = COLORS[cov]
            ax.barh(i, pct, left=x, height=bar_h, color=color, alpha=0.93, zorder=2)
            if pct >= 6.0:
                ax.text(x + pct / 2, i, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=8.5,
                        color="#0f1115", fontweight="bold", zorder=3)
            x += pct

    # Ejes (sin ticks: los logos hacen de etiqueta)
    ax.set_yticks([])
    ax.set_xlabel("% de snaps defensivos con cobertura registrada", fontsize=11,
                  labelpad=8, color=FG)
    ax.set_title(
        f"Coberturas defensivas por equipo  |  NFL {season}",
        fontsize=15, weight="bold", pad=46, color=FG
    )
    ax.grid(False)
    ax.axvline(0, color="#2a2f3a", linewidth=0.8)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2f3a")

    # Logos
    for i, team in enumerate(teams):
        add_logo(ax, team, i)

    # Leyenda
    handles = [mpatches.Patch(color=COLORS[c], label=c) for c in ORDER]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.55, 1.045), ncol=4,
              frameon=False, fontsize=10, labelcolor=FG)

    # Fuente abajo a la izquierda (arriba pisaba la leyenda)
    ax.text(0.01, 0.01, "Fuente: FTN Data via nflverse pbp_participation",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, color="#666666", fontstyle="italic")

    # Firma @CuartayDato
    ax.text(0.99, 0.01, "@CuartayDato",
            transform=ax.transAxes, ha="right", va="bottom",
            color="#888888", fontsize=9, alpha=0.85, fontstyle="italic")

    out = salida(f"coberturas_defensivas_{season}.png", SEASON)
    plt.tight_layout(rect=[0, 0.02, 1, 0.87])
    plt.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out}")


# ─────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    part, SEASON = load_coverage(SEASON)
    pct_df = compute_pct(part)

    print(f"\nEquipos procesados: {len(pct_df)}")
    print(pct_df.sort_values("COVER 3", ascending=False).head(5).to_string())

    plot_coberturas(pct_df, SEASON)
