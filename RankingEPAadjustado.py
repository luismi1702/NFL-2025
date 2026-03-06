# RankingEPAadjustado.py
# Ranking ofensivo y defensivo con EPA/play ajustado — logos, color por rendimiento y marca @CuartayDato

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap

# === Config ===
SEASON    = 2025
MIN_WEEK  = 1     # semana inicial del rango analizado
MAX_WEEK  = 18    # semana final   (18 = temporada completa; ajustar mid-season)
MIN_PLAYS = 100   # mínimo de jugadas para incluir un equipo en el ranking

URL       = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
LOGOS_DIR = "logos"

OUT_OFF = f"ranking_ofensivo_ajustado_{SEASON}.png"
OUT_DEF = f"ranking_defensivo_ajustado_{SEASON}.png"

# Tema oscuro
BG, FG, GRID = "#0f1115", "#EDEDED", "#2a2f3a"
# Paleta rojo → amarillo → verde
RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_logos_to_positions(ax, teams, y_positions,
                           base_zoom=0.045,
                           x_offset_fraction=0.10):
    """Dibuja logos a la izquierda (sin texto) ajustando tamaño según aspecto."""
    xmin, xmax = ax.get_xlim()
    xrng = xmax - xmin if xmax > xmin else 1.0
    x_logo = xmin - x_offset_fraction * xrng

    hard_penalty = {"NYJ": 4.5}  # Jets muy ancho

    for team, y in zip(teams, y_positions):
        path = os.path.join(LOGOS_DIR, f"{team}.png")
        if not os.path.exists(path):
            continue
        try:
            img = plt.imread(path)
            h, w = img.shape[:2]
            aspect = (w / float(h)) if h else 1.0

            if team in hard_penalty:
                zoom = base_zoom / hard_penalty[team]
            else:
                if aspect <= 1.3:
                    divisor = 1.0
                else:
                    divisor = 1.0 + 0.6 * (aspect - 1.3)
                divisor = np.clip(divisor, 1.0, 2.2)
                zoom = base_zoom / divisor

            ab = AnnotationBbox(
                OffsetImage(img, zoom=zoom, resample=True),
                (x_logo, y),
                frameon=False,
                xycoords=("data", "data"),
                box_alignment=(0.5, 0.5),
                pad=0
            )
            ax.add_artist(ab)
        except Exception:
            pass

    ax.set_xlim(x_logo - 0.02 * xrng, xmax)


def colors_by_percentile(values: np.ndarray, higher_is_better: bool) -> list:
    """Colores R→Y→G por percentil; invertidos si lower-is-better."""
    vals = np.asarray(values, dtype=float)
    ranks = vals.argsort().argsort()
    p = ranks / max(len(vals) - 1, 1)
    if not higher_is_better:
        p = 1.0 - p
    return [RYG(pi) for pi in p]


def plot_ranking(data_df, title, subtitle, outfile, higher_is_better=True):
    """Genera el gráfico de ranking EPA/play ajustado."""
    fig, ax = plt.subplots(figsize=(14, 10), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)

    dfp = data_df.sort_values("EPA_ajustado", ascending=not higher_is_better)
    y_pos = np.arange(len(dfp))
    bar_colors = colors_by_percentile(dfp["EPA_ajustado"].values, higher_is_better=higher_is_better)

    vmin, vmax = float(dfp["EPA_ajustado"].min()), float(dfp["EPA_ajustado"].max())
    rng = max(vmax - vmin, 1e-6)
    left_pad = 0.012 * rng
    right_pad = 0.11 * rng
    ax.set_xlim(vmin - left_pad, vmax + right_pad)

    ax.barh(y_pos, dfp["EPA_ajustado"], color=bar_colors, height=0.54)
    ax.set_yticks([])
    ax.set_xlabel("EPA/play ajustado", fontsize=12, color=FG, labelpad=8)

    ax.grid(axis="x", linestyle="--", alpha=0.25, color=GRID)
    ax.axvline(0, color=GRID, linewidth=1)

    # Valores al lado de cada barra
    x_right = ax.get_xlim()[1]
    for y, v in zip(y_pos, dfp["EPA_ajustado"]):
        ax.text(min(v + 0.004 * rng, x_right - 0.01 * rng),
                y, f"{v:+.3f}",
                va="center", ha="left", fontsize=10, color=FG)

    # Logos
    add_logos_to_positions(ax, dfp["Equipo"], y_pos,
                           base_zoom=0.045, x_offset_fraction=0.09)
    ax.invert_yaxis()

    # Título y subtítulo
    fig.text(0.5, 0.97, title,
             ha="center", va="top",
             fontsize=17, fontweight="bold", color=FG)
    fig.text(0.5, 0.92, subtitle,
             ha="center", va="top",
             fontsize=9.5, color="#888888", fontstyle="italic")

    # Fuente
    fig.text(0.01, 0.01,
             "Fuente: nflverse-data (nflverse)  ·  Solo jugadas de pase y carrera",
             ha="left", va="bottom",
             fontsize=7.5, color="#555555", fontstyle="italic")

    # Firma
    fig.text(0.99, 0.01, "@CuartayDato",
             ha="right", va="bottom",
             color="#888888", fontsize=9, alpha=0.85, fontstyle="italic")

    plt.savefig(outfile, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main():
    print(f"Descargando datos NFL {SEASON} semanas {MIN_WEEK}-{MAX_WEEK}...")
    df = pd.read_csv(URL, low_memory=False, compression="infer")
    to_num(df, ["epa", "week"])

    # Filtro de semana y jugadas válidas
    df = df[
        df["week"].between(MIN_WEEK, MAX_WEEK) &
        df["play_type"].isin(["pass", "run"]) &
        df["posteam"].notna() &
        df["defteam"].notna()
    ].copy()

    # Filtro de muestra mínima
    off_counts = df.groupby("posteam")["epa"].count()
    def_counts = df.groupby("defteam")["epa"].count()
    valid_off = off_counts[off_counts >= MIN_PLAYS].index
    valid_def = def_counts[def_counts >= MIN_PLAYS].index
    print(f"Equipos con >={MIN_PLAYS} jugadas — ataque: {len(valid_off)}, defensa: {len(valid_def)}")

    # Baselines sin ajuste
    off_mean = df.groupby("posteam")["epa"].mean()
    def_mean = df.groupby("defteam")["epa"].mean()

    # Ajuste ofensivo
    df["opp_def_allow_mean"] = df["defteam"].map(def_mean)
    df["epa_off_adj"] = df["epa"] - df["opp_def_allow_mean"]
    off_rank = (df[df["posteam"].isin(valid_off)]
                .groupby("posteam")["epa_off_adj"].mean()
                .sort_values(ascending=False))
    off_df = off_rank.to_frame(name="EPA_ajustado").reset_index(names="Equipo")

    # Ajuste defensivo
    df["opp_off_gen_mean"] = df["posteam"].map(off_mean)
    df["epa_def_adj"] = df["epa"] - df["opp_off_gen_mean"]
    def_rank = (df[df["defteam"].isin(valid_def)]
                .groupby("defteam")["epa_def_adj"].mean()
                .sort_values(ascending=True))
    def_df = def_rank.to_frame(name="EPA_ajustado").reset_index(names="Equipo")

    # Gráficos
    week_range = f"S{MIN_WEEK}-S{MAX_WEEK}" if MAX_WEEK < 18 else "Temporada completa"
    plot_ranking(
        data_df=off_df,
        title=f"Ranking ofensivo NFL {SEASON} — EPA/play ajustado  |  {week_range}",
        subtitle="EPA por jugada ajustado por la calidad de la defensa rival  ·  Más alto = mejor ataque",
        outfile=OUT_OFF,
        higher_is_better=True
    )
    print(f"Guardado: {OUT_OFF}")

    plot_ranking(
        data_df=def_df,
        title=f"Ranking defensivo NFL {SEASON} — EPA/play ajustado  |  {week_range}",
        subtitle="EPA por jugada ajustado por la calidad del ataque rival  ·  Más bajo = mejor defensa",
        outfile=OUT_DEF,
        higher_is_better=False
    )
    print(f"Guardado: {OUT_DEF}")


if __name__ == "__main__":
    main()
