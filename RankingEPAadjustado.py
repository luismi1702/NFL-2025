# adjusted_epa_rankings_split_logos_only.py
# Ranking ofensivo y defensivo con EPA/play ajustado — logos pegados, color por rendimiento y marca @CuartayDato

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap

URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.csv.gz"
LOGOS_DIR = "logos"

OUT_OFF = "ranking_ofensivo_ajustado_2025.png"
OUT_DEF = "ranking_defensivo_ajustado_2025.png"

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


def plot_ranking(data_df, title, outfile, higher_is_better=True):
    """Genera el gráfico de ranking EPA/play ajustado."""
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": FG,
        "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
        "text.color": FG, "grid.color": GRID,
    })

    fig, ax = plt.subplots(figsize=(14, 10), dpi=180)
    fig.patch.set_facecolor(BG)
    plt.subplots_adjust(left=0.20, right=0.96, top=0.92, bottom=0.08)

    dfp = data_df.sort_values("EPA_ajustado", ascending=not higher_is_better)
    y_pos = np.arange(len(dfp))
    bar_colors = colors_by_percentile(dfp["EPA_ajustado"].values, higher_is_better=higher_is_better)

    vmin, vmax = float(dfp["EPA_ajustado"].min()), float(dfp["EPA_ajustado"].max())
    rng = max(vmax - vmin, 1e-6)
    left_pad = 0.012 * rng
    right_pad = 0.09 * rng
    ax.set_xlim(vmin - left_pad, vmax + right_pad)

    ax.barh(y_pos, dfp["EPA_ajustado"], color=bar_colors, height=0.54)
    ax.set_yticks([])
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("EPA/play ajustado", fontsize=12)

    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.axvline(0, color=GRID, linewidth=1)

    x_right = ax.get_xlim()[1]
    for y, v in zip(y_pos, dfp["EPA_ajustado"]):
        ax.text(min(v + 0.004 * rng, x_right - 0.01 * rng),
                y, f"{v:+.3f}".replace("+", " "),
                va="center", ha="left", fontsize=10, color=FG)

    xoff_frac = max(left_pad / rng * 0.60, 0.006)
    add_logos_to_positions(ax, dfp["Equipo"], y_pos,
                           base_zoom=0.045, x_offset_fraction=xoff_frac)

    ax.invert_yaxis()

    # --- Marca de agua sutil con @CuartayDato ---
    ax.text(0.99, 0.01, "@CuartayDato",
            transform=ax.transAxes,
            ha="right", va="bottom",
            color="#888888", fontsize=9, alpha=0.8,
            fontstyle="italic")

    plt.savefig(outfile, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main():
    print("Descargando datos NFL 2025 (nflverse)...")
    df = pd.read_csv(URL, low_memory=False, compression="infer")
    to_num(df, ["epa"])

    # Jugadas válidas
    df = df[df["play_type"].isin(["pass", "run"]) & df["posteam"].notna() & df["defteam"].notna()].copy()

    # Baselines sin ajuste
    off_mean = df.groupby("posteam")["epa"].mean()
    def_mean = df.groupby("defteam")["epa"].mean()

    # Ajuste ofensivo
    df["opp_def_allow_mean"] = df["defteam"].map(def_mean)
    df["epa_off_adj"] = df["epa"] - df["opp_def_allow_mean"]
    off_rank = df.groupby("posteam")["epa_off_adj"].mean().sort_values(ascending=False)
    off_df = off_rank.to_frame(name="EPA_ajustado").reset_index(names="Equipo")

    # Ajuste defensivo
    df["opp_off_gen_mean"] = df["posteam"].map(off_mean)
    df["epa_def_adj"] = df["epa"] - df["opp_off_gen_mean"]
    def_rank = df.groupby("defteam")["epa_def_adj"].mean().sort_values(ascending=True)
    def_df = def_rank.to_frame(name="EPA_ajustado").reset_index(names="Equipo")

    # Gráficos
    plot_ranking(
        data_df=off_df,
        title="ATAQUE — EPA/play ajustado por rival (más alto = mejor)",
        outfile=OUT_OFF,
        higher_is_better=True
    )
    print(f"✔ PNG ofensivo guardado: {OUT_OFF}")

    plot_ranking(
        data_df=def_df,
        title="DEFENSA — EPA/play ajustado por rival (más bajo = mejor)",
        outfile=OUT_DEF,
        higher_is_better=False
    )
    print(f"✔ PNG defensivo guardado: {OUT_DEF}")


if __name__ == "__main__":
    main()
