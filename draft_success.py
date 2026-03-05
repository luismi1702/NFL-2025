# draft_success.py
# Tasa de éxito por posición y ronda del draft NFL (2000-2019)
# Éxito = jugó 5+ temporadas (segundo contrato tras el rookie deal)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import nflreadpy

# === Config ===
SEASONS   = list(range(2000, 2020))   # 20 años → rookie deals expirados
BG        = "#0f1115"
CARD      = "#151924"
FG        = "#EDEDED"
ACCENT    = "#2d6cdf"
DPI       = 200
FIGSIZE   = (13, 8)

# Paleta rojo → amarillo → verde
RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

# Agrupación de posiciones
POS_MAP = {
    "QB": "QB",
    "RB": "RB", "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "T":  "OL", "G": "OL", "C": "OL", "OL": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB",
    "CB": "DB", "S": "DB", "DB": "DB",
}

# Orden de filas en el heatmap (de skill positions a trenches)
POS_ORDER  = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"]
ROUND_LABELS = ["Ronda 1", "Ronda 2", "Ronda 3", "Ronda 4", "Ronda 5", "Ronda 6", "Ronda 7"]


# ─────────────────────────────────────────────
# 1. Datos
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print(f"Cargando draft picks {SEASONS[0]}–{SEASONS[-1]}...")
    df = nflreadpy.load_draft_picks(seasons=SEASONS).to_pandas()

    df["pos_group"] = df["position"].map(POS_MAP)
    df = df[df["pos_group"].notna() & df["to"].notna() & df["round"].notna()].copy()
    df["round"] = df["round"].astype(int)
    df["season"] = df["season"].astype(int)
    df["to"] = df["to"].astype(int)

    # Éxito = jugó 5+ temporadas → casi seguro obtuvo segundo contrato
    df["success"] = (df["to"] >= df["season"] + 4).astype(int)

    return df


def compute_heatmap(df: pd.DataFrame):
    rate  = df.groupby(["pos_group", "round"])["success"].mean().mul(100).unstack()
    count = df.groupby(["pos_group", "round"])["success"].count().unstack()
    rate  = rate.reindex(POS_ORDER).reindex(columns=range(1, 8))
    count = count.reindex(POS_ORDER).reindex(columns=range(1, 8))
    return rate, count


# ─────────────────────────────────────────────
# 2. Gráfico
# ─────────────────────────────────────────────
def plot_heatmap(rate: pd.DataFrame, count: pd.DataFrame):
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "text.color": FG, "axes.edgecolor": "#2a2f3a",
    })

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    n_pos   = len(POS_ORDER)     # 8 filas
    n_rnd   = 7                  # 7 columnas
    cell_w  = 1.0
    cell_h  = 1.0

    for r, pos in enumerate(POS_ORDER):
        for c, rnd in enumerate(range(1, 8)):
            val   = rate.loc[pos, rnd]   if not pd.isna(rate.loc[pos, rnd])   else np.nan
            n     = count.loc[pos, rnd]  if not pd.isna(count.loc[pos, rnd])  else 0

            color = RYG(val / 100) if not np.isnan(val) else "#2a2f3a"

            rect = plt.Rectangle((c * cell_w, (n_pos - 1 - r) * cell_h),
                                  cell_w, cell_h,
                                  linewidth=0.5, edgecolor=BG,
                                  facecolor=color)
            ax.add_patch(rect)

            if not np.isnan(val):
                # Porcentaje grande
                ax.text(c * cell_w + cell_w / 2,
                        (n_pos - 1 - r) * cell_h + cell_h * 0.58,
                        f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="#0f1115" if val > 45 else FG)
                # N picks pequeño
                ax.text(c * cell_w + cell_w / 2,
                        (n_pos - 1 - r) * cell_h + cell_h * 0.28,
                        f"n={int(n)}",
                        ha="center", va="center",
                        fontsize=8, color="#0f1115" if val > 45 else "#888888")

    # Ejes
    ax.set_xlim(0, n_rnd)
    ax.set_ylim(0, n_pos)
    ax.set_xticks([i + 0.5 for i in range(n_rnd)])
    ax.set_xticklabels(ROUND_LABELS, fontsize=11, color=FG)
    ax.set_yticks([n_pos - 1 - i + 0.5 for i in range(n_pos)])
    ax.set_yticklabels(POS_ORDER, fontsize=13, fontweight="bold", color=FG)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Título
    ax.set_title(
        "Tasa de éxito en el Draft NFL por posición y ronda\n"
        "Éxito = jugó 5+ temporadas (segundo contrato tras el rookie deal)  |  2000–2019",
        fontsize=13, weight="bold", color=FG, pad=14, loc="left"
    )

    # Barra de color (leyenda)
    sm = plt.cm.ScalarMappable(cmap=RYG, norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.025, pad=0.02, aspect=20)
    cbar.ax.tick_params(colors=FG, labelsize=9)
    cbar.outline.set_edgecolor("#2a2f3a")
    cbar.set_label("% éxito", color=FG, fontsize=9)

    # Nota fuente
    ax.text(0, -0.6, "Fuente: Pro Football Reference via nflreadpy  |  n = picks totales por celda",
            fontsize=8, color="#666666", fontstyle="italic", va="top")

    # Firma @CuartayDato
    ax.text(0.99, 0.01, "@CuartayDato",
            transform=ax.transAxes, ha="right", va="bottom",
            color="#888888", fontsize=9, alpha=0.85, fontstyle="italic")

    plt.tight_layout()
    out = "draft_success_por_posicion.png"
    plt.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out}")


# ─────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()

    print(f"\nTotal picks analizados: {len(df)}")
    print("\nTasa de éxito global por posición (%):")
    print(df.groupby("pos_group")["success"].mean()
            .mul(100).round(1)
            .reindex(POS_ORDER)
            .to_string())

    rate, count = compute_heatmap(df)
    plot_heatmap(rate, count)
