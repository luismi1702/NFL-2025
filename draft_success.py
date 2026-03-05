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
FIGSIZE   = (15, 10)

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
    n_pos  = len(POS_ORDER)   # 8
    n_rnd  = 7

    # Layout de celdas con huecos
    CW, CH   = 0.86, 0.78     # tamaño celda
    GX, GY   = 0.14, 0.22     # gap entre celdas
    SX, SY   = CW + GX, CH + GY  # step (siempre 1.0)

    # Márgenes en coordenadas de datos
    LEFT_MARGIN  = 1.5    # espacio para etiquetas de posición
    TOP_MARGIN   = 1.2    # espacio para cabeceras de ronda
    BOTTOM_EXTRA = 0.9    # espacio para leyenda y fuente

    total_w = LEFT_MARGIN + n_rnd * SX
    total_h = TOP_MARGIN  + n_pos * SY + BOTTOM_EXTRA

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    # ── Título ──────────────────────────────────────────────────
    fig.text(0.03, 0.97,
             "Tasa de éxito en el Draft NFL",
             ha="left", va="top",
             fontsize=20, fontweight="bold", color=FG)
    fig.text(0.03, 0.91,
             "% de jugadores que jugaron 5+ temporadas (segundo contrato tras el rookie deal)  ·  Drafts 2000–2019",
             ha="left", va="top",
             fontsize=10, color="#888888", fontstyle="italic")

    # ── Cabeceras de ronda ───────────────────────────────────────
    rnd_labels_short = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    for c, lbl in enumerate(rnd_labels_short):
        cx = LEFT_MARGIN + c * SX + CW / 2
        cy = total_h - TOP_MARGIN * 0.45
        # Píldora de fondo
        pill = plt.Rectangle((LEFT_MARGIN + c * SX, cy - 0.28),
                              CW, 0.52,
                              linewidth=0, facecolor=CARD,
                              transform=ax.transData, zorder=1)
        ax.add_patch(pill)
        ax.text(cx, cy, lbl,
                ha="center", va="center",
                fontsize=12, fontweight="bold", color=ACCENT, zorder=2)

    # ── Celdas ──────────────────────────────────────────────────
    for r, pos in enumerate(POS_ORDER):
        row_y = total_h - TOP_MARGIN - (r + 1) * SY + GY / 2

        # Etiqueta de posición
        ax.text(LEFT_MARGIN - 0.18, row_y + CH / 2,
                pos,
                ha="right", va="center",
                fontsize=13, fontweight="bold", color=FG)

        for c, rnd in enumerate(range(1, 8)):
            val = rate.loc[pos, rnd]  if not pd.isna(rate.loc[pos, rnd])  else np.nan
            n   = count.loc[pos, rnd] if not pd.isna(count.loc[pos, rnd]) else 0

            cell_color = RYG(val / 100) if not np.isnan(val) else "#1e2330"
            cx = LEFT_MARGIN + c * SX
            cy = row_y

            # Celda redondeada
            cell = mpatches.FancyBboxPatch(
                (cx, cy), CW, CH,
                boxstyle="round,pad=0.01,rounding_size=0.07",
                linewidth=0, facecolor=cell_color,
                transform=ax.transData, zorder=2
            )
            ax.add_patch(cell)

            # Borde especial para 100%
            if not np.isnan(val) and val >= 99:
                border = mpatches.FancyBboxPatch(
                    (cx - 0.025, cy - 0.025), CW + 0.05, CH + 0.05,
                    boxstyle="round,pad=0.01,rounding_size=0.09",
                    linewidth=2, edgecolor="#ffd700",
                    facecolor="none",
                    transform=ax.transData, zorder=3
                )
                ax.add_patch(border)

            if not np.isnan(val):
                ink = "#0f1115" if val > 50 else FG
                ink_sub = "#1a2a1a" if val > 50 else "#666666"
                # Porcentaje
                ax.text(cx + CW / 2, cy + CH * 0.60,
                        f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=15, fontweight="bold", color=ink, zorder=4)
                # N picks
                ax.text(cx + CW / 2, cy + CH * 0.25,
                        f"n={int(n)}",
                        ha="center", va="center",
                        fontsize=8, color=ink_sub, zorder=4)

    # ── Leyenda de color (gradiente horizontal) ──────────────────
    leg_y  = BOTTOM_EXTRA * 0.55
    leg_x0 = LEFT_MARGIN
    leg_w  = n_rnd * SX * 0.55
    leg_h  = 0.22
    n_seg  = 200
    for i in range(n_seg):
        seg_x = leg_x0 + i * (leg_w / n_seg)
        ax.add_patch(plt.Rectangle(
            (seg_x, leg_y), leg_w / n_seg + 0.01, leg_h,
            linewidth=0, facecolor=RYG(i / n_seg), zorder=2
        ))
    for pct, label in [(0, "0%"), (25, "25%"), (50, "50%"), (75, "75%"), (100, "100%")]:
        ax.text(leg_x0 + (pct / 100) * leg_w, leg_y - 0.12,
                label, ha="center", va="top", fontsize=8, color="#888888")
    ax.text(leg_x0 - 0.1, leg_y + leg_h / 2,
            "% éxito", ha="right", va="center", fontsize=8, color="#888888")

    # ── Fuente ──────────────────────────────────────────────────
    ax.text(LEFT_MARGIN, BOTTOM_EXTRA * 0.18,
            "Fuente: Pro Football Reference via nflreadpy  ·  n = nº de picks analizados por celda",
            ha="left", va="center", fontsize=7.5,
            color="#555555", fontstyle="italic")

    # ── Firma ────────────────────────────────────────────────────
    ax.text(total_w - 0.05, BOTTOM_EXTRA * 0.18,
            "@CuartayDato",
            ha="right", va="center",
            fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

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
