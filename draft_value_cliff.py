# draft_value_cliff.py
# ¿A partir de qué ronda es un riesgo draftear cada posición?
# Line chart: tasa de éxito (segundo contrato) por posición y ronda · 2011-2022

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nflreadpy

# === Config ===
SEASONS = list(range(2011, 2023))
BG      = "#0f1115"
CARD    = "#151924"
FG      = "#EDEDED"
GRID_C  = "#2a2f3a"
ACCENT  = "#2d6cdf"
DPI     = 180
OUT     = "draft_value_cliff.png"

POS_MAP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "T": "OL", "G": "OL", "C": "OL", "OL": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB",
    "CB": "DB", "S": "DB", "DB": "DB",
}
POS_ORDER = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"]

POS_COLORS = {
    "QB": "#e63946",   # rojo
    "RB": "#ff8c42",   # naranja
    "WR": "#ffbe0b",   # ámbar
    "TE": "#06d6a0",   # verde menta
    "OL": "#4cc9f0",   # azul cielo
    "DL": "#a855f7",   # púrpura
    "LB": "#f72585",   # rosa/magenta
    "DB": "#3a86ff",   # azul eléctrico
}

NICK_TO_ABBR = {
    "Cardinals":"ARI","Falcons":"ATL","Ravens":"BAL","Bills":"BUF","Panthers":"CAR",
    "Bears":"CHI","Bengals":"CIN","Browns":"CLE","Cowboys":"DAL","Broncos":"DEN",
    "Lions":"DET","Packers":"GB","Texans":"HOU","Colts":"IND","Jaguars":"JAX",
    "Chiefs":"KC","Rams":"LA","Chargers":"LAC","Raiders":"LV","Dolphins":"MIA",
    "Vikings":"MIN","Patriots":"NE","Saints":"NO","Giants":"NYG","Jets":"NYJ",
    "Eagles":"PHI","Steelers":"PIT","Seahawks":"SEA","49ers":"SF",
    "Buccaneers":"TB","Titans":"TEN","Commanders":"WAS",
    "Redskins":"WAS","Football Team":"WAS","Oilers":"TEN",
}
ABBR_NORM = {
    "OAK":"LV","LVR":"LV","SD":"LAC","STL":"LA","KAN":"KC","SFO":"SF","GNB":"GB",
    "TAM":"TB","JAC":"JAX","NOR":"NO","PHO":"ARI","HST":"HOU","NWE":"NE",
    "ARI":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR",
    "CHI":"CHI","CIN":"CIN","CLE":"CLE","DAL":"DAL","DEN":"DEN",
    "DET":"DET","GB":"GB","HOU":"HOU","IND":"IND","JAX":"JAX",
    "KC":"KC","LA":"LA","LAC":"LAC","LAR":"LA","LV":"LV","MIA":"MIA",
    "MIN":"MIN","NE":"NE","NO":"NO","NYG":"NYG","NYJ":"NYJ",
    "PHI":"PHI","PIT":"PIT","SEA":"SEA","SF":"SF","TB":"TB",
    "TEN":"TEN","WAS":"WAS",
}

def normalize_team(raw):
    if not isinstance(raw, str): return None
    base = raw.split("/")[0].strip()
    if base in ABBR_NORM: return ABBR_NORM[base]
    if base in NICK_TO_ABBR: return NICK_TO_ABBR[base]
    return None


# ─────────────────────────────────────────────
# 1. Datos
# ─────────────────────────────────────────────
def load_data():
    print("Cargando draft picks 2011-2022...")
    draft = nflreadpy.load_draft_picks(seasons=SEASONS).to_pandas()
    draft["pos_group"] = draft["position"].map(POS_MAP)
    draft = draft[draft["pos_group"].notna() & draft["gsis_id"].notna()].copy()
    draft["season"] = draft["season"].astype(int)
    draft["round"]  = draft["round"].astype(int)

    print("Cargando contratos OTC...")
    c = nflreadpy.load_contracts().to_pandas()
    c = c.dropna(subset=["gsis_id","draft_team","team","year_signed","years","draft_year"]).copy()
    c["year_signed"] = c["year_signed"].astype(int)
    c["years"]       = c["years"].astype(float)
    c["draft_year"]  = c["draft_year"].astype(int)
    c["team_norm"]   = c["team"].apply(normalize_team)
    c["dt_norm"]     = c["draft_team"].apply(normalize_team)

    second = c[
        c["team_norm"].notna() & c["dt_norm"].notna() &
        (c["team_norm"] == c["dt_norm"]) &
        (c["year_signed"] >= c["draft_year"] + 3) &
        (c["years"] >= 2)
    ]["gsis_id"].unique()

    draft["success"] = draft["gsis_id"].isin(second).astype(int)
    return draft


# ─────────────────────────────────────────────
# 2. Gráfico
# ─────────────────────────────────────────────
def plot_cliff(df, pos_filter=None):
    """
    pos_filter=None  → modo grid: todas las posiciones con igual peso
    pos_filter="QB"  → modo posicion: QB destacada, resto apagadas
    """
    rate  = df.groupby(["pos_group","round"])["success"].mean().mul(100).unstack()
    rate  = rate.reindex(POS_ORDER).reindex(columns=range(1, 8))

    rounds     = list(range(1, 8))
    rnd_labels = ["R1","R2","R3","R4","R5","R6","R7"]

    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for pos in POS_ORDER:
        vals  = [rate.loc[pos, r] if not pd.isna(rate.loc[pos, r]) else np.nan for r in rounds]

        highlighted = (pos_filter is None or pos == pos_filter)

        if highlighted:
            color  = POS_COLORS[pos]
            lw     = 3.2 if pos_filter else 2.5
            ms     = 9   if pos_filter else 7
            alpha  = 1.0
            zorder = 4
        else:
            color  = "#2a3245"   # gris azulado muy apagado
            lw     = 1.2
            ms     = 4
            alpha  = 0.45
            zorder = 2

        ax.plot(rounds, vals,
                color=color, linewidth=lw, alpha=alpha,
                marker='o', markersize=ms,
                markerfacecolor=color, markeredgecolor=BG, markeredgewidth=1.2,
                label=pos, zorder=zorder)

        # Anotación de caída: solo para la posición destacada (o todas en grid)
        if highlighted:
            best_drop = (0, None, None)
            for i in range(len(vals) - 1):
                if not np.isnan(vals[i]) and not np.isnan(vals[i+1]):
                    drop = vals[i] - vals[i+1]
                    if drop > best_drop[0]:
                        best_drop = (drop, i, vals[i+1])
            drop_mag, drop_idx, drop_end_val = best_drop
            min_drop = 5 if pos_filter else 8
            if drop_mag > min_drop and drop_idx is not None:
                mid_x = rounds[drop_idx] + 0.5
                mid_y = (vals[drop_idx] + drop_end_val) / 2
                fs    = 9 if pos_filter else 7
                ax.text(mid_x, mid_y + 1.5, f"↓{drop_mag:.0f}%",
                        ha='center', va='bottom',
                        color=POS_COLORS[pos], fontsize=fs,
                        fontweight='bold' if pos_filter else 'normal',
                        alpha=0.95, zorder=5)

    # ── Estilo ejes ─────────────────────────────────────────────
    ax.yaxis.grid(True, color=GRID_C, linewidth=0.7, zorder=1)
    ax.xaxis.grid(True, color=GRID_C, linewidth=0.3, alpha=0.5, zorder=1)
    ax.set_axisbelow(True)

    ax.set_xticks(rounds)
    ax.set_xticklabels(rnd_labels, color=FG, fontsize=12, fontweight='bold')
    max_rate = float(rate.max().max())
    y_top    = int(np.ceil(max_rate / 10) * 10) + 8
    y_ticks  = list(range(0, y_top + 1, 10))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y}%" for y in y_ticks], color="#888888", fontsize=9)
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(-2, y_top)
    ax.tick_params(axis='both', length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)

    # ── Leyenda ──────────────────────────────────────────────────
    if pos_filter is None:
        # Modo grid: leyenda completa arriba a la derecha
        ax.legend(loc='upper right', frameon=True,
                  facecolor=CARD, edgecolor=GRID_C,
                  labelcolor=FG, fontsize=9,
                  ncol=2, handlelength=1.5, handleheight=0.8,
                  borderpad=0.7, labelspacing=0.5, columnspacing=1.0)
    else:
        # Modo posición: nombre grande como marca de agua en el fondo
        ax.text(0.97, 0.95, pos_filter,
                transform=ax.transAxes, ha='right', va='top',
                fontsize=72, fontweight='bold',
                color=POS_COLORS[pos_filter], alpha=0.08, zorder=1)

    # ── Título ───────────────────────────────────────────────────
    if pos_filter is None:
        titulo   = "¿A partir de qué ronda es un riesgo draftear cada posición?"
        subtitulo = "% de picks que firmaron segundo contrato (≥2 años) con su equipo de draft  ·  2011–2022"
    else:
        titulo   = f"Value Cliff — {pos_filter}: ¿a partir de qué ronda es un riesgo?"
        subtitulo = "% de picks que firmaron segundo contrato (≥2 años) con su equipo de draft  ·  2011–2022"

    ax.text(0.01, 1.055, titulo,
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=15, fontweight='bold', color=FG)
    ax.text(0.01, 1.01, subtitulo,
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=9, color="#888888", fontstyle='italic')

    # ── Fuente y marca de agua ───────────────────────────────────
    ax.text(0.01, 0.01, "Fuente: PFR & OverTheCap via nflreadpy",
            transform=ax.transAxes, ha="left", va="bottom",
            color="#555555", fontsize=7.5, fontstyle="italic")
    ax.text(0.99, 0.01, "@CuartayDato",
            transform=ax.transAxes, ha="right", va="bottom",
            color="#888888", fontsize=9, alpha=0.8, fontstyle="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {OUT}")


# ─────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    modo    = input("¿Grid o Posición? (grid/posicion): ").strip().lower()
    pos_raw = input("Posición (QB/RB/WR/TE/OL/DL/LB/DB) — vacío si grid: ").strip().upper()

    df = load_data()

    if modo.startswith("p") and pos_raw in POS_ORDER:
        plot_cliff(df, pos_filter=pos_raw)
    else:
        plot_cliff(df, pos_filter=None)
