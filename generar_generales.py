# generar_generales.py
# Genera los 3 PNGs de los días generales 20-22 abril 2026:
#   20 → value_cliff_ataque.png   (QB, RB, WR, TE)
#   21 → value_cliff_defensa.png  (OL, DL, LB, DB)
#   22 → draft_success_r1_por_eleccion.png

import os, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import nflreadpy

# ── Config ────────────────────────────────────────────────────────────────────
SEASONS = list(range(2011, 2023))
BG      = "#0f1115"
CARD    = "#151924"
FG      = "#EDEDED"
GRID_C  = "#2a2f3a"
ACCENT  = "#2d6cdf"
DPI     = 180

POS_MAP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "T": "OL", "G": "OL", "C": "OL", "OL": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB",
    "CB": "CB", "S": "S", "DB": "CB",
}
POS_COLORS = {
    "QB": "#e63946", "RB": "#f4a261", "WR": "#2ec4b6", "TE": "#a8dadc",
    "OL": "#457b9d", "DL": "#6a4c93", "LB": "#f1c453", "DB": "#06d6a0",
    "CB": "#06d6a0", "S": "#f72585",
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALENDAR = os.path.join(BASE_DIR, "draft_calendar")

# ── 1. Carga de datos (una sola vez) ─────────────────────────────────────────
def load_draft_data():
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


# ── 2. Value Cliff (días 20 y 21) ────────────────────────────────────────────
def plot_value_cliff(df, pos_subset, titulo_grupo, out_path):
    rounds     = list(range(1, 8))
    rnd_labels = ["R1","R2","R3","R4","R5","R6","R7"]
    rate = df.groupby(["pos_group","round"])["success"].mean().mul(100).unstack()
    rate = rate.reindex(pos_subset).reindex(columns=range(1, 8))

    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for pos in pos_subset:
        vals = [rate.loc[pos, r] if not pd.isna(rate.loc[pos, r]) else np.nan for r in rounds]
        color = POS_COLORS[pos]
        ax.plot(rounds, vals,
                color=color, linewidth=3.0,
                marker='o', markersize=8,
                markerfacecolor=color, markeredgecolor=BG, markeredgewidth=1.4,
                label=pos, zorder=4)

        # Anotar mayor caída
        best_drop = (0, None, None)
        for i in range(len(vals) - 1):
            if not np.isnan(vals[i]) and not np.isnan(vals[i+1]):
                drop = vals[i] - vals[i+1]
                if drop > best_drop[0]:
                    best_drop = (drop, i, vals[i+1])
        drop_mag, drop_idx, drop_end_val = best_drop
        if drop_mag > 5 and drop_idx is not None:
            mid_x = rounds[drop_idx] + 0.5
            mid_y = (vals[drop_idx] + drop_end_val) / 2
            ax.text(mid_x, mid_y + 1.5, f"↓{drop_mag:.0f}%",
                    ha='center', va='bottom',
                    color=color, fontsize=9, fontweight='bold',
                    alpha=0.95, zorder=5)

    ax.yaxis.grid(True, color=GRID_C, linewidth=0.7, zorder=1)
    ax.xaxis.grid(True, color=GRID_C, linewidth=0.3, alpha=0.5, zorder=1)
    ax.set_axisbelow(True)
    ax.set_xticks(rounds)
    ax.set_xticklabels(rnd_labels, color=FG, fontsize=12, fontweight='bold')
    max_rate = float(rate.max().max())
    y_top = int(np.ceil(max_rate / 10) * 10) + 8
    ax.set_yticks(range(0, y_top + 1, 10))
    ax.set_yticklabels([f"{y}%" for y in range(0, y_top + 1, 10)], color="#888888", fontsize=9)
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(-2, y_top)
    ax.tick_params(axis='both', length=0)
    for spine in ax.spines.values(): spine.set_edgecolor(GRID_C)

    ax.legend(loc='upper right', frameon=True,
              facecolor=CARD, edgecolor=GRID_C,
              labelcolor=FG, fontsize=11,
              handlelength=1.8, handleheight=0.9,
              borderpad=0.8, labelspacing=0.6)

    titulo   = f"Value Cliff — {titulo_grupo}: ¿a partir de qué ronda es un riesgo draftear?"
    subtitulo = "% de picks que firmaron segundo contrato (≥2 años) con su equipo de draft  ·  2011–2022"
    ax.text(0.01, 1.055, titulo,
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=14, fontweight='bold', color=FG)
    ax.text(0.01, 1.01, subtitulo,
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=9, color="#888888", fontstyle='italic')
    ax.text(0.01, 0.01, "Fuente: PFR & OverTheCap via nflreadpy",
            transform=ax.transAxes, ha="left", va="bottom",
            color="#555555", fontsize=7.5, fontstyle="italic")
    ax.text(0.99, 0.01, "@CuartayDato",
            transform=ax.transAxes, ha="right", va="bottom",
            color="#888888", fontsize=9, alpha=0.8, fontstyle="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"OK: {out_path}")


# ── 3. Draft Success R1 (día 22) ─────────────────────────────────────────────
PICK_BINS   = [0, 5, 10, 15, 20, 25, 32]
PICK_LABELS = ["1–5", "6–10", "11–15", "16–20", "21–25", "26–32"]
RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])
POS_ORDER_R1 = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "CB", "S"]

def plot_success_r1(df, out_path):
    df_r1 = df[df["round"] == 1].copy()
    pick_col = next((c for c in ["pick", "pick_no", "draft_pick"] if c in df_r1.columns), None)
    df_r1["pick_num"] = pd.to_numeric(df_r1[pick_col], errors="coerce")
    df_r1 = df_r1[df_r1["pick_num"].notna()].copy()
    df_r1["pick_group"] = pd.cut(df_r1["pick_num"], bins=PICK_BINS, labels=PICK_LABELS, right=True)

    rate  = df_r1.groupby(["pos_group","pick_group"], observed=True)["success"].mean().mul(100).unstack()
    count = df_r1.groupby(["pos_group","pick_group"], observed=True)["success"].count().unstack()
    rate  = rate.reindex(POS_ORDER_R1).reindex(columns=PICK_LABELS)
    count = count.reindex(POS_ORDER_R1).reindex(columns=PICK_LABELS)

    n_pos, n_col = len(POS_ORDER_R1), len(PICK_LABELS)
    CW, CH = 0.90, 0.78
    GX, GY = 0.14, 0.22
    SX, SY = CW + GX, CH + GY
    LEFT_MARGIN  = 1.5
    TOP_MARGIN   = 2.6
    BOTTOM_EXTRA = 0.9
    RIGHT_MARGIN = 1.0
    total_w = LEFT_MARGIN + n_col * SX + RIGHT_MARGIN
    total_h = TOP_MARGIN  + n_pos * SY + BOTTOM_EXTRA

    FIGSIZE = (14, 10)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    ax.text(LEFT_MARGIN, total_h - 0.25,
            "Tasa de éxito en el Draft NFL — Primera Ronda",
            ha="left", va="top",
            fontsize=20, fontweight="bold", color=FG, zorder=5)
    ax.text(LEFT_MARGIN, total_h - 1.05,
            "% de picks que firmaron un segundo contrato (≥2 años) con el mismo equipo que los drafteó  ·  Drafts 2011–2022",
            ha="left", va="top",
            fontsize=10, color="#888888", fontstyle="italic", zorder=5)

    for c, lbl in enumerate(PICK_LABELS):
        cx = LEFT_MARGIN + c * SX + CW / 2
        cy = total_h - TOP_MARGIN * 0.72
        pill = plt.Rectangle((LEFT_MARGIN + c * SX, cy - 0.28), CW, 0.52,
                              linewidth=0, facecolor=CARD, transform=ax.transData, zorder=1)
        ax.add_patch(pill)
        ax.text(cx, cy, f"Pick\n{lbl}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=ACCENT, zorder=2, linespacing=1.2)

    for r, pos in enumerate(POS_ORDER_R1):
        row_y = total_h - TOP_MARGIN - (r + 1) * SY + GY / 2
        ax.text(LEFT_MARGIN - 0.18, row_y + CH / 2, pos,
                ha="right", va="center", fontsize=13, fontweight="bold", color=FG)
        for c, lbl in enumerate(PICK_LABELS):
            val = rate.loc[pos, lbl]  if not pd.isna(rate.loc[pos, lbl])  else np.nan
            n   = count.loc[pos, lbl] if not pd.isna(count.loc[pos, lbl]) else 0
            cell_color = RYG(val / 100) if not np.isnan(val) else "#1e2330"
            cx, cy = LEFT_MARGIN + c * SX, row_y
            cell = mpatches.FancyBboxPatch((cx, cy), CW, CH,
                                           boxstyle="round,pad=0.01,rounding_size=0.07",
                                           linewidth=0, facecolor=cell_color,
                                           transform=ax.transData, zorder=2)
            ax.add_patch(cell)
            if not np.isnan(val) and val >= 99:
                border = mpatches.FancyBboxPatch((cx-0.025, cy-0.025), CW+0.05, CH+0.05,
                                                 boxstyle="round,pad=0.01,rounding_size=0.09",
                                                 linewidth=2, edgecolor="#ffd700", facecolor="none",
                                                 transform=ax.transData, zorder=3)
                ax.add_patch(border)
            if not np.isnan(val):
                ink     = "#0f1115" if val > 50 else FG
                ink_sub = "#1a2a1a" if val > 50 else "#666666"
                ax.text(cx + CW/2, cy + CH*0.60, f"{val:.0f}%",
                        ha="center", va="center", fontsize=15, fontweight="bold", color=ink, zorder=4)
                ax.text(cx + CW/2, cy + CH*0.25, f"n={int(n)}",
                        ha="center", va="center", fontsize=8, color=ink_sub, zorder=4)

    leg_y  = BOTTOM_EXTRA * 0.55
    leg_x0 = LEFT_MARGIN
    leg_w  = n_col * SX * 0.55
    leg_h  = 0.22
    for i in range(200):
        seg_x = leg_x0 + i * (leg_w / 200)
        ax.add_patch(plt.Rectangle((seg_x, leg_y), leg_w/200 + 0.01, leg_h,
                                   linewidth=0, facecolor=RYG(i/200), zorder=2))
    for pct, label in [(0,"0%"),(25,"25%"),(50,"50%"),(75,"75%"),(100,"100%")]:
        ax.text(leg_x0 + (pct/100)*leg_w, leg_y - 0.12,
                label, ha="center", va="top", fontsize=8, color="#888888")
    ax.text(leg_x0 - 0.1, leg_y + leg_h/2,
            "% éxito", ha="right", va="center", fontsize=8, color="#888888")

    ax.text(LEFT_MARGIN, BOTTOM_EXTRA*0.18,
            "Fuente: Pro Football Reference via nflreadpy  ·  n = nº de picks analizados por celda",
            ha="left", va="center", fontsize=7.5, color="#555555", fontstyle="italic")
    ax.text(total_w - 0.05, BOTTOM_EXTRA*0.18, "@CuartayDato",
            ha="right", va="center", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

    plt.savefig(out_path, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"OK: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_draft_data()

    d20 = os.path.join(CALENDAR, "2026-04-20_Generales")
    d21 = os.path.join(CALENDAR, "2026-04-21_Generales")
    d22 = os.path.join(CALENDAR, "2026-04-22_Generales")

    print("\n--- Dia 20: Value Cliff Ataque ---")
    plot_value_cliff(df,
        pos_subset=["QB", "RB", "WR", "TE", "OL"],
        titulo_grupo="Ataque (QB - RB - WR - TE - OL)",
        out_path=os.path.join(d20, "value_cliff_ataque.png"))

    print("\n--- Dia 21: Value Cliff Defensa ---")
    plot_value_cliff(df,
        pos_subset=["DL", "LB", "CB", "S"],
        titulo_grupo="Defensa (DL - LB - CB - S)",
        out_path=os.path.join(d21, "value_cliff_defensa.png"))

    print("\n--- Dia 22: Draft Success R1 ---")
    plot_success_r1(df,
        out_path=os.path.join(d22, "draft_success_r1_por_eleccion.png"))

    print("\nTodos los PNGs generados.")
