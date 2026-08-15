# draft_eval_2026.py
# Evaluación estadística del Draft NFL 2026
# Metodología: tasa histórica de éxito por posición × ronda (2011-2022)
# Éxito = firmó extensión ≥2 años con el mismo equipo, 3+ años después del draft

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import os

from draft_success import load_data, compute_heatmap, BG, CARD, FG, ACCENT

DPI     = 180
FIGSIZE = (14, 18)
RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

# ── Picks del Draft 2026 (round, pick_overall, team, pos_raw) ─────────────────
DRAFT_2026 = [
    # R1
    (1,1,"LV","QB"),(1,2,"NYJ","LB"),(1,3,"ARI","RB"),(1,4,"TEN","WR"),
    (1,5,"NYG","LB"),(1,6,"KC","CB"),(1,7,"WAS","LB"),(1,8,"NO","WR"),
    (1,9,"CLE","T"),(1,10,"NYG","G"),(1,11,"DAL","S"),(1,12,"MIA","T"),
    (1,13,"LA","QB"),(1,14,"BAL","G"),(1,15,"TB","DE"),(1,16,"NYJ","TE"),
    (1,17,"DET","T"),(1,18,"MIN","DE"),(1,19,"CAR","T"),(1,20,"PHI","WR"),
    (1,21,"PIT","T"),(1,22,"LAC","LB"),(1,23,"DAL","DE"),(1,24,"CLE","WR"),
    (1,25,"CHI","S"),(1,26,"HOU","G"),(1,27,"MIA","CB"),(1,28,"NE","T"),
    (1,29,"KC","DT"),(1,30,"NYJ","WR"),(1,31,"TEN","DE"),(1,32,"SEA","RB"),
    # R2
    (2,33,"SF","WR"),(2,34,"ARI","G"),(2,35,"BUF","DE"),(2,36,"HOU","DT"),
    (2,37,"NYG","CB"),(2,38,"LV","S"),(2,39,"CLE","WR"),(2,40,"KC","DE"),
    (2,41,"CIN","DE"),(2,42,"NO","DT"),(2,43,"MIA","LB"),(2,44,"DET","DE"),
    (2,45,"BAL","OLB"),(2,46,"TB","LB"),(2,47,"PIT","WR"),(2,48,"ATL","CB"),
    (2,49,"CAR","DT"),(2,50,"NYJ","CB"),(2,51,"MIN","LB"),(2,52,"GB","CB"),
    (2,53,"IND","LB"),(2,54,"PHI","TE"),(2,55,"NE","DE"),(2,56,"JAX","TE"),
    (2,57,"CHI","C"),(2,58,"CLE","S"),(2,59,"HOU","TE"),(2,60,"TEN","LB"),
    (2,61,"LA","TE"),(2,62,"BUF","CB"),(2,63,"LAC","C"),(2,64,"SEA","S"),
    # R3
    (3,65,"SF","QB"),(3,66,"DEN","DT"),(3,67,"LV","DE"),(3,68,"PHI","T"),
    (3,69,"CHI","TE"),(3,70,"SF","OLB"),(3,71,"WAS","WR"),(3,72,"CIN","CB"),
    (3,73,"NO","TE"),(3,74,"NYG","WR"),(3,75,"MIA","WR"),(3,76,"PIT","QB"),
    (3,77,"GB","DT"),(3,78,"IND","S"),(3,79,"ATL","WR"),(3,80,"BAL","WR"),
    (3,81,"JAX","DT"),(3,82,"MIN","DT"),(3,83,"CAR","WR"),(3,84,"TB","WR"),
    (3,85,"PIT","CB"),(3,86,"CLE","T"),(3,87,"MIA","TE"),(3,88,"JAX","G"),
    (3,89,"CHI","WR"),(3,90,"SF","RB"),(3,91,"LV","C"),(3,92,"DAL","OLB"),
    (3,93,"LA","T"),(3,94,"MIA","WR"),(3,95,"NE","TE"),(3,96,"PIT","T"),
    (3,97,"MIN","T"),(3,98,"MIN","S"),(3,99,"SEA","CB"),(3,100,"JAX","S"),
    # R4
    (4,101,"LV","CB"),(4,102,"BUF","T"),(4,103,"NYJ","DT"),(4,104,"ARI","DT"),
    (4,105,"LAC","WR"),(4,106,"HOU","G"),(4,107,"SF","DT"),(4,108,"DEN","RB"),
    (4,109,"KC","CB"),(4,110,"NYJ","QB"),(4,111,"DEN","T"),(4,112,"DAL","T"),
    (4,113,"IND","G"),(4,114,"DAL","CB"),(4,115,"BAL","WR"),(4,116,"TB","CB"),
    (4,117,"LAC","T"),(4,118,"DET","LB"),(4,119,"JAX","DE"),(4,120,"GB","DE"),
    (4,121,"PIT","WR"),(4,122,"LV","RB"),(4,123,"HOU","LB"),(4,124,"CHI","CB"),
    (4,125,"BUF","WR"),(4,126,"BUF","LB"),(4,127,"SF","T"),(4,128,"CIN","C"),
    (4,129,"CAR","CB"),(4,130,"MIA","DE"),(4,131,"LAC","S"),(4,132,"NO","G"),
    (4,133,"BAL","TE"),(4,134,"ATL","LB"),(4,135,"IND","LB"),(4,136,"NO","WR"),
    (4,137,"DAL","DT"),(4,138,"MIA","LB"),(4,139,"SF","CB"),(4,140,"CIN","WR"),
    # R5
    (5,141,"HOU","S"),(5,142,"TEN","G"),(5,143,"ARI","WR"),(5,144,"CAR","C"),
    (5,145,"LAC","DT"),(5,146,"CLE","C"),(5,147,"WAS","DE"),(5,148,"SEA","G"),
    (5,149,"CLE","LB"),(5,150,"LV","S"),(5,151,"CAR","S"),(5,152,"DEN","TE"),
    (5,153,"GB","C"),(5,154,"SF","LB"),(5,155,"TB","DT"),(5,156,"IND","DE"),
    (5,157,"DET","CB"),(5,158,"MIA","S"),(5,159,"MIN","FB"),(5,160,"TB","G"),
    (5,161,"KC","RB"),(5,162,"BAL","CB"),(5,163,"MIN","CB"),(5,164,"JAX","TE"),
    (5,165,"TEN","RB"),(5,166,"CHI","LB"),(5,167,"BUF","S"),(5,168,"DET","WR"),
    (5,169,"PIT","TE"),(5,170,"CLE","TE"),(5,171,"NE","CB"),(5,172,"NO","S"),
    (5,173,"BAL","TE"),(5,174,"BAL","RB"),(5,175,"LV","CB"),(5,176,"KC","WR"),
    (5,177,"MIA","WR"),(5,178,"PHI","QB"),(5,179,"SF","T"),(5,180,"MIA","TE"),
    (5,181,"BUF","DT"),
    # R6
    (6,182,"CLE","QB"),(6,183,"ARI","LB"),(6,184,"TEN","DT"),(6,185,"TB","TE"),
    (6,186,"NYG","DT"),(6,187,"WAS","RB"),(6,188,"NYJ","G"),(6,189,"CIN","C"),
    (6,190,"NO","WR"),(6,191,"JAX","WR"),(6,192,"NYG","T"),(6,193,"NYG","LB"),
    (6,194,"TEN","C"),(6,195,"LV","WR"),(6,196,"NE","T"),(6,197,"LA","WR"),
    (6,198,"MIN","RB"),(6,199,"SEA","WR"),(6,200,"MIA","G"),(6,201,"GB","CB"),
    (6,202,"LAC","G"),(6,203,"JAX","WR"),(6,204,"HOU","WR"),(6,205,"DET","DT"),
    (6,206,"LAC","G"),(6,207,"PHI","G"),(6,208,"ATL","DT"),(6,209,"WAS","C"),
    (6,210,"PIT","DT"),(6,212,"NE","LB"),(6,213,"CHI","DT"),
    (6,214,"IND","DE"),(6,215,"ATL","LB"),
    # R7
    (7,217,"ARI","T"),(7,218,"DAL","WR"),(7,219,"NO","CB"),(7,220,"BUF","CB"),
    (7,221,"CIN","TE"),(7,222,"DET","DE"),(7,223,"WAS","QB"),(7,224,"PIT","S"),
    (7,225,"TEN","TE"),(7,226,"CIN","DT"),(7,227,"CAR","LB"),(7,228,"NYJ","S"),
    (7,229,"LV","DT"),(7,230,"PIT","RB"),(7,231,"ATL","T"),(7,232,"LA","DT"),
    (7,233,"JAX","DE"),(7,234,"NE","QB"),(7,235,"MIN","C"),(7,236,"SEA","CB"),
    (7,237,"IND","RB"),(7,238,"MIA","DE"),(7,240,"JAX","LB"),
    (7,241,"BUF","G"),(7,242,"SEA","DT"),(7,243,"HOU","LB"),(7,244,"PHI","S"),
    (7,245,"NE","RB"),(7,246,"DEN","S"),(7,247,"NE","DE"),(7,248,"CLE","TE"),
    (7,249,"KC","QB"),(7,250,"BAL","DT"),(7,251,"PHI","DT"),(7,252,"PHI","DE"),
]

POS_MAP = {
    "QB":"QB","RB":"RB","FB":"RB","WR":"WR","TE":"TE",
    "T":"OL","G":"OL","C":"OL","OL":"OL",
    "DE":"DL","DT":"DL","NT":"DL",
    "LB":"LB","ILB":"LB","OLB":"LB",
    "CB":"DB","S":"DB",
    "P":None,"K":None,
}

ALL_TEAMS = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE",
    "DAL","DEN","DET","GB","HOU","IND","JAX","KC",
    "LA","LAC","LV","MIA","MIN","NE","NO","NYG",
    "NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS",
]


def get_logo(team, base_zoom=0.042):
    path = f"logos/{team}.png"
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGBA")
    # Recorta el margen transparente (NYJ es un wordmark con mucho aire)
    caja = img.getbbox()
    if caja:
        img = img.crop(caja)
    w, h = img.size
    # Normaliza por el area de tinta real, no por el lienzo
    zoom = base_zoom * 500.0 / max((h * w) ** 0.5, 1.0)
    if w * zoom > 900.0 * base_zoom:
        zoom = 900.0 * base_zoom / w
    return OffsetImage(img, zoom=zoom)


def build_expected(rate: pd.DataFrame) -> pd.DataFrame:
    records = []
    for rnd, pick, team, pos_raw in DRAFT_2026:
        pg = POS_MAP.get(pos_raw)
        if pg is None:
            continue
        if pg in rate.index and rnd in rate.columns:
            val = rate.loc[pg, rnd]
            ev = val / 100 if not np.isnan(val) else 0.0
        else:
            ev = 0.0
        records.append({"team": team, "round": rnd, "pos_group": pg, "ev": ev})
    df = pd.DataFrame(records)
    summary = df.groupby("team").agg(
        total_ev=("ev", "sum"),
        n_picks=("ev", "count"),
    ).reset_index()
    summary["ev_per_pick"] = summary["total_ev"] / summary["n_picks"]
    for t in ALL_TEAMS:
        if t not in summary["team"].values:
            summary = pd.concat([summary, pd.DataFrame([{"team": t, "total_ev": 0, "n_picks": 0, "ev_per_pick": 0}])], ignore_index=True)
    summary = summary.sort_values("total_ev", ascending=True).reset_index(drop=True)
    return summary


MEDAL_COLORS = {0: "#FFD700", 1: "#C0C0C0", 2: "#CD7F32"}  # oro, plata, bronce (top 3 desde arriba)
BOTTOM_COLOR = "#d84a4a"
DEFAULT_COLOR = "#2d6cdf"

def plot(summary: pd.DataFrame):
    n = len(summary)  # sorted ascending → index 0 = peor, index n-1 = mejor
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    max_ev = summary["total_ev"].max()
    bar_h = 0.60
    league_avg = summary["total_ev"].mean()

    for i, row in summary.iterrows():
        rank = n - i  # rank 1 = mejor (i = n-1)

        # Color según posición
        if rank <= 3:
            color = MEDAL_COLORS[rank - 1]
        elif rank >= n - 2:
            color = BOTTOM_COLOR
        else:
            color = DEFAULT_COLOR

        # Barra
        ax.barh(i, row["total_ev"], height=bar_h, color=color,
                left=0.0, zorder=2, linewidth=0, alpha=0.9)

        # Valor al final de la barra
        ax.text(row["total_ev"] + 0.03, i + 0.08,
                f"{row['total_ev']:.2f}",
                va="center", ha="left", fontsize=9.5, color=FG, fontweight="bold", zorder=4)

        # N picks
        ax.text(row["total_ev"] + 0.03, i - 0.18,
                f"{int(row['n_picks'])} picks",
                va="center", ha="left", fontsize=7.5, color="#888888", zorder=4)

        # Logo
        logo = get_logo(row["team"])
        if logo:
            ab = AnnotationBbox(logo, (-0.10, i), frameon=False,
                                box_alignment=(1, 0.5), zorder=5)
            ax.add_artist(ab)
        else:
            ax.text(-0.06, i, row["team"], va="center", ha="right",
                    fontsize=8, color=FG, zorder=4)

        # Ranking número
        rank_color = MEDAL_COLORS.get(rank - 1, BOTTOM_COLOR if rank >= n - 1 else "#666666")
        ax.text(-0.20, i, f"#{rank}", va="center", ha="right",
                fontsize=9, color=rank_color if rank <= 3 or rank >= n - 1 else "#555555",
                fontweight="bold" if rank <= 3 else "normal", zorder=4)

    # Grid vertical suave
    ax.xaxis.grid(True, color="#2a2f3a", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Línea media liga
    ax.axvline(league_avg, color="#888888", linewidth=1.0, linestyle="--", zorder=3, alpha=0.5)
    ax.text(league_avg + 0.02, n - 0.3,
            f"media\n{league_avg:.2f}", color="#888888", fontsize=7.5, va="top", ha="left")

    # Ejes
    ax.set_xlim(-0.32, max_ev * 1.20)
    ax.set_ylim(-0.8, n + 0.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors="#555555", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#2a2f3a")

    # Título
    fig.text(0.04, 0.975,
             "¿Quién ganó el Draft 2026 en papel?",
             fontsize=19, fontweight="bold", color=FG, va="top")
    fig.text(0.04, 0.963,
             "Éxitos esperados por equipo · Tasas históricas de renovación por posición × ronda (2011–2022)",
             fontsize=9, color="#888888", fontstyle="italic", va="top")

    # Leyenda
    legend_items = [
        mpatches.Patch(color=MEDAL_COLORS[0], label="Top 3"),
        mpatches.Patch(color=DEFAULT_COLOR, label="Resto"),
        mpatches.Patch(color=BOTTOM_COLOR, label="Bottom 2"),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=False,
              labelcolor=FG, fontsize=8)

    # Fuente y marca de agua
    fig.text(0.04, 0.012,
             "Fuente: nflreadpy · Contratos OTC · Drafts 2011–2022",
             fontsize=7.5, color="#555555", fontstyle="italic")
    fig.text(0.97, 0.012, "@CuartayDato",
             ha="right", fontsize=9, color="#888888", alpha=0.8, fontstyle="italic")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out = "draft_eval_2026.png"
    plt.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out}")
    return out


if __name__ == "__main__":
    print("Cargando datos históricos...")
    df = load_data()
    rate, _ = compute_heatmap(df)

    print("Calculando valor esperado por equipo...")
    summary = build_expected(rate)

    print("\n=== TOP 10 ===")
    top = summary.sort_values("total_ev", ascending=False).head(10)
    for _, r in top.iterrows():
        print(f"{r['team']:4s}  EV={r['total_ev']:.2f}  picks={int(r['n_picks'])}  EV/pick={r['ev_per_pick']:.3f}")

    print("\n=== BOTTOM 5 ===")
    bot = summary.sort_values("total_ev").head(5)
    for _, r in bot.iterrows():
        print(f"{r['team']:4s}  EV={r['total_ev']:.2f}  picks={int(r['n_picks'])}  EV/pick={r['ev_per_pick']:.3f}")

    plot(summary)
