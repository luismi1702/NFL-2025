# draft_equipos.py
# Tasa de éxito en el draft por equipo NFL (2011-2022)
# Éxito = jugador renovó contrato (≥2 años) con el mismo equipo que lo drafteó

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap
import nflreadpy

# === Config ===
SEASONS   = list(range(2011, 2023))
LOGOS_DIR = "logos"
OUT       = "draft_success_por_equipo.png"
BG        = "#0f1115"
FG        = "#EDEDED"
GRID      = "#2a2f3a"
DPI       = 200
RYG       = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

# Mapa posiciones
POS_MAP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "T":  "OL", "G": "OL", "C": "OL", "OL": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB",
    "CB": "DB", "S": "DB", "DB": "DB",
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

def normalize_team(raw: str) -> str:
    if not isinstance(raw, str):
        return None
    base = raw.split("/")[0].strip()
    if base in ABBR_NORM:
        return ABBR_NORM[base]
    if base in NICK_TO_ABBR:
        return NICK_TO_ABBR[base]
    return None


# ─────────────────────────────────────────────
# 1. Datos
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("Cargando draft picks 2011-2022...")
    draft = nflreadpy.load_draft_picks(seasons=SEASONS).to_pandas()
    draft["pos_group"] = draft["position"].map(POS_MAP)
    draft = draft[draft["pos_group"].notna() & draft["gsis_id"].notna()].copy()
    draft["season"] = draft["season"].astype(int)
    draft["team_norm"] = draft["team"].apply(normalize_team)

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


def compute_team_rates(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("team_norm")["success"].agg(["mean", "count"]).reset_index()
    grp.columns = ["team", "rate", "n"]
    grp["rate_pct"] = (grp["rate"] * 100).round(1)
    return grp.sort_values("rate_pct", ascending=True).reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. Logo helper
# ─────────────────────────────────────────────
BASE_ZOOM    = 0.055          # zoom base (calibrado para fig 13x14 a DPI 200)
HARD_PENALTY = {"NYJ": 4.5}  # NYJ es 4096x4096, necesita reducción extra

def add_logo(ax, team: str, y: float, x_logo: float):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        ax.text(x_logo, y, team, ha="center", va="center",
                fontsize=8, color=FG, fontweight="bold")
        return
    try:
        img = plt.imread(path)
        h, w = img.shape[:2]
        aspect = (w / float(h)) if h else 1.0

        if team in HARD_PENALTY:
            zoom = BASE_ZOOM / HARD_PENALTY[team]
        else:
            # Logos anchos se reducen proporcionalmente para no sobrepasar el espacio
            divisor = 1.0 + 0.6 * max(0.0, aspect - 1.3)
            divisor = np.clip(divisor, 1.0, 2.2)
            zoom = BASE_ZOOM / divisor

        ab = AnnotationBbox(
            OffsetImage(img, zoom=zoom, resample=True),
            (x_logo, y), frameon=False, xycoords="data",
            box_alignment=(0.5, 0.5), pad=0
        )
        ax.add_artist(ab)
    except Exception:
        ax.text(x_logo, y, team, ha="center", va="center",
                fontsize=8, color=FG, fontweight="bold")


# ─────────────────────────────────────────────
# 3. Gráfico
# ─────────────────────────────────────────────
def plot_ranking(team_df: pd.DataFrame):
    n = len(team_df)
    nfl_avg = (team_df["rate_pct"] * team_df["n"]).sum() / team_df["n"].sum()

    # Colores por percentil
    ranks  = team_df["rate_pct"].values.argsort().argsort()
    colors = [RYG(r / max(n - 1, 1)) for r in ranks]

    fig, ax = plt.subplots(figsize=(13, 14), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    y_pos = np.arange(n)
    bar_h = 0.62

    # Barras
    ax.barh(y_pos, team_df["rate_pct"], color=colors, height=bar_h, zorder=2)

    # Valores dentro/fuera de barra
    xmax = team_df["rate_pct"].max()
    for y, (_, row) in zip(y_pos, team_df.iterrows()):
        val = row["rate_pct"]
        txt = f"{val:.1f}%  (n={int(row['n'])})"
        offset = xmax * 0.012
        ax.text(val + offset, y, txt,
                va="center", ha="left", fontsize=9, color=FG, zorder=3)

    # Línea media NFL
    ax.axvline(nfl_avg, color="#ffd700", linewidth=1.4,
               linestyle="--", zorder=4, alpha=0.85)
    ax.text(nfl_avg + xmax * 0.005, n - 0.1,
            f"Media NFL\n{nfl_avg:.1f}%",
            color="#ffd700", fontsize=8.5, va="top", ha="left", zorder=5)

    # Logos — se colocan DESPUÉS de fijar xlim para que x_logo esté bien calculado
    ax.set_xlim(-xmax * 0.20, xmax * 1.18)
    xmin_data = ax.get_xlim()[0]
    xmax_data = ax.get_xlim()[1]
    x_logo = xmin_data + (xmax_data - xmin_data) * 0.07   # 7% desde el borde izq

    for y, (_, row) in zip(y_pos, team_df.iterrows()):
        add_logo(ax, row["team"], y, x_logo)

    # Ejes y estilo
    ax.set_yticks([])
    ax.set_xlabel("% de picks que renovaron con su equipo de draft", fontsize=11,
                  labelpad=8, color=FG)
    ax.tick_params(colors=FG)
    ax.grid(axis="x", linestyle="--", alpha=0.2, zorder=1)
    ax.axvline(0, color=GRID, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    # Título
    ax.set_title(
        "Éxito en el Draft NFL por equipo  |  Drafts 2011–2022",
        fontsize=16, fontweight="bold", pad=14, color=FG
    )

    # Subtítulo
    fig.text(0.5, 0.965,
             "% de picks que firmaron un segundo contrato (≥2 años) con el mismo equipo que los drafteó",
             ha="center", va="top", fontsize=9.5, color="#888888", fontstyle="italic")

    # Fuente
    ax.text(0.01, -0.03,
            "Fuente: Pro Football Reference & OverTheCap via nflreadpy  ·  n = nº de picks por equipo",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color="#555555", fontstyle="italic")

    # Firma
    ax.text(0.99, -0.03, "@CuartayDato",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

    ax.invert_yaxis()

    plt.savefig(OUT, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {OUT}")


# ─────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df       = load_data()
    team_df  = compute_team_rates(df)

    print(f"\nTotal picks: {df['success'].count()} | Equipos: {len(team_df)}")
    print(f"Media NFL: {(team_df['rate_pct'] * team_df['n']).sum() / team_df['n'].sum():.1f}%\n")
    print(team_df.sort_values("rate_pct", ascending=False).to_string(index=False))

    plot_ranking(team_df)
