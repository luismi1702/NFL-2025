# draft_grid.py
# Cuadrícula 4×8: mini-heatmap de éxito en el draft por equipo, posición y ronda
# Éxito = segundo contrato (≥2 años) con el mismo equipo que lo drafteó  ·  2011-2022

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap
import nflreadpy

# === Config ===
SEASONS   = list(range(2011, 2023))
LOGOS_DIR = "logos"
OUT       = "draft_grid_equipos.png"
BG        = "#0f1115"
CARD      = "#151924"
FG        = "#EDEDED"
ACCENT    = "#2d6cdf"
DPI       = 150
RYG       = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

POS_ORDER    = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"]
NCOLS        = 4   # columnas de equipos

# Agrupación de rondas
ROUND_GROUPS = [
    ([1, 2], "R1-2"),
    ([3, 4], "R3-4"),
    ([5, 6, 7], "R5-7"),
]
RND_LABELS   = [lbl for _, lbl in ROUND_GROUPS]
NRND         = len(ROUND_GROUPS)  # 3

POS_MAP = {
    "QB":"QB","RB":"RB","FB":"RB","WR":"WR","TE":"TE",
    "T":"OL","G":"OL","C":"OL","OL":"OL",
    "DE":"DL","DT":"DL","NT":"DL","DL":"DL",
    "LB":"LB","ILB":"LB","OLB":"LB",
    "CB":"DB","S":"DB","DB":"DB",
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
    draft["pos_group"]  = draft["position"].map(POS_MAP)
    draft["team_norm"]  = draft["team"].apply(normalize_team)
    draft = draft[
        draft["pos_group"].notna() &
        draft["gsis_id"].notna() &
        draft["team_norm"].notna()
    ].copy()
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

    # Asignar grupo de ronda
    rnd_map = {}
    for rounds, lbl in ROUND_GROUPS:
        for r in rounds:
            rnd_map[r] = lbl
    draft["rnd_group"] = draft["round"].map(rnd_map)

    return draft


# ─────────────────────────────────────────────
# 2. Logo helper
# ─────────────────────────────────────────────
BASE_ZOOM    = 0.030
HARD_PENALTY = {"NYJ": 4.5}

def add_logo(ax, team, x, y):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return
    try:
        img = plt.imread(path)
        h, w = img.shape[:2]
        aspect = (w / float(h)) if h else 1.0
        if team in HARD_PENALTY:
            zoom = BASE_ZOOM / HARD_PENALTY[team]
        else:
            divisor = np.clip(1.0 + 0.6 * max(0.0, aspect - 1.3), 1.0, 2.2)
            zoom = BASE_ZOOM / divisor
        ab = AnnotationBbox(
            OffsetImage(img, zoom=zoom, resample=True),
            (x, y), frameon=False, xycoords="data",
            box_alignment=(0.5, 0.5), pad=0
        )
        ax.add_artist(ab)
    except Exception:
        pass


# ─────────────────────────────────────────────
# 3. Gráfico
# ─────────────────────────────────────────────
def plot_grid(df: pd.DataFrame):
    # Tasa global por equipo → ordenar de mejor a peor
    team_rate = (df.groupby("team_norm")["success"]
                   .mean().mul(100)
                   .sort_values(ascending=False))
    teams_sorted = team_rate.index.tolist()
    nrows = int(np.ceil(len(teams_sorted) / NCOLS))  # 8

    # ── Dimensiones de cada mini-heatmap (coord de datos) ──────
    CW, CH = 1.72, 0.62     # celda (más ancha al tener sólo 3 columnas)
    GX, GY = 0.14, 0.14     # gap
    SX, SY = CW + GX, CH + GY   # step = 1.86, 0.76

    LM  = 0.70   # left margin dentro del bloque (etiquetas pos)
    TM  = 1.35   # top margin dentro del bloque (logo + rondas)
    RM  = 0.20   # right margin
    BM  = 0.20   # bottom margin
    BGX = 0.45   # gap horizontal entre bloques
    BGY = 0.45   # gap vertical entre bloques

    BW = LM + NRND * SX + RM      # ancho de bloque ≈ 6.48
    BH = TM + 8 * SY + BM         # alto de bloque  ≈ 7.63

    TITLE_H  = 1.8   # zona global de título
    SOURCE_H = 0.8   # zona de fuente/firma

    total_w = NCOLS * BW + (NCOLS - 1) * BGX
    total_h = TITLE_H + nrows * BH + (nrows - 1) * BGY + SOURCE_H

    fig, ax = plt.subplots(figsize=(16, 32), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    # ── Título global ────────────────────────────────────────────
    ax.text(total_w / 2, total_h - 0.30,
            "Éxito en el Draft NFL por equipo, posición y ronda  |  2011–2022",
            ha="center", va="top", fontsize=18, fontweight="bold", color=FG, zorder=5)
    ax.text(total_w / 2, total_h - 1.05,
            "% de picks que firmaron segundo contrato (≥2 años) con su equipo de draft  ·  Rondas agrupadas: R1-2 · R3-4 · R5-7",
            ha="center", va="top", fontsize=9, color="#888888", fontstyle="italic", zorder=5)

    # ── Mini-heatmaps ────────────────────────────────────────────
    for idx, team in enumerate(teams_sorted):
        col = idx % NCOLS
        row = idx // NCOLS

        # Origen del bloque (esquina inferior-izquierda)
        bx = col * (BW + BGX)
        by = total_h - TITLE_H - (row + 1) * BH - row * BGY

        # Fondo del bloque
        bg_rect = mpatches.FancyBboxPatch(
            (bx, by), BW, BH,
            boxstyle="round,pad=0.01,rounding_size=0.15",
            linewidth=0, facecolor=CARD, zorder=1
        )
        ax.add_patch(bg_rect)

        # --- Logo + nombre + tasa global ---
        logo_x = bx + LM / 2
        logo_y = by + BH - TM * 0.35
        add_logo(ax, team, logo_x, logo_y)

        rate_global = team_rate[team]
        ax.text(bx + LM + NRND * SX / 2 + RM / 2,
                by + BH - TM * 0.30,
                f"{team}  {rate_global:.1f}%",
                ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=FG, zorder=5)

        # --- Cabeceras de grupo de ronda ---
        for c_idx, lbl in enumerate(RND_LABELS):
            cx = bx + LM + c_idx * SX + CW / 2
            cy = by + BH - TM * 0.75
            ax.text(cx, cy, lbl,
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color=ACCENT, zorder=4)

        # --- Separador ---
        ax.plot([bx + 0.1, bx + BW - 0.1],
                [by + BH - TM + 0.12, by + BH - TM + 0.12],
                color="#2a2f3a", linewidth=0.6, zorder=3)

        # --- Celdas ---
        team_df = df[df["team_norm"] == team]
        rate_m  = team_df.groupby(["pos_group","rnd_group"])["success"].mean().mul(100).unstack()
        count_m = team_df.groupby(["pos_group","rnd_group"])["success"].count().unstack()
        rate_m  = rate_m.reindex(POS_ORDER).reindex(columns=RND_LABELS)
        count_m = count_m.reindex(POS_ORDER).reindex(columns=RND_LABELS)

        for r_idx, pos in enumerate(POS_ORDER):
            cell_y = by + BH - TM - (r_idx + 1) * SY + GY / 2

            # Etiqueta posición
            ax.text(bx + LM - 0.08, cell_y + CH / 2,
                    pos, ha="right", va="center",
                    fontsize=7.5, fontweight="bold", color=FG, zorder=4)

            for c_idx, lbl in enumerate(RND_LABELS):
                cell_x = bx + LM + c_idx * SX

                val = rate_m.loc[pos, lbl]  if pos in rate_m.index  else np.nan
                n   = count_m.loc[pos, lbl] if pos in count_m.index else 0
                n   = 0 if pd.isna(n) else int(n)

                if n == 0:
                    cell_color = "#1a1f2b"
                else:
                    cell_color = RYG(val / 100) if not np.isnan(val) else "#1a1f2b"

                cell = mpatches.FancyBboxPatch(
                    (cell_x, cell_y), CW, CH,
                    boxstyle="round,pad=0.01,rounding_size=0.07",
                    linewidth=0, facecolor=cell_color, zorder=2
                )
                ax.add_patch(cell)

                if n > 0 and not np.isnan(val):
                    ink = "#0f1115" if val > 50 else FG
                    ax.text(cell_x + CW / 2, cell_y + CH * 0.62,
                            f"{val:.0f}%",
                            ha="center", va="center",
                            fontsize=8, fontweight="bold", color=ink, zorder=4)
                    ax.text(cell_x + CW / 2, cell_y + CH * 0.22,
                            f"n={n}",
                            ha="center", va="center",
                            fontsize=6, color=ink if val > 50 else "#666666", zorder=4)

    # ── Fuente y firma ───────────────────────────────────────────
    ax.text(0.5, SOURCE_H * 0.55,
            "Fuente: Pro Football Reference & OverTheCap via nflreadpy  ·  Celdas oscuras = 0 picks en esa combinación",
            ha="center", va="center",
            fontsize=7.5, color="#555555", fontstyle="italic", zorder=5)
    ax.text(total_w - 0.2, SOURCE_H * 0.55,
            "@CuartayDato",
            ha="right", va="center",
            fontsize=9, color="#888888", alpha=0.85, fontstyle="italic", zorder=5)

    plt.savefig(OUT, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {OUT}")


# ─────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    print(f"\nTotal picks: {len(df)} | Equipos: {df['team_norm'].nunique()}")
    plot_grid(df)
