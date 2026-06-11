# draft_ranking_equipos.py
# Modo ranking : bar chart de 32 equipos ordenados por tasa de éxito en el draft
# Modo equipo  : heatmap detallado de un equipo (posición × ronda R1-R7)
# Éxito = segundo contrato (≥2 años) con el mismo equipo · Drafts 2011-2022

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import nflreadpy

# === Config ===
SEASONS   = list(range(2011, 2023))
LOGOS_DIR = "logos"
BG        = "#0f1115"
CARD      = "#151924"
FG        = "#EDEDED"
GRID_C    = "#2a2f3a"
ACCENT    = "#2d6cdf"
DPI       = 180

RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

POS_MAP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "T": "OL", "G": "OL", "C": "OL", "OL": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB",
    "CB": "DB", "S": "DB", "DB": "DB",
}
POS_ORDER    = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"]
ROUND_LABELS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]

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
    draft["team_norm"] = draft["team"].apply(normalize_team)
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
    return draft


# ─────────────────────────────────────────────
# 2. Helper: logo
# ─────────────────────────────────────────────
BASE_ZOOM    = 0.042
HARD_PENALTY = {"NYJ": 4.5}

def add_logo(ax, team, x, y):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path): return
    try:
        img    = plt.imread(path)
        h, w   = img.shape[:2]
        aspect = w / float(h) if h else 1.0
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
# 3. Modo ranking
# ─────────────────────────────────────────────
def plot_ranking(df):
    team_rate  = df.groupby("team_norm")["success"].mean().mul(100)
    team_count = df.groupby("team_norm")["success"].count()
    team_rate  = team_rate.sort_values(ascending=True)   # ascendente → mejor arriba en barh
    league_avg = float(team_rate.mean())

    teams  = team_rate.index.tolist()
    values = team_rate.values
    n      = len(teams)

    # Normalizar colores sobre el rango real (peor=rojo, mejor=verde)
    v_min, v_max = float(values.min()), float(values.max())
    v_range = max(v_max - v_min, 1.0)

    fig, ax = plt.subplots(figsize=(11, 14), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bar_h   = 0.62
    logo_x  = -4.5   # posición X de los logos (a la izquierda del eje)

    for i, (team, val) in enumerate(zip(teams, values)):
        color = RYG((val - v_min) / v_range)
        ax.barh(i, val, height=bar_h, color=color, zorder=3)

        # Porcentaje al final de la barra
        ax.text(val + 0.6, i, f"{val:.1f}%",
                va='center', ha='left', fontsize=8.5,
                fontweight='bold', color=color, zorder=4)

        # Número de picks (n)
        n_picks = int(team_count.get(team, 0))
        ax.text(val + 5.5, i, f"n={n_picks}",
                va='center', ha='left', fontsize=7,
                color="#555555", zorder=4)

        # Logo a la izquierda
        add_logo(ax, team, logo_x, i)

    # Línea de media de la liga
    ax.axvline(league_avg, color='white', linewidth=1.2,
               linestyle='--', alpha=0.45, zorder=2)
    ax.text(league_avg + 0.3, n - 0.3, f"Media: {league_avg:.1f}%",
            fontsize=7.5, color='white', alpha=0.55, va='top', zorder=4)

    # Ejes
    ax.set_xlim(-7, 80)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(teams, fontsize=8.5, color=FG)
    ax.set_xticks(range(0, 81, 10))
    ax.set_xticklabels([f"{x}%" for x in range(0, 81, 10)],
                       color="#888888", fontsize=8)
    ax.tick_params(axis='both', length=0)
    ax.xaxis.grid(True, color=GRID_C, linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)

    # Título
    ax.text(0.01, 1.028,
            "Ranking de equipos NFL por éxito en el draft · 2011–2022",
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=14, fontweight='bold', color=FG)
    ax.text(0.01, 1.005,
            "% de picks que firmaron segundo contrato (≥2 años) con el mismo equipo que los drafteó",
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8.5, color="#888888", fontstyle='italic')

    ax.text(0.01, 0.003, "Fuente: PFR & OverTheCap via nflreadpy",
            transform=ax.transAxes, ha="left", va="bottom",
            color="#555555", fontsize=7.5, fontstyle="italic")
    ax.text(0.99, 0.003, "@CuartayDato",
            transform=ax.transAxes, ha="right", va="bottom",
            color="#888888", fontsize=9, alpha=0.8, fontstyle="italic")

    plt.tight_layout()
    out = "draft_ranking_equipos.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {out}")


# ─────────────────────────────────────────────
# 4. Modo spotlight — ranking con equipo destacado
# ─────────────────────────────────────────────
def plot_ranking_spotlight(df, team):
    team = team.upper()
    team_rate  = df.groupby("team_norm")["success"].mean().mul(100)
    team_count = df.groupby("team_norm")["success"].count()
    team_rate  = team_rate.sort_values(ascending=True)
    league_avg = float(team_rate.mean())

    teams  = team_rate.index.tolist()
    values = team_rate.values
    n      = len(teams)
    v_min, v_max = float(values.min()), float(values.max())
    v_range = max(v_max - v_min, 1.0)

    rank = sorted(team_rate.index.tolist(), key=lambda t: team_rate[t], reverse=True).index(team) + 1

    fig, ax = plt.subplots(figsize=(11, 14), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bar_h  = 0.62
    logo_x = -4.5

    for i, (t, val) in enumerate(zip(teams, values)):
        if t == team:
            color = RYG((val - v_min) / v_range)
            alpha = 1.0
        else:
            color = "#2a2f3a"
            alpha = 0.55

        ax.barh(i, val, height=bar_h, color=color, alpha=alpha, zorder=3)

        if t == team:
            ax.text(val + 0.6, i, f"{val:.1f}%",
                    va='center', ha='left', fontsize=8.5,
                    fontweight='bold', color=color, zorder=4)

        add_logo(ax, t, logo_x, i)

    # Línea de media de la liga
    ax.axvline(league_avg, color='white', linewidth=1.2,
               linestyle='--', alpha=0.45, zorder=2)
    ax.text(league_avg + 0.3, n - 0.3, f"Media: {league_avg:.1f}%",
            fontsize=7.5, color='white', alpha=0.55, va='top', zorder=4)

    # Rank badge
    ax.text(0.99, 0.02, f"#{rank} de 32",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=22, fontweight='bold',
            color=RYG((team_rate[team] - v_min) / v_range), alpha=0.9, zorder=5)

    ax.set_xlim(-7, 80)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(teams, fontsize=8.5, color=FG)
    for lbl, t in zip(ax.get_yticklabels(), teams):
        lbl.set_color(FG if t == team else "#444c5e")
    ax.set_xticks(range(0, 81, 10))
    ax.set_xticklabels([f"{x}%" for x in range(0, 81, 10)],
                       color="#888888", fontsize=8)
    ax.tick_params(axis='both', length=0)
    ax.xaxis.grid(True, color=GRID_C, linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)

    ax.text(0.01, 1.028,
            f"{team} en el ranking de draft NFL · 2011–2022",
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=14, fontweight='bold', color=FG)
    ax.text(0.01, 1.005,
            "% de picks que firmaron segundo contrato (>=2 años) con el mismo equipo que los drafteó",
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8.5, color="#888888", fontstyle='italic')

    ax.text(0.01, 0.003, "Fuente: PFR & OverTheCap via nflreadpy",
            transform=ax.transAxes, ha="left", va="bottom",
            color="#555555", fontsize=7.5, fontstyle="italic")
    ax.text(0.99, 0.003, "@CuartayDato",
            transform=ax.transAxes, ha="right", va="bottom",
            color="#888888", fontsize=9, alpha=0.8, fontstyle="italic")

    plt.tight_layout()
    out = "draft_ranking_spotlight.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {out}")


# ─────────────────────────────────────────────
# 5. Modo equipo — heatmap detallado (posición × ronda)
# ─────────────────────────────────────────────
def plot_equipo(df, team):
    team    = team.upper()
    team_df = df[df["team_norm"] == team]
    if team_df.empty:
        print(f"No se encontraron picks para {team}. Verifica la sigla.")
        return

    rate  = team_df.groupby(["pos_group","round"])["success"].mean().mul(100).unstack()
    count = team_df.groupby(["pos_group","round"])["success"].count().unstack()
    rate  = rate.reindex(POS_ORDER).reindex(columns=range(1, 8))
    count = count.reindex(POS_ORDER).reindex(columns=range(1, 8))

    # Media de la liga para comparar (base del color)
    lg_rate = df.groupby(["pos_group","round"])["success"].mean().mul(100).unstack()
    lg_rate = lg_rate.reindex(POS_ORDER).reindex(columns=range(1, 8))

    MAX_DELTA = 20.0   # ±20pp cubre los extremos del gradiente (np.clip satura en rojo/verde puro)

    n_pos = len(POS_ORDER)
    n_rnd = 7

    CW, CH = 1.10, 0.82
    GX, GY = 0.16, 0.24
    SX, SY = CW + GX, CH + GY

    LEFT_MARGIN  = 1.8
    TOP_MARGIN   = 3.2
    BOTTOM_EXTRA = 1.1
    RIGHT_MARGIN = 1.2
    total_w = LEFT_MARGIN + n_rnd * SX + RIGHT_MARGIN
    total_h = TOP_MARGIN + n_pos * SY + BOTTOM_EXTRA

    fig, ax = plt.subplots(figsize=(13, 11), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    # ── Título ────────────────────────────────────────────────────
    ax.text(LEFT_MARGIN, total_h - 0.28,
            f"{team} — Tasa de éxito en el draft por posición y ronda",
            ha="left", va="top", fontsize=17, fontweight="bold", color=FG, zorder=5)
    ax.text(LEFT_MARGIN, total_h - 1.05,
            "% de picks con segundo contrato (≥2 años) · Drafts 2011–2022  ·  color = diferencia vs media liga  ·  △ = pp de diferencia",
            ha="left", va="top", fontsize=9, color="#888888", fontstyle="italic", zorder=5)

    # ── Logo del equipo ───────────────────────────────────────────
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if os.path.exists(path):
        try:
            img    = plt.imread(path)
            h, w   = img.shape[:2]
            aspect = w / float(h) if h else 1.0
            zoom   = 0.10 if team != "NYJ" else 0.10 / 4.5
            divisor = np.clip(1.0 + 0.6 * max(0.0, aspect - 1.3), 1.0, 2.2)
            zoom = zoom / divisor
            ab = AnnotationBbox(
                OffsetImage(img, zoom=zoom, resample=True),
                (LEFT_MARGIN - 1.0, total_h - 0.85),
                frameon=False, xycoords="data",
                box_alignment=(0.5, 0.5), pad=0
            )
            ax.add_artist(ab)
        except Exception:
            pass

    # ── Cabeceras de ronda ────────────────────────────────────────
    for c, lbl in enumerate(ROUND_LABELS):
        cx = LEFT_MARGIN + c * SX + CW / 2
        cy = total_h - TOP_MARGIN * 0.73
        pill = plt.Rectangle(
            (LEFT_MARGIN + c * SX, cy - 0.30), CW, 0.56,
            linewidth=0, facecolor=CARD, transform=ax.transData, zorder=1
        )
        ax.add_patch(pill)
        ax.text(cx, cy, lbl, ha="center", va="center",
                fontsize=12, fontweight="bold", color=ACCENT, zorder=2)

    # ── Celdas ────────────────────────────────────────────────────
    for r, pos in enumerate(POS_ORDER):
        row_y = total_h - TOP_MARGIN - (r + 1) * SY + GY / 2

        # Etiqueta posición
        ax.text(LEFT_MARGIN - 0.18, row_y + CH / 2, pos,
                ha="right", va="center",
                fontsize=13, fontweight="bold", color=FG)

        for c, rnd in enumerate(range(1, 8)):
            val    = rate.loc[pos, rnd]  if not pd.isna(rate.loc[pos, rnd])  else np.nan
            n_val  = count.loc[pos, rnd] if not pd.isna(count.loc[pos, rnd]) else 0
            lg_val = lg_rate.loc[pos, rnd] if not pd.isna(lg_rate.loc[pos, rnd]) else np.nan

            if not np.isnan(val):
                if not np.isnan(lg_val):
                    delta      = val - lg_val
                    norm_delta = np.clip(0.5 + delta / (2 * MAX_DELTA), 0.0, 1.0)
                else:
                    delta      = np.nan
                    norm_delta = 0.5
                cell_color = RYG(norm_delta)
            else:
                delta      = np.nan
                norm_delta = 0.0
                cell_color = "#1e2330"
            cx = LEFT_MARGIN + c * SX
            cy = row_y

            # Celda redondeada
            cell = mpatches.FancyBboxPatch(
                (cx, cy), CW, CH,
                boxstyle="round,pad=0.01,rounding_size=0.08",
                linewidth=0, facecolor=cell_color,
                transform=ax.transData, zorder=2
            )
            ax.add_patch(cell)

            if not np.isnan(val):
                ink     = "#0f1115" if norm_delta > 0.35 else FG
                ink_sub = "#1a2a1a" if norm_delta > 0.35 else "#555555"

                # Porcentaje (grande, centrado)
                ax.text(cx + CW / 2, cy + CH * 0.62,
                        f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=14, fontweight="bold", color=ink, zorder=4)

                # n picks
                ax.text(cx + CW / 2, cy + CH * 0.28,
                        f"n={int(n_val)}",
                        ha="center", va="center",
                        fontsize=7.5, color=ink_sub, zorder=4)

                # Delta vs media liga (esquina superior derecha de la celda)
                if not np.isnan(delta):
                    d_ink = "#0f1115" if norm_delta > 0.35 else FG
                    sign  = "+" if delta >= 0 else ""
                    ax.text(cx + CW - 0.06, cy + CH - 0.06,
                            f"{sign}{delta:.0f}",
                            ha="right", va="top",
                            fontsize=6.5, color=d_ink, zorder=4, alpha=0.9)
            else:
                # Celda vacía: mostrar n=0 si hay datos de liga
                if not np.isnan(lg_val):
                    ax.text(cx + CW / 2, cy + CH / 2, "—",
                            ha="center", va="center",
                            fontsize=12, color="#2a3245", zorder=4)

    # ── Leyenda de color ──────────────────────────────────────────
    leg_y  = BOTTOM_EXTRA * 0.52
    leg_x0 = LEFT_MARGIN
    leg_w  = n_rnd * SX * 0.52
    leg_h  = 0.22
    n_seg  = 200
    for i in range(n_seg):
        seg_x = leg_x0 + i * (leg_w / n_seg)
        ax.add_patch(plt.Rectangle(
            (seg_x, leg_y), leg_w / n_seg + 0.01, leg_h,
            linewidth=0, facecolor=RYG(i / n_seg), zorder=2
        ))
    for frac, label in [(0.0, f"−{int(MAX_DELTA)}pp"), (0.5, "= Liga"), (1.0, f"+{int(MAX_DELTA)}pp")]:
        ax.text(leg_x0 + frac * leg_w, leg_y - 0.12,
                label, ha="center", va="top", fontsize=8, color="#888888")
    ax.text(leg_x0 - 0.1, leg_y + leg_h / 2,
            "vs liga", ha="right", va="center", fontsize=8, color="#888888")

    # Nota delta
    ax.text(leg_x0 + leg_w + 0.4, leg_y + leg_h / 2,
            "△ = pp de diferencia exacta",
            ha="left", va="center", fontsize=7.5, color="#555555", fontstyle="italic")

    # ── Fuente y firma ────────────────────────────────────────────
    ax.text(LEFT_MARGIN, BOTTOM_EXTRA * 0.16,
            "Fuente: Pro Football Reference & OverTheCap via nflreadpy",
            ha="left", va="center", fontsize=7.5, color="#555555", fontstyle="italic")
    ax.text(total_w - 0.1, BOTTOM_EXTRA * 0.16,
            "@CuartayDato",
            ha="right", va="center",
            fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

    out = "draft_heatmap_equipo.png"
    plt.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out}")


# ─────────────────────────────────────────────
# 5. Modo grid (32 equipos)
# ─────────────────────────────────────────────
def plot_grid_equipos(df):
    lg_rate = df.groupby(["pos_group", "round"])["success"].mean().mul(100).unstack()
    lg_rate = lg_rate.reindex(POS_ORDER).reindex(columns=range(1, 8))
    MAX_DELTA = 20.0

    all_teams = sorted(df["team_norm"].unique())

    NCOLS, NROWS = 4, 8
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(18, 38), dpi=120)
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.995, "Tasa de éxito en el draft — Los 32 equipos · 2011–2022",
             ha='center', va='top', fontsize=20, fontweight='bold', color=FG)
    fig.text(0.5, 0.984, "Color = diferencia vs media de la liga por posición y ronda  ·  Verde = mejor  ·  Rojo = peor",
             ha='center', va='top', fontsize=9.5, color="#888888", fontstyle='italic')

    for idx, team in enumerate(all_teams):
        row = idx // NCOLS
        col = idx % NCOLS
        ax  = axes[row][col]
        ax.set_facecolor(CARD)

        team_df = df[df["team_norm"] == team]
        rate = team_df.groupby(["pos_group", "round"])["success"].mean().mul(100).unstack()
        rate = rate.reindex(POS_ORDER).reindex(columns=range(1, 8))

        matrix = np.full((len(POS_ORDER), 7), np.nan)
        for r, pos in enumerate(POS_ORDER):
            for c, rnd in enumerate(range(1, 8)):
                val = rate.loc[pos, rnd]    if not pd.isna(rate.loc[pos, rnd])    else np.nan
                lg  = lg_rate.loc[pos, rnd] if not pd.isna(lg_rate.loc[pos, rnd]) else np.nan
                if not np.isnan(val) and not np.isnan(lg):
                    delta = val - lg
                    matrix[r, c] = np.clip(0.5 + delta / (2 * MAX_DELTA), 0.0, 1.0)

        masked = np.ma.masked_invalid(matrix)
        cmap = RYG.copy()
        cmap.set_bad(color="#1e2330")
        ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        ax.set_facecolor("#1e2330")

        ax.set_xticks(range(7))
        ax.set_xticklabels(ROUND_LABELS, fontsize=5, color="#666666")
        ax.set_yticks(range(len(POS_ORDER)))
        ax.set_yticklabels(POS_ORDER, fontsize=5.5, color=FG)
        ax.tick_params(axis='both', length=0, pad=1)
        ax.set_title(team, fontsize=9, fontweight='bold', color=FG, pad=4)

        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
            spine.set_linewidth(0.4)

    for idx in range(len(all_teams), NROWS * NCOLS):
        row = idx // NCOLS
        col = idx % NCOLS
        axes[row][col].set_visible(False)

    plt.subplots_adjust(top=0.978, bottom=0.015, hspace=0.65, wspace=0.45)

    fig.text(0.99, 0.007, "@CuartayDato",
             ha='right', va='bottom', fontsize=9, color="#888888", alpha=0.8, fontstyle='italic')
    fig.text(0.01, 0.007, "Fuente: PFR & OverTheCap via nflreadpy  ·  Drafts 2011–2022",
             ha='left', va='bottom', fontsize=7.5, color="#555555", fontstyle='italic')

    out = "draft_grid_heatmap.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {out}")


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    modo    = input("¿Ranking, Equipo o Grid? (ranking/equipo/grid): ").strip().lower()
    eq_raw  = input("Equipo (ej: SF) — vacío si ranking o grid: ").strip().upper()

    df = load_data()

    if modo.startswith("e") and eq_raw:
        plot_equipo(df, eq_raw)
    elif modo.startswith("g"):
        plot_grid_equipos(df)
    else:
        plot_ranking(df)
