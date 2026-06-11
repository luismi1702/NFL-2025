"""
run_gap.py
- Sin equipo (Enter): heatmap 32 equipos
- Con equipo (ej: SF): fan chart por equipo (estilo raycarp)
NFL 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pbp_loader import cargar_pbp

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON   = None   # None = auto-detectar última temporada
BG       = "#0f1115"
FG       = "#EDEDED"
GRID     = "#2a2f3a"
DPI      = 170
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}
MIN_CARRIES  = 15

GAP_ORDER  = ["LE", "LT", "LG", "C", "RG", "RT", "RE"]
GAP_LABELS = {
    "LE": "LE\nLeft End",
    "LT": "LT\nLeft Tackle",
    "LG": "LG\nLeft Guard",
    "C":  "C\nCenter",
    "RG": "RG\nRight Guard",
    "RT": "RT\nRight Tackle",
    "RE": "RE\nRight End",
}

# Círculos de referencia OL (no tienen datos, son puntos de formación)
OL_POS = {
    "LT": (-2.8, 2.5),
    "LG": (-1.5, 2.5),
    "C":  ( 0.0, 2.5),
    "RG": ( 1.5, 2.5),
    "RT": ( 2.8, 2.5),
}
OL_R = 0.37

# Endpoints de los huecos: las líneas del RB terminan ENTRE los OL
# Posiciones calculadas para que cada línea pase entre los círculos OL
GAP_XY = {
    "LE": (-4.0,  2.5),   # fuera del LT — sin etiqueta
    "LT": (-2.15, 3.0),   # entre LT y LG
    "LG": (-0.94, 3.0),   # entre LG y C
    "C":  ( 0.0,  3.3),   # por el centro (pasa a través del círculo C)
    "RG": ( 0.94, 3.0),   # entre C y RG
    "RT": ( 2.15, 3.0),   # entre RG y RT
    "RE": ( 4.0,  2.5),   # fuera del RT — sin etiqueta
}
RB_XY        = (0.0, 0.55)
RB_R         = 0.42
ARROW_LEN    = 0.30    # longitud del triángulo de flecha
ARROW_W      = 0.15    # semi-ancho de la base del triángulo
CTRL_FACTOR  = 0.85    # control point cerca del x del endpoint → curvas hacia afuera
CTRL_Y       = 0.95    # control point cerca del nivel del RB → curvas suben después

COL_POS = "#06d6a0"   # verde — EPA positivo (buena carrera)
COL_NEG = "#d84a4a"   # rojo  — EPA negativo (mala carrera)
COL_RB  = "#E5C070"   # dorado — RB


# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_logo(team, base_zoom=0.038):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        h, w = img.shape[:2]
        aspect = w / float(h) if h else 1.0
        if team in HARD_PENALTY:
            zoom = base_zoom / HARD_PENALTY[team]
        else:
            div = np.clip(1.0 + 0.6 * max(0.0, aspect - 1.3), 1.0, 2.2)
            zoom = base_zoom / div
        return OffsetImage(img, zoom=zoom, resample=True)
    except Exception:
        return None


def classify_gap(loc, gap):
    if pd.isna(loc):
        return None
    if loc == "middle":
        return "C"
    if pd.isna(gap):
        return None
    if loc == "left":
        if gap == "end":    return "LE"
        if gap == "tackle": return "LT"
        if gap == "guard":  return "LG"
    if loc == "right":
        if gap == "guard":  return "RG"
        if gap == "tackle": return "RT"
        if gap == "end":    return "RE"
    return None


def get_ol_starters(team):
    """Devuelve {pos: jersey_number} para los titulares de la OL del equipo."""
    try:
        url = (f"https://github.com/nflverse/nflverse-data/releases/download/"
               f"depth_charts/depth_charts_{SEASON}.csv")
        dc = pd.read_csv(url, low_memory=False)
        ol_pos = {"LT", "LG", "C", "RG", "RT"}
        team_col = next((c for c in dc.columns if "club" in c.lower() or c == "team"), None)
        pos_col  = next((c for c in dc.columns if c in ("position", "depth_chart_position")), None)
        if team_col is None or pos_col is None:
            return {}
        sub = dc[(dc[team_col] == team) & (dc[pos_col].isin(ol_pos))]
        if "depth_team" in sub.columns:
            sub = sub[sub["depth_team"] == 1]
        if "week" in sub.columns:
            sub = sub[sub["week"] == sub["week"].max()]
        result = {}
        for _, row in sub.drop_duplicates(subset=[pos_col]).iterrows():
            j = row.get("jersey_number", np.nan)
            if pd.notna(j):
                result[row[pos_col]] = int(j)
        return result
    except Exception:
        return {}


# ── FAN CHART ──────────────────────────────────────────────────────────────────
def draw_fan_chart(plays_team, team):
    carries = len(plays_team)
    if carries == 0:
        print(f"No hay datos de carrera para {team}")
        return

    yards = plays_team["yards_gained"].sum()
    ypc   = yards / carries if carries > 0 else 0.0
    tds   = int(plays_team["rush_touchdown"].fillna(0).sum()) \
            if "rush_touchdown" in plays_team.columns else 0
    bigs  = int((plays_team["yards_gained"] >= 10).sum())
    fums  = int(plays_team["fumble"].fillna(0).sum()) \
            if "fumble" in plays_team.columns else 0

    gap_data = {}
    for gap in GAP_ORDER:
        sub = plays_team[plays_team["gap"] == gap]["epa"]
        n   = len(sub)
        gap_data[gap] = {"epa": sub.mean() if n >= MIN_CARRIES else np.nan, "n": n}

    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(-4.8, 4.8)
    ax.set_ylim(-0.6, 5.0)

    rx, ry = RB_XY

    ol_numbers = get_ol_starters(team)

    # ── Curvas Bézier RB → base de flecha ────────────────────────────────────
    for gap in GAP_ORDER:
        px, py = GAP_XY[gap]
        epa = gap_data[gap]["epa"]
        col = COL_POS if (not np.isnan(epa) and epa >= 0) else COL_NEG
        if np.isnan(epa):
            col = GRID
        # Punto de control: se queda cerca del eje central, a altura CTRL_Y
        # → todas las curvas salen casi verticales del RB y se abren en abanico
        ctrl_x = rx + (px - rx) * CTRL_FACTOR
        ctrl_y = CTRL_Y
        # Tangente en el endpoint = dirección P2-P1 del Bézier cuadrático
        tan_x, tan_y = px - ctrl_x, py - ctrl_y
        tan_len = np.hypot(tan_x, tan_y)
        ux, uy  = tan_x / tan_len, tan_y / tan_len
        # La curva termina en la base del triángulo de flecha
        bx, by  = px - ux * ARROW_LEN, py - uy * ARROW_LEN
        lw      = 3.5 if gap not in ("LE", "RE") else 2.2
        verts   = [(rx, ry), (ctrl_x, ctrl_y), (bx, by)]
        codes   = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]
        patch   = mpatches.PathPatch(
            MPath(verts, codes),
            facecolor="none", edgecolor=col,
            linewidth=lw, zorder=2, capstyle="round"
        )
        ax.add_patch(patch)

    # ── Círculos OL de referencia (encima de las líneas) ──────────────────────
    for pos, (cx, cy) in OL_POS.items():
        ol_circle = plt.Circle((cx, cy), OL_R,
                               facecolor="#1c2535", edgecolor=GRID,
                               linewidth=1.2, zorder=5)
        ax.add_patch(ol_circle)
        jersey = ol_numbers.get(pos)
        if jersey:
            ax.text(cx, cy + 0.10, pos, ha="center", va="center",
                    color=FG, fontsize=9, fontweight="bold", zorder=6)
            ax.text(cx, cy - 0.11, f"#{jersey}", ha="center", va="center",
                    color="#aaaaaa", fontsize=7, zorder=6)
        else:
            ax.text(cx, cy, pos, ha="center", va="center",
                    color=FG, fontsize=10, fontweight="bold", zorder=6)

    # ── Flechas (triángulos) + valores EPA ────────────────────────────────────
    for gap in GAP_ORDER:
        px, py = GAP_XY[gap]
        epa    = gap_data[gap]["epa"]
        n      = gap_data[gap]["n"]
        col    = COL_POS if (not np.isnan(epa) and epa >= 0) else COL_NEG
        if np.isnan(epa):
            col = "#3a4050"

        # Recalcular tangente (igual que en la sección de curvas)
        ctrl_x  = rx + (px - rx) * CTRL_FACTOR
        tan_x, tan_y = px - ctrl_x, py - CTRL_Y
        tan_len = np.hypot(tan_x, tan_y)
        ux, uy  = tan_x / tan_len, tan_y / tan_len

        perp = np.array([-uy, ux])
        tip  = np.array([px, py])
        base = tip - np.array([ux, uy]) * ARROW_LEN
        v1   = base + perp * ARROW_W
        v2   = base - perp * ARROW_W
        ax.add_patch(plt.Polygon([tip, v1, v2], color=col, zorder=7))

        if not np.isnan(epa):
            sign = "+" if epa >= 0 else ""
            ax.text(px, py + 0.20, f"{sign}{epa:.2f}",
                    ha="center", va="bottom",
                    color=col, fontsize=9.5, fontweight="bold", zorder=8)
        else:
            ax.text(px, py + 0.16, "N/D",
                    ha="center", va="bottom",
                    color="#555555", fontsize=8, zorder=8)

        if n > 0:
            ax.text(px, base[1] - 0.10, f"n={n}",
                    ha="center", va="top",
                    color="#666666", fontsize=6.5, zorder=8)

    # ── Círculo RB ────────────────────────────────────────────────────────────
    rb_circle = plt.Circle(RB_XY, RB_R, color=COL_RB, zorder=8)
    ax.add_patch(rb_circle)
    ax.text(rx, ry, "RB", ha="center", va="center",
            color=BG, fontsize=12, fontweight="bold", zorder=9)

    # ── Logo del equipo (izquierda del RB) ────────────────────────────────────
    logo = load_logo(team, base_zoom=0.075)
    if logo:
        ab = AnnotationBbox(logo, (rx - 1.6, ry),
                            frameon=False, zorder=3,
                            box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        ax.text(rx - 1.6, ry, team, ha="center", va="center",
                color=FG, fontsize=13, fontweight="bold")

    # Barra de stats (parte superior)
    STATS = [
        (f"{carries}",    "CAR"),
        (f"{int(yards)}", "YDS"),
        (f"{ypc:.1f}",    "YPC"),
        (f"{tds}",        "TD"),
        (f"{bigs}",       "BIG"),
        (f"{fums}",       "FUM"),
    ]
    xs = np.linspace(-3.3, 3.3, len(STATS))
    sy = 4.55
    for i, (val, lbl) in enumerate(STATS):
        ax.text(xs[i], sy + 0.18, val,
                ha="center", va="bottom",
                color=FG, fontsize=11, fontweight="bold", zorder=5)
        ax.text(xs[i], sy + 0.02, lbl,
                ha="center", va="top",
                color="#888888", fontsize=8, zorder=5)
        if i < len(STATS) - 1:
            xsep = (xs[i] + xs[i + 1]) / 2
            ax.plot([xsep, xsep], [sy - 0.12, sy + 0.45],
                    color=GRID, linewidth=0.8, zorder=3)

    # Títulos
    fig.text(0.5, 0.97,
             f"Run Gap Ofensivo — {team} | NFL {SEASON}",
             ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
    fig.text(0.5, 0.945,
             "EPA/acarreo por hueco  |  Verde = EPA positivo (buena carrera)  |  n = acarreos por hueco  |  Excluye scrambles QB",
             ha="center", va="top", fontsize=8, color="#888888", fontstyle="italic")
    fig.text(0.01, 0.01, f"Fuente: nflverse-data  |  NFL {SEASON}",
             ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
    fig.text(0.99, 0.01, "@CuartayDato",
             ha="right", va="bottom", fontsize=9, color="#888888",
             alpha=0.85, fontstyle="italic")

    outfile = f"run_gap_off_{team}_{SEASON}.png"
    fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {outfile}")


# ── HEATMAP 32 EQUIPOS ────────────────────────────────────────────────────────
def draw_heatmap(epa_piv, n_piv):
    teams   = epa_piv.index.tolist()
    n_teams = len(teams)
    n_cols  = len(GAP_ORDER)
    cell_w  = 1.4
    cell_h  = 0.52
    logo_w  = 1.2
    fig_w   = logo_w + n_cols * cell_w + 1.2
    fig_h   = max(8, n_teams * cell_h + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(-logo_w, n_cols)
    ax.set_ylim(-1, n_teams + 0.8)

    valid = epa_piv.values[~np.isnan(epa_piv.values)]
    v_abs = max(abs(valid.min()), abs(valid.max()), 0.05) if len(valid) else 0.3
    norm  = Normalize(vmin=-v_abs, vmax=v_abs)
    cmap  = plt.cm.RdYlGn

    for row_i, team in enumerate(teams):
        y = n_teams - row_i - 1
        for col_j, gap in enumerate(GAP_ORDER):
            val = epa_piv.loc[team, gap]
            n   = n_piv.loc[team, gap]
            x   = col_j
            bg_color = cmap(norm(val)) if not np.isnan(val) else "#1e2430"
            rect = plt.Rectangle((x, y), 1, 1, color=bg_color,
                                  linewidth=0.4, edgecolor=BG, zorder=1)
            ax.add_patch(rect)
            if not np.isnan(val):
                sign      = "+" if val >= 0 else ""
                txt_color = "#0a0e13" if 0.3 < norm(val) < 0.7 else FG
                ax.text(x + 0.5, y + 0.60, f"{sign}{val:.3f}",
                        ha="center", va="center",
                        color=txt_color, fontsize=8, fontweight="bold", zorder=2)
                ax.text(x + 0.5, y + 0.25, f"n={n}",
                        ha="center", va="center",
                        color=txt_color, fontsize=6.5, zorder=2)
            else:
                ax.text(x + 0.5, y + 0.5, "—",
                        ha="center", va="center",
                        color="#444444", fontsize=10, zorder=2)

    for row_i, team in enumerate(teams):
        y   = n_teams - row_i - 1
        img = load_logo(team, base_zoom=0.036)
        if img is not None:
            ab = AnnotationBbox(img, (-logo_w / 2, y + 0.5),
                                frameon=False, zorder=3,
                                box_alignment=(0.5, 0.5))
            ax.add_artist(ab)
        else:
            ax.text(-logo_w / 2, y + 0.5, team,
                    ha="center", va="center",
                    color=FG, fontsize=7.5, fontweight="bold")

    for col_j, gap in enumerate(GAP_ORDER):
        ax.text(col_j + 0.5, n_teams + 0.35,
                GAP_LABELS[gap],
                ha="center", va="center",
                color=FG, fontsize=8, fontweight="bold", linespacing=1.3)
    ax.axhline(n_teams, color=GRID, linewidth=0.8, zorder=3)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.70])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("EPA por acarreo\n(+ = mejor carrera)", color=FG, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=FG, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)
    cb.outline.set_edgecolor(GRID)

    fig.text(0.5, 0.99,
             f"Eficiencia en carrera por hueco | NFL {SEASON}",
             ha="center", va="top", fontsize=14, fontweight="bold", color=FG)
    fig.text(0.5, 0.975,
             "EPA/acarreo por direccion de hueco  |  Ordenado por EPA global (mejor arriba)  |  Excluye scrambles QB  |  — = menos de 15 acarreos",
             ha="center", va="top", fontsize=8.5, color="#888888", fontstyle="italic")
    fig.text(0.01, 0.005, f"Fuente: nflverse-data  |  NFL {SEASON}",
             ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
    fig.text(0.90, 0.005, "@CuartayDato",
             ha="right", va="bottom", fontsize=9, color="#888888",
             alpha=0.85, fontstyle="italic")

    outfile = f"run_gap_{SEASON}.png"
    fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {outfile}")


# ── INPUT ──────────────────────────────────────────────────────────────────────
team_input = input("Equipo (siglas, ej: SF — Enter = todos los equipos): ").strip().upper()

# ── CARGA DE DATOS ─────────────────────────────────────────────────────────────
pbp, SEASON = cargar_pbp(SEASON, columns=["play_type", "posteam", "epa",
                                          "run_location", "run_gap", "qb_scramble",
                                          "yards_gained", "rush_touchdown", "fumble"])
for col in ["epa", "qb_scramble", "yards_gained", "rush_touchdown", "fumble"]:
    pbp[col] = pd.to_numeric(pbp[col], errors="coerce")
pbp["qb_scramble"] = pbp["qb_scramble"].fillna(0)
print(f"PBP filas: {len(pbp):,}")

plays = pbp[
    (pbp["play_type"] == "run") &
    (pbp["qb_scramble"] == 0) &
    pbp["posteam"].notna() &
    pbp["epa"].notna() &
    pbp["run_location"].notna()
].copy()
print(f"Acarreos diseñados: {len(plays):,}")

plays["gap"] = plays.apply(
    lambda r: classify_gap(r["run_location"], r["run_gap"]), axis=1
)
plays = plays[plays["gap"].isin(GAP_ORDER)].copy()
print(f"Con hueco clasificado: {len(plays):,}")

# ── MODO SINGLE TEAM → FAN CHART ──────────────────────────────────────────────
if team_input:
    all_teams = plays["posteam"].dropna().unique()
    if team_input not in all_teams:
        print(f"Equipo '{team_input}' no encontrado. Disponibles: {', '.join(sorted(all_teams))}")
    else:
        draw_fan_chart(plays[plays["posteam"] == team_input].copy(), team_input)

# ── MODO TODOS → HEATMAP ──────────────────────────────────────────────────────
else:
    grp     = plays.groupby(["posteam", "gap"])["epa"].agg(
                  epa_mean="mean", n_carries="count").reset_index()
    epa_piv = grp.pivot(index="posteam", columns="gap", values="epa_mean")
    n_piv   = grp.pivot(index="posteam", columns="gap", values="n_carries").fillna(0).astype(int)

    for col in GAP_ORDER:
        if col not in epa_piv.columns: epa_piv[col] = np.nan
        if col not in n_piv.columns:   n_piv[col]   = 0
    epa_piv = epa_piv[GAP_ORDER]
    n_piv   = n_piv[GAP_ORDER]
    epa_piv = epa_piv.where(n_piv >= MIN_CARRIES)

    overall_epa      = plays.groupby("posteam")["epa"].mean()
    epa_piv["_sort"] = overall_epa
    epa_piv          = epa_piv.sort_values("_sort", ascending=False).drop(columns="_sort")
    n_piv            = n_piv.loc[epa_piv.index]

    # Consola
    print(f"\n{'='*70}")
    print(f"  EPA carrera por hueco | NFL {SEASON}  (+ = mejor ataque de carrera)")
    print(f"{'='*70}")
    header = f"{'Off':<5}" + "".join(f"  {g:>8}" for g in GAP_ORDER)
    print(header); print("-" * len(header))
    for tm in epa_piv.index:
        row  = epa_piv.loc[tm]
        vals = "".join(
            f"  {row[g]:+7.3f}" if not np.isnan(row[g]) else f"  {'N/D':>7}"
            for g in GAP_ORDER
        )
        print(f"{tm:<5}{vals}")

    draw_heatmap(epa_piv, n_piv)
