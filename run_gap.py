"""
run_gap.py
- Sin equipo (Enter): heatmap 32 equipos
- Con equipo (ej: SF): fan chart por equipo (estilo raycarp, diseño antiguo)
- --equipo KC: fan chart con el estilo nuevo (% de éxito y puesto por hueco)
- --cruce KC SEA: el ataque de KC y, alineado debajo, lo que concede SEA en
  cada hueco (duelo del sábado). Con --week 6 se corta en esa semana.
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
from pbp_loader import cargar_pbp, salida, season_cli, sello, week_cli

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON   = season_cli()   # None = auto-detectar última temporada
BG       = "#0f1115"
FG       = "#EDEDED"
GRID     = "#2a2f3a"
DPI      = 170
LOGOS_DIR    = "logos"
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
        # Recorta margenes transparentes: algunos archivos traen mucho aire
        # (NYJ: tinta 3768x1186 en lienzo 4096x4096) y sin recorte salen enanos
        if img.ndim == 3 and img.shape[2] == 4:
            ys, xs = np.where(img[:, :, 3] > 0.02)
            if len(ys):
                img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = img.shape[:2]
        # Normaliza por el area de tinta real; los wordmarks apaisados
        # pueden ensancharse hasta 1.8x para compensar su poca altura
        zoom = base_zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * zoom > 900.0 * (base_zoom):
            zoom = 900.0 * (base_zoom) / w
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

    # Etiquetas de los huecos exteriores (las flechas LE/RE no tienen circulo OL)
    for _gap in ("LE", "RE"):
        _gx, _gy = GAP_XY[_gap]
        ax.text(_gx, _gy - 0.55, _gap, ha="center", va="top",
                color="#8a93a6", fontsize=9, fontweight="bold", zorder=6)

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
    fig.text(0.01, 0.01, f"Fuente: nflverse-data  |  {sello(SEASON)}",
             ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
    fig.text(0.99, 0.01, "@CuartayDato",
             ha="right", va="bottom", fontsize=9, color="#888888",
             alpha=0.85, fontstyle="italic")

    outfile = salida(f"run_gap_off_{team}_{SEASON}.png", SEASON)
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
                _r, _g, _b = bg_color[0], bg_color[1], bg_color[2]
                txt_color = "#0a0e13" if (0.299*_r + 0.587*_g + 0.114*_b) > 0.45 else FG
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
    fig.text(0.01, 0.005, f"Fuente: nflverse-data  |  {sello(SEASON)}",
             ha="left", va="bottom", fontsize=7.5, color="#555555", fontstyle="italic")
    fig.text(0.90, 0.005, "@CuartayDato",
             ha="right", va="bottom", fontsize=9, color="#888888",
             alpha=0.85, fontstyle="italic")

    outfile = salida(f"run_gap_{SEASON}.png", SEASON)
    fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {outfile}")


# ── FAN CHART CON ESTILO NUEVO: un equipo, o su cruce con una defensa ─────────
# (24-oct-2026, duelo KC @ SEA). % de exito (EPA > 0) en vez de EPA, para que
# cuadre con el resto de piezas; puesto entre los 32 en cada hueco; todos los
# huecos pintados con su muestra (los de menos de MIN_VISIBLE, apagados, nunca
# en blanco). Lado = el del ataque (run_location de nflverse), asi ataque y
# defensa se leen columna a columna.
MIN_VISIBLE = 10
GAP_TXT = {"LE": "por fuera, a la izquierda", "LT": "por el tackle izquierdo",
           "LG": "por el guard izquierdo", "C": "por el centro", "RG": "por el guard derecho",
           "RT": "por el tackle derecho", "RE": "por fuera, a la derecha"}
GAP_CORTO = {"LE": "Fuera izq.", "LT": "Tackle izq.", "LG": "Guard izq.", "C": "Centro",
             "RG": "Guard der.", "RT": "Tackle der.", "RE": "Fuera der."}


def tabla_huecos(plays, col, team, asc):
    g = plays.groupby([col, "gap"]).agg(n=("success", "size"), e=("success", "mean")).reset_index()
    g["rk"] = g.groupby("gap").e.rank(ascending=asc, method="min").astype(int)
    g = g[g[col] == team].set_index("gap")
    return {gap: (g.loc[gap] if gap in g.index else None) for gap in GAP_ORDER}


def fan_estilo(plays, atk, dfn=None, hasta=None):
    import estilo as E
    A = tabla_huecos(plays, "posteam", atk, asc=False)
    D = tabla_huecos(plays, "defteam", dfn, asc=True) if dfn else None
    na = E.nombre_equipo(atk)
    nd = E.nombre_equipo(dfn) if dfn else None
    ok = lambda d: d is not None and d.n >= MIN_VISIBLE   # noqa: E731
    # el titular y el contexto solo se apoyan en huecos con MIN_CARRIES o mas
    fiable = lambda d: d is not None and d.n >= MIN_CARRIES   # noqa: E731
    cap = lambda t: t[:1].upper() + t[1:]                      # noqa: E731
    best_a = min((g for g in GAP_ORDER if fiable(A[g])), key=lambda g: A[g].rk)
    print(f"\n{atk} por hueco:")
    for gap in GAP_ORDER:
        a = A[gap]
        d = D[gap] if D else None
        print(f"  {gap:>2}  {atk} {a.e:.0%} ({a.rk}º, {a.n})" if a is not None else f"  {gap:>2}  -",
              f"| {dfn} {d.e:.0%} ({d.rk}º, {d.n})" if d is not None else "")

    fig = E.lienzo()
    E.antetitulo(fig, f"Carrera por hueco · {atk} @ {dfn}" if dfn else f"Carrera por hueco · {atk}")
    # Titular automatico: solo afirma lo que se calcula, y solo si es top 5
    tit, ctx = None, [(f"{cap(na)} tiene éxito en el", None), (f"{A[best_a].e * 100:.0f}%", "fuerte"),
                      (f"de sus carreras {GAP_TXT[best_a]} ({A[best_a].rk}º de la NFL).", None)]
    if D:
        best_d = min((g for g in GAP_ORDER if fiable(D[g])), key=lambda g: D[g].rk)
        ctx += [(f"{cap(nd)} solo concede el", None), (f"{D[best_d].e * 100:.0f}%", "fuerte"),
                (f"{GAP_TXT[best_d]} ({D[best_d].rk}ª).", None)]
        if A[best_a].rk <= 5 and D[best_d].rk <= 5:
            tit = (f"{cap(na)} corre {GAP_TXT[best_a]}. "
                   f"{cap(nd)} cierra {GAP_TXT[best_d].replace('por ', '', 1)}.")
            if best_a == best_d:
                tit = f"Lo mejor de {na} contra lo mejor de {nd}: {GAP_TXT[best_a]}."
    elif A[best_a].rk <= 5:
        tit = f"{cap(na)} hace daño {GAP_TXT[best_a]}."
    y = E.titular(fig, tit, y=0.925) - 0.012 if tit else 0.90
    y = E.contexto(fig, ctx, y=y)
    y_top = y - 0.025

    # ── Diagrama: eje con aspecto 1:1 dentro del lienzo
    X0, X1 = -4.7, 4.7
    ax_h = 0.34
    w_in = (E.R - E.L) * E.ANCHO
    h_in = ax_h * E.ALTO
    yr = (X1 - X0) * h_in / w_in
    ax_y0 = y_top - 0.035 - ax_h
    ax = fig.add_axes([E.L, ax_y0, E.R - E.L, ax_h])
    ax.axis("off")
    ax.set_xlim(X0, X1)
    ax.set_ylim(4.75 - yr, 4.75)
    ax.patch.set_alpha(0)
    rx, ry = 0.0, 0.75
    oy = 2.5
    gxy = {g: (x, y_ + 0.0) for g, (x, y_) in GAP_XY.items()}
    gris = "#3a4050"
    for gap in GAP_ORDER:
        a = A[gap]
        px, py = gxy[gap]
        col = E.color_rango(a.rk) if a is not None else gris
        col = gris if col == E.GRIS1 else col
        alpha = 1.0 if ok(a) else 0.35
        ctrl_x = rx + (px - rx) * CTRL_FACTOR
        tx, ty = px - ctrl_x, py - CTRL_Y
        tl = np.hypot(tx, ty)
        ux, uy = tx / tl, ty / tl
        bx, by = px - ux * ARROW_LEN, py - uy * ARROW_LEN
        lw = 1.2 + 4.0 * min((a.n if a is not None else 0) / 50, 1)
        ax.add_patch(mpatches.PathPatch(MPath([(rx, ry), (ctrl_x, CTRL_Y), (bx, by)],
                                              [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]),
                                        fc="none", ec=col, lw=lw, alpha=alpha, zorder=2,
                                        capstyle="round"))
        perp = np.array([-uy, ux])
        tip = np.array([px, py])
        base = tip - np.array([ux, uy]) * ARROW_LEN
        ax.add_patch(plt.Polygon([tip, base + perp * ARROW_W, base - perp * ARROW_W], color=col,
                                 alpha=alpha, zorder=7))
        if a is not None:
            ax.text(px, py + 0.62, f"{a.e * 100:.0f}%", ha="center", va="bottom", family=E.DISP,
                    weight=800, size=17, color=E.FG, alpha=alpha, zorder=8)
            ax.text(px, py + 0.38, f"{a.rk}º", ha="center", va="bottom", family=E.TXT, weight="bold",
                    size=7.5, color=E.color_rango(a.rk), alpha=alpha, zorder=8)
            ax.text(px, py + 0.17, f"{a.n} carr.", ha="center", va="bottom", family=E.TXT, size=6,
                    color=E.GRIS2, zorder=8)
    for pos, (cx, cy) in OL_POS.items():
        ax.add_patch(plt.Circle((cx, cy), OL_R, fc="#1c2535", ec=E.GRID, lw=1, zorder=5))
        ax.text(cx, cy, pos, ha="center", va="center", family=E.TXT, weight="bold", size=7.5,
                color=E.FG, zorder=6)
    ax.add_patch(plt.Circle((rx, ry), RB_R, color=E.color_equipo(atk), zorder=8))
    ax.text(rx, ry, "RB", ha="center", va="center", family=E.TXT, weight="bold", size=9,
            color="white", zorder=9)
    E.poner_logo(fig, atk, E.L + 0.06, ax_y0 + 0.05, zoom=0.03)
    E.etiqueta(fig, f"{na} · éxito por carrera", E.L + 0.01, y_top - 0.02)

    # ── Franja de la defensa, alineada con las puntas de flecha
    def fx(x):
        return E.L + (x - X0) / (X1 - X0) * (E.R - E.L)

    yb = ax_y0 - 0.01
    if D:
        E.linea_h(fig, yb)
        E.poner_logo(fig, dfn, E.L + 0.025, yb - 0.03, zoom=0.016)
        E.etiqueta(fig, f"{nd} · éxito concedido en el mismo hueco", E.L + 0.055, yb - 0.03)
        for gap in GAP_ORDER:
            d = D[gap]
            x = fx(gxy[gap][0])
            fig.text(x, yb - 0.065, GAP_CORTO[gap].upper(), ha="center", va="center", family=E.TXT,
                     weight="semibold", size=5.8, color=E.GRIS1)
            if d is None:
                continue
            al = 1.0 if ok(d) else 0.35
            fig.text(x, yb - 0.098, f"{d.e * 100:.0f}%", ha="center", va="center", family=E.DISP,
                     weight=800, size=17, color=E.FG, alpha=al)
            fig.text(x, yb - 0.128, f"{d.rk}ª", ha="center", va="center", family=E.TXT, weight="bold",
                     size=7.5, color=E.color_rango(d.rk), alpha=al)
            fig.text(x, yb - 0.147, f"{d.n} carr.", ha="center", va="center", family=E.TXT, size=6,
                     color=E.GRIS2)
        yb -= 0.17
    else:
        for gap in GAP_ORDER:
            fig.text(fx(gxy[gap][0]), yb - 0.01, GAP_CORTO[gap].upper(), ha="center", va="center",
                     family=E.TXT, weight="semibold", size=5.8, color=E.GRIS1)
        yb -= 0.04
    E.tarjeta(fig, yb, y_top)
    sello_txt = f"NFL {SEASON} · datos hasta sem. {hasta}" if hasta else sello(SEASON)
    E.pie(fig, f"Carreras diseñadas, sin scrambles · Lado = el del ataque · Éxito = EPA > 0 · "
               f"Apagados: menos de {MIN_VISIBLE} carreras · Grosor = uso",
          f"{sello_txt} · nflverse PBP")
    nombre = f"run_gap_cruce_{atk}_{dfn}_{SEASON}.png" if dfn else f"run_gap_off_{atk}_{SEASON}.png"
    E.guardar(fig, salida(nombre, SEASON))


# ── INPUT ──────────────────────────────────────────────────────────────────────
import argparse                                    # noqa: E402
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--cruce", nargs=2, metavar=("ATAQUE", "DEFENSA"))
_ap.add_argument("--equipo")
_args, _ = _ap.parse_known_args()
HASTA = week_cli()
if _args.cruce or _args.equipo:
    team_input = ""
else:
    team_input = input("Equipo (siglas, ej: SF — Enter = todos los equipos): ").strip().upper()

# ── CARGA DE DATOS ─────────────────────────────────────────────────────────────
pbp, SEASON = cargar_pbp(SEASON, columns=["play_type", "posteam", "defteam", "epa", "week",
                                          "run_location", "run_gap", "qb_scramble",
                                          "yards_gained", "rush_touchdown", "fumble"])
if HASTA:
    pbp = pbp[pbp["week"] <= HASTA]
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

plays["success"] = (plays["epa"] > 0).astype(float)

# ── MODO CRUCE / EQUIPO CON ESTILO NUEVO ─────────────────────────────────────
if _args.cruce:
    fan_estilo(plays, _args.cruce[0].upper(), _args.cruce[1].upper(), HASTA)
elif _args.equipo:
    fan_estilo(plays, _args.equipo.upper(), None, HASTA)

# ── MODO SINGLE TEAM → FAN CHART ──────────────────────────────────────────────
elif team_input:
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
