# draft_r1_radar.py
# Radar de barras polares: inversión en primera ronda por posición
# Modo equipo: radar de un equipo · Modo grid: 32 equipos en cuadrícula 8×4
# Datos: picks de R1 · 2011-2022

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import nflreadpy

# === Config ===
SEASONS   = list(range(2011, 2026))
LOGOS_DIR = "logos"
BG        = "#0f1115"
CARD      = "#151924"
FG        = "#EDEDED"
ACCENT    = "#2d6cdf"
GRID_C    = "#2a2f3a"
DPI_TEAM  = 180
DPI_GRID  = 120

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
    print("Cargando picks 2011-2022 (todas las rondas)...")
    draft = nflreadpy.load_draft_picks(seasons=SEASONS).to_pandas()
    draft["pos_group"] = draft["position"].map(POS_MAP)
    draft["team_norm"] = draft["team"].apply(normalize_team)
    draft = draft[
        draft["pos_group"].notna() &
        draft["gsis_id"].notna() &
        draft["team_norm"].notna()
    ].copy()
    print(f"Total picks cargados: {len(draft)}")
    return draft


def league_avg(df):
    """Picks medios por equipo por posición (promedio de los 32 equipos)."""
    by_team = df.groupby(["team_norm", "pos_group"]).size().unstack(fill_value=0)
    return by_team.mean().to_dict()


# ─────────────────────────────────────────────
# 2. Helpers
# ─────────────────────────────────────────────
BOTTOM_RATIO = 0.22   # fracción del max_val usada como radio interior (donut)


def add_logo(fig, ax, team, zoom=0.18):
    """Añade el logo del equipo centrado en el polar plot."""
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return
    try:
        img = plt.imread(path)
        h, w = img.shape[:2]
        aspect = w / float(h) if h else 1.0
        if team == "NYJ":
            z = zoom / 4.5
        else:
            divisor = np.clip(1.0 + 0.6 * max(0.0, aspect - 1.3), 1.0, 2.2)
            z = zoom / divisor
        ab = AnnotationBbox(
            OffsetImage(img, zoom=z, resample=True),
            (0.5, 0.5),
            frameon=False, xycoords='axes fraction',
            box_alignment=(0.5, 0.5), pad=0
        )
        ax.add_artist(ab)
    except Exception:
        pass


def draw_radar(ax, counts, max_val, avg_counts=None, show_labels=True, label_fs=11, count_fs=10):
    """Dibuja el radar polar de barras en el axes dado (efecto donut).
    avg_counts: dict pos→float con la media de la liga (se dibuja como polígono de referencia).
    """
    n      = len(POS_ORDER)
    theta  = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width  = (2 * np.pi / n) * 0.72
    BOTTOM = max_val * BOTTOM_RATIO

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Círculos guía
    guide_ticks = [BOTTOM + v for v in np.linspace(0, max_val, 5)[1:]]
    ax.set_yticks(guide_ticks)
    ax.yaxis.set_ticklabels([])
    ax.yaxis.grid(True, color=GRID_C, linewidth=0.5, linestyle='--', alpha=0.5, zorder=1)
    ax.xaxis.grid(False)

    # Barras con bottom → efecto donut
    ax.bar(
        theta,
        [counts.get(p, 0) for p in POS_ORDER],
        width=width, bottom=BOTTOM,
        color=[POS_COLORS[p] for p in POS_ORDER],
        alpha=0.85, align='center', zorder=3
    )

    # ── Media de la liga: línea horizontal dentro de cada barra ──
    if avg_counts is not None:
        for i, pos in enumerate(POS_ORDER):
            avg_val = avg_counts.get(pos, 0)
            r_line  = BOTTOM + avg_val
            # Arco que recorre el ancho de la barra
            t_start = theta[i] - width / 2
            t_end   = theta[i] + width / 2
            t_arc   = np.linspace(t_start, t_end, 30)
            ax.plot(t_arc, [r_line] * 30,
                    color='white', linewidth=2.0, alpha=0.75, zorder=5)

    ax.set_xticks([])
    ax.set_ylim(0, (BOTTOM + max_val) * 1.30)

    if show_labels:
        for i, pos in enumerate(POS_ORDER):
            cnt = counts.get(pos, 0)
            r_lbl = (BOTTOM + max_val) * 1.18
            ax.text(theta[i], r_lbl, pos,
                    ha='center', va='center',
                    fontsize=label_fs, fontweight='bold',
                    color=POS_COLORS[pos], zorder=5)
            if cnt > 0:
                ax.text(theta[i], BOTTOM + cnt + max_val * 0.07, str(int(cnt)),
                        ha='center', va='bottom',
                        fontsize=count_fs, fontweight='bold',
                        color=FG, zorder=5)
    else:
        for i, pos in enumerate(POS_ORDER):
            cnt = counts.get(pos, 0)
            if cnt > 0:
                ax.text(theta[i], BOTTOM + cnt + max_val * 0.10, str(int(cnt)),
                        ha='center', va='bottom',
                        fontsize=count_fs, color=FG, zorder=5)


# ─────────────────────────────────────────────
# 3. Modo equipo
# ─────────────────────────────────────────────
def plot_team(df, team):
    team = team.upper()
    team_df = df[df["team_norm"] == team]
    if team_df.empty:
        print(f"No se encontraron picks para {team}. Verifica la sigla.")
        return

    counts   = team_df.groupby("pos_group").size().to_dict()
    avg      = league_avg(df)
    max_val  = max(max(counts.values(), default=1), max(avg.values(), default=1), 3)

    fig = plt.figure(figsize=(9, 9), dpi=DPI_TEAM)
    fig.patch.set_facecolor(BG)

    ax = fig.add_subplot(111, projection='polar')
    ax.set_facecolor(BG)
    ax.spines['polar'].set_color(GRID_C)
    ax.spines['polar'].set_linewidth(0.5)

    draw_radar(ax, counts, max_val, avg_counts=avg, show_labels=True, label_fs=12, count_fs=11)
    add_logo(fig, ax, team, zoom=0.18)

    # Leyenda de referencia
    fig.text(0.98, 0.08, "—  Media liga",
             ha='right', va='bottom', fontsize=8, color='white', alpha=0.6, fontstyle='italic')

    total = sum(counts.values())
    fig.text(0.5, 0.96, f"{team} — Capital de draft por posición · 2011–2025",
             ha='center', va='top', fontsize=14, fontweight='bold', color=FG)
    fig.text(0.5, 0.925, f"{total} picks totales · tamaño = nº de elecciones · línea = media de la liga",
             ha='center', va='top', fontsize=9, color="#888888", fontstyle='italic')
    fig.text(0.98, 0.02, "@CuartayDato",
             ha='right', va='bottom', fontsize=9, color="#888888", alpha=0.8, fontstyle='italic')
    fig.text(0.02, 0.02, "Fuente: PFR via nflreadpy  ·  Drafts 2011–2025",
             ha='left', va='bottom', fontsize=7.5, color="#555555", fontstyle='italic')

    out = "draft_radar_equipo.png"
    plt.savefig(out, dpi=DPI_TEAM, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {out}")


# ─────────────────────────────────────────────
# 4. Modo grid (32 equipos)
# ─────────────────────────────────────────────
def plot_grid(df):
    counts_all   = df.groupby(["team_norm","pos_group"]).size().unstack(fill_value=0)
    teams_sorted = sorted(counts_all.index.tolist())
    avg          = league_avg(df)
    max_val      = max(int(counts_all.max().max()), max(avg.values(), default=1), 3)

    NCOLS, NROWS = 4, 8
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(18, 38), dpi=DPI_GRID,
                             subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.994, "Capital de draft por posición · 32 equipos · 2011–2022",
             ha='center', va='top', fontsize=22, fontweight='bold', color=FG)
    fig.text(0.5, 0.983, "Picks totales por posición · línea blanca = media de la liga · ordenados alfabéticamente  ·  2011–2025",
             ha='center', va='top', fontsize=10, color="#888888", fontstyle='italic')

    for idx, team in enumerate(teams_sorted):
        row = idx // NCOLS
        col = idx % NCOLS
        ax  = axes[row][col]
        ax.set_facecolor(CARD)
        ax.spines['polar'].set_color(GRID_C)
        ax.spines['polar'].set_linewidth(0.4)

        counts = counts_all.loc[team].to_dict() if team in counts_all.index else {}
        draw_radar(ax, counts, max_val, avg_counts=avg, show_labels=False, count_fs=6)
        ax.set_title(team, fontsize=10, fontweight='bold', color=FG, pad=5)

    # Ocultar celdas vacías (si hay <32 equipos)
    for idx in range(len(teams_sorted), NROWS * NCOLS):
        row = idx // NCOLS
        col = idx % NCOLS
        axes[row][col].set_visible(False)

    # Leyenda de posiciones
    patches = [mpatches.Patch(color=POS_COLORS[p], label=p) for p in POS_ORDER]
    fig.legend(handles=patches, loc='lower center', ncol=8,
               frameon=True, facecolor=CARD, edgecolor=GRID_C,
               labelcolor=FG, fontsize=10, bbox_to_anchor=(0.5, 0.003))

    fig.text(0.98, 0.005, "@CuartayDato",
             ha='right', va='bottom', fontsize=9, color="#888888", alpha=0.8, fontstyle='italic')
    fig.text(0.02, 0.005, "Fuente: PFR via nflreadpy  ·  Drafts 2011–2025",
             ha='left', va='bottom', fontsize=7.5, color="#555555", fontstyle='italic')

    plt.subplots_adjust(top=0.978, bottom=0.022, hspace=0.40, wspace=0.15)
    out = "draft_r1_radar_grid.png"
    plt.savefig(out, dpi=DPI_GRID, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {out}")


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    modo      = input("¿Equipo o Grid? (equipo/grid): ").strip().lower()
    equipo_raw = input("Equipo (ej: SF) — vacío si modo grid: ").strip()

    df = load_data()

    if modo.startswith("e") and equipo_raw:
        plot_team(df, equipo_raw)
    else:
        plot_grid(df)
