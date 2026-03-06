"""
matchup.py
Analisis de emparejamiento entre dos equipos NFL.
Un PNG con dos columnas: ATAQUE A / DEFENSA B | ATAQUE B / DEFENSA A
Dimensiones: personal ofensivo, coberturas, man/zona, presion, red zone, 3er down.
NFL 2025
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON   = 2025
URL_PBP  = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
URL_PART = f"https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{SEASON}.parquet"
BG       = "#0f1115"
CARD     = "#151924"
CARD2    = "#1a2030"   # fondo alterno para separar ataque/defensa
FG       = "#EDEDED"
GRID     = "#2a2f3a"
DPI      = 200
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}
RYG = LinearSegmentedColormap.from_list("ryg", ["#c0392b", "#e8b84b", "#27ae60"])

MIN_SNAPS = 15
TOP_PKG   = 4
TOP_COV   = 4
BAR_H     = 0.38

MAX_PKG = 5   # maximo de filas en panel de personal
MAX_COV = 5   # maximo de filas en panel de coberturas

OFF_PKG_LABEL = {
    "11": "11 personal", "12": "12 personal",
    "21": "21 personal", "10": "10 personal",
    "13": "13 personal", "22": "22 personal",
}
COV_LABEL = {
    "COVER_0": "Cover 0", "COVER_1": "Cover 1", "2_MAN":   "2-Man",
    "COVER_2": "Cover 2", "COVER_3": "Cover 3", "COVER_4": "Cover 4",
    "COVER_6": "Cover 6", "COVER_9": "Cover 9",
}

# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_logo(team, base_zoom=0.07):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        h, w = img.shape[:2]
        aspect = w / float(h) if h else 1.0
        zoom = base_zoom / HARD_PENALTY[team] if team in HARD_PENALTY else \
               base_zoom / np.clip(1.0 + 0.6 * max(0.0, aspect - 1.3), 1.0, 2.2)
        return OffsetImage(img, zoom=zoom, resample=True)
    except Exception:
        return None


def count_dbs(s):
    if pd.isna(s): return None
    total = sum(int(m.group(1))
                for pos in ["CB", "FS", "SS", "DB"]
                for m in [re.search(r"(\d+)\s+" + pos + r"(?:[,\s]|$)", str(s))]
                if m)
    return total if total > 0 else None

def classify_def_pkg(n):
    if n is None: return None
    if n <= 4:    return "Base"
    if n == 5:    return "Nickel"
    if n == 6:    return "Dime"
    return "Dollar+"

def parse_off_pkg(s):
    if pd.isna(s): return None
    rb = re.search(r"(\d+)\s*RB", str(s), re.I)
    te = re.search(r"(\d+)\s*TE", str(s), re.I)
    return f"{rb.group(1)}{te.group(1)}" if rb and te else None

def man_zone_label(x):
    if pd.isna(x) or x == "": return None
    return "Hombre" if "MAN" in str(x) else ("Zona" if "ZONE" in str(x) else None)

def pressure_label(x):
    if pd.isna(x): return None
    return "Bajo presion" if x else "Pocket limpio"

def epa_stat(df, mask=None, col=None, cat=None):
    sub = df.copy()
    if mask is not None: sub = sub[mask]
    if col  is not None and cat is not None: sub = sub[sub[col] == cat]
    sub = sub.dropna(subset=["epa"])
    if len(sub) < MIN_SNAPS: return np.nan, len(sub)
    return sub["epa"].mean(), len(sub)


# ── CARGA DE DATOS ─────────────────────────────────────────────────────────────
team_a = input("Equipo A: ").strip().upper()
team_b = input("Equipo B: ").strip().upper()

print(f"Descargando PBP {SEASON}...")
pbp = pd.read_csv(URL_PBP, low_memory=False, compression="infer")
pbp["epa"]     = pd.to_numeric(pbp["epa"],     errors="coerce")
pbp["play_id"] = pd.to_numeric(pbp["play_id"], errors="coerce")

print(f"Descargando participacion {SEASON}...")
part = pd.read_parquet(URL_PART, columns=[
    "nflverse_game_id", "play_id",
    "offense_personnel", "defense_personnel",
    "defense_coverage_type", "defense_man_zone_type",
    "was_pressure",
])
part = part.rename(columns={"nflverse_game_id": "game_id"})
part["play_id"] = pd.to_numeric(part["play_id"], errors="coerce")

merged    = pbp.merge(part, on=["game_id", "play_id"], how="left")
all_plays = merged[merged["play_type"].isin(["pass", "run"]) & merged["epa"].notna()].copy()
print(f"Jugadas totales: {len(all_plays):,}")

all_plays["off_pkg"]    = all_plays["offense_personnel"].apply(parse_off_pkg)
all_plays["def_pkg"]    = all_plays["defense_personnel"].apply(count_dbs).apply(classify_def_pkg)
all_plays["man_zone"]   = all_plays["defense_man_zone_type"].apply(man_zone_label)
all_plays["pressure"]   = all_plays["was_pressure"].apply(pressure_label)
all_plays["red_zone"]   = all_plays["yardline_100"].le(20)
all_plays["third_down"] = all_plays["down"].eq(3)

off_a = all_plays[all_plays["posteam"] == team_a].copy()
def_b = all_plays[all_plays["defteam"] == team_b].copy()
off_b = all_plays[all_plays["posteam"] == team_b].copy()
def_a = all_plays[all_plays["defteam"] == team_a].copy()

for tm, df in [(team_a, off_a), (team_b, off_b)]:
    if df.empty: raise SystemExit(f"No se encontraron jugadas para {tm}.")
print(f"{team_a}: {len(off_a):,} jugadas of. | {team_b}: {len(off_b):,} jugadas of.")

# ── CATEGORIAS DINAMICAS (union de top-N de ambos equipos) ─────────────────
def _union_top(series_a, series_b, n, max_cats):
    """Top-n de cada serie; union preservando orden de frecuencia combinada."""
    top_a = series_a.dropna().value_counts().head(n).index.tolist()
    top_b = series_b.dropna().value_counts().head(n).index.tolist()
    seen, result = set(), []
    for cat in top_a + top_b:
        if cat not in seen:
            seen.add(cat)
            result.append(cat)
        if len(result) == max_cats:
            break
    return result

DYN_PKG = _union_top(off_a["off_pkg"], off_b["off_pkg"], TOP_PKG, MAX_PKG)
DYN_COV = _union_top(
    def_b["defense_coverage_type"], def_a["defense_coverage_type"], TOP_COV, MAX_COV
)


# ── BUILD COLUMN DATA ─────────────────────────────────────────────────────────
def build_column(off_df, def_df):
    panels = {}

    # 1. Personal ofensivo (categorias dinamicas: union top-N ambos equipos)
    rows = []
    for cat in DYN_PKG:
        o, on = epa_stat(off_df, col="off_pkg", cat=cat)
        d, dn = epa_stat(def_df, col="off_pkg", cat=cat)
        rows.append(dict(label=OFF_PKG_LABEL.get(cat, cat),
                         off_val=o, off_n=on, def_val=d, def_n=dn))
    panels["pkg"] = rows

    # 2. Coberturas (categorias dinamicas: union top-N ambos defensas)
    rows = []
    for cat in DYN_COV:
        o, on = epa_stat(off_df, col="defense_coverage_type", cat=cat)
        d, dn = epa_stat(def_df, col="defense_coverage_type", cat=cat)
        rows.append(dict(label=COV_LABEL.get(cat, cat),
                         off_val=o, off_n=on, def_val=d, def_n=dn))
    panels["cov"] = rows

    # 3. Man / Zona
    rows = []
    for cat in ["Hombre", "Zona"]:
        o, on = epa_stat(off_df, col="man_zone", cat=cat)
        d, dn = epa_stat(def_df, col="man_zone", cat=cat)
        rows.append(dict(label=cat,
                         off_val=o, off_n=on, def_val=d, def_n=dn))
    panels["mz"] = rows

    # 4. Presion
    rows = []
    for cat in ["Pocket limpio", "Bajo presion"]:
        o, on = epa_stat(off_df, col="pressure", cat=cat)
        d, dn = epa_stat(def_df, col="pressure", cat=cat)
        rows.append(dict(label=cat,
                         off_val=o, off_n=on, def_val=d, def_n=dn))
    panels["pres"] = rows

    # 5. Situaciones especiales
    o_rz, on_rz = epa_stat(off_df, mask=off_df["red_zone"])
    d_rz, dn_rz = epa_stat(def_df, mask=def_df["red_zone"])
    o_3d, on_3d = epa_stat(off_df, mask=off_df["third_down"])
    d_3d, dn_3d = epa_stat(def_df, mask=def_df["third_down"])
    panels["sit"] = [
        dict(label="Red Zone",  off_val=o_rz, off_n=on_rz, def_val=d_rz, def_n=dn_rz),
        dict(label="3er Down",  off_val=o_3d, off_n=on_3d, def_val=d_3d, def_n=dn_3d),
    ]
    return panels


col_left  = build_column(off_a, def_b)
col_right = build_column(off_b, def_a)


# ── DRAW MATCHUP PANEL ────────────────────────────────────────────────────────
def draw_matchup_panel(ax, rows, atk_team, def_team):
    """
    Por cada dimension hay DOS filas:
      fila superior  = barra del ATACANTE  (EPA ofensivo en esa situacion)
      fila inferior  = barra del DEFENSOR  (EPA permitido en esa situacion)
    Color: verde = EPA positivo (bueno para el atacante) /
           rojo  = EPA negativo o bajo.
    Para la barra defensiva el colormap esta invertido:
           verde = permite poco EPA (buena defensa).
    """
    if not rows:
        ax.axis("off")
        return

    # Una fila visual por cada dimension, con 2 sub-barras dentro
    n_dims = len(rows)
    # Posiciones: cada dimension ocupa 1 unidad en y
    # Dentro de cada unidad: off_bar en +0.22, def_bar en -0.22
    OFF_Y  =  0.21
    DEF_Y  = -0.21

    # Norma: simetrica alrededor de 0, basada en los valores del panel
    all_vals = [v for r in rows for v in [r["off_val"], r["def_val"]] if not np.isnan(v)]
    v_abs = max(abs(min(all_vals)), abs(max(all_vals)), 0.05) if all_vals else 0.3
    norm  = Normalize(vmin=-v_abs, vmax=v_abs)

    # Maximo para umbral inside/outside
    off_abs = [abs(r["off_val"]) for r in rows if not np.isnan(r["off_val"])]
    def_abs = [abs(r["def_val"]) for r in rows if not np.isnan(r["def_val"])]
    off_max = max(off_abs) if off_abs else 0.1
    def_max = max(def_abs) if def_abs else 0.1

    for i, r in enumerate(rows):
        y_center = n_dims - i - 1   # 0 abajo, n_dims-1 arriba
        y_off    = y_center + OFF_Y
        y_def    = y_center + DEF_Y

        # Fondo alternado por dimension para facilitar lectura
        bg = CARD if i % 2 == 0 else CARD2
        ax.barh(y_center, v_abs * 2.2, left=-v_abs * 1.1,
                height=1.0, color=bg, zorder=0, edgecolor="none")

        # ── Barra ATAQUE ──────────────────────────────────────────────────────
        v = r["off_val"]
        if not np.isnan(v):
            color = RYG(norm(v))
            ax.barh(y_off, v, height=BAR_H, color=color,
                    zorder=3, edgecolor="none", alpha=0.95)
            sign = "+" if v >= 0 else ""
            txt  = f"{atk_team} {sign}{v:.3f}"
            inside = abs(v) >= off_max * 0.25
            if inside:
                ax.text(v / 2, y_off, txt, ha="center", va="center",
                        color="#0a0e13", fontsize=6.5, fontweight="bold", zorder=5)
            else:
                x_out = v + off_max * 0.05 if v >= 0 else v - off_max * 0.05
                ax.text(x_out, y_off, txt,
                        ha="left" if v >= 0 else "right", va="center",
                        color=FG, fontsize=6.5, fontweight="bold", zorder=5)
        else:
            ax.text(0, y_off, f"{atk_team} n/d", ha="center", va="center",
                    color="#555555", fontsize=6.5)

        # ── Barra DEFENSA ─────────────────────────────────────────────────────
        v = r["def_val"]
        if not np.isnan(v):
            color = RYG.reversed()(norm(v))   # verde = poco EPA permitido
            ax.barh(y_def, v, height=BAR_H, color=color,
                    zorder=3, edgecolor="none", alpha=0.95)
            sign = "+" if v >= 0 else ""
            txt  = f"{def_team} {sign}{v:.3f}"
            inside = abs(v) >= def_max * 0.25
            if inside:
                ax.text(v / 2, y_def, txt, ha="center", va="center",
                        color="#0a0e13", fontsize=6.5, fontweight="bold", zorder=5)
            else:
                x_out = v + def_max * 0.05 if v >= 0 else v - def_max * 0.05
                ax.text(x_out, y_def, txt,
                        ha="left" if v >= 0 else "right", va="center",
                        color=FG, fontsize=6.5, fontweight="bold", zorder=5)
        else:
            ax.text(0, y_def, f"{def_team} n/d", ha="center", va="center",
                    color="#555555", fontsize=6.5)

        # Linea separadora entre dimensiones
        if i < n_dims - 1:
            ax.axhline(y_center - 0.5, color=GRID, linewidth=0.6, alpha=0.5, zorder=4)

    # ── Eje Y: sin etiquetas (se muestran en columna central) ─────────────────
    ax.set_yticks([])
    ax.set_ylim(-0.55, n_dims - 0.45)

    # ── Eje X y estilo ─────────────────────────────────────────────────────────
    ax.axvline(0, color=FG, linewidth=0.7, alpha=0.5, zorder=2)
    ax.grid(axis="x", color=GRID, linewidth=0.4, alpha=0.3, zorder=1)
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors=FG, length=0, labelsize=6.5)


# ── DRAW CENTER LABELS ────────────────────────────────────────────────────────
def draw_center_labels(ax, rows, n_dims, title=""):
    """Columna central: etiquetas de dimension compartidas por ambos paneles."""
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, n_dims - 0.45)
    for i, r in enumerate(rows):
        y  = n_dims - i - 1
        bg = CARD if i % 2 == 0 else CARD2
        ax.add_patch(plt.Rectangle((0, y - 0.5), 1, 1, color=bg, zorder=0))
        ax.text(0.5, y, r["label"],
                ha="center", va="center",
                color=FG, fontsize=7, fontweight="bold", zorder=1)
        if i < n_dims - 1:
            ax.axhline(y - 0.5, color=GRID, linewidth=0.4, alpha=0.3, zorder=2)
    if title:
        ax.set_title(title, color=FG, fontsize=8, pad=5,
                     fontweight="bold", loc="center")


# ── FIGURA ────────────────────────────────────────────────────────────────────
PANEL_ORDER  = ["pkg", "cov", "mz", "pres", "sit"]
PANEL_TITLES = {
    "pkg":  "Personal ofensivo",
    "cov":  "Coberturas del defensor",
    "mz":   "Man / Zona",
    "pres": "Presion",
    "sit":  "Red Zone & 3er Down",
}
HR = [max(len(col_left[k]), len(col_right[k])) for k in PANEL_ORDER]

fig = plt.figure(figsize=(16, 13), facecolor=BG)
gs  = gridspec.GridSpec(
    len(PANEL_ORDER) + 1, 3,
    figure=fig,
    height_ratios=[0.75] + HR,
    hspace=0.65, wspace=0.05,
    left=0.04, right=0.97,
    top=0.90, bottom=0.05,
    width_ratios=[1.0, 0.35, 1.2],
)

# ── CABECERAS DE COLUMNA ──────────────────────────────────────────────────────
for gs_col, (atk, dfn) in zip([0, 2], [(team_a, team_b), (team_b, team_a)]):
    ax_h = fig.add_subplot(gs[0, gs_col])
    ax_h.set_facecolor(CARD)
    ax_h.axis("off")
    logo = load_logo(atk, base_zoom=0.065)
    if logo:
        ab = AnnotationBbox(logo, (0.10, 0.50),
                            xycoords="axes fraction",
                            frameon=False, zorder=3)
        ax_h.add_artist(ab)
    ax_h.text(0.58, 0.65, f"ATAQUE {atk}", ha="center", va="center",
              color=FG, fontsize=11, fontweight="bold", transform=ax_h.transAxes)
    ax_h.text(0.58, 0.18, f"vs  DEFENSA {dfn}", ha="center", va="center",
              color="#aaaaaa", fontsize=9, transform=ax_h.transAxes)
    for sp in ax_h.spines.values(): sp.set_edgecolor(GRID)

# Celda central del header (decorativa)
ax_hc = fig.add_subplot(gs[0, 1])
ax_hc.set_facecolor(BG)
ax_hc.axis("off")

# ── PANELES ───────────────────────────────────────────────────────────────────
for p_idx, key in enumerate(PANEL_ORDER):
    rows_l = col_left[key]
    rows_r = col_right[key]
    n_dims = len(rows_l)

    ax_l = fig.add_subplot(gs[p_idx + 1, 0])
    draw_matchup_panel(ax_l, rows_l, team_a, team_b)

    ax_c = fig.add_subplot(gs[p_idx + 1, 1])
    draw_center_labels(ax_c, rows_l, n_dims, title=PANEL_TITLES[key])

    ax_r = fig.add_subplot(gs[p_idx + 1, 2])
    draw_matchup_panel(ax_r, rows_r, team_b, team_a)

# ── LEYENDA GLOBAL ─────────────────────────────────────────────────────────────
off_patch = mpatches.Patch(color=RYG(0.85),          label="Barra atacante (EPA generado)")
def_patch  = mpatches.Patch(color=RYG.reversed()(0.85), label="Barra defensora (EPA permitido)")
pos_patch  = mpatches.Patch(color=RYG(0.82),          label="Verde = EPA positivo")
neg_patch  = mpatches.Patch(color="#c0392b",           label="Rojo = EPA negativo")
fig.legend(
    handles=[off_patch, def_patch, pos_patch, neg_patch],
    loc="lower center", ncol=4,
    fontsize=7, labelcolor=FG,
    facecolor=CARD, edgecolor=GRID, framealpha=0.8,
    bbox_to_anchor=(0.5, 0.005),
)

# ── TITULOS ───────────────────────────────────────────────────────────────────
fig.text(0.5, 0.975,
         f"{team_a}  vs  {team_b}  |  Analisis de Matchup  |  NFL {SEASON}",
         ha="center", va="top", fontsize=15, fontweight="bold", color=FG)
fig.text(0.5, 0.954,
         "Barra superior = EPA generado en ataque  |  Barra inferior = EPA permitido en defensa  |  Verde mejor, Rojo peor",
         ha="center", va="top", fontsize=8, color="#888888", fontstyle="italic")
fig.text(0.01, 0.010, f"Fuente: nflverse-data + NGS participation  |  NFL {SEASON}",
         ha="left", va="bottom", fontsize=7, color="#555555", fontstyle="italic")
fig.text(0.99, 0.010, "@CuartayDato",
         ha="right", va="bottom", fontsize=9, color="#888888", alpha=0.85, fontstyle="italic")

# ── GUARDAR ───────────────────────────────────────────────────────────────────
outfile = f"matchup_{team_a}_vs_{team_b}_{SEASON}.png"
fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Guardado: {outfile}")
