"""
informe_equipo.py
Informe completo de un equipo NFL: dos PNGs (ataque y defensa).
  PNG 1 ATAQUE  : personal ofensivo usado | vs personal defensivo
                  | vs cobertura | man/zona | presion
  PNG 2 DEFENSA : personal defensivo usado | vs personal ofensivo
                  | cobertura usada | man/zona | presion generada
Datos: nflverse PBP + NGS participation.
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

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON   = 2025
URL_PBP  = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.csv.gz"
URL_PART = f"https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{SEASON}.parquet"
BG       = "#0f1115"
CARD     = "#151924"
FG       = "#EDEDED"
GRID     = "#2a2f3a"
DPI      = 200
LOGOS_DIR    = "logos"
HARD_PENALTY = {"NYJ": 4.5}
RYG = LinearSegmentedColormap.from_list("ryg", ["#c0392b", "#e8b84b", "#27ae60"])

MIN_SNAPS = 15

OFF_PKG_ORDER = ["11", "12", "21", "10", "13", "22"]
OFF_PKG_LABEL = {"11":"11 (1RB·1TE·3WR)", "12":"12 (1RB·2TE·2WR)",
                 "21":"21 (2RB·1TE·2WR)", "10":"10 (1RB·0TE·4WR)",
                 "13":"13 (1RB·3TE·1WR)", "22":"22 (2RB·2TE·1WR)"}
DEF_PKG_ORDER = ["Base", "Nickel", "Dime", "Dollar+"]
COV_ORDER  = ["COVER_0","COVER_1","2_MAN","COVER_2","COVER_3","COVER_4","COVER_6","COVER_9"]
COV_LABELS = {"COVER_0":"Cover 0 (blitz total)", "COVER_1":"Cover 1 (man 1 alto)",
              "2_MAN":"2-Man (man 2 altos)",     "COVER_2":"Cover 2 (zona)",
              "COVER_3":"Cover 3 (zona 3 altos)", "COVER_4":"Cover 4 (quarters)",
              "COVER_6":"Cover 6 (mix)",          "COVER_9":"Cover 9"}
MZ_ORDER   = ["Zona", "Hombre"]
PRES_ORDER = ["Pocket limpio", "Bajo presion"]

# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_logo(team, base_zoom=0.055):
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
    if pd.isna(s):
        return None
    total = sum(int(m.group(1))
                for pos in ["CB","FS","SS","DB"]
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


def epa_group(df, col, order):
    grp = (df[df[col].notna()].groupby(col)["epa"]
           .agg(epa="mean", n="count").reset_index().rename(columns={col:"cat"}))
    grp = grp[grp["n"] >= MIN_SNAPS]
    grp["cat"] = pd.Categorical(grp["cat"], categories=order, ordered=True)
    return grp.sort_values("cat").dropna(subset=["cat"]).reset_index(drop=True)


def snap_pct(df, col, total):
    """Devuelve dict {cat: snap%} para las categorias con >= MIN_SNAPS."""
    counts = df[df[col].notna()][col].value_counts()
    return {k: v / total * 100 for k, v in counts.items()}


def get_team_ranks(all_plays, team_col, cat_col, cat_order, team, ascending=False):
    """
    Ranking del equipo en cada categoria vs el resto de la liga.
    ascending=False → rank 1 = mejor ataque (EPA mas alto)
    ascending=True  → rank 1 = mejor defensa (EPA mas bajo)
    Devuelve dict {cat: rank_int_o_None}.
    """
    df = all_plays[all_plays[cat_col].notna()]
    grp = (df.groupby([team_col, cat_col])["epa"]
             .agg(epa="mean", n="count")
             .reset_index())
    grp = grp[grp["n"] >= MIN_SNAPS]
    ranks = {}
    for cat in cat_order:
        sub = grp[grp[cat_col] == cat].copy()
        if sub.empty or team not in sub[team_col].values:
            ranks[cat] = None
            continue
        sub = sub.sort_values("epa", ascending=ascending).reset_index(drop=True)
        idx = sub.index[sub[team_col] == team].tolist()
        ranks[cat] = idx[0] + 1 if idx else None
    return ranks


def draw_panel(ax, labels, epa_vals, n_vals, title,
               cmap, cats=None, league_avgs=None,
               snap_pcts=None, xlabel="EPA / jugada", rankings=None):
    """Panel horizontal de barras con EPA dentro, marcador de liga y snap% en label."""
    # Color por desviacion respecto a la media de liga:
    #   verde = mejor que la media | rojo = peor que la media
    positions = np.arange(len(labels))
    if league_avgs and cats is not None:
        devs = [v - league_avgs.get(cats[i], 0) if not np.isnan(v) else np.nan
                for i, v in enumerate(epa_vals)]
    else:
        devs = list(epa_vals)
    valid_devs = [d for d in devs if not np.isnan(d)]
    d_abs = max(abs(min(valid_devs)), abs(max(valid_devs)), 0.03) if valid_devs else 0.2
    norm_obj = Normalize(vmin=-d_abs, vmax=d_abs)
    colors = [cmap(norm_obj(d)) if not np.isnan(d) else "#1e2430"
              for d in devs]

    ax.barh(positions, np.where(np.isnan(epa_vals), 0, epa_vals),
            color=colors, height=0.65, zorder=3, edgecolor="none")

    # Umbral relativo: barras < 22% del maximo van fuera
    valid_abs = [abs(v) for v in epa_vals if not np.isnan(v)]
    max_bar   = max(valid_abs) if valid_abs else 0.1
    INSIDE_MIN = max_bar * 0.22

    for i, (val, n) in enumerate(zip(epa_vals, n_vals)):
        y_mid = positions[i]
        if np.isnan(val):
            ax.text(0, y_mid, "n/d", ha="center", va="center",
                    color="#555555", fontsize=7, zorder=4)
            continue
        sign = "+" if val >= 0 else ""
        cat  = cats[i] if cats is not None else None
        rank = rankings.get(cat) if rankings and cat else None
        r_str = f"  R={rank}" if rank is not None else ""
        # Dos lineas: EPA arriba, n= R= abajo juntos
        txt = f"{sign}{val:.3f}\nn={int(n)}{r_str}"

        if abs(val) >= INSIDE_MIN:
            ax.text(val / 2, y_mid, txt,
                    ha="center", va="center",
                    color="#0a0e13", fontsize=6.5, fontweight="bold",
                    zorder=5, linespacing=1.3)
        else:
            # Texto fuera: a la derecha si val>=0, a la izquierda si val<0
            offset = max_bar * 0.04
            x_out  = val + offset if val >= 0 else val - offset
            ha_out = "left" if val >= 0 else "right"
            ax.text(x_out, y_mid, txt,
                    ha=ha_out, va="center",
                    color=FG, fontsize=6.5, fontweight="bold",
                    zorder=5, linespacing=1.3)

        if league_avgs is not None and cat is not None:
            lg = league_avgs.get(cat)
            if lg is not None:
                ax.scatter([lg], [y_mid], marker="|", s=200,
                           color="white", linewidths=2, zorder=6, alpha=0.85)

    # Y labels: add snap% if provided
    ylabels = []
    for i, lbl in enumerate(labels):
        cat = cats[i] if cats is not None else None
        pct = snap_pcts.get(cat, None) if snap_pcts and cat else None
        ylabels.append(f"{lbl}\n({pct:.0f}%)" if pct is not None else lbl)

    ax.set_yticks(positions)
    ax.set_yticklabels(ylabels, color=FG, fontsize=7.5)
    ax.axvline(0, color=FG, linewidth=0.7, alpha=0.4, zorder=2)
    ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_xlabel(xlabel, color=FG, fontsize=8)
    ax.set_title(title, color=FG, fontsize=9.5, pad=6, loc="left", fontweight="bold")
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=FG, length=0)

    # Legenda marcador liga (solo si hay league_avgs)
    if league_avgs:
        ax.scatter([], [], marker="|", s=120, color="white", linewidths=2,
                   label="| Media liga", alpha=0.85)
        ax.legend(fontsize=6.5, labelcolor=FG, facecolor=CARD,
                  edgecolor=GRID, framealpha=0.5, loc="lower right")


def style_axes(axes):
    for ax in axes:
        ax.set_facecolor(CARD)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.tick_params(colors=FG)


def add_logo_title(fig, team, subtitle):
    """Logo a la izquierda, título centrado."""
    logo = load_logo(team, base_zoom=0.10)
    if logo is not None:
        lax = fig.add_axes([0.03, 0.912, 0.065, 0.082])
        lax.imshow(logo.get_data())
        lax.axis("off")
    fig.text(0.5, 0.975, subtitle,
             ha="center", va="top", fontsize=15, fontweight="bold", color=FG)
    fig.text(0.5, 0.930,
             "Fuente: nflverse PBP + NGS participation  |  | = Media de liga",
             ha="center", va="top", fontsize=8, color="#888888", fontstyle="italic")
    fig.text(0.99, 0.008, "@CuartayDato",
             ha="right", va="bottom", fontsize=9, color="#888888",
             alpha=0.85, fontstyle="italic")
    fig.text(0.01, 0.008, f"NFL {SEASON}",
             ha="left", va="bottom", fontsize=8, color="#555555", fontstyle="italic")


# ── INPUT ──────────────────────────────────────────────────────────────────────
team = input("Equipo (siglas, p.ej. KC): ").strip().upper()

# ── CARGA DE DATOS ─────────────────────────────────────────────────────────────
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

merged = pbp.merge(part, on=["game_id", "play_id"], how="left")
all_plays = merged[
    merged["play_type"].isin(["pass", "run"]) &
    merged["epa"].notna()
].copy()
print(f"Jugadas totales: {len(all_plays):,}")

# ── CLASIFICAR COLUMNAS ────────────────────────────────────────────────────────
for df in [all_plays]:
    df["db_count"]  = df["defense_personnel"].apply(count_dbs)
    df["def_pkg"]   = df["db_count"].apply(classify_def_pkg)
    df["off_pkg"]   = df["offense_personnel"].apply(parse_off_pkg)
    df["man_zone"]  = df["defense_man_zone_type"].apply(man_zone_label)
    df["pressure"]  = df["was_pressure"].apply(pressure_label)

# ── SPLIT EQUIPO ──────────────────────────────────────────────────────────────
off  = all_plays[all_plays["posteam"] == team].copy()
defn = all_plays[all_plays["defteam"] == team].copy()

if off.empty:
    raise SystemExit(f"No se encontraron jugadas para {team}.")
print(f"{team} — jugadas ofensivas: {len(off):,}  |  defensivas: {len(defn):,}")

# ── MEDIAS DE LIGA ────────────────────────────────────────────────────────────
def lg_avg(col, order):
    return all_plays.groupby(col)["epa"].mean().reindex(order).to_dict()

lg_cov     = lg_avg("defense_coverage_type", COV_ORDER)
lg_mz      = lg_avg("man_zone",   MZ_ORDER)
lg_pres    = lg_avg("pressure",   PRES_ORDER)
lg_def_pkg = lg_avg("def_pkg",    DEF_PKG_ORDER)
lg_off_pkg = lg_avg("off_pkg",    OFF_PKG_ORDER)

# Liga para defensa (EPA permitido por cobertura/paquete cuando defienden)
lg_cov_def  = all_plays.groupby("defense_coverage_type")["epa"].mean().to_dict()
lg_mz_def   = all_plays.groupby("man_zone")["epa"].mean().to_dict()
lg_pres_def = all_plays.groupby("pressure")["epa"].mean().to_dict()
lg_def_used = all_plays.groupby("def_pkg")["epa"].mean().to_dict()
lg_off_vs   = all_plays.groupby("off_pkg")["epa"].mean().to_dict()

# ── RANKINGS DE LIGA ──────────────────────────────────────────────────────────
# Ataque: rank 1 = mayor EPA (mejor ataque)
rk_cov_off  = get_team_ranks(all_plays, "posteam", "defense_coverage_type", COV_ORDER,     team, ascending=False)
rk_opkg     = get_team_ranks(all_plays, "posteam", "off_pkg",               OFF_PKG_ORDER, team, ascending=False)
rk_dpkg_off = get_team_ranks(all_plays, "posteam", "def_pkg",               DEF_PKG_ORDER, team, ascending=False)
rk_mz_off   = get_team_ranks(all_plays, "posteam", "man_zone",              MZ_ORDER,      team, ascending=False)
rk_pres_off = get_team_ranks(all_plays, "posteam", "pressure",              PRES_ORDER,    team, ascending=False)
# Defensa: rank 1 = menor EPA (mejor defensa)
rk_vs_off   = get_team_ranks(all_plays, "defteam", "off_pkg",               OFF_PKG_ORDER, team, ascending=True)
rk_dpkg_def = get_team_ranks(all_plays, "defteam", "def_pkg",               DEF_PKG_ORDER, team, ascending=True)
rk_cov_def2 = get_team_ranks(all_plays, "defteam", "defense_coverage_type", COV_ORDER,     team, ascending=True)
rk_mz_def2  = get_team_ranks(all_plays, "defteam", "man_zone",              MZ_ORDER,      team, ascending=True)
rk_pres_def2= get_team_ranks(all_plays, "defteam", "pressure",              PRES_ORDER,    team, ascending=True)

# ── NORMALIZACIONES ───────────────────────────────────────────────────────────
def build_norm(vals):
    v = [v for v in vals if v is not None and not np.isnan(v)]
    if not v: return Normalize(vmin=-0.2, vmax=0.2)
    v_abs = max(abs(min(v)), abs(max(v)), 0.05)
    return Normalize(vmin=-v_abs, vmax=v_abs)


# ── FUNCIÓN PARA CONSTRUIR DATOS DE PANEL ─────────────────────────────────────
def panel_data(df, col, order, label_map=None):
    """Devuelve (labels, epa_vals, n_vals, cats) para un panel."""
    grp = epa_group(df, col, order)
    cats   = grp["cat"].tolist()
    epa    = grp["epa"].values.astype(float)
    n      = grp["n"].values.astype(int)
    labels = [label_map.get(c, c) if label_map else c for c in cats]
    return labels, epa, n, cats


# ══════════════════════════════════════════════════════════════════════════════
# PNG 1: ATAQUE
# ══════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(16, 11), facecolor=BG)
gs1  = gridspec.GridSpec(2, 3, figure=fig1,
                          width_ratios=[2.5, 1.5, 1.5],
                          height_ratios=[1.15, 1],
                          hspace=0.52, wspace=0.38,
                          left=0.05, right=0.97,
                          top=0.87, bottom=0.06)

ax_cov   = fig1.add_subplot(gs1[:, 0])   # izquierda: vs cobertura (full height)
ax_opkg  = fig1.add_subplot(gs1[0, 1])   # top center: personal ofensivo usado
ax_dpkg  = fig1.add_subplot(gs1[0, 2])   # top right: vs personal defensivo
ax_mz    = fig1.add_subplot(gs1[1, 1])   # bottom center: man/zona
ax_pres  = fig1.add_subplot(gs1[1, 2])   # bottom right: presion

style_axes([ax_cov, ax_opkg, ax_dpkg, ax_mz, ax_pres])

cmap_off = RYG          # verde = EPA alto = buen ataque

# ── Panel: vs Cobertura ──────────────────────────────────────────────────────
labels, epa, n, cats = panel_data(off, "defense_coverage_type", COV_ORDER, COV_LABELS)
sp_cov = snap_pct(off, "defense_coverage_type", len(off))
draw_panel(ax_cov, labels, epa, n,
           title="EPA vs cobertura defensiva rival",
           cmap=cmap_off, cats=cats, league_avgs=lg_cov,
           snap_pcts=sp_cov, rankings=rk_cov_off)

# ── Panel: Personal ofensivo usado ───────────────────────────────────────────
labels, epa, n, cats = panel_data(off, "off_pkg", OFF_PKG_ORDER, OFF_PKG_LABEL)
sp_opkg = snap_pct(off, "off_pkg", len(off))
draw_panel(ax_opkg, labels, epa, n,
           title="Personal ofensivo usado",
           cmap=cmap_off, cats=cats, league_avgs=lg_off_pkg,
           snap_pcts=sp_opkg, rankings=rk_opkg)

# ── Panel: vs Personal defensivo ─────────────────────────────────────────────
labels, epa, n, cats = panel_data(off, "def_pkg", DEF_PKG_ORDER)
draw_panel(ax_dpkg, labels, epa, n,
           title="EPA vs personal defensivo rival",
           cmap=cmap_off, cats=cats, league_avgs=lg_def_pkg, rankings=rk_dpkg_off)

# ── Panel: Man / Zona ────────────────────────────────────────────────────────
labels, epa, n, cats = panel_data(off, "man_zone", MZ_ORDER)
sp_mz = snap_pct(off, "man_zone", len(off))
draw_panel(ax_mz, labels, epa, n,
           title="EPA vs Hombre / Zona",
           cmap=cmap_off, cats=cats, league_avgs=lg_mz,
           snap_pcts=sp_mz, rankings=rk_mz_off)

# ── Panel: Presion ───────────────────────────────────────────────────────────
labels, epa, n, cats = panel_data(off, "pressure", PRES_ORDER)
sp_pr = snap_pct(off, "pressure", len(off))
draw_panel(ax_pres, labels, epa, n,
           title="EPA bajo presion vs pocket limpio",
           cmap=cmap_off, cats=cats, league_avgs=lg_pres,
           snap_pcts=sp_pr, rankings=rk_pres_off)

add_logo_title(fig1, team,
               f"{team} | Informe Ofensivo | NFL {SEASON}")

out1 = f"informe_ataque_{team}_{SEASON}.png"
fig1.savefig(out1, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig1)
print(f"Guardado: {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# PNG 2: DEFENSA
# ══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(16, 11), facecolor=BG)
gs2  = gridspec.GridSpec(2, 3, figure=fig2,
                          width_ratios=[2.5, 1.5, 1.5],
                          height_ratios=[1.15, 1],
                          hspace=0.52, wspace=0.38,
                          left=0.05, right=0.97,
                          top=0.87, bottom=0.06)

ax_vs_off  = fig2.add_subplot(gs2[:, 0])   # izquierda: vs personal ofensivo rival
ax_dpkg2   = fig2.add_subplot(gs2[0, 1])   # top center: personal defensivo usado
ax_cov2    = fig2.add_subplot(gs2[0, 2])   # top right: cobertura usada
ax_mz2     = fig2.add_subplot(gs2[1, 1])   # bottom center: man/zona jugada
ax_pres2   = fig2.add_subplot(gs2[1, 2])   # bottom right: presion generada

style_axes([ax_vs_off, ax_dpkg2, ax_cov2, ax_mz2, ax_pres2])

cmap_def = RYG.reversed()  # verde = EPA bajo/negativo = buena defensa

# ── Panel: vs Personal ofensivo rival ────────────────────────────────────────
labels, epa, n, cats = panel_data(defn, "off_pkg", OFF_PKG_ORDER, OFF_PKG_LABEL)
draw_panel(ax_vs_off, labels, epa, n,
           title="EPA permitido vs personal ofensivo rival",
           cmap=cmap_def, cats=cats, league_avgs=lg_off_vs,
           rankings=rk_vs_off,
           xlabel="EPA permitido (- = mejor defensa)")

# ── Panel: Personal defensivo usado ──────────────────────────────────────────
labels, epa, n, cats = panel_data(defn, "def_pkg", DEF_PKG_ORDER)
sp_dpkg = snap_pct(defn, "def_pkg", len(defn))
draw_panel(ax_dpkg2, labels, epa, n,
           title="Personal defensivo usado",
           cmap=cmap_def, cats=cats, league_avgs=lg_def_used,
           snap_pcts=sp_dpkg, rankings=rk_dpkg_def,
           xlabel="EPA permitido (- = mejor defensa)")

# ── Panel: Cobertura usada ────────────────────────────────────────────────────
labels, epa, n, cats = panel_data(defn, "defense_coverage_type", COV_ORDER, COV_LABELS)
sp_cov2 = snap_pct(defn, "defense_coverage_type", len(defn))
draw_panel(ax_cov2, labels, epa, n,
           title="Cobertura usada",
           cmap=cmap_def, cats=cats, league_avgs=lg_cov_def,
           snap_pcts=sp_cov2, rankings=rk_cov_def2,
           xlabel="EPA permitido (- = mejor defensa)")

# ── Panel: Man / Zona jugada ─────────────────────────────────────────────────
labels, epa, n, cats = panel_data(defn, "man_zone", MZ_ORDER)
sp_mz2 = snap_pct(defn, "man_zone", len(defn))
draw_panel(ax_mz2, labels, epa, n,
           title="EPA permitido: Hombre / Zona",
           cmap=cmap_def, cats=cats, league_avgs=lg_mz_def,
           snap_pcts=sp_mz2, rankings=rk_mz_def2,
           xlabel="EPA permitido (- = mejor defensa)")

# ── Panel: Presion generada ───────────────────────────────────────────────────
labels, epa, n, cats = panel_data(defn, "pressure", PRES_ORDER)
sp_pr2 = snap_pct(defn, "pressure", len(defn))
draw_panel(ax_pres2, labels, epa, n,
           title="EPA permitido segun presion generada",
           cmap=cmap_def, cats=cats, league_avgs=lg_pres_def,
           snap_pcts=sp_pr2, rankings=rk_pres_def2,
           xlabel="EPA permitido (- = mejor defensa)")

add_logo_title(fig2, team,
               f"{team} | Informe Defensivo | NFL {SEASON}")

out2 = f"informe_defensa_{team}_{SEASON}.png"
fig2.savefig(out2, dpi=DPI, facecolor=BG, bbox_inches="tight")
plt.close(fig2)
print(f"Guardado: {out2}")
