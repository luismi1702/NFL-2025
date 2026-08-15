"""
informe_equipo.py
Informe completo de un equipo NFL: dos PNGs (ataque y defensa).

Rediseño jul-2026 — "team card" presentable:
  - Cabecera con 4 KPIs grandes y su ranking de liga
  - Columna izquierda: cada faceta como punto sobre una pista de ranking 1→32
    (un solo eje universal: izquierda = top de la liga, derecha = cola)
  - Columna derecha: fortalezas y debilidades autogeneradas + identidad
Datos: nflverse PBP + NGS participation (coberturas, personal, presión).
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pbp_loader import cargar_pbp, cargar_participation, cargar_ftn, salida, season_cli, sello

sys.stdout.reconfigure(encoding="utf-8")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEASON    = season_cli()
BG        = "#0f1115"
CARD      = "#151924"
FG        = "#EDEDED"
GRID      = "#2a2f3a"
DPI       = 170
LOGOS_DIR = "logos"
RYG = LinearSegmentedColormap.from_list("ryg", ["#d84a4a", "#ffd166", "#06d6a0"])

MIN_SNAPS = 15
MIN_USO   = 3.0   # % mínimo de snaps para que una faceta entre en claves

OFF_PKG_ORDER = ["11", "12", "21", "10", "13", "22"]
OFF_PKG_LABEL = {"11": "11 (3WR)", "12": "12 (2TE)", "21": "21 (2RB)",
                 "10": "10 (4WR)", "13": "13 (3TE)", "22": "22 (2RB·2TE)"}
DEF_PKG_ORDER = ["Base", "Nickel", "Dime", "Dollar+"]
COV_ORDER  = ["COVER_0", "COVER_1", "2_MAN", "COVER_2", "COVER_3",
              "COVER_4", "COVER_6", "COVER_9"]
COV_LABELS = {"COVER_0": "Cover 0", "COVER_1": "Cover 1", "2_MAN": "2-Man",
              "COVER_2": "Cover 2", "COVER_3": "Cover 3", "COVER_4": "Cover 4",
              "COVER_6": "Cover 6", "COVER_9": "Cover 9"}
MZ_ORDER   = ["Hombre", "Zona"]
PRES_ORDER = ["Bajo presion", "Pocket limpio"]
PRES_LABEL = {"Bajo presion": "Bajo presión", "Pocket limpio": "Pocket limpio"}


# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_logo(team, base_zoom=0.055):
    path = os.path.join(LOGOS_DIR, f"{team}.png")
    if not os.path.exists(path):
        return None
    try:
        img = plt.imread(path)
        if img.ndim == 3 and img.shape[2] == 4:
            ys, xs = np.where(img[:, :, 3] > 0.02)
            if len(ys):
                img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = img.shape[:2]
        z = base_zoom * 500.0 / max((h * w) ** 0.5, 1.0)
        if w * z > 900.0 * base_zoom:
            z = 900.0 * base_zoom / w
        return OffsetImage(img, zoom=z, resample=True)
    except Exception:
        return None


def count_dbs(s):
    if pd.isna(s):
        return None
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
    s = str(s)
    rb = re.search(r"(\d+)\s*RB", s, re.I)
    fb = re.search(r"(\d+)\s*FB", s, re.I)
    te = re.search(r"(\d+)\s*TE", s, re.I)
    rb_n = (int(rb.group(1)) if rb else 0) + (int(fb.group(1)) if fb else 0)
    te_n = int(te.group(1)) if te else 0
    return f"{rb_n}{te_n}" if rb_n > 0 else None


def man_zone_label(x):
    if pd.isna(x) or x == "": return None
    return "Hombre" if "MAN" in str(x) else ("Zona" if "ZONE" in str(x) else None)


def pressure_label(x):
    if pd.isna(x): return None
    return "Bajo presion" if int(float(x)) else "Pocket limpio"


def rank_color(rank, n_teams):
    """Verde = top de la liga, rojo = cola."""
    if rank is None or n_teams <= 1:
        return "#3a4050"
    return RYG(1.0 - (rank - 1) / (n_teams - 1))


def facet_table(all_plays, team_col, cat_col, team, ascending):
    """{cat: dict(epa, n, rank, n_teams, lg)} para las categorías de un split.
    ascending=True → rank 1 = EPA más bajo (mejor defensa)."""
    df = all_plays[all_plays[cat_col].notna()]
    grp = (df.groupby([team_col, cat_col])["epa"]
             .agg(epa="mean", n="count").reset_index())
    grp = grp[grp["n"] >= MIN_SNAPS]
    lg = df.groupby(cat_col)["epa"].mean().to_dict()
    out = {}
    for cat, sub in grp.groupby(cat_col):
        sub = sub.sort_values("epa", ascending=ascending).reset_index(drop=True)
        idx = sub.index[sub[team_col] == team].tolist()
        if not idx:
            continue
        r = sub.loc[idx[0]]
        out[cat] = dict(epa=float(r["epa"]), n=int(r["n"]),
                        rank=idx[0] + 1, n_teams=len(sub),
                        lg=float(lg.get(cat, np.nan)))
    return out


def team_rank(series_by_team, team, ascending):
    s = series_by_team.dropna().sort_values(ascending=ascending)
    teams = s.index.tolist()
    return (teams.index(team) + 1 if team in teams else None), len(teams), \
           float(s.get(team, np.nan))


def snap_pct(df, col):
    """% sobre las jugadas CON dato en esa faceta (no sobre todas las jugadas):
    've zona 70%' = de los pases con cobertura charteada, no diluido por
    carreras sin charting. Así el % de presión cuadra además con el KPI."""
    counts = df[df[col].notna()][col].value_counts()
    total  = counts.sum()
    if total == 0:
        return {}
    return {k: v / total * 100 for k, v in counts.items()}


def ordinal(rank):
    return f"{rank}º"


# ── INPUT Y DATOS ──────────────────────────────────────────────────────────────
team = input("Equipo (siglas, p.ej. KC): ").strip().upper()

pbp, SEASON = cargar_pbp(SEASON)
pbp["epa"]     = pd.to_numeric(pbp["epa"],     errors="coerce")
pbp["play_id"] = pd.to_numeric(pbp["play_id"], errors="coerce")
print(f"PBP {SEASON}: {len(pbp):,} jugadas REG")

part, _ = cargar_participation(SEASON)
part = part[["nflverse_game_id", "play_id", "offense_personnel",
             "defense_personnel", "defense_coverage_type",
             "defense_man_zone_type", "was_pressure"]]
part = part.rename(columns={"nflverse_game_id": "game_id"})
part["play_id"] = pd.to_numeric(part["play_id"], errors="coerce")

merged = pbp.merge(part, on=["game_id", "play_id"], how="left")

# FTN charting (play-action; solo 2022+) para la identidad de play-calling
try:
    ftn, _ = cargar_ftn(SEASON)
    ftn = ftn[["nflverse_game_id", "nflverse_play_id", "is_play_action"]].rename(
        columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    merged = merged.merge(ftn, on=["game_id", "play_id"], how="left")
except Exception as e:
    print(f"  Aviso: FTN no disponible ({e}) — sin play-action%")
    merged["is_play_action"] = np.nan

all_plays = merged[
    merged["play_type"].isin(["pass", "run"]) & merged["epa"].notna()
].copy()
for c in ["shotgun", "no_huddle"]:
    all_plays[c] = pd.to_numeric(all_plays[c], errors="coerce")
print(f"Jugadas totales: {len(all_plays):,}")

all_plays["db_count"] = all_plays["defense_personnel"].apply(count_dbs)
all_plays["def_pkg"]  = all_plays["db_count"].apply(classify_def_pkg)
all_plays["off_pkg"]  = all_plays["offense_personnel"].apply(parse_off_pkg)
all_plays["man_zone"] = all_plays["defense_man_zone_type"].apply(man_zone_label)
all_plays["pressure"] = all_plays["was_pressure"].apply(pressure_label)

off  = all_plays[all_plays["posteam"] == team].copy()
defn = all_plays[all_plays["defteam"] == team].copy()
if off.empty:
    raise SystemExit(f"No se encontraron jugadas para {team}.")
print(f"{team} — jugadas ofensivas: {len(off):,}  |  defensivas: {len(defn):,}")

pases_lg    = all_plays[all_plays["play_type"] == "pass"]
carreras_lg = all_plays[all_plays["play_type"] == "run"]


# ── SECCIONES DEL INFORME ─────────────────────────────────────────────────────
# (título de sección, columna, orden, etiquetas, snap% del propio equipo)
def secciones(side):
    """side: 'off' (posteam, rank desc) o 'def' (defteam, rank asc)."""
    team_col  = "posteam" if side == "off" else "defteam"
    asc       = side == "def"
    df_team   = off if side == "off" else defn
    cfg = [
        ("PERSONAL OFENSIVO" + (" PROPIO" if side == "off" else " RIVAL"),
         "off_pkg", OFF_PKG_ORDER, OFF_PKG_LABEL),
        ("PERSONAL DEFENSIVO" + (" RIVAL" if side == "off" else " PROPIO"),
         "def_pkg", DEF_PKG_ORDER, {}),
        ("COBERTURAS" + (" QUE LE PRESENTAN" if side == "off" else " QUE JUEGA"),
         "defense_coverage_type", COV_ORDER, COV_LABELS),
        ("HOMBRE / ZONA", "man_zone", MZ_ORDER, {}),
        ("PRESIÓN" + (" SUFRIDA" if side == "off" else " GENERADA"),
         "pressure", PRES_ORDER, PRES_LABEL),
    ]
    out = []
    for titulo, col, orden, lmap in cfg:
        tabla = facet_table(all_plays, team_col, col, team, ascending=asc)
        uso   = snap_pct(df_team, col)
        items = []
        for cat in orden:
            if cat not in tabla:
                continue
            d = tabla[cat]
            items.append(dict(cat=cat, label=lmap.get(cat, cat),
                              uso=uso.get(cat, 0.0), **d))
        if items:
            out.append((titulo, items))
    return out


# ── KPIs DE CABECERA ──────────────────────────────────────────────────────────
def kpis(side):
    if side == "off":
        base = [("EPA / JUGADA", all_plays.groupby("posteam")["epa"].mean(), False, "epa"),
                ("EPA / PASE",   pases_lg.groupby("posteam")["epa"].mean(),  False, "epa"),
                ("EPA / CARRERA", carreras_lg.groupby("posteam")["epa"].mean(), False, "epa"),
                ("PRESIÓN SUFRIDA",
                 pases_lg.groupby("posteam")["was_pressure"].mean() * 100, True, "pct")]
    else:
        base = [("EPA PERMITIDO", all_plays.groupby("defteam")["epa"].mean(), True, "epa"),
                ("EPA/PASE PERM.", pases_lg.groupby("defteam")["epa"].mean(), True, "epa"),
                ("EPA/CARR. PERM.", carreras_lg.groupby("defteam")["epa"].mean(), True, "epa"),
                ("PRESIÓN GENERADA",
                 pases_lg.groupby("defteam")["was_pressure"].mean() * 100, False, "pct")]
    out = []
    for nombre, serie, asc, fmt in base:
        rank, nt, val = team_rank(serie, team, ascending=asc)
        out.append(dict(nombre=nombre, rank=rank, n_teams=nt, val=val, fmt=fmt))
    return out


# ── CLAVES AUTOMÁTICAS ────────────────────────────────────────────────────────
def claves(secs, n_top=3):
    """(fortalezas, debilidades) por ranking, entre facetas con uso suficiente.
    Umbral por fracción del nº REAL de equipos con muestra en esa faceta
    (un rank 5 de 9 equipos no es ni élite ni cola)."""
    pool = []
    for titulo, items in secs:
        for it in items:
            if it["uso"] >= MIN_USO and it["n"] >= MIN_SNAPS:
                pool.append(dict(it, seccion=titulo))
    corte = lambda nt: int(np.ceil(nt * 0.25))
    fort = sorted([p for p in pool if p["rank"] <= corte(p["n_teams"])],
                  key=lambda d: d["rank"] / d["n_teams"])[:n_top]
    debs = sorted([p for p in pool
                   if p["rank"] >= p["n_teams"] - corte(p["n_teams"]) + 1],
                  key=lambda d: -d["rank"] / d["n_teams"])[:n_top]
    return fort, debs


# ── DIBUJO ────────────────────────────────────────────────────────────────────
HALO = [pe.withStroke(linewidth=2.2, foreground=CARD)]


def draw_informe(side, outfile):
    secs = secciones(side)
    kpi  = kpis(side)
    fort, debs = claves(secs)
    es_off = side == "off"

    fig, ax = plt.subplots(figsize=(15, 11), facecolor=BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # ── Cabecera ──────────────────────────────────────────────────────────────
    ax.add_patch(plt.Rectangle((0, 91.5), 100, 8.5, color=CARD, zorder=0))
    logo = load_logo(team, base_zoom=0.085)
    if logo is not None:
        ab = AnnotationBbox(logo, (5.5, 95.7), frameon=False, zorder=3)
        ax.add_artist(ab)
    ax.text(50, 97.4, f"{team}  ·  INFORME {'OFENSIVO' if es_off else 'DEFENSIVO'}  ·  NFL {SEASON}",
            ha="center", va="center", fontsize=16, fontweight="bold", color=FG)
    ax.text(50, 93.4,
            "Cada faceta situada en su ranking de liga  ·  izquierda = top NFL  ·  "
            "tamaño de fuente del uso = peso real de esa situación",
            ha="center", va="center", fontsize=8, color="#888888", fontstyle="italic")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kx = [11, 30, 49, 68]
    for (x, k) in zip(kx, kpi):
        ax.add_patch(plt.Rectangle((x - 8.5, 83.2), 17, 6.9, color=CARD, zorder=1))
        if k["fmt"] == "epa":
            vtxt = f"{k['val']:+.3f}"
        else:
            vtxt = f"{k['val']:.1f}%"
        ax.text(x, 88.2, k["nombre"], ha="center", va="center",
                fontsize=7.5, color="#9aa3b5", zorder=2)
        ax.text(x - 2.2, 85.4, vtxt, ha="center", va="center", fontsize=13,
                fontweight="bold", color=FG, zorder=2)
        col = rank_color(k["rank"], k["n_teams"])
        ax.add_patch(plt.Circle((x + 4.6, 85.4), 1.55, color=col, zorder=2))
        ax.text(x + 4.6, 85.4, f"{k['rank']}" if k["rank"] else "—",
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="#0a0e13", zorder=3)
    ax.text(84.5, 86.8, "1 = mejor\nde 32", ha="left", va="center",
            fontsize=7, color="#666666", linespacing=1.4)

    # ── Columna izquierda: pistas de ranking ──────────────────────────────────
    ax.add_patch(plt.Rectangle((1.2, 1.5), 60.6, 80.2, color=CARD, zorder=0))
    x_lbl, x_t0, x_t1, x_val = 14.5, 17.5, 46.0, 47.5

    # Espaciado dinámico: todo debe caber entre y_top y y_bot pase lo que pase
    y_top, y_bot = 79.5, 3.6
    n_items    = sum(len(items) for _, items in secs)
    cab, gap   = 3.4, 1.1                      # cabecera de sección + hueco
    row_h = (y_top - y_bot - len(secs) * (cab + gap)) / max(n_items, 1)
    row_h = min(max(row_h, 2.4), 3.6)

    y = y_top
    for titulo, items in secs:
        ax.text(2.5, y, titulo, ha="left", va="center", fontsize=8.5,
                fontweight="bold", color="#8fa1c0", zorder=3)
        y -= cab * 0.55
        for frac, lbl in [(0.0, "mejor"), (1.0, "peor")]:
            ax.text(x_t0 + frac * (x_t1 - x_t0), y, lbl, ha="center",
                    va="center", fontsize=6, color="#555555", zorder=3)
        y -= cab * 0.45
        for it in items:
            # etiqueta con uso: tamaño de fuente ∝ uso real
            fs = 7.0 + min(it["uso"], 45) * 0.055
            ax.text(x_lbl, y + row_h * 0.10, f"{it['label']}", ha="right",
                    va="center", fontsize=fs, color=FG, zorder=3)
            ax.text(x_lbl, y - row_h * 0.34, f"{it['uso']:.0f}% uso",
                    ha="right", va="center", fontsize=6, color="#777777", zorder=3)
            # pista
            ax.plot([x_t0, x_t1], [y, y], color="#252c3b", linewidth=3.5,
                    zorder=2, solid_capstyle="round")
            for frac in [0.0, 0.5, 1.0]:
                xx = x_t0 + frac * (x_t1 - x_t0)
                ax.plot([xx, xx], [y - 0.55, y + 0.55], color="#333a49",
                        linewidth=0.8, zorder=2)
            # punto en el rank
            if it["n_teams"] > 1:
                fx = (it["rank"] - 1) / (it["n_teams"] - 1)
            else:
                fx = 0.5
            px = x_t0 + fx * (x_t1 - x_t0)
            col = rank_color(it["rank"], it["n_teams"])
            ax.add_patch(plt.Circle((px, y), 1.1, color=col, zorder=4))
            ax.text(px, y, f"{it['rank']}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="#0a0e13", zorder=5)
            # valor EPA + n a la derecha (con nº de equipos si no son 32)
            extra = f" · de {it['n_teams']} eq." if it["n_teams"] < 30 else ""
            ax.text(x_val, y + row_h * 0.16, f"{it['epa']:+.2f} EPA",
                    ha="left", va="center", fontsize=7.5, fontweight="bold",
                    color=col, zorder=3)
            ax.text(x_val, y - row_h * 0.28,
                    f"liga {it['lg']:+.2f} · n={it['n']}{extra}",
                    ha="left", va="center", fontsize=6, color="#777777", zorder=3)
            y -= row_h
        y -= gap

    # ── Columna derecha: claves + identidad ───────────────────────────────────
    def card(y0, y1, titulo, color):
        ax.add_patch(plt.Rectangle((63.2, y0), 35.4, y1 - y0, color=CARD, zorder=0))
        ax.text(64.6, y1 - 1.9, titulo, ha="left", va="center", fontsize=9.5,
                fontweight="bold", color=color, zorder=3)

    def linea_clave(yy, it, color):
        verbo_uso = "uso" if it["seccion"].startswith(("PERSONAL OFENSIVO PROPIO",
                                                       "PERSONAL DEFENSIVO PROPIO",
                                                       "COBERTURAS QUE JUEGA")) else "visto"
        ax.text(64.6, yy, f"{it['label']}", ha="left", va="center",
                fontsize=9, fontweight="bold", color=FG, zorder=3)
        ax.add_patch(plt.Circle((96.2, yy), 1.15, color=rank_color(it["rank"], it["n_teams"]),
                                zorder=4))
        ax.text(96.2, yy, f"{it['rank']}", ha="center", va="center", fontsize=7,
                fontweight="bold", color="#0a0e13", zorder=5)
        if it["n_teams"] < 30:
            ax.text(96.2, yy - 2.0, f"de {it['n_teams']}", ha="center",
                    va="center", fontsize=5.5, color="#777777", zorder=5)
        ax.text(64.6, yy - 1.6,
                f"{it['epa']:+.2f} EPA (liga {it['lg']:+.2f})  ·  "
                f"{it['uso']:.0f}% {verbo_uso}  ·  n={it['n']}",
                ha="left", va="center", fontsize=6.8, color="#9aa3b5", zorder=3)

    # Alturas dinámicas: DOMINA/SUFRE ocupan solo lo que necesitan sus items;
    # todo el espacio que sobra se lo queda IDENTIDAD (que es la más densa)
    TITLE_H, ROW_H, PAD_BOT, GAP = 4.2, 6.6, 1.3, 1.6
    def card_h(items):
        body = len(items) * ROW_H if items else 3.5
        return TITLE_H + body + PAD_BOT

    y_top = 81.7
    h_dom = card_h(fort)
    y0_dom = y_top - h_dom
    card(y0_dom, y_top, "✓  DONDE DOMINA", "#06d6a0")
    yy = y_top - TITLE_H - 1.5
    if not fort:
        ax.text(64.6, yy, "Sin facetas top-8 con uso relevante", ha="left",
                va="center", fontsize=8, color="#777777")
    for it in fort:
        linea_clave(yy, it, "#06d6a0")
        yy -= ROW_H

    y1_suf = y0_dom - GAP
    h_suf = card_h(debs)
    y0_suf = y1_suf - h_suf
    card(y0_suf, y1_suf, "✗  DONDE SUFRE", "#d84a4a")
    yy = y1_suf - TITLE_H - 1.5
    if not debs:
        ax.text(64.6, yy, "Sin facetas en la cola con uso relevante", ha="left",
                va="center", fontsize=8, color="#777777")
    for it in debs:
        linea_clave(yy, it, "#d84a4a")
        yy -= ROW_H

    # Identidad: se queda con todo el espacio restante hasta el pie
    y1_id, y0_id = y0_suf - GAP, 1.5
    card(y0_id, y1_id, "IDENTIDAD", "#8fa1c0")
    yy = y1_id - TITLE_H - 1.5
    if es_off:
        uso_pkg  = snap_pct(off, "off_pkg")
        lg_pkg   = snap_pct(all_plays, "off_pkg")

        def pct_propio(df_team, col):
            """% de jugadas propias con esa cualidad (decisión del equipo,
            no lo que le hace el rival) + su media de liga."""
            t  = df_team[col].mean() * 100 if col in df_team else np.nan
            lg = all_plays[col].mean() * 100
            return (0.0 if pd.isna(t) else t), lg

        sg_t,  sg_lg  = pct_propio(off, "shotgun")
        nh_t,  nh_lg  = pct_propio(off, "no_huddle")
        pa_t,  pa_lg  = pct_propio(off, "is_play_action")
        uc_t,  uc_lg  = (100 - sg_t if sg_t else np.nan), (100 - sg_lg)
        top_pkgs = sorted(
            [p for p in OFF_PKG_ORDER if uso_pkg.get(p, 0) >= 5],
            key=lambda p: -uso_pkg.get(p, 0))
        pares = [("Shotgun", sg_t, sg_lg),
                 ("Bajo centro", uc_t, uc_lg),
                 ("No-huddle", nh_t, nh_lg),
                 ("Play-action", pa_t, pa_lg)] + [
                 (f"Personal {p}", uso_pkg.get(p, 0), lg_pkg.get(p, 0))
                 for p in top_pkgs]
    else:
        uso_pkg  = snap_pct(defn, "def_pkg")
        lg_pkg   = snap_pct(all_plays, "def_pkg")
        jug_mz   = snap_pct(defn, "man_zone")
        lg_mz    = snap_pct(all_plays, "man_zone")
        pr       = snap_pct(defn, "pressure")
        lg_pr    = snap_pct(all_plays, "pressure")
        pares = [("Juega zona", jug_mz.get("Zona", 0), lg_mz.get("Zona", 0)),
                 ("Genera presión", pr.get("Bajo presion", 0), lg_pr.get("Bajo presion", 0))] + [
                 (f"Juega {p}", uso_pkg.get(p, 0), lg_pkg.get(p, 0))
                 for p in DEF_PKG_ORDER if uso_pkg.get(p, 0) >= 5]

    # Nº de filas que caben realmente en el hueco disponible
    n_rows_max = max(int((yy - y0_id - 2.0) / 3.9) + 1, 1)
    for nombre, pct, lg in pares[:n_rows_max]:
        ax.text(64.6, yy, nombre, ha="left", va="center", fontsize=8,
                color=FG, zorder=3)
        # barra de uso vs liga
        bx0, bx1 = 78.5, 93.5
        ax.plot([bx0, bx1], [yy, yy], color="#252c3b", linewidth=4.5,
                zorder=2, solid_capstyle="round")
        ax.plot([bx0, bx0 + (bx1 - bx0) * min(pct, 100) / 100], [yy, yy],
                color="#2d6cdf", linewidth=4.5, zorder=3, solid_capstyle="round")
        lx = bx0 + (bx1 - bx0) * min(lg, 100) / 100
        ax.plot([lx, lx], [yy - 0.8, yy + 0.8], color="white", linewidth=1.4,
                zorder=4, alpha=0.85)
        ax.text(94.5, yy, f"{pct:.0f}%", ha="left", va="center", fontsize=7.5,
                fontweight="bold", color=FG, zorder=3)
        yy -= 3.9
    ax.text(64.6, y0_id + 0.9, "barra azul = este equipo  ·  marca blanca = media NFL",
            ha="left", va="center", fontsize=6.3, color="#666666",
            fontstyle="italic", zorder=3)

    # ── Pie ───────────────────────────────────────────────────────────────────
    fig.text(0.012, 0.008,
             f"Fuente: nflverse PBP + NGS participation  |  {sello(SEASON)}  |  "
             f"mín. {MIN_SNAPS} snaps por faceta",
             ha="left", va="bottom", fontsize=7.5, color="#555555",
             fontstyle="italic")
    fig.text(0.988, 0.008, "@CuartayDato", ha="right", va="bottom", fontsize=9,
             color="#888888", alpha=0.85, fontstyle="italic")

    fig.savefig(outfile, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {outfile}")


draw_informe("off", salida(f"informe_ataque_{team}_{SEASON}.png", SEASON))
draw_informe("def", salida(f"informe_defensa_{team}_{SEASON}.png", SEASON))
