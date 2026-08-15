"""
informe_equipo.py
Informe completo de un equipo NFL: dos PNGs (ataque y defensa).

Rediseño jul-2026 — "team card" presentable:
  - Banda superior: 3 KPIs con su ranking + DONDE DOMINA en fila
  - Columna izquierda: cada faceta como punto sobre una pista de ranking 1→32
    (un solo eje universal: izquierda = top de la liga, derecha = cola)
  - Columna derecha: debilidades, identidad y un bloque PRESION que reune
    el KPI, el % y el origen (mini-campo con las cuatro flechas)
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
        # La presión ya no es una faceta suelta al final de la columna: vive en
        # el bloque PRESIÓN de la derecha, junto al KPI y al origen
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


# ── ORIGEN DE LA PRESIÓN ──────────────────────────────────────────────────────
ORIGENES = ["INT", "EXT", "LB", "DB"]
ORIGEN_LABEL = {"INT": "Interior", "EXT": "Exterior",
                "LB": "Blitz LB", "DB": "Blitz DB"}
ORIGEN_COLOR = {"INT": "#2d6cdf", "EXT": "#06d6a0",
                "LB": "#ffd166", "DB": "#d84a4a"}

_MAPA_DEPTH = {
    "DT": "INT", "NT": "INT",
    "DE": "EXT", "OLB": "EXT", "EDGE": "EXT", "RUSH": "EXT",
    "JACK": "EXT", "LEO": "EXT",
    "LB": "LB", "ILB": "LB", "MLB": "LB", "WLB": "LB", "SLB": "LB",
    "MIKE": "LB", "WILL": "LB", "SAM": "LB",
    "CB": "DB", "DB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "NB": "DB",
}
_MAPA_POS = {"DT": "INT", "NT": "INT", "DE": "EXT", "OLB": "EXT", "EDGE": "EXT",
             "LB": "LB", "ILB": "LB", "MLB": "LB",
             "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB"}
PESO_INTERIOR = 280   # un DE de 3-4 juega por dentro; ver dline_presion_origen


def _clasificar_rusher(depth, pos, peso=None):
    d = depth.upper() if isinstance(depth, str) else ""
    if d == "DE" and peso and float(peso) >= PESO_INTERIOR:
        return "INT"
    if d in _MAPA_DEPTH:
        return _MAPA_DEPTH[d]
    if isinstance(pos, str) and pos.upper() in _MAPA_POS:
        return _MAPA_POS[pos.upper()]
    return None


def _mapa_origen():
    """gsis_id → origen, por depth chart (única fuente fiable para edge/interior)."""
    from pbp_loader import cargar_rosters
    ros, _ = cargar_rosters(SEASON)
    m = {}
    for _, r in ros.iterrows():
        gid = r.get("gsis_id")
        if pd.isna(gid):
            continue
        cl = _clasificar_rusher(r.get("depth_chart_position"), r.get("position"),
                                r.get("weight"))
        if cl:
            m[gid] = cl
    return m


def origen_presion(side):
    """Reparto de la presión por origen, del equipo y de la liga.

    Cada cara usa la misma fuente que su script dedicado, para que los números
    del informe coincidan con el gráfico grande:
      - defensa: presiones reales de PFR      -> igual que dline_presion_origen
      - ataque:  atribución de sacks+QB hits  -> igual que oline_presion_origen
        (los hurries no traen autor en datos públicos, por eso aquí es
        atribución y no presión completa)
    """
    try:
        if side == "def":
            from pbp_loader import cargar_pfr, cargar_rosters
            pfr, _ = cargar_pfr("def", SEASON)
            pfr = pfr[pfr["tm"] != "3TM"].copy()
            pfr["prss"] = pd.to_numeric(pfr["prss"], errors="coerce").fillna(0)
            ros, _ = cargar_rosters(SEASON)
            por_nombre = {}
            for _, r in ros.iterrows():
                cl = _clasificar_rusher(r.get("depth_chart_position"),
                                        r.get("position"), r.get("weight"))
                if cl:
                    por_nombre.setdefault(_clave_nombre(r.get("full_name")), cl)
            pfr["origen"] = pfr["player"].map(_clave_nombre).map(por_nombre)
            ev = pfr.dropna(subset=["origen"])
            equipo_col, mi = "tm", team
            tot = ev.groupby([equipo_col, "origen"])["prss"].sum().unstack(fill_value=0)
        else:
            # Quién presionó a este ataque: autores de sacks y QB hits
            m = _mapa_origen()
            filas = []
            for col_id, col_eq in [("sack_player_id", "posteam"),
                                   ("qb_hit_1_player_id", "posteam"),
                                   ("qb_hit_2_player_id", "posteam"),
                                   ("half_sack_1_player_id", "posteam"),
                                   ("half_sack_2_player_id", "posteam")]:
                if col_id not in pbp.columns:
                    continue
                sub = pbp[pbp[col_id].notna()][[col_id, col_eq]].copy()
                sub.columns = ["pid", "equipo"]
                filas.append(sub)
            if not filas:
                return None
            ev = pd.concat(filas, ignore_index=True)
            ev["origen"] = ev["pid"].map(m)
            ev = ev.dropna(subset=["origen", "equipo"])
            tot = ev.groupby(["equipo", "origen"]).size().unstack(fill_value=0)
            mi = team

        tot = tot.reindex(columns=ORIGENES, fill_value=0)

        # TASA (por 100 dropbacks), no reparto. El reparto compara la
        # composición contra la de la liga y eso engaña: SF sacaba "60%
        # exterior vs 49% de la liga" (parece que presiona mucho por fuera)
        # cuando su tasa exterior es 10.1 contra 11.4 de la liga, o sea MENOS.
        # El 60% solo decía que su poca presión se concentra ahí.
        # Ventaja añadida: las cuatro tasas suman el KPI del bloque.
        col_eq = "posteam" if side == "off" else "defteam"
        drop = pbp[pd.to_numeric(pbp["qb_dropback"], errors="coerce") == 1] \
            .groupby(col_eq).size()
        tasa = tot.div(drop, axis=0).dropna(how="all") * 100
        if mi not in tasa.index:
            return None
        return {"equipo": {o: float(tasa.loc[mi, o]) for o in ORIGENES},
                "liga":   {o: float(tasa[o].mean()) for o in ORIGENES},
                "n":      int(tot.loc[mi].sum())}
    except Exception as e:
        print(f"  Aviso: sin origen de presión ({type(e).__name__}: {str(e)[:60]})")
        return None


def _clave_nombre(n):
    n = str(n).lower().replace(".", " ").replace("'", "").replace("-", " ")
    p = [x for x in n.split() if x not in ("jr", "sr", "ii", "iii", "iv", "v")]
    if not p:
        return ""
    return p[0] if len(p) == 1 else p[0][0] + " " + p[-1]


def pares_identidad(es_off):
    """Filas de IDENTIDAD: (nombre, % del equipo, % de la liga).

    Se calcula aparte del dibujo porque su numero de filas decide cuanto
    espacio le queda al bloque PRESION."""
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
    return pares


def presion_por_equipo(es_off):
    """(serie de presión por equipo, etiqueta). Misma fuente que el reparto.

    Defensa: presiones acreditadas (PFR) por dropback encajado.
    Ataque:  veces presionado el QB (PFR) por dropback.
    Se usa PFR y no `was_pressure` para que el KPI y el desglose por origen
    salgan del mismo pozo y el reparto divida de verdad ese número.
    """
    try:
        from pbp_loader import cargar_pfr
        drop = pbp[pd.to_numeric(pbp["qb_dropback"], errors="coerce") == 1]
        if es_off:
            pfr, _ = cargar_pfr("pass", SEASON)
            pfr = pfr[pfr["team"] != "3TM"].copy()
            pfr["times_pressured"] = pd.to_numeric(pfr["times_pressured"],
                                                   errors="coerce").fillna(0)
            tot = pfr.groupby("team")["times_pressured"].sum()
            den = drop.groupby("posteam").size()
            etiqueta = "presiones sufridas por 100 dropbacks"
        else:
            pfr, _ = cargar_pfr("def", SEASON)
            pfr = pfr[pfr["tm"] != "3TM"].copy()
            pfr["prss"] = pd.to_numeric(pfr["prss"], errors="coerce").fillna(0)
            tot = pfr.groupby("tm")["prss"].sum()
            den = drop.groupby("defteam").size()
            etiqueta = "presiones por 100 dropbacks"
        return (tot / den * 100).dropna(), etiqueta
    except Exception as e:
        print(f"  Aviso: KPI de presión desde PBP ({type(e).__name__})")
        col = "posteam" if es_off else "defteam"
        return (pases_lg.groupby(col)["was_pressure"].mean() * 100,
                "de los dropbacks")


def dibujar_presion(ax, side, y0, y1, card):
    """Bloque PRESIÓN: el KPI, y bajo él un mini-campo con las cuatro flechas.

    Versión reducida del diagrama de dline_presion_origen / oline_presion_origen:
    sin línea ofensiva ni nombres, solo el QB y los cuatro orígenes con su %.
    El grosor de la flecha es el % del reparto, igual que en el grande.
    """
    import matplotlib.patches as mpatches
    from matplotlib.path import Path as MPath

    es_off = side == "off"
    titulo = "PRESIÓN SUFRIDA" if es_off else "PRESIÓN GENERADA"
    card(y0, y1, titulo, "#e0a458")

    # KPI de presión desde PFR, la MISMA fuente que el reparto de abajo: con
    # was_pressure (% de dropbacks presionados) el KPI y el desglose salían de
    # pozos distintos y no se podía decir "de esa presión, el 60% viene por
    # fuera" — dos jugadores pueden acreditarse presión en la misma jugada.
    serie, etiqueta = presion_por_equipo(es_off)
    rank, nt, val = team_rank(serie, team, ascending=es_off)
    ax.text(64.6, y1 - 5.6, f"{val:.1f}" if pd.notna(val) else "N/D",
            ha="left", va="center", fontsize=15, fontweight="bold",
            color=FG, zorder=3)
    ax.text(70.6, y1 - 5.6, etiqueta, ha="left", va="center",
            fontsize=6.8, color="#9aa3b5", zorder=3)
    if rank:
        ax.add_patch(plt.Circle((85.0, y1 - 5.6), 1.5,
                                color=rank_color(rank, nt), zorder=3))
        ax.text(85.0, y1 - 5.6, str(rank), ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#0a0e13", zorder=4)
        ax.text(87.4, y1 - 5.6, f"de {nt}", ha="left", va="center",
                fontsize=6.5, color="#767E90", zorder=3)

    datos = origen_presion(side)
    if not datos:
        ax.text(64.6, y0 + 6.0, "Origen de la presión no disponible",
                ha="left", va="center", fontsize=7.5, color="#777777", zorder=3)
        return

    # ── Mini-campo apaisado ───────────────────────────────────────────────
    # La geometría se calcula sobre el alto REAL del bloque: con posiciones
    # fijas desde y0, el campo quedaba aplastado abajo y las etiquetas se
    # pisaban entre sí.
    top_c = y1 - 8.6          # bajo el KPI
    bot_c = y0 + 2.8          # sobre la nota del pie
    h_c   = max(top_c - bot_c, 6.0)

    qx, qy = 81.0, bot_c + h_c * 0.06
    ax.add_patch(plt.Circle((qx, qy), 1.7, color="#E5C070", zorder=8))
    ax.text(qx, qy, "QB", ha="center", va="center", color=BG,
            fontsize=7, fontweight="bold", zorder=9)

    # Los cuatro orígenes en FILA a la misma altura, no en arco: en 35 unidades
    # de ancho el arco hacía que las etiquetas se pisaran con las flechas y que
    # "Exterior" se saliera por el borde derecho de la tarjeta.
    X_ORI = {"DB": 67.4, "LB": 76.2, "INT": 85.0, "EXT": 93.8}
    y_ori = bot_c + h_c * 0.80

    for o in ORIGENES:
        px, py = X_ORI[o], y_ori
        # El ángulo de llegada se deduce de la posición: siempre correcto,
        # sin tablas de ángulos que reajustar si se mueve un origen
        ang_deg = np.degrees(np.arctan2(py - qy, px - qx))
        pct  = datos["equipo"][o]
        lg   = datos["liga"][o]
        col  = ORIGEN_COLOR[o]
        # El grosor es la TASA, no el reparto: así la tinta total del abanico
        # es proporcional a la presión real y una defensa floja tiene las
        # cuatro flechas finas, en vez de una gorda por concentración
        lw   = 0.8 + 5.2 * min(pct / 14.0, 1.0)
        ang  = np.radians(ang_deg)
        ux, uy = np.cos(ang), np.sin(ang)
        destino = np.array([qx, qy]) + np.array([ux, uy]) * 2.1
        ctrl = (qx + (px - qx) * 0.85, qy + h_c * 0.30)
        ax.add_patch(mpatches.PathPatch(
            MPath([(px, py), ctrl, tuple(destino)],
                  [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]),
            facecolor="none", edgecolor=col, linewidth=lw, zorder=3,
            capstyle="round", alpha=0.95))
        perp = np.array([-uy, ux])
        tip  = destino - np.array([ux, uy]) * 0.15
        base = tip + np.array([ux, uy]) * (0.7 + lw * 0.09)
        an   = 0.34 + lw * 0.07
        ax.add_patch(plt.Polygon([tip, base + perp * an, base - perp * an],
                                 color=col, zorder=6))
        # Etiqueta sobre el origen: nombre, % y REFERENCIA de la liga.
        # Se muestra el valor de la liga, no la diferencia: un "-4 liga" no
        # dice la unidad (son puntos porcentuales), no da la referencia y se
        # lee tan fácil como "la liga es -4". El color ya indica si supera la
        # media, así que la resta no aporta nada.
        ax.text(px, py + 3.4, ORIGEN_LABEL[o], ha="center", va="center",
                fontsize=6.8, fontweight="bold", color=FG, zorder=9)
        ax.text(px, py + 1.9, f"{pct:.1f}", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=col, zorder=9)
        ax.text(px, py + 0.6, f"liga {lg:.1f}", ha="center", va="center",
                fontsize=6, zorder=9,
                color="#06d6a0" if pct >= lg else "#767E90")

    nota = ("atribución de sacks y QB hits"
            if es_off else "presiones reales (PFR)")
    ax.text(64.6, y0 + 0.9,
            f"presiones por 100 dropbacks — las cuatro suman el total  ·  "
            f"{nota}  ·  n={datos['n']}",
            ha="left", va="center", fontsize=6.3, color="#666666",
            fontstyle="italic", zorder=3)


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

    # ── Cabecera (comprimida: era 8.5 de alto para un logo y dos líneas) ──────
    ax.add_patch(plt.Rectangle((0, 94.6), 100, 5.4, color=CARD, zorder=0))
    logo = load_logo(team, base_zoom=0.062)
    if logo is not None:
        ab = AnnotationBbox(logo, (4.6, 97.3), frameon=False, zorder=3)
        ax.add_artist(ab)
    ax.text(50, 98.2, f"{team}  ·  INFORME {'OFENSIVO' if es_off else 'DEFENSIVO'}  ·  NFL {SEASON}",
            ha="center", va="center", fontsize=15, fontweight="bold", color=FG)
    ax.text(50, 95.7,
            "Cada faceta situada en su ranking de liga  ·  izquierda = top NFL  ·  "
            "tamaño de fuente del uso = peso real de esa situación",
            ha="center", va="center", fontsize=7.5, color="#888888", fontstyle="italic")

    # ── Banda superior: 3 KPIs (izquierda) + DONDE DOMINA (derecha) ───────────
    # El 4º KPI era la presión y se ha ido al bloque PRESIÓN del final, donde
    # está el resto del tema. La leyenda "1 = mejor / de 32" se quitó: el badge
    # verde ya se entiende solo y ocupaba un hueco que ahora usa DONDE DOMINA.
    Y0_BANDA, Y1_BANDA = 84.2, 93.6
    kx = [11.5, 31.5, 51.5]
    for (x, k) in zip(kx, kpi):
        ax.add_patch(plt.Rectangle((x - 9.3, Y0_BANDA), 18.6, Y1_BANDA - Y0_BANDA,
                                   color=CARD, zorder=1))
        vtxt = f"{k['val']:+.3f}" if k["fmt"] == "epa" else f"{k['val']:.1f}%"
        ax.text(x, Y1_BANDA - 2.0, k["nombre"], ha="center", va="center",
                fontsize=7.5, color="#9aa3b5", zorder=2)
        ax.text(x - 2.4, Y0_BANDA + 3.1, vtxt, ha="center", va="center",
                fontsize=13.5, fontweight="bold", color=FG, zorder=2)
        col = rank_color(k["rank"], k["n_teams"])
        ax.add_patch(plt.Circle((x + 5.0, Y0_BANDA + 3.1), 1.6, color=col, zorder=2))
        ax.text(x + 5.0, Y0_BANDA + 3.1, f"{k['rank']}" if k["rank"] else "—",
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="#0a0e13", zorder=3)

    # DONDE DOMINA: las 3 facetas en fila, no apiladas — en la banda no caben
    # a 6.6 de alto cada una, pero en horizontal entran de sobra
    ax.add_patch(plt.Rectangle((63.2, Y0_BANDA), 35.4, Y1_BANDA - Y0_BANDA,
                               color=CARD, zorder=1))
    ax.text(64.6, Y1_BANDA - 1.9, "✓  DONDE DOMINA", ha="left", va="center",
            fontsize=9, fontweight="bold", color="#06d6a0", zorder=3)
    if not fort:
        ax.text(64.6, Y0_BANDA + 3.0, "Sin facetas top-8 con uso relevante",
                ha="left", va="center", fontsize=7.5, color="#777777", zorder=3)
    else:
        ancho = 35.4 / max(len(fort), 1)
        for i, it in enumerate(fort):
            cx = 63.2 + ancho * (i + 0.5)
            ax.text(cx, Y0_BANDA + 4.3, it["label"][:16], ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=FG, zorder=3)
            ax.add_patch(plt.Circle((cx - 3.4, Y0_BANDA + 1.8), 1.15,
                                    color=rank_color(it["rank"], it["n_teams"]), zorder=4))
            ax.text(cx - 3.4, Y0_BANDA + 1.8, f"{it['rank']}", ha="center",
                    va="center", fontsize=7, fontweight="bold", color="#0a0e13", zorder=5)
            ax.text(cx - 1.6, Y0_BANDA + 1.8,
                    f"{it['epa']:+.2f} EPA · {it['uso']:.0f}%",
                    ha="left", va="center", fontsize=6.8, color="#9aa3b5", zorder=3)

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

    # DONDE DOMINA ya se ha dibujado arriba, en la banda de los KPIs
    y1_suf = 82.9
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

    # IDENTIDAD toma solo lo que necesitan sus filas y el bloque PRESIÓN se
    # queda TODO el resto: con altura fija a PRESIÓN le faltaba aire para el
    # mini-campo mientras a IDENTIDAD le sobraba media tarjeta.
    pares = pares_identidad(es_off)
    ROW_ID = 3.9
    y1_id  = y0_suf - GAP
    h_id   = TITLE_H + len(pares) * ROW_ID + 2.6
    y0_id  = max(y1_id - h_id, 30.5)          # suelo: PRESIÓN nunca baja de ~28

    # Bloque PRESIÓN: reúne el KPI, el % y el origen, que antes estaban en tres
    # sitios alejados de la tarjeta (cabecera, columna izquierda y ninguno)
    dibujar_presion(ax, side, 1.5, y0_id - GAP, card)

    card(y0_id, y1_id, "IDENTIDAD", "#8fa1c0")
    yy = y1_id - TITLE_H - 1.5
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
