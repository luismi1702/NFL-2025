"""
informe_equipo.py
Informe completo de un equipo NFL: dos PNGs (ataque y defensa).

Rediseño jul-2026 — "team card" presentable:
  - Banda superior: 3 KPIs con su ranking + DONDE DOMINA en fila; debajo,
    DONDE SUFRE con el mismo formato (las dos son la misma clase de dato)
  - Columna izquierda: cada faceta como punto sobre una pista de ranking 1→32
    (un solo eje universal: izquierda = top de la liga, derecha = cola)
  - Columna izquierda: al pie, franja de CARRERA POR HUECO. En ataque son los
    huecos propios (LE->RE); en defensa se nombra al DEFENSOR que cubre cada
    uno y se invierte el orden, porque la izquierda del ataque es la derecha
    de la defensa
  - Columna derecha: debilidades, identidad y un bloque PRESION que reune
    el KPI, el % y el origen (mini-campo con las cuatro flechas)
  - DONDE DOMINA / SUFRE tiene 19 candidatos OCULTOS que no se dibujan en
    ninguna seccion: solo salen si el equipo es extremo, para que el resumen
    aporte algo que no esta ya a la vista. Requisito para entrar: que el
    ranking signifique MEJOR o PEOR — PROE o el uso de personal no valen,
    porque ahi ser primero no es bueno, es distinto (eso va en IDENTIDAD)
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


def color_epa(valor, escala, mas_es_mejor):
    """Color por VALOR de EPA, no por puesto.

    En la franja de huecos el ranking engañaba: donde toda la liga defiende
    parecido, un -0.07 y un -0.09 quedaban a 12 puestos de distancia y salían
    de colores muy distintos pese a ser el mismo rendimiento.
    """
    if not escala or pd.isna(valor):
        return "#3a4050"
    lo, hi = escala
    if hi <= lo:
        return RYG(0.5)
    t = float(np.clip((valor - lo) / (hi - lo), 0, 1))
    return RYG(t if mas_es_mejor else 1.0 - t)


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
    ftn = ftn[["nflverse_game_id", "nflverse_play_id", "is_play_action",
               "n_blitzers"]].rename(
        columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    merged = merged.merge(ftn, on=["game_id", "play_id"], how="left")
except Exception as e:
    print(f"  Aviso: FTN no disponible ({e}) — sin play-action%")
    merged["is_play_action"] = np.nan
    merged["n_blitzers"]     = np.nan

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
def presion_como_clave(es_off):
    """La presión, como candidato a DOMINA/SUFRE.

    Estaba en el pool cuando era una sección de la columna izquierda; al
    moverla al bloque de la derecha se quedó fuera sin querer. Si el punto
    fuerte de un equipo es generar presión, tiene que poder salir arriba.
    """
    try:
        serie, _ = presion_por_equipo(es_off)
        rank, nt, val = team_rank(serie, team, ascending=es_off)
        if rank is None or pd.isna(val):
            return None
        return dict(label="Presión sufrida" if es_off else "Presión generada",
                    cat="presion", epa=float(val), lg=float(serie.mean()),
                    rank=rank, n_teams=nt, n=MIN_SNAPS, uso=100.0,
                    seccion="PRESIÓN", es_pct=True)
    except Exception:
        return None


def metricas_extra(es_off):
    """Candidatos OCULTOS para DONDE DOMINA / DONDE SUFRE.

    No se dibujan en ninguna sección: solo aparecen arriba si el equipo es
    extremo en ellas. La idea es que la tarjeta pueda decir algo que no está
    ya a la vista — antes DOMINA/SUFRE era un resumen de la columna izquierda
    y por tanto no aportaba información nueva.

    Son las métricas de equipo de otros scripts del catálogo (zona roja,
    3er/4º down, explosivas, turnovers, play-action, three-and-out), todas
    calculables con el PBP que el informe ya tiene cargado.
    """
    col = "posteam" if es_off else "defteam"
    p   = all_plays
    fuera = []

    def añadir(label, serie, mejor_alto, n_serie=None, min_n=25):
        """mejor_alto=True → rank 1 al valor más alto."""
        s = serie.dropna()
        if n_serie is not None:
            s = s[n_serie.reindex(s.index).fillna(0) >= min_n]
        if team not in s.index or len(s) < 8:
            return
        orden = s.sort_values(ascending=not mejor_alto)
        fuera.append(dict(
            label=label, cat=label, epa=float(s.loc[team]),
            lg=float(s.mean()), rank=list(orden.index).index(team) + 1,
            n_teams=len(orden),
            n=int(n_serie.get(team, 0)) if n_serie is not None else MIN_SNAPS,
            uso=100.0, es_pct=label.endswith("%")))

    # Zona roja
    rz = p[p["yardline_100"].le(20)]
    añadir("EPA en zona roja", rz.groupby(col)["epa"].mean(), es_off,
           rz.groupby(col).size())

    # 3er down y 4º down
    for down, etiqueta, minimo in ((3, "3er down %", 40), (4, "4º down %", 12)):
        d = p[p["down"] == down]
        conv = pd.to_numeric(d.get(f"{'third' if down==3 else 'fourth'}_down_converted"),
                             errors="coerce")
        if conv is None or conv.isna().all():
            continue
        d = d.assign(_c=conv.fillna(0))
        añadir(etiqueta, d.groupby(col)["_c"].mean() * 100, es_off,
               d.groupby(col).size(), min_n=minimo)

    # Jugadas explosivas (pase 15+, carrera 10+)
    y = pd.to_numeric(p["yards_gained"], errors="coerce")
    exp = ((p["play_type"].eq("pass") & y.ge(15)) |
           (p["play_type"].eq("run") & y.ge(10)))
    añadir("Explosivas %", p.assign(_e=exp).groupby(col)["_e"].mean() * 100,
           es_off, p.groupby(col).size(), min_n=200)

    # Turnovers: EPA de intercepciones y balones sueltos perdidos
    to = p[(pd.to_numeric(p["interception"], errors="coerce").fillna(0) == 1) |
           (pd.to_numeric(p["fumble_lost"], errors="coerce").fillna(0) == 1)]
    if len(to):
        añadir("EPA en turnovers", to.groupby(col)["epa"].sum(), es_off,
               to.groupby(col).size(), min_n=8)

    # Play-action (solo lado ofensivo: es una decisión propia)
    if es_off and "is_play_action" in p.columns:
        pa = p[pd.to_numeric(p["is_play_action"], errors="coerce").fillna(0) == 1]
        añadir("EPA con play-action", pa.groupby(col)["epa"].mean(), True,
               pa.groupby(col).size(), min_n=60)

    # EPA ajustado por la calidad del rival (misma fórmula que
    # RankingEPAadjustado: restar la media del oponente a cada jugada).
    # Un equipo 8º en bruto y 3º ajustado tiene una historia que hoy no sale.
    rival = "defteam" if es_off else "posteam"
    media_rival = p.groupby(rival)["epa"].mean()
    aj = p["epa"] - p[rival].map(media_rival)
    añadir("EPA ajustado por rival", p.assign(_a=aj).groupby(col)["_a"].mean(),
           es_off, p.groupby(col).size(), min_n=200)

    # Clutch: 4º cuarto y prórroga con el partido a 7 puntos o menos
    cl = p[(pd.to_numeric(p["qtr"], errors="coerce") >= 4) &
           (pd.to_numeric(p["score_differential"], errors="coerce").abs() <= 7)]
    añadir("EPA en clutch", cl.groupby(col)["epa"].mean(), es_off,
           cl.groupby(col).size(), min_n=60)

    # Tendencia: últimas 4 jornadas contra el resto. Es la métrica más útil EN
    # temporada — responde "¿va de menos a más?", que no se ve en el acumulado.
    if "week" in p.columns:
        w = pd.to_numeric(p["week"], errors="coerce")
        ult = int(w.max()) if w.notna().any() else 0
        if ult >= 8:
            fin = p[w > ult - 4]
            ini = p[w <= ult - 4]
            delta = (fin.groupby(col)["epa"].mean() - ini.groupby(col)["epa"].mean())
            añadir("Tendencia últimas 4", delta, es_off,
                   fin.groupby(col).size(), min_n=100)

    # Success rate: % de jugadas con EPA positivo. Otra lente que el EPA medio
    # — un equipo puede tener buen EPA por cuatro explosivas y ser irregular.
    añadir("Success rate %", p.assign(_s=p["epa"] > 0).groupby(col)["_s"].mean() * 100,
           es_off, p.groupby(col).size(), min_n=200)

    # EPA en 1er down: marca el ritmo de toda la serie
    d1 = p[p["down"] == 1]
    añadir("EPA en 1er down", d1.groupby(col)["epa"].mean(), es_off,
           d1.groupby(col).size(), min_n=120)

    # Distancia media a superar en 3er down (consecuencia de lo anterior)
    d3 = p[p["down"] == 3]
    añadir("Distancia en 3er down", d3.groupby(col)["ydstogo"].mean(), not es_off,
           d3.groupby(col).size(), min_n=40)

    # Sack rate sobre dropbacks
    if "qb_dropback" in p.columns:
        dr = p[pd.to_numeric(p["qb_dropback"], errors="coerce").fillna(0) == 1]
        añadir("Sack rate %",
               dr.assign(_k=pd.to_numeric(dr["sack"], errors="coerce").fillna(0))
                 .groupby(col)["_k"].mean() * 100,
               not es_off, dr.groupby(col).size(), min_n=150)

    # Penalizaciones: yardas por jugada (en defensa, las que comete el rival)
    if "penalty_yards" in p.columns and "penalty_team" in p.columns:
        py = pd.to_numeric(p["penalty_yards"], errors="coerce").fillna(0)
        propias = p["penalty_team"] == p[col]
        añadir("Yds de penalización/jugada",
               p.assign(_p=py.where(propias, 0)).groupby(col)["_p"].mean(),
               False, p.groupby(col).size(), min_n=200)

    # Red zone TD%: el resultado concreto, distinto del EPA en zona roja
    if "touchdown" in p.columns:
        rz_td = rz.assign(_t=pd.to_numeric(rz["touchdown"], errors="coerce").fillna(0))
        añadir("TD% en zona roja", rz_td.groupby(col)["_t"].mean() * 100, es_off,
               rz.groupby(col).size(), min_n=40)

    # Two-minute: últimos 2 minutos de cada mitad
    if "half_seconds_remaining" in p.columns:
        tm = p[pd.to_numeric(p["half_seconds_remaining"], errors="coerce") <= 120]
        añadir("EPA en two-minute", tm.groupby(col)["epa"].mean(), es_off,
               tm.groupby(col).size(), min_n=50)

    # Contra blitz (5+ rushers, de FTN charting)
    if "n_blitzers" in p.columns:
        bl = p[pd.to_numeric(p["n_blitzers"], errors="coerce").fillna(0) >= 1]
        añadir("EPA contra blitz" if es_off else "EPA con blitz",
               bl.groupby(col)["epa"].mean(), es_off,
               bl.groupby(col).size(), min_n=60)

    # Three-and-out: series que acaban sin primer down
    if "fixed_drive" in p.columns:
        d = p.dropna(subset=["fixed_drive"])
        por_drive = d.groupby([col, "game_id", "fixed_drive"]).agg(
            jugadas=("epa", "size"),
            primeros=("first_down", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()))
        tres = (por_drive["primeros"] == 0) & (por_drive["jugadas"] <= 3)
        añadir("Three-and-out %", tres.groupby(level=0).mean() * 100,
               not es_off, por_drive.groupby(level=0).size(), min_n=80)

        # Drives que acaban en puntos: el resultado que de verdad importa
        if "drive_ended_with_score" in d.columns:
            pts = d.groupby([col, "game_id", "fixed_drive"])["drive_ended_with_score"].max()
            pts = pd.to_numeric(pts, errors="coerce").fillna(0)
            añadir("Drives con puntos %", pts.groupby(level=0).mean() * 100,
                   es_off, por_drive.groupby(level=0).size(), min_n=80)

    return fuera


def claves(secs, extra=(), n_top=3):
    """(fortalezas, debilidades) por ranking, entre facetas con uso suficiente.
    Umbral por fracción del nº REAL de equipos con muestra en esa faceta
    (un rank 5 de 9 equipos no es ni élite ni cola)."""
    pool = []
    for titulo, items in secs:
        for it in items:
            if it["uso"] >= MIN_USO and it["n"] >= MIN_SNAPS:
                pool.append(dict(it, seccion=titulo))
    for it in extra:
        pool.append(dict(it, seccion=it.get("seccion", "EXTRA")))

    corte = lambda nt: int(np.ceil(nt * 0.25))

    def top(cands, key):
        """Uno por sección mientras haya variedad, y si no se completa con los
        siguientes mejores.

        El tope existe porque los huecos son 7 celdas y los extras 6 métricas:
        sin él una sola familia se comía las 3 plazas. Pero aplicado a rajatabla
        dejaba tarjetas con una sola clave (o ninguna) cuando los candidatos se
        concentraban en pocas secciones, que es peor que repetir familia.
        """
        ordenados = sorted(cands, key=key)
        vistas, out = set(), []
        for c in ordenados:
            if c["seccion"] in vistas:
                continue
            vistas.add(c["seccion"])
            out.append(c)
            if len(out) == n_top:
                return out
        for c in ordenados:                      # relleno sin repetir faceta
            if c not in out:
                out.append(c)
                if len(out) == n_top:
                    break
        return out

    fort = top([p for p in pool if p["rank"] <= corte(p["n_teams"])],
               key=lambda d: d["rank"] / d["n_teams"])
    debs = top([p for p in pool
                if p["rank"] >= p["n_teams"] - corte(p["n_teams"]) + 1],
               key=lambda d: -d["rank"] / d["n_teams"])
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


# ── CARRERA POR HUECO ─────────────────────────────────────────────────────────
GAP_ORDER = ["LE", "LT", "LG", "C", "RG", "RT", "RE"]

# En ataque los huecos son los de la propia línea, de izquierda a derecha.
GAP_CORTO = {"LE": "LE", "LT": "LT", "LG": "LG", "C": "C",
             "RG": "RG", "RT": "RT", "RE": "RE"}

# En defensa hay que hacer dos cosas que no son evidentes:
#   1. Nombrar al DEFENSOR que cubre el hueco, no al liniero rival: el que
#      defiende el hueco del left guard es un DT, no un guard.
#   2. DAR LA VUELTA al orden, porque la izquierda del ataque es la derecha de
#      la defensa. El hueco "LT" del rival lo cubre el DE DERECHO nuestro.
# Es la misma traducción que ya hace DL_DISPLAY en run_gap_defensa.py.
GAP_ORDER_DEF = ["RE", "RT", "RG", "C", "LG", "LT", "LE"]   # izq→dcha DEFENSIVA
GAP_DEF_LABEL = {
    "RE": "Exterior\nizq",  "RT": "DE\nizq",  "RG": "DT\nizq",
    "C":  "NT",
    "LG": "DT\ndcha",       "LT": "DE\ndcha", "LE": "Exterior\ndcha",
}


def classify_gap(loc, gap):
    """Misma clasificación que run_gap.py, para que los dos coincidan."""
    if pd.isna(loc):
        return None
    if loc == "middle":
        return "C"
    if pd.isna(gap):
        return None
    if loc == "left":
        return {"end": "LE", "tackle": "LT", "guard": "LG"}.get(gap)
    if loc == "right":
        return {"guard": "RG", "tackle": "RT", "end": "RE"}.get(gap)
    return None


def carrera_por_hueco(es_off):
    """EPA/acarreo por hueco, con rank de liga. Solo acarreos DISEÑADOS:
    los scrambles del QB no dicen nada de cómo corre el equipo por un hueco."""
    try:
        col = "posteam" if es_off else "defteam"
        need = ["run_location", "run_gap", "qb_scramble", "play_type", "epa", col]
        if any(c not in pbp.columns for c in need):
            return None
        r = pbp[(pbp["play_type"] == "run") &
                (pd.to_numeric(pbp["qb_scramble"], errors="coerce").fillna(0) == 0) &
                pbp["epa"].notna()].copy()
        r["hueco"] = [classify_gap(l, g) for l, g in
                      zip(r["run_location"], r["run_gap"])]
        r = r[r["hueco"].notna()]
        if r.empty:
            return None

        out, todos = {}, []
        for h in GAP_ORDER:
            sub = r[r["hueco"] == h]
            g = sub.groupby(col)["epa"].agg(epa="mean", n="count")
            g = g[g["n"] >= 10]                    # muestra mínima por hueco
            todos.extend(g["epa"].tolist())        # para la escala de color
            if team not in g.index:
                out[h] = None
                continue
            # rank 1 = mejor: más EPA en ataque, menos EPA permitido en defensa
            orden = g.sort_values("epa", ascending=not es_off)
            out[h] = dict(epa=float(g.loc[team, "epa"]),
                          n=int(g.loc[team, "n"]),
                          rank=list(orden.index).index(team) + 1,
                          n_teams=len(orden),
                          lg=float(r[r["hueco"] == h]["epa"].mean()))
        if not any(v for v in out.values()):
            return None, None
        # Escala de color por EPA, no por ranking: con el ranking, un -0.07 y un
        # -0.09 salían de colores distintos solo porque la liga entera defiende
        # bien ese hueco y los puestos se separan mucho con valores casi
        # iguales. Percentiles 5-95 para que un outlier no aplaste el resto.
        # Va aparte y no dentro de `out`: colarla ahí obliga a que cada
        # consumidor sepa filtrarla, y es una trampa fácil de pisar.
        escala = ((float(np.percentile(todos, 5)),
                   float(np.percentile(todos, 95))) if todos else None)
        return out, escala
    except Exception as e:
        print(f"  Aviso: sin carrera por hueco ({type(e).__name__})")
        return None, None


def dibujar_huecos(ax, es_off, y0, y1, x0, x1):
    """Franja de 7 celdas: los huecos van LE→RE, o sea de izquierda a derecha
    igual que la línea ofensiva, así que la franja se lee como el campo."""
    datos, escala = carrera_por_hueco(es_off)
    titulo = "CARRERA POR HUECO" if es_off else "CARRERA PERMITIDA POR HUECO"
    ax.text(x0 + 1.3, y1 - 1.0, titulo, ha="left", va="center", fontsize=8.5,
            fontweight="bold", color="#8fa1c0", zorder=3)
    if not datos:
        ax.text(x0 + 1.3, (y0 + y1) / 2 - 1.2, "Sin muestra por hueco",
                ha="left", va="center", fontsize=7.5, color="#777777", zorder=3)
        return

    # El título va en su propia línea: pegado a las celdas chocaba con "LE"
    orden  = GAP_ORDER if es_off else GAP_ORDER_DEF
    etiqs  = GAP_CORTO if es_off else GAP_DEF_LABEL
    w = (x1 - x0 - 2.6) / len(orden)
    y_cell1, y_cell0 = y1 - 4.8, y0 + 2.6
    for i, h in enumerate(orden):
        cx0 = x0 + 1.3 + i * w
        d = datos.get(h)
        if not d:
            ax.add_patch(plt.Rectangle((cx0, y_cell0), w - 0.25,
                                       y_cell1 - y_cell0, facecolor="#1e2430",
                                       edgecolor=BG, linewidth=0.6, zorder=1))
            ax.text(cx0 + w / 2, (y_cell0 + y_cell1) / 2, "—", ha="center",
                    va="center", color="#444444", fontsize=8, zorder=2)
        else:
            bgc = color_epa(d["epa"], escala, es_off)
            ax.add_patch(plt.Rectangle((cx0, y_cell0), w - 0.25,
                                       y_cell1 - y_cell0, facecolor=bgc,
                                       edgecolor=BG, linewidth=0.6, zorder=1))
            lum = 0.299 * bgc[0] + 0.587 * bgc[1] + 0.114 * bgc[2]
            txt = "#0a0e13" if lum > 0.45 else FG
            ax.text(cx0 + w / 2, (y_cell0 + y_cell1) / 2 + 0.75,
                    f"{d['epa']:+.2f}", ha="center", va="center", color=txt,
                    fontsize=8, fontweight="bold", zorder=2)
            ax.text(cx0 + w / 2, (y_cell0 + y_cell1) / 2 - 0.75,
                    f"#{d['rank']} · n={d['n']}", ha="center", va="center",
                    color=txt, fontsize=6, zorder=2)
        ax.text(cx0 + w / 2, y_cell1 + 1.35, etiqs[h], ha="center",
                va="center", color=FG, fontsize=7.2, fontweight="bold",
                zorder=3, linespacing=1.15)

    ax.text(x0 + 1.3, y0 + 1.0,
            ("EPA por acarreo diseñado  ·  color = EPA contra la liga  ·  "
             "los huecos van de izquierda a derecha como la línea ofensiva"
             if es_off else
             "EPA por acarreo diseñado permitido  ·  color = EPA contra la liga  ·  "
             "visto desde la defensa: el hueco del LT rival lo cubre nuestro DE derecho"),
            ha="left", va="center", fontsize=6.2, color="#666666",
            fontstyle="italic", zorder=3)


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
            # Solo las presiones CLASIFICADAS por origen, no todas: si el KPI
            # suma también las de jugadores que no cruzan con el roster, las
            # cuatro flechas del mini-campo no suman el número de arriba
            # (pasaba en BAL, DAL y SEA) y además no cuadraba con
            # dline_presion_origen, que suma igualmente solo lo clasificado.
            pfr, _ = cargar_pfr("def", SEASON)
            pfr = pfr[pfr["tm"] != "3TM"].copy()
            pfr["prss"] = pd.to_numeric(pfr["prss"], errors="coerce").fillna(0)
            from pbp_loader import cargar_rosters
            ros, _ = cargar_rosters(SEASON)
            por_nombre = {}
            for _, r in ros.iterrows():
                cl = _clasificar_rusher(r.get("depth_chart_position"),
                                        r.get("position"), r.get("weight"))
                if cl:
                    por_nombre.setdefault(_clave_nombre(r.get("full_name")), cl)
            pfr = pfr[pfr["player"].map(_clave_nombre).map(por_nombre).notna()]
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
    es_off = side == "off"
    secs = secciones(side)
    kpi  = kpis(side)
    # Candidatos que NO estan en la columna izquierda pero si en la tarjeta:
    # presion (se fue al bloque de la derecha) y carrera por hueco (franja).
    # Sin esto, la tarjeta mostraba dos cosas que su propio resumen no veia.
    extra = metricas_extra(es_off)
    huecos, _ = carrera_por_hueco(es_off)
    huecos = huecos or {}
    etiqs_h = GAP_CORTO if es_off else GAP_DEF_LABEL
    for h, d in huecos.items():
        if d:
            extra.append(dict(d, label=f"Carrera {etiqs_h[h].replace(chr(10), ' ')}",
                              cat=h, uso=100.0, seccion="HUECOS"))
    pres = presion_como_clave(es_off)
    if pres:
        extra.append(pres)
    fort, debs = claves(secs, extra)

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

    # DOMINA y SUFRE comparten formato: las 3 facetas EN FILA, no apiladas.
    # Antes SUFRE era una lista vertical y quedaban dos secciones de la misma
    # naturaleza dibujadas de forma distinta.
    def banda_claves(y0, y1, titulo, color, items, vacio):
        ax.add_patch(plt.Rectangle((63.2, y0), 35.4, y1 - y0, color=CARD, zorder=1))
        ax.text(64.6, y1 - 1.9, titulo, ha="left", va="center",
                fontsize=9, fontweight="bold", color=color, zorder=3)
        if not items:
            ax.text(64.6, y0 + 3.0, vacio, ha="left", va="center",
                    fontsize=7.5, color="#777777", zorder=3)
            return
        ancho = 35.4 / max(len(items), 1)
        for i, it in enumerate(items):
            cx = 63.2 + ancho * (i + 0.5)
            # Sin truncar: "Carrera Exterior izq" perdía el lado, que es justo
            # lo que distingue el dato. Se encoge la fuente si hace falta.
            etq = it["label"]
            fs  = 8.5 if len(etq) <= 17 else (7.6 if len(etq) <= 22 else 6.9)
            ax.text(cx, y0 + 4.3, etq, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=FG, zorder=3)
            ax.add_patch(plt.Circle((cx - 3.4, y0 + 1.8), 1.15,
                                    color=rank_color(it["rank"], it["n_teams"]), zorder=4))
            ax.text(cx - 3.4, y0 + 1.8, f"{it['rank']}", ha="center",
                    va="center", fontsize=7, fontweight="bold", color="#0a0e13", zorder=5)
            if it["n_teams"] < 30:
                ax.text(cx - 3.4, y0 - 0.1, f"de {it['n_teams']}", ha="center",
                        va="center", fontsize=5.5, color="#777777", zorder=5)
            es_extra = it["seccion"] in ("EXTRA", "HUECOS", "PRESIÓN")
            unidad = "%" if it.get("es_pct") else " EPA"
            ax.text(cx - 1.6, y0 + 1.8,
                    f"{it['epa']:+.2f}{unidad}" if es_extra
                    else f"{it['epa']:+.2f} EPA · {it['uso']:.0f}%",
                    ha="left", va="center", fontsize=6.8, color="#9aa3b5", zorder=3)

    banda_claves(Y0_BANDA, Y1_BANDA, "✓  DONDE DOMINA", "#06d6a0", fort,
                 "Sin facetas top con uso relevante")

    # ── Columna izquierda: pistas de ranking ──────────────────────────────────
    ax.add_patch(plt.Rectangle((1.2, 1.5), 60.6, 80.2, color=CARD, zorder=0))
    x_lbl, x_t0, x_t1, x_val = 14.5, 17.5, 46.0, 47.5

    # La franja de huecos se reserva al pie de la columna; las secciones de
    # ranking se reparten lo que queda. Aprieta las filas de 3.22 a ~2.63, aun
    # por encima del suelo de 2.4 que necesitan para no solaparse.
    H_HUECOS = 11.4
    y0_huecos, y1_huecos = 2.6, 2.6 + H_HUECOS
    dibujar_huecos(ax, es_off, y0_huecos, y1_huecos, 1.2, 61.8)

    # Espaciado dinámico: todo debe caber entre y_top y y_bot pase lo que pase
    y_top, y_bot = 79.5, y1_huecos + 1.6
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

    TITLE_H, GAP = 4.2, 1.6

    # DONDE SUFRE, justo debajo y con el MISMO formato que DOMINA. Como banda
    # ocupa 9.4 en vez de las ~25 de la lista vertical, así que además libera
    # sitio para IDENTIDAD y para el mini-campo de presión.
    y1_suf = 82.9
    y0_suf = y1_suf - (Y1_BANDA - Y0_BANDA)
    banda_claves(y0_suf, y1_suf, "✗  DONDE SUFRE", "#d84a4a", debs,
                 "Sin facetas en la cola con uso relevante")

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
